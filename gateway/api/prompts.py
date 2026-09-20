"""Answering Hermes's human-in-the-loop prompts, and cancelling a running turn."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from adapters.hermes import (
    APPROVAL_CHOICE_FOR_APPROVED,
    APPROVAL_CHOICES,
    HermesAdapter,
    HermesError,
)
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)

logger = logging.getLogger(__name__)

prompts_router = APIRouter(tags=["prompts"])

PROMPTS_PATH_PREFIX = "/api/prompts"

_STATUS_OK = "ok"
_STATUS_EXPIRED = "expired"

_RESOLVED_KEY = "resolved"

_INTERRUPT_STATUS = "interrupted"

_MAX_REQUEST_ID_LEN = 128

_ALREADY_RESOLVED_DETAIL = (
    "no prompt is pending for request_id {request_id!r}: it was already "
    "answered, superseded by a newer prompt, or it expired while the card was "
    "on screen. Hermes does not distinguish these, and it is a normal race, "
    "not a failure -- the turn has already moved on."
)


def _validate_request_id(request_id: str) -> str:
    """Reject an obviously-unusable request id before any Hermes round trip."""
    cleaned = request_id.strip()
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="request_id must be the non-empty id carried on the prompt event",
        )
    if len(cleaned) > _MAX_REQUEST_ID_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"request_id must be at most {_MAX_REQUEST_ID_LEN} characters",
        )
    return cleaned


def _already_resolved(request_id: str) -> HTTPException:
    """409 for the answer-arrived-too-late race. Deliberately not a 404."""
    return HTTPException(
        status_code=409, detail=_ALREADY_RESOLVED_DETAIL.format(request_id=request_id)
    )


def _unrecognized_upstream_answer(method: str, answer: Any) -> HTTPException:
    """502 for a reply this gateway cannot read as success *or* as expired."""
    return HTTPException(
        status_code=502,
        detail=(
            f"Hermes answered {method} with something this gateway does not "
            f"recognize ({answer!r}); the prompt may or may not have been "
            "resolved, so it is not being reported as answered"
        ),
    )


def _upstream_failure(exc: HermesError, *, redact: str | None = None) -> HTTPException:
    """Map a `HermesError` from a request-id-keyed responder onto a 502."""
    return HTTPException(status_code=502, detail=_redacted(str(exc), redact))


def _redacted(text: str, secret: str | None) -> str:
    """`text` with every occurrence of `secret` replaced by a marker."""
    if not secret:
        return text
    return text.replace(secret, "[redacted]")


def _respond_status(result: Any, method: str, request_id: str, *, redact: str | None = None) -> str:
    """Read the `{"status": ...}` reply shared by clarify/sudo/secret.respond."""
    status = result.get("status") if isinstance(result, dict) else None
    if isinstance(status, str):
        cleaned = status.strip().lower()
        if cleaned == _STATUS_OK:
            return _STATUS_OK
        if cleaned == _STATUS_EXPIRED:
            raise _already_resolved(request_id)
    raise _unrecognized_upstream_answer(method, _redacted(repr(status), redact))


_RESOLUTION_ANSWERED = "answered"
_RESOLVED_BY_APP = "app"


def _record_resolution(
    app_state: Any,
    *,
    event_type: str,
    request_id: str,
    profile: str,
    stored_id: str | None = None,
    **detail: Any,
) -> bool:
    """Append `<kind>.resolved` to the open run this answer belongs to."""
    recorder = getattr(app_state, "run_recorder", None)
    if recorder is None:
        return False
    try:
        session_id = stored_id or recorder.stored_id_for_request(request_id, profile)
        if session_id is None:
            logger.info(
                "%s %s answered with no open run to record it on (profile=%r)",
                event_type,
                request_id,
                profile,
            )
            return False
        payload = {
            "request_id": request_id,
            "resolution": _RESOLUTION_ANSWERED,
            "by": _RESOLVED_BY_APP,
            **detail,
        }
        return bool(recorder.record_synthetic(session_id, event_type, payload, profile))
    except Exception:
        logger.warning(
            "could not record %s for %s; the answer landed, the ledger has a gap",
            event_type,
            request_id,
            exc_info=True,
        )
        return False


class ApprovalResponse(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/approvals/{request_id}`."""

    model_config = ConfigDict(extra="forbid")

    choice: str | None = None
    approved: bool | None = None

    @field_validator("choice")
    @classmethod
    def _known_choice(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in APPROVAL_CHOICES:
            raise ValueError(
                f"choice must be one of {list(APPROVAL_CHOICES)}; Hermes treats "
                "an unrecognized choice as 'deny', so an unchecked one here "
                "would silently deny the owner's approval"
            )
        return cleaned

    @model_validator(mode="after")
    def _exactly_one_form(self) -> ApprovalResponse:
        if (self.choice is None) == (self.approved is None):
            raise ValueError(
                "send exactly one of 'choice' (one of "
                f"{list(APPROVAL_CHOICES)}) or 'approved' (a boolean, which "
                "maps to 'once'/'deny' and cannot express 'session' or 'always')"
            )
        return self

    def resolved_choice(self) -> str:
        """The single `choice` string to put on the wire."""
        if self.choice is not None:
            return self.choice
        return APPROVAL_CHOICE_FOR_APPROVED[bool(self.approved)]


class ClarifyResponse(BaseModel):
    """Body for `POST /api/prompts/{request_id}/clarify`."""

    model_config = ConfigDict(extra="forbid")

    answer: str | None = Field(default=None, min_length=1)
    answers: dict[str, str] | None = None

    @field_validator("answer")
    @classmethod
    def _reject_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "answer must contain non-whitespace characters: a blank answer "
                "resumes the agent having told it nothing"
            )
        return value

    @field_validator("answers")
    @classmethod
    def _reject_blank_batch(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("answers must name at least one question")
        for qid, answer in value.items():
            if not qid.strip():
                raise ValueError("every answer must name the question it answers")
            if not answer.strip():
                raise ValueError(
                    f"answer for {qid!r} must contain non-whitespace characters: a blank "
                    "answer resumes the agent having told it nothing"
                )
        return value

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> ClarifyResponse:
        """One form or the other, never both and never neither."""
        if (self.answer is None) == (self.answers is None):
            raise ValueError(
                "send exactly one of `answer` (a single question) or `answers` "
                "(a batch, keyed by question id)"
            )
        return self


class SudoResponse(BaseModel):
    """Body for `POST /api/prompts/{request_id}/sudo` -- **carries a password**."""

    model_config = ConfigDict(extra="forbid")

    password: SecretStr

    @field_validator("password")
    @classmethod
    def _reject_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("password must not be empty")
        return value


class SecretResponse(BaseModel):
    """Body for `POST /api/prompts/{request_id}/secret` -- **carries a secret**."""

    model_config = ConfigDict(extra="forbid")

    value: SecretStr

    @field_validator("value")
    @classmethod
    def _reject_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("value must not be empty")
        return value


_VALUE_BEARING_ERROR_KEYS = ("input", "ctx")


async def scrub_prompt_validation_errors(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """FastAPI's 422 handler, with the rejected value removed on prompt routes."""
    errors: list[dict[str, Any]] = jsonable_encoder(exc.errors())
    if request.url.path.startswith(PROMPTS_PATH_PREFIX):
        errors = [
            {key: value for key, value in error.items() if key not in _VALUE_BEARING_ERROR_KEYS}
            for error in errors
        ]
    return JSONResponse(status_code=422, content={"detail": errors})


@prompts_router.post("/sessions/{stored_session_id}/approvals/{request_id}")
async def respond_to_approval(
    stored_session_id: str,
    request_id: str,
    body: ApprovalResponse,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Answer an `approval.requested` prompt -- allow or refuse a gated action."""
    stored_id = _validate_stored_session_id(stored_session_id)
    rid = _validate_request_id(request_id)
    choice = body.resolved_choice()
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: adapter.approval_respond(live, choice, request_id=rid),
                profile=profile,
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    resolved = result.get(_RESOLVED_KEY) if isinstance(result, dict) else None
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise _unrecognized_upstream_answer("approval.respond", resolved)
    if resolved < 1:
        raise _already_resolved(rid)

    logger.info(
        "approval %s on session %s answered %r (resolved=%d)", rid, stored_id, choice, resolved
    )
    _record_resolution(
        request.app.state,
        event_type="approval.resolved",
        request_id=rid,
        profile=profile,
        stored_id=stored_id,
        decision=choice,
    )
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "request_id": rid,
        "choice": choice,
        "resolved": resolved,
    }


@prompts_router.post("/prompts/{request_id}/clarify")
async def respond_to_clarify(
    request_id: str,
    body: ClarifyResponse,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Answer a clarify prompt: one free-text answer, or a whole batch."""
    rid = _validate_request_id(request_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    try:
        result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: adapter.clarify_respond(rid, body.answer or "", answers=body.answers),
        )
    except HermesError as exc:
        raise _upstream_failure(exc) from exc

    status = _respond_status(result, "clarify.respond", rid)
    logger.info("clarify %s answered", rid)
    _record_resolution(
        request.app.state,
        event_type="clarify.resolved",
        request_id=rid,
        profile=profile,
        answer=body.answer if body.answer is not None else f"{len(body.answers or {})} answers",
    )
    return {"request_id": rid, "status": status}


@prompts_router.post("/prompts/{request_id}/sudo")
async def respond_to_sudo(
    request_id: str,
    body: SudoResponse,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Supply the sudo password a `sudo.requested` prompt asked for."""
    rid = _validate_request_id(request_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    password = body.password.get_secret_value()
    try:
        result = await _with_reconnect(
            request.app.state, adapter, lambda: adapter.sudo_respond(rid, password)
        )
    except HermesError as exc:
        raise _upstream_failure(exc, redact=password) from exc

    status = _respond_status(result, "sudo.respond", rid, redact=password)
    logger.info("sudo prompt %s answered", rid)
    _record_resolution(
        request.app.state, event_type="sudo.resolved", request_id=rid, profile=profile
    )
    return {"request_id": rid, "status": status}


@prompts_router.post("/prompts/{request_id}/secret")
async def respond_to_secret(
    request_id: str,
    body: SecretResponse,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Supply the secret a `secret.requested` prompt asked the user for."""
    rid = _validate_request_id(request_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    value = body.value.get_secret_value()
    try:
        result = await _with_reconnect(
            request.app.state, adapter, lambda: adapter.secret_respond(rid, value)
        )
    except HermesError as exc:
        raise _upstream_failure(exc, redact=value) from exc

    status = _respond_status(result, "secret.respond", rid, redact=value)
    logger.info("secret prompt %s answered", rid)
    _record_resolution(
        request.app.state, event_type="secret.resolved", request_id=rid, profile=profile
    )
    return {"request_id": rid, "status": status}


@prompts_router.post("/sessions/{stored_session_id}/interrupt")
async def interrupt_session(
    stored_session_id: str, request: Request, profile: str = Query(default="default")
) -> dict:
    """Stop the turn currently running on a session -- the cancel button."""
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter, cache, stored_id, adapter.session_interrupt, profile=profile
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    raw_status = result.get("status") if isinstance(result, dict) else None
    status = raw_status.strip() if isinstance(raw_status, str) and raw_status.strip() else "unknown"
    logger.info("interrupt requested for session %s -> %r", stored_id, status)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "interrupt_status": status,
        "interrupt_status_known": status == _INTERRUPT_STATUS,
        "interrupt": result,
    }
