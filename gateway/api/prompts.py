"""Answering Hermes's human-in-the-loop prompts, and cancelling a running turn.

This is B-37. The gateway already *forwards* `approval.requested` /
`clarify.requested` / `sudo.requested` / `secret.requested` to the phone and
`HermesAdapter` already knows how to answer each one -- but until this module
existed no HTTP route reached any of them, so a prompt that arrived on the
phone was unanswerable and the turn simply blocked until Hermes gave up. A
long or runaway turn could not be stopped either.

Five routes, and they do not all have the same shape, because the four
upstream RPCs do not (`docs/PROTOCOL_VERIFIED.md`, "The four human-in-the-loop
prompts"):

| Route | Upstream | Keyed on | Verified? |
|---|---|---|---|
| `POST /api/sessions/{stored}/approvals/{request_id}` | `approval.respond` | **session** (live handle) + request_id | live |
| `POST /api/prompts/{request_id}/clarify` | `clarify.respond` | request_id | live |
| `POST /api/prompts/{request_id}/sudo` | `sudo.respond` | request_id | **source-read only** |
| `POST /api/prompts/{request_id}/secret` | `secret.respond` | request_id | **source-read only** |
| `POST /api/sessions/{stored}/interrupt` | `session.interrupt` | session (live handle) | live |

Three properties this module is responsible for:

**A 200 must mean the answer landed.** Hermes reports "there was nothing to
answer" as an ordinary success -- `{"resolved": 0}` from `approval.respond`,
`{"status": "expired"}` from the other three. Both are the *normal* outcome of
a race the phone cannot avoid: the owner taps just after Hermes gave up
waiting, or a second device answered first. That is a 409 here with a plain
explanation, never a silent 200 telling the owner their tap landed when it did
not, and never a 500 -- it is not an error, it is a race.

**Two of these carry a credential, not a decision.** `sudo.request` is not
"approve running sudo", it is "type your sudo password", and `secret.request`
asks the user to supply a secret outright. Both fall under
`docs/ARCHITECTURE.md` §14: the value is ephemeral and request-scoped. It is
never written to the event log, never persisted, never logged, never echoed in
a response body, and never placed in a URL or query string. See
"The secret path" below -- the guarantees are structural, not conventions.

**Stored ids in, live handles resolved per call.** The two session-scoped
routes take the **STORED / durable** session id, exactly like every other
route, and resolve the live handle through the one shared
`_with_live_handle()`. A live handle is process-local to the current Hermes
connection and is never accepted from a client, never persisted, and never
cached beyond that connection (`LiveHandleCache` keys on the adapter's
connection generation). The other three routes need no session at all: Hermes
keys them on `request_id` alone.

## The secret path

`SudoResponse.password` and `SecretResponse.value` are `SecretStr`, so the
model's own `repr()` -- the thing that would show up in a traceback, a log
record built from `%r`, or an error rendered by a framework -- is
`SecretStr('**********')` and not the credential.

Beyond that:

* **Never in a URL.** The value is a JSON body field. `request_id` is the only
  thing in the path, and it is an opaque identifier, not a credential.
* **Never logged.** `HermesAdapter.request()` logs no parameters, and the
  audit line these routes emit names the `request_id` and nothing else --
  enough to prove the prompt was answered, carrying nothing worth stealing.
* **Never persisted.** Neither route emits an event, so the value cannot
  reach the `/ws/events` fan-out. The one thing either route writes to the
  workspace database is the B-189 resolution row below, whose payload is
  built from the `request_id` and fixed strings only -- the value is not in
  scope where that payload is assembled.

## Recording that a prompt was answered (B-189)

An answered prompt produces no `*.resolved` frame on the wire (Hermes emits
one only for a sudo/secret expiry), so before this the run ledger saw a turn
block on `<kind>.requested` and never saw it unblock. Each route, once
Hermes has confirmed the answer landed, asks `RunRecorder.record_synthetic`
to append a `<kind>.resolved` row (`resolution: answered`, `by: app`) to the
session's OPEN run. Best-effort and after the fact: a recorder problem is
logged and the 200 stands, because the answer *did* land. The three
session-less routes find their run through the recorder's own memory of
which open run carried that `request_id` (`stored_id_for_request`).
* **Never echoed.** The success body is `{"request_id", "status"}`. Unlike
  `POST /turns`, Hermes's raw result is deliberately *not* forwarded, and the
  upstream error text is passed through `_redacted()` before it becomes a 502
  detail, so even a Hermes that quoted the value back cannot bounce it to the
  client.
* **Never echoed by a 422 either.** Pydantic puts the rejected value in
  `input` (and sometimes `ctx`) on every validation error, and FastAPI's
  default handler serializes those straight into the response body -- so a
  mistyped secret would come back in the 422. `scrub_prompt_validation_errors`
  is installed on the app for exactly this and strips both keys for every path
  under `PROMPTS_PATH_PREFIX`.
"""

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

# Every request-id-keyed route lives under this prefix, and the validation-error
# scrubber keys off it. `test_prompts.py` asserts the registered routes really
# do, so renaming one without renaming the other is a test failure rather than a
# silently unscrubbed 422 carrying somebody's password.
PROMPTS_PATH_PREFIX = "/api/prompts"

# Hermes's own answers, verified live where marked in PROTOCOL_VERIFIED.md.
# `ok` is the only one that means the value was accepted.
_STATUS_OK = "ok"
# Returned (as a normal result, not an error) for an unknown, stale or
# already-answered request_id. Hermes does this deliberately -- `allow_expired
# =True` -- because a prompt can time out server-side while its card is still
# on the phone's screen.
_STATUS_EXPIRED = "expired"

# The `{"resolved": n}` count `approval.respond` returns. Zero means nothing
# was pending: a no-op, not a success.
_RESOLVED_KEY = "resolved"

# `session.interrupt` -> `{"status": "interrupted"}`, verified live mid-turn.
_INTERRUPT_STATUS = "interrupted"

_MAX_REQUEST_ID_LEN = 128

# What the client is told for the "there was nothing to answer" race. Shared so
# all four responders phrase it the same way and the phone can match on the
# status code alone.
_ALREADY_RESOLVED_DETAIL = (
    "no prompt is pending for request_id {request_id!r}: it was already "
    "answered, superseded by a newer prompt, or it expired while the card was "
    "on screen. Hermes does not distinguish these, and it is a normal race, "
    "not a failure -- the turn has already moved on."
)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _validate_request_id(request_id: str) -> str:
    """Reject an obviously-unusable request id before any Hermes round trip.

    Observed shapes differ per prompt type -- 32 hex for an approval, 8 hex for
    a clarify -- so this deliberately does *not* enforce a format. It only
    refuses what cannot possibly be one: empty/whitespace, or absurdly long.
    """
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
    """409 for the answer-arrived-too-late race. Deliberately not a 404.

    404 would say "there is no such thing", which is a claim this gateway
    cannot make: Hermes returns the identical answer for an id it never had and
    for one it had and has since retired, so the two are indistinguishable from
    here. 409 says what is actually known -- the prompt is not in a state where
    it can be answered -- and the detail spells out both possibilities.
    """
    return HTTPException(
        status_code=409, detail=_ALREADY_RESOLVED_DETAIL.format(request_id=request_id)
    )


def _unrecognized_upstream_answer(method: str, answer: Any) -> HTTPException:
    """502 for a reply this gateway cannot read as success *or* as expired.

    The alternative is guessing, and guessing here means telling the owner
    their approval landed when nothing is known about whether it did. Same
    discipline as `_normalize_submit_status()` in `api/main.py`: an
    unrecognized status is reported, never quietly promoted to success.
    """
    return HTTPException(
        status_code=502,
        detail=(
            f"Hermes answered {method} with something this gateway does not "
            f"recognize ({answer!r}); the prompt may or may not have been "
            "resolved, so it is not being reported as answered"
        ),
    )


def _upstream_failure(exc: HermesError, *, redact: str | None = None) -> HTTPException:
    """Map a `HermesError` from a request-id-keyed responder onto a 502.

    These three RPCs take no session, so the 404 branch of
    `_http_error_from_hermes()` cannot apply -- there is no session id to name.

    `redact` is the credential the call carried, when it carried one. Hermes's
    error messages are not believed to quote parameters back, but "not believed
    to" is not a guarantee worth resting a password on, so the text is scrubbed
    before it is allowed into a response body.
    """
    return HTTPException(status_code=502, detail=_redacted(str(exc), redact))


def _redacted(text: str, secret: str | None) -> str:
    """`text` with every occurrence of `secret` replaced by a marker.

    A no-op when there is no secret (the approval/clarify paths). Never logs,
    never returns the secret, and does nothing clever: a plain replace, so an
    empty or absent secret cannot turn into a match-everything pattern.
    """
    if not secret:
        return text
    return text.replace(secret, "[redacted]")


def _respond_status(result: Any, method: str, request_id: str, *, redact: str | None = None) -> str:
    """Read the `{"status": ...}` reply shared by clarify/sudo/secret.respond.

    Returns `"ok"`; raises 409 for `expired`, 502 for anything else. Never puts
    Hermes's raw result in the exception, so an unexpected shape cannot carry a
    credential back out through the 502 either.
    """
    status = result.get("status") if isinstance(result, dict) else None
    if isinstance(status, str):
        cleaned = status.strip().lower()
        if cleaned == _STATUS_OK:
            return _STATUS_OK
        if cleaned == _STATUS_EXPIRED:
            raise _already_resolved(request_id)
    raise _unrecognized_upstream_answer(method, _redacted(repr(status), redact))


#: `payload.resolution` / `payload.by` on every synthesized row (B-189). The
#: wire's own expiry rows carry `resolution: "expired"` and no `by`.
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
    """Append `<kind>.resolved` to the open run this answer belongs to (B-189).

    `stored_id` is known on the session-keyed approval route; the other three
    resolve it from the recorder's request-id memory. `detail` is what the
    route may say about the answer -- the approval `decision`, the clarify
    `answer` -- and is NEVER a credential: the sudo/secret routes pass none.
    Returns whether a row was written; never raises, since the Hermes call
    that matters has already succeeded by the time this runs.
    """
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


# ----------------------------------------------------------------------
# Request bodies
# ----------------------------------------------------------------------


class ApprovalResponse(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/approvals/{request_id}`.

    **An approval is not a boolean.** Hermes offers four answers -- `once`,
    `session`, `always`, `deny` -- and the two middle ones are what stop the
    owner being asked the same question forever. `choice` is therefore the
    real field.

    `approved` exists only so a client that genuinely can express nothing but
    yes/no still has a correct way to say it: `true` maps to `once` (the
    *narrowest* yes -- a client that cannot say which yes it means must not be
    granted the permanent one) and `false` maps to `deny`. Exactly one of the
    two fields must be present; sending both, or neither, is a 422 rather than
    a silent precedence rule.

    `extra="forbid"`, so nothing else from the wire can reach the adapter.
    """

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
    """Body for `POST /api/prompts/{request_id}/clarify`.

    The answer is **free text**, not an index into the event's `choices`: the
    list Hermes puts on `clarify.request` is presentation (it appends things
    like " (Recommended)"), so echoing a choice string back verbatim is the
    client's job and any other text is equally valid.

    Blank is refused for the same reason `POST /turns` refuses it (B-24): a
    whitespace answer resumes the agent with nothing, which looks to it like
    the user said nothing at all.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "answer must contain non-whitespace characters: a blank answer "
                "resumes the agent having told it nothing"
            )
        return value


class SudoResponse(BaseModel):
    """Body for `POST /api/prompts/{request_id}/sudo` -- **carries a password**.

    `sudo.request` asks the user to *type their sudo password*; it is not an
    approve/deny. `password` is a `SecretStr` so this model's `repr()` cannot
    leak it into a traceback or a log record, and §14 handling applies to it
    end to end -- see this module's docstring.

    An empty password is refused rather than forwarded: Hermes would accept it
    as the answer, and "the user submitted nothing" and "the user submitted an
    empty password" are not the same event. Cancelling/denying a sudo prompt is
    a distinct upstream path that has not been verified and is not exposed here.
    """

    model_config = ConfigDict(extra="forbid")

    password: SecretStr

    @field_validator("password")
    @classmethod
    def _reject_empty(cls, value: SecretStr) -> SecretStr:
        # Note what this does NOT do: it never interpolates the value into the
        # message. A validator that said "password {value!r} is invalid" would
        # put the credential straight into a 422 body.
        if not value.get_secret_value():
            raise ValueError("password must not be empty")
        return value


class SecretResponse(BaseModel):
    """Body for `POST /api/prompts/{request_id}/secret` -- **carries a secret**.

    `secret.request` asks the user to supply a credential the agent needs. Same
    `SecretStr` handling and same §14 rules as `SudoResponse`.
    """

    model_config = ConfigDict(extra="forbid")

    value: SecretStr

    @field_validator("value")
    @classmethod
    def _reject_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("value must not be empty")
        return value


# ----------------------------------------------------------------------
# 422 scrubbing -- installed on the app in `api/main.py`
# ----------------------------------------------------------------------

# Pydantic attaches the rejected input to every validation error, and FastAPI's
# default handler serializes it into the 422 body. For `/api/prompts/*` that
# input can be the owner's password.
_VALUE_BEARING_ERROR_KEYS = ("input", "ctx")


async def scrub_prompt_validation_errors(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """FastAPI's 422 handler, with the rejected value removed on prompt routes.

    Behaviour is unchanged for every other path in the service -- `input` is
    genuinely useful when debugging a malformed projects or turns request, and
    quietly changing the error shape everywhere would be a worse trade than
    scrubbing the two routes that need it.

    Under `PROMPTS_PATH_PREFIX` the `input` and `ctx` keys are dropped from
    every error entry. `loc`, `msg` and `type` remain, so a client still learns
    *which* field was wrong and *why* -- it just is not told its own secret
    back. The scrub is by path prefix rather than per-route because a
    `RequestValidationError` is raised before the route function exists to ask.
    """
    errors: list[dict[str, Any]] = jsonable_encoder(exc.errors())
    if request.url.path.startswith(PROMPTS_PATH_PREFIX):
        errors = [
            {key: value for key, value in error.items() if key not in _VALUE_BEARING_ERROR_KEYS}
            for error in errors
        ]
    return JSONResponse(status_code=422, content={"detail": errors})


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


@prompts_router.post("/sessions/{stored_session_id}/approvals/{request_id}")
async def respond_to_approval(
    stored_session_id: str,
    request_id: str,
    body: ApprovalResponse,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Answer an `approval.requested` prompt -- allow or refuse a gated action.

    `{stored_session_id}` is the **STORED / durable** id, the same one the
    event stream stamps on every frame as `_stored_session_id`, so the phone
    already has it when the approval card appears. `{request_id}` is the id
    carried on the `approval.request` payload.

    Unlike the other three prompts, `approval.respond` is *session*-keyed:
    Hermes resolves the session before it looks at the choice, so a request id
    on its own fails `[4001] session not found`. The live handle is resolved
    here through the same `_with_live_handle()` every other session route uses
    -- from the current connection's cache when known, otherwise a fresh
    `session.resume` (which works fine on a session with an approval
    outstanding; the resume reply even carries `pending_approval`). It is never
    accepted from the client and never persisted.

    `request_id` is always sent even though Hermes would default to the oldest
    pending approval without it: the phone may be answering a card that has
    since been superseded, and resolving the wrong one would allow an action
    the owner never saw.

    Body: `{"choice": "once"|"session"|"always"|"deny"}`, or `{"approved":
    true|false}` for a client that can only express yes/no (mapping to
    `once`/`deny`). Exactly one of the two.

    Responses:

    * **200** -- `resolved` is Hermes's own count and is `>= 1`. The approval
      landed.
    * **409** -- Hermes answered `{"resolved": 0}`: nothing was pending under
      that id. Normal race, not an error (see `_already_resolved`).
    * **404** -- no session with that stored id.
    * **502** -- any other Hermes failure, or an answer this gateway cannot
      read.

    Nothing announces an approval's resolution on the event stream -- Hermes
    emits no `approval.resolved` at all -- so this response is the only signal
    that it landed, and a second device holding the same card open must
    reconcile from `session.resume`'s `pending_approval`.
    """
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
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    resolved = result.get(_RESOLVED_KEY) if isinstance(result, dict) else None
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise _unrecognized_upstream_answer("approval.respond", resolved)
    if resolved < 1:
        raise _already_resolved(rid)

    # Audit-safe on purpose: an approval is a policy decision, not a
    # credential, and §14 asks for a lightweight audit trail of exactly this.
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
    """Answer a `clarify.requested` prompt with free text.

    Keyed on `request_id` alone -- no session, no live handle. Hermes looks the
    pending request up directly. `?profile=` (B-136) says which *connection*
    to look it up on, since `request_id`'s namespace is per-Hermes-process,
    not global: a prompt raised on `kimi25` is unknown to `default`.

    The answer goes on the wire under `answer`. That matters more than it
    looks: sending it under `response` instead is accepted with
    `{"status": "ok"}` and the clarify tool then completes with
    `user_response: ""`, so the agent resumes having discarded what the owner
    typed. That was B-39, observed live; `HermesAdapter.clarify_respond()` owns
    the correct field name and a test pins it.

    Responses: **200** answered; **409** unknown/stale/already answered
    (Hermes's `{"status": "expired"}`); **502** upstream failure or an
    unrecognized reply.
    """
    rid = _validate_request_id(request_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    try:
        result = await _with_reconnect(
            request.app.state, adapter, lambda: adapter.clarify_respond(rid, body.answer)
        )
    except HermesError as exc:
        raise _upstream_failure(exc) from exc

    status = _respond_status(result, "clarify.respond", rid)
    # The answer itself is the owner's words, not a credential -- but there is
    # no reason to put it in the log either, so this records only that the
    # prompt was answered.
    logger.info("clarify %s answered", rid)
    _record_resolution(
        request.app.state,
        event_type="clarify.resolved",
        request_id=rid,
        profile=profile,
        answer=body.answer,
    )
    return {"request_id": rid, "status": status}


@prompts_router.post("/prompts/{request_id}/sudo")
async def respond_to_sudo(
    request_id: str,
    body: SudoResponse,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Supply the sudo password a `sudo.requested` prompt asked for.

    **This is not an approve/deny.** `sudo.request` asks the user to type their
    sudo password, so the body carries a real credential and every §14 rule in
    this module's docstring applies: not logged, not persisted, not in an
    event, not in the URL, not echoed back, and scrubbed out of a 422.

    **The wire field (`password`) is source-read, not measured.** Provoking a
    real `sudo.request` means producing a real credential on the owner's
    machine, so it was deliberately never captured live -- see
    `docs/PROTOCOL_VERIFIED.md`, "`sudo.request` / `secret.request` -- NOT
    captured, and why". B-39 is the standing warning about what that costs:
    a wrong field name here would come back `{"status": "ok"}` with the
    password silently dropped. Verify against a live instance with a throwaway
    credential before trusting a 200 from this route.

    A matching `sudo.expire` event carries this same `request_id` and clears
    only this prompt -- never every pending prompt.

    Responses: **200** `{"request_id", "status": "ok"}` and nothing else --
    the raw Hermes result is deliberately not forwarded; **409** expired or
    already answered; **502** upstream failure (with the password scrubbed out
    of the message).
    """
    rid = _validate_request_id(request_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    # Held in a local for the length of one request and nothing longer. It is
    # not stored on `request.state`, not returned, and not closed over by
    # anything that outlives the call.
    password = body.password.get_secret_value()
    try:
        result = await _with_reconnect(
            request.app.state, adapter, lambda: adapter.sudo_respond(rid, password)
        )
    except HermesError as exc:
        raise _upstream_failure(exc, redact=password) from exc

    status = _respond_status(result, "sudo.respond", rid, redact=password)
    # request_id only. Enough to prove the prompt was answered; carries nothing.
    logger.info("sudo prompt %s answered", rid)
    # The fact, never the value: `_record_resolution` is handed the request
    # id and nothing else from this scope.
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
    """Supply the secret a `secret.requested` prompt asked the user for.

    Same handling as the sudo route above in every respect -- the value is a
    real credential under `docs/ARCHITECTURE.md` §14 and is ephemeral and
    request-scoped.

    **The wire field (`value`) is source-read, not measured**, for the same
    reason: answering one live requires a real secret. Hermes's
    `{"status": "expired"}` reply for an unknown `request_id` *was* confirmed
    live, using an obviously-fake placeholder.

    A matching `secret.expire` event carries this same `request_id` and clears
    only this prompt.

    Responses: **200** `{"request_id", "status": "ok"}`; **409** expired or
    already answered; **502** upstream failure (value scrubbed).
    """
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
    """Stop the turn currently running on a session -- the cancel button.

    `{stored_session_id}` is the **STORED / durable** id. `session.interrupt`
    needs the **LIVE** handle, which is resolved here through the shared
    `_with_live_handle()`: cache hit on the current Hermes connection, else a
    `session.resume`. A handle from an earlier connection is unreachable by
    construction (`LiveHandleCache` is keyed on the adapter's connection
    generation) and one Hermes has forgotten self-heals with a single
    re-resolve. No live handle is ever accepted from the client or persisted --
    `[4001] session not found` is exactly what a stored id passed here would
    get, which is the recurring bug this route must not reintroduce.

    Verified live 2026-08-30, mid-turn while tokens were streaming:
    `{"session_id": <LIVE>}` -> `{"status": "interrupted"}`.

    Takes no body -- there is nothing to say beyond "stop".

    `ARCHITECTURE.md` §11.1 sketches this as `POST /api/turns/{id}/interrupt`.
    It is session-scoped instead because that is what Hermes implements: there
    is no turn id in the protocol, and a session runs one turn at a time.

    Responses: **200** always carries `interrupt_status` (Hermes's own string,
    verbatim) plus `interrupt_status_known`, which is False for anything other
    than `interrupted` -- the same rule `POST /turns` follows, so an
    unrecognized answer is reported rather than read as success. **404** for an
    unknown stored id, **502** for any other Hermes failure.

    Interrupting a session with nothing running has not been measured; whatever
    Hermes answers is reported as-is rather than guessed at.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(adapter, cache, stored_id, adapter.session_interrupt),
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
