"""The session routes: list, resume, messages, one row, submit a turn, new + file."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import (
    HermesAdapter,
    HermesError,
)
from adapters.hermes.client import SERVER_REQUEST_ID_FIELD
from api.projects import (
    _load_project,
    file_stored_session,
    filed_project_index,
    workspace_db,
)
from api.validators import reject_blank_text, reject_rewind_fields
from config.settings import get_settings
from domain.background_ledger import append_finished_background_results
from domain.chat_store import ChatStore
from domain.event_stream import KNOWN_SUBMIT_STATUSES, SUBMIT_STATUS_UNKNOWN
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _resume_for_live_id,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)
from domain.live_handles import LiveHandleCache
from domain.project_workspace import instructions_file_exists
from domain.project_workspace import workspace_dir as project_workspace_dir
from domain.sandbox_fs import sandbox_fs_for
from domain.tool_result_backfill import attach_captured_results
from domain.transcript import (
    TRANSCRIPT_DETAIL_FULL,
    TRANSCRIPT_DETAIL_LIGHT,
    _find_transcript_row,
    _project_message,
    _project_transcript,
    _transcript,
)

logger = logging.getLogger(__name__)

sessions_router = APIRouter(tags=["sessions"])


def _attach_captured_tool_results(
    app_state: Any, profile: str, stored_id: str, messages: Any
) -> int:
    """B-156: give a reloaded transcript's tool rows the results the live
    stream had. See `domain/tool_result_backfill.py`. Best-effort by
    contract -- a store failure must never cost the transcript itself."""
    store = getattr(app_state, "chat_store", None)
    if store is None or not isinstance(messages, list):
        return 0
    try:
        captured = store.tool_rows(profile=profile or "default", stored_session_id=stored_id)
        return attach_captured_results(messages, captured)
    except Exception:  # pragma: no cover - defensive
        logger.exception("could not attach captured tool results; transcript served without them")
        return 0


def _finish_transcript(
    app_state: Any, profile: str, stored_id: str, messages: Any, *, light: bool
) -> tuple[int, Any]:
    """The transcript's blocking SQLite reads and projection, for `asyncio.to_thread`.

    Returns `(background rows appended, projected messages)`.
    """
    _attach_captured_tool_results(app_state, profile, stored_id, messages)
    added = append_finished_background_results(app_state, stored_id, messages)
    return added, _project_transcript(messages, light=light)


@sessions_router.get("/sessions")
async def list_sessions(request: Request, profile: str | None = Query(default=None)) -> dict:
    """List Hermes sessions, backed by `HermesAdapter.session_list()`."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    scope: str | None = None
    if profile is not None:
        scope = profile.strip()
        if not scope:
            raise HTTPException(
                status_code=422,
                detail=(
                    "`profile` must be a non-empty Hermes profile name "
                    "(GET /api/profiles lists them); omit it for all sessions"
                ),
            )
    try:
        result = await _with_reconnect(
            request.app.state, adapter, lambda: adapter.session_list(profile=scope)
        )
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _annotate_filing_status(request.app.state, result)
    return result


def _annotate_filing_status(app_state: Any, result: Any) -> None:
    """Stamp `filed` / `project_id` / `project_title` onto each session, in place."""
    if not isinstance(result, dict):
        return
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        return

    stored_ids = [
        row["id"]
        for row in sessions
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
    ]

    factory = getattr(app_state, "db_sessions", None)
    index: dict[str, dict[str, Any]] = {}
    if factory is not None and stored_ids:
        try:
            with factory() as db:
                index = filed_project_index(db, stored_ids)
        except Exception:
            logger.warning(
                "could not read the workspace filing index; serving the session "
                "list with everything marked unfiled",
                exc_info=True,
            )
            index = {}

    for row in sessions:
        if not isinstance(row, dict):
            continue
        stored_id = row.get("id")
        filing = index.get(stored_id) if isinstance(stored_id, str) else None
        row["filed"] = filing is not None
        row["project_id"] = filing["project_id"] if filing else None
        row["project_title"] = filing["project_title"] if filing else None
        row["workspace_session_id"] = filing["workspace_session_id"] if filing else None
        row["archived_at"] = filing.get("archived_at") if filing else None


# Every {stored_session_id} path segment is Hermes's durable id, never a live handle.
@sessions_router.post("/sessions/{stored_session_id}/resume")
async def resume_session(
    stored_session_id: str,
    request: Request,
    detail: Literal["full", "light"] = Query(default=TRANSCRIPT_DETAIL_FULL),
    profile: str = Query(default="default"),
) -> dict:
    """Resume a saved Hermes session and hand back both of its ids."""
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache: LiveHandleCache = resolve_live_handle_cache(request.app.state, profile)
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _resume_for_live_id(adapter, stored_id, cache, profile=profile),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    message_count, messages = _transcript(result, "message_count")
    # Hermes keeps a tool call's args but not its result; captured rows restore them.
    # A background turn leaves no Hermes transcript row; the ledger is the only record.
    added, projected = await asyncio.to_thread(
        _finish_transcript, request.app.state, profile, stored_id, messages,
        light=detail == TRANSCRIPT_DETAIL_LIGHT,
    )
    message_count += added
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "message_count": message_count,
        "messages": projected,
        "pending_approval": _pending_prompt(result, "approval"),
        "pending_clarify": _pending_prompt(result, "clarify"),
    }


def pending_prompt_from_open_requests(result: Any, method: str) -> dict | None:
    """The oldest unanswered `method` request on a resumed session, or `None`."""
    if not isinstance(result, dict):
        return None
    for entry in result.get("open_requests") or []:
        if not isinstance(entry, dict) or entry.get("method") != method:
            continue
        params = entry.get("params")
        payload = dict(params) if isinstance(params, dict) else {}
        srq_id = entry.get("id")
        if not isinstance(srq_id, str) or not srq_id:
            continue
        payload[SERVER_REQUEST_ID_FIELD] = srq_id
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            payload["request_id"] = srq_id
        return payload
    return None


# Newer Hermes replays a clarify only in open_requests; older builds still send the key.
def _pending_prompt(result: Any, method: str) -> dict | None:
    """Hermes's own `pending_<method>` key when this build still sends one,
    the `open_requests` replay otherwise.
    """
    if isinstance(result, dict):
        own = result.get(f"pending_{method}")
        if own:
            return own
    return pending_prompt_from_open_requests(result, method)


@sessions_router.get("/sessions/{stored_session_id}/messages")
async def session_messages(
    stored_session_id: str,
    request: Request,
    detail: Literal["full", "light"] = Query(default=TRANSCRIPT_DETAIL_FULL),
    profile: str = Query(default="default"),
) -> dict:
    """Message history for a session -- `ARCHITECTURE.md` §11.1's
    `GET /api/sessions/{id}/messages`.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache: LiveHandleCache = resolve_live_handle_cache(request.app.state, profile)
    try:
        live_id, history = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter, cache, stored_id, adapter.session_history, profile=profile
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    count, messages = _transcript(history, "count")
    added, projected = await asyncio.to_thread(
        _finish_transcript, request.app.state, profile, stored_id, messages,
        light=detail == TRANSCRIPT_DETAIL_LIGHT,
    )
    count += added
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "count": count,
        "messages": projected,
    }


@sessions_router.get("/sessions/{stored_session_id}/messages/{row_id}")
async def session_message_detail(
    stored_session_id: str,
    row_id: str,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """One transcript row, at full fidelity -- B-86's on-demand body fetch."""
    stored_id = _validate_stored_session_id(stored_session_id)
    wanted_row_id = row_id.strip()
    if not wanted_row_id:
        raise HTTPException(
            status_code=422,
            detail="row_id must be a non-empty Hermes transcript row id",
        )
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache: LiveHandleCache = resolve_live_handle_cache(request.app.state, profile)
    try:
        _live_id, history = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter, cache, stored_id, adapter.session_history, profile=profile
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    _count, messages = _transcript(history, "count")
    row = _find_transcript_row(messages, wanted_row_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"session {stored_id!r} has no transcript row with row_id "
                f"{wanted_row_id!r} (tool rows carry no row_id and are not "
                f"addressable here)"
            ),
        )
    return {"message": _project_message(row, light=False)}


class TurnSubmission(BaseModel):
    """Request body for `POST /api/sessions/{stored_session_id}/turns`."""

    # Closed schema: no rewind/truncate key can reach prompt.submit from the wire.
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only text with a 422, before Hermes is touched."""
        return reject_blank_text(
            value,
            message=(
                "text must contain non-whitespace characters: a blank turn "
                "would still start a real turn on the Hermes session"
            ),
        )

    @model_validator(mode="before")
    @classmethod
    def _reject_rewind_fields(cls, data: Any) -> Any:
        return reject_rewind_fields(data, submission="turn")


# Hermes returns four different 200 outcomes; an unknown status must never read as sent.
def _normalize_submit_status(acknowledgement: Any) -> tuple[str, bool]:
    """Pull the real outcome out of a `prompt.submit` acknowledgement (B-05o)."""
    raw = acknowledgement.get("status") if isinstance(acknowledgement, dict) else None
    if isinstance(raw, str) and raw.strip():
        status = raw.strip()
        return status, status in KNOWN_SUBMIT_STATUSES
    return SUBMIT_STATUS_UNKNOWN, False


@sessions_router.post("/sessions/{stored_session_id}/turns")
async def submit_turn(
    stored_session_id: str,
    submission: TurnSubmission,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Submit one user turn to a saved session -- the chat send path."""
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache: LiveHandleCache = resolve_live_handle_cache(request.app.state, profile)
    chat_store: ChatStore | None = getattr(request.app.state, "chat_store", None)

    chat_message_id: str | None = None
    if chat_store is not None:
        try:
            chat_message_id = chat_store.capture_submitted_user_row(
                profile=profile, stored_session_id=stored_id, text=submission.text
            )
        except Exception:
            logger.exception("could not write the submitted row to ChatStore; sending anyway")
            chat_message_id = None

    try:
        live_id, acknowledgement = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: adapter.prompt_submit(live, submission.text),
                profile=profile,
            ),
        )
    except HermesError as exc:
        if chat_message_id is not None:
            with contextlib.suppress(Exception):
                chat_store.discard_row(chat_message_id)
        raise _http_error_from_hermes(exc, stored_id) from exc

    status, known = _normalize_submit_status(acknowledgement)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "submit_status": status,
        "submit_status_known": known,
        "submit": acknowledgement,
        "chat_message_id": chat_message_id,
    }


class NewFiledSession(BaseModel):
    """Body for `POST /api/projects/{project_id}/sessions/new` (P2-17)."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    # Hermes persists a session only once it has content, so a first turn is required.
    first_message: str = Field(min_length=1)
    profile: str = Field(default="default")

    @field_validator("title", "first_message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace characters")
        return value


@sessions_router.post("/projects/{project_id}/sessions/new", status_code=201)
async def create_and_file_session(
    project_id: str,
    body: NewFiledSession,
    request: Request,
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """The "+ New session" pill (P2-17): mint a brand-new Hermes session, filed
    to this project immediately.
    """
    project = _load_project(db, project_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, body.profile)
    cache: LiveHandleCache = resolve_live_handle_cache(request.app.state, body.profile)

    settings = get_settings()
    create_kwargs: dict[str, Any] = {}
    # Honoured since Hermes 0.21.3; without it the session lands on the default profile.
    if body.profile and body.profile != "default":
        create_kwargs["profile"] = body.profile
    if await instructions_file_exists(
        sandbox_fs_for(request.app.state, adapter), settings, project.id
    ):
        create_kwargs["cwd"] = project_workspace_dir(settings, project.id)

    try:
        create_result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: adapter.session_create(body.title, **create_kwargs),
        )
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    stored_id = create_result.get("stored_session_id") if isinstance(create_result, dict) else None
    live_id = create_result.get("session_id") if isinstance(create_result, dict) else None
    if (
        not isinstance(stored_id, str)
        or not stored_id
        or not isinstance(live_id, str)
        or not live_id
    ):
        raise HTTPException(
            status_code=502,
            detail=f"session.create returned no usable ids: {create_result!r}",
        )

    cache.put(stored_id, live_id)

    try:
        acknowledgement = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: adapter.prompt_submit(live_id, body.first_message),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    filing_row, _created = file_stored_session(
        db, project, stored_id, body.title, profile=body.profile
    )

    status, known = _normalize_submit_status(acknowledgement)
    return {
        "created": True,
        "live_session_id": live_id,
        "submit_status": status,
        "submit_status_known": known,
        **filing_row,
    }
