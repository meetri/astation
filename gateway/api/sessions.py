"""The session routes: list, resume, messages, one row, submit a turn, new + file.

Moved out of `api/main.py` (CLEANUP_PLAN step 3.2) as a router included from
there like every other surface. `{stored_session_id}` on every path here is
the Hermes **STORED / durable** id (`20260829_182532_991e3f`), never a live
handle and never a workspace primary key -- see `CLAUDE.md`, "Hermes has two
session id spaces". Live handles are resolved per request through
`LiveHandleCache` (`domain/hermes_runtime.py`'s `_with_live_handle`).

The transcript projection the read routes apply is `domain/transcript.py`
; the `prompt.submit` outcome vocabulary is
`domain/event_stream.py` (B-05o).
"""

from __future__ import annotations

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


@sessions_router.get("/sessions")
async def list_sessions(request: Request, profile: str | None = Query(default=None)) -> dict:
    """List Hermes sessions, backed by `HermesAdapter.session_list()`.

    `?profile=<name>` scopes the listing to one Hermes profile -- the operator's
    "agent"/"channel" (`GET /api/profiles` lists them). Verified live
    2026-09-01 (PV "Profiles"): unscoped returned 130 sessions and
    `{"profile": "mlx"}` returned 8, so the filter is genuinely honoured
    upstream. Omitted (the default) is the unscoped listing below, unchanged.
    A blank value is a 422 rather than a silently-unscoped list; an unknown
    profile name is whatever Hermes answers (measured: an empty-ish list, not
    an error), forwarded as-is.

    This is a **lens, not a picker**: sessions cannot be created into a
    profile or moved between profiles (`session.create` accepts `profile` and
    ignores it -- measured), so filtering is the whole of what is honest here.
    See `api/instance.py`.

    **This is the All-sessions view and it stays exactly what it was.** Hermes
    owns the sessions; the workspace is an index over them, so every session on
    the instance appears here whether it has been filed into a project or not.
    Filing is gradual and optional, and nothing a user does in the workspace can
    make a session unreachable from this list. The operator had 47 sessions the day
    projects arrived and all 47 are still here.

    What is added -- and it is only ever *added*, no key was removed, renamed or
    reordered -- is the filing status of each session, so the UI can show "in
    project X" without a second round trip:

    | Key | Meaning |
    |---|---|
    | `filed` | whether a workspace `Session` row references this stored id |
    | `project_id` / `project_title` | which project, or `null` when unfiled |
    | `workspace_session_id` | our own `sess_...` row id, or `null`. Never an id Hermes knows |

    The annotation is **best-effort and never fatal**: if the workspace DB is
    missing, unmigrated or unreadable, every session comes back `filed: false`
    and the list is served anyway. A local storage problem must not be able to
    take away the view of the user's own research sessions.

    Hermes's payload is otherwise forwarded verbatim, and an element that is not
    a dict, or that has no `id`, is left untouched -- the B-34 rule that this
    gateway does not assume a row shape it has not measured.

    Any failure to reach/authenticate against Hermes (unreachable host, bad
    credentials, protocol mismatch, RPC error) is reported as a clean 502
    with the adapter's own error message -- never an unhandled crash or a
    bare 500.

    Reads its adapter from `request.app.state`, like every other route: the
    module-global `app` is the same object today, but a route that reaches for
    it directly stops working the moment this router is mounted anywhere else
.
    """
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
    """Stamp `filed` / `project_id` / `project_title` onto each session, in place.

    Best-effort by design (see `list_sessions`). Every failure mode -- no
    engine, no schema, an unreadable file, a payload shaped differently from
    what Hermes sends today -- degrades to "nothing is filed" rather than to an
    error, because this annotation is a convenience and the session list is not.
    """
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
        # P6-3: the user's archive flag, iso `Z` or null. The row stays in this
        # list either way -- archiving never removes a session from All sessions.
        row["archived_at"] = filing.get("archived_at") if filing else None


@sessions_router.post("/sessions/{stored_session_id}/resume")
async def resume_session(
    stored_session_id: str,
    request: Request,
    detail: Literal["full", "light"] = Query(default=TRANSCRIPT_DETAIL_FULL),
    profile: str = Query(default="default"),
) -> dict:
    """Resume a saved Hermes session and hand back both of its ids.

    `{stored_session_id}` is the runtime's **STORED / durable** id (the `id`
    from `GET /api/sessions`, e.g. "20260829_182532_991e3f") -- not a live
    handle and not a workspace primary key.

    Returns `stored_session_id`, the freshly-minted `live_session_id`, and the
    transcript `session.resume` loads in the same call (`message_count` /
    `messages`). The `live_session_id` is what every subsequent live-handle
    call needs (`session.history`, `prompt.submit`, `session.interrupt`), but
    it is **ephemeral**: valid only for the gateway's current Hermes
    connection. Clients should treat it as a short-lived handle and re-resume
    rather than storing it.

    This route always performs a real `session.resume` -- the transcript it
    returns *is* the point of the call, and that transcript only comes back
    from `session.resume` itself. It seeds the gateway's live-handle cache on
    the way through, so the message-fetch and send that typically follow it
    are free of a second transcript download.

    `messages` is forwarded element for element, in Hermes's own order, and no
    row is reshaped: 58% of real transcript rows are tool calls with no `text`,
    no `row_id` and no `timestamp`. See `_transcript()`. The one thing
    that changes per row is B-86's omission rule -- a `reasoning_content` that
    is byte-identical to `reasoning` is dropped (41% of the measured payload,
    a copy no client has ever read), and `?detail=light` additionally omits
    the reasoning bodies in favour of `has_reasoning` / `reasoning_chars`,
    with the body then reachable one row at a time via
    `GET /sessions/{id}/messages/{row_id}`. `args` is never omitted -- a tool
    row carries no `row_id`, so there would be no way to fetch it back. See
    `_project_message()`. Without the parameter this route answers exactly
    what it always did, minus the duplicate.

    Everything else in the response is untouched by `detail`:
    `stored_session_id`, `live_session_id`, `message_count`,
    `pending_approval` and `pending_clarify` are load-bearing on both settings.

    **`pending_approval` is part of the response schema (P2-5/P2-9g).**
    Measured (PV "Reconnect replay: a pending approval is on the resumed
    session"): `session.resume` on a session with an approval outstanding
    carries `pending_approval` holding exactly the `approval.request`
    payload, `choices` included -- it is how a client that reopens the app
    finds out it owes an answer, since nothing re-emits the approval event.
    This route used to build its response from four hand-picked keys and
    silently *stripped* it, which left the app unable to restore the
    approval card after being backgrounded. It is now forwarded verbatim;
    the key is always present, `null` when Hermes sent none (an explicit
    nothing, same convention as `_stored_session_id`).

    **`pending_clarify` is part of the response schema too (P2-17/B-74).**
    Hermes keeps the identical replay contract for a pending clarify as for
    a pending approval -- live-probed 2026-09-01
    (`.scratch/polish2/p2_17/probe_clarify_resume.py`, PV "Reconnect replay:
    a pending approval is on the resumed session"): a session left with an
    unanswered `clarify.request` and then resumed carries a top-level
    `pending_clarify` key holding exactly the `clarify.request` payload
    (`question`, `choices`, `request_id`). Forwarded verbatim, same
    convention as `pending_approval` -- always present, `null` when Hermes
    sent none.

    Errors are structured: 404 for an unknown stored id, 502 for any other
    Hermes failure (unreachable, auth, protocol) -- never a bare 500.
    """
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
    # Hermes keeps a call's args but never its result; the chat store
    # captured the live `tool.completed` frames, so a reload gets them back.
    _attach_captured_tool_results(request.app.state, profile, stored_id, messages)
    # P2-1: Hermes writes NO transcript row for a background turn, so finished
    # background results exist only in the gateway's ledger. Appended here (and
    # in GET /messages) as synthesized rows so a reload still shows the task
    # ran -- the live-stream injection alone only covers clients subscribed at
    # the instant the completion fired. Best-effort; the count stays equal to
    # what the list actually holds.
    message_count += append_finished_background_results(request.app.state, stored_id, messages)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "message_count": message_count,
        "messages": _project_transcript(messages, light=detail == TRANSCRIPT_DETAIL_LIGHT),
        # Verbatim, never reshaped: the app seeds its PendingPromptStore from
        # this (P2-8), and the measured payload is the `approval.request`
        # event's own shape, `choices` included.
        "pending_approval": _pending_prompt(result, "approval"),
        # Same treatment for a pending clarify (P2-17/B-74): verbatim
        # `clarify.request` payload shape, forwarded unreshaped.
        "pending_clarify": _pending_prompt(result, "clarify"),
    }


def pending_prompt_from_open_requests(result: Any, method: str) -> dict | None:
    """The oldest unanswered `method` request on a resumed session, or `None`.

    Hermes 0.21.3 stopped replaying a pending prompt under its own key
    and replays every unanswered server -> client request together instead:

        "open_requests": [{"id": "srq-<12 hex>", "method": "clarify",
                           "params": {"session_id": ..., "questions": [...]}}]

    `pending_clarify` is simply gone from that build's resume payload, and
    reopening a session with a question outstanding is the ONE way a client
    that was backgrounded ever learns it owes an answer -- nothing re-emits
    the request. So this reshapes the entry back into the payload shape the
    rest of this system already speaks, exactly as the live frame is
    reshaped in `HermesAdapter._dispatch_server_request`: `request_id` names
    what answers it, and `_srq_id` is the open request's own id.

    Oldest first is Hermes's own order (`server_requests.open_requests`), and
    the oldest is the one blocking the turn.
    """
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


def _pending_prompt(result: Any, method: str) -> dict | None:
    """Hermes's own `pending_<method>` key when this build still sends one,
    the `open_requests` replay otherwise.

    Both, in that order, because 0.21.3 still replays `pending_approval` and
    dropped only `pending_clarify` -- and this repo is shared, so an older
    Hermes must keep working unchanged.
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

    **`{id}` here is the Hermes STORED session id, not a workspace primary
    key.** That is a Phase 0 stand-in and will change: once projects exist
    (Phase 1), this path segment becomes our own `sess_...` `sessions.id`, and
    the Hermes stored id moves behind it as the `sessions.runtime_session_id`
    column (`domain/models.py`). `ARCHITECTURE.md` §21 is explicit that Hermes
    session ids must not become workspace primary keys -- nothing should start
    treating this path parameter as one in the meantime.

    Resolution order is exactly `docs/PROTOCOL_VERIFIED.md`'s: stored id ->
    `session.resume` -> use the returned live handle for `session.history`.
    The handle comes from `LiveHandleCache` when one was already resolved on
    this same Hermes connection, so a repeat fetch costs one `session.history`
    instead of a whole extra transcript download; a handle from any earlier
    connection is unreachable, and one Hermes has forgotten self-heals via a
    single re-resume (see `_with_live_handle`).

    `messages` is forwarded element for element with no row reshaped -- see
    `_transcript()` and B-34 -- under B-86's omission rule, identical to
    `POST /resume`'s: a `reasoning_content` byte-identical to `reasoning` is
    dropped always, and `?detail=light` omits the reasoning bodies in favour of
    `has_reasoning` / `reasoning_chars`, keeping `args` (a tool row has no
    `row_id` to fetch it back by). Both routes share `_project_message()`, so
    there is exactly one place where a row can lose anything. Absent the
    parameter this answers what it always did, minus the duplicate.

    Errors are structured: 404 for an unknown stored id, 502 for any other
    Hermes failure -- never a bare 500.
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
    _attach_captured_tool_results(request.app.state, profile, stored_id, messages)  # B-156
    # P2-1: same injection as POST /resume -- a background turn has no Hermes
    # row, so its result rides in from the ledger or nowhere.
    count += append_finished_background_results(request.app.state, stored_id, messages)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "count": count,
        "messages": _project_transcript(messages, light=detail == TRANSCRIPT_DETAIL_LIGHT),
    }


@sessions_router.get("/sessions/{stored_session_id}/messages/{row_id}")
async def session_message_detail(
    stored_session_id: str,
    row_id: str,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """One transcript row, at full fidelity -- B-86's on-demand body fetch.

    `{"message": {...}}` and nothing else: the single row exactly as Hermes
    stored it, `reasoning` / `args` / `text` / `context` complete, so a client
    that took the cheap `?detail=light` listing can fill in a body when the
    reader actually opens one. B-86's duplicate rule still applies here (a
    `reasoning_content` identical to `reasoning` is dropped); nothing else is
    omitted, whatever the row's shape.

    `{row_id}` is **Hermes's own row identifier**, the `row_id` already on the
    row -- not an index into `messages`, not a workspace key, and never
    anything to do with `truncate_before_row_id`, which rewrites history. This
    route only reads.

    **Not every row is addressable, by Hermes's design.** 58% of real rows are
    tool calls carrying no `row_id` at all (B-34; measured again 2026-09-01 on
    the B-86 session -- all 799 tool rows lacked one, against 525 assistant and
    55 user rows that had one), so an unmatched id gets a 404 that says exactly
    that rather than implying the session is gone. Nothing is lost by it: a
    tool row's `args`, `context` and `name` all ride in the light listing
    already, precisely because they could not be fetched back here.

    The session is resolved the same way `GET /messages` resolves it -- stored
    id -> cached live handle or one `session.resume` -> `session.history`, via
    `_with_live_handle` / `_with_reconnect` -- and it selects the row
    server-side, so the client never re-downloads the transcript to find one.
    Finished background results are deliberately *not* injected: those rows are
    the gateway's own synthesis and carry no `row_id`, so there is nothing here
    to address them by, and their text already arrives whole in the listing.

    Errors are structured: 404 for an unknown stored id or an unknown row,
    502 for any other Hermes failure.
    """
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
    """Request body for `POST /api/sessions/{stored_session_id}/turns`.

    Deliberately a closed schema: `extra="forbid"` means the *only* thing a
    client can put on the wire is `text`. That is the structural guarantee
    that no rewind/truncate parameter can be smuggled in from the network and
    reach `prompt_submit()` -- `truncate_before_row_id`,
    `truncate_before_user_ordinal`, `confirm_truncate` and
    `confirm_empty_truncate` perform a *destructive rewrite* of the user's
    real session history (see `docs/PROTOCOL_VERIFIED.md`, "Rewind / edit
    semantics"). A rewind is a separate, explicit, user-initiated action and
    will need its own endpoint with its own confirmation; it is not something
    an ordinary send may ever do by accident.

    The `model_validator(mode="before")` runs ahead of pydantic's own
    extra-field check purely so the rewind case gets a message that names the
    hazard instead of a generic "extra inputs are not permitted".

    `text` must contain something a person actually typed. `min_length=1`
    alone accepts `"   "` and `"\\n"`, which would submit a blank turn
    to a real research session -- Hermes starts a turn for it and the model
    answers whitespace. The app cannot send one (`ConversationModel.send()`
    trims and `canSend` requires non-empty), but the gateway must not depend
    on its client for that.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only text with a 422, before Hermes is touched.

        Returns the caller's string **verbatim** when it passes (see
        `api.validators.reject_blank_text`): silently rewriting a user's
        message is not this endpoint's job, and the transcript should hold
        what was sent.
        """
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


def _normalize_submit_status(acknowledgement: Any) -> tuple[str, bool]:
    """Pull the real outcome out of a `prompt.submit` acknowledgement (B-05o).

    Returns `(status, is_known)`.

    Passes Hermes's own string through untouched (apart from surrounding
    whitespace) -- this must not invent, rename, or collapse statuses, since
    the set is Hermes's to define and it may grow. What it will not do is
    guess: an absent status, a `null`, a non-string, or an empty string comes
    back as `SUBMIT_STATUS_UNKNOWN` with `is_known=False`, and an unfamiliar
    string comes back verbatim, also with `is_known=False`.

    The whole point is that "HTTP 200" and "the user's message will be
    answered" are different claims. `redirected` -- observed live -- is a
    200 whose turn never streams a reply of its own.
    """
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
    """Submit one user turn to a saved session -- the chat send path.

    **Chat history (P1/B-136).** The user's row is written to the gateway's
    own durable `ChatStore` (`source: "submit"`) BEFORE Hermes is asked to do
    anything, so a durable copy of what was typed exists even if the call
    below fails outright. If it does fail, that row is discarded
    (`ChatStore.discard_row`) -- mirroring this route's existing
    error-handling style, a submission that never reached Hermes must not
    look like a persisted, sent message. `?profile=` scopes which profile's
    history the row belongs to (`GET /api/profiles` lists them) and, via
    `resolve_profile_adapter` (`domain/hermes_runtime.py`), which connection the
    call below actually goes out on; omitted, it defaults to `default`,
    unchanged from before multi-profile support existed. A profile that
    isn't connected right now is a 503 naming it, never a silent fall-back
    to the default connection.

    `{stored_session_id}` is the **STORED / durable** id. The live handle
    `prompt.submit` requires is resolved through `_with_live_handle()`: from
    the current connection's cache when one is already known, otherwise via
    `session.resume`. A handle from an earlier connection cannot be returned
    (the cache is keyed on the adapter's connection generation), and one
    Hermes has forgotten is re-resolved once automatically.

    `adapter.prompt_submit()` is called with exactly two arguments -- the
    resolved live handle and the body's `text`. Nothing else from the request
    is forwarded, and `TurnSubmission` refuses any rewind/truncate key
    outright (422), so this endpoint cannot trigger a destructive history
    rewrite no matter what a client sends. It also refuses an empty or
    whitespace-only `text` (422, B-24) -- that never reaches Hermes.

    **The response says what actually happened to the prompt (B-05o).**
    `prompt.submit` has four distinct success answers, and a 200 here does
    *not* mean the user's message will be answered:

    | `submit_status` | What Hermes did |
    |---|---|
    | `streaming`   | A new turn started and will stream back. The normal case. |
    | `redirected`  | A turn was already in flight; this text was applied as a **correction** to it (`display.busy_input_mode` defaults to `interrupt`). No separate reply to this message. |
    | `steered`     | Injected into the running turn at the next atomic action boundary. Again no separate reply. |
    | `queued`      | Accepted; it will run **after** the current turn finishes. |
    | `unknown`     | Hermes sent no status, a null, a non-string, or something this gateway has never seen. Must not be read as success. |

    All four of Hermes's are HTTP 200 -- which is exactly the bug: the app
    treated any 200 as "sent" and a `redirected` message silently never got
    a reply (observed live 2026-08-29). So the response carries:

    * `submit_status` -- the status string Hermes returned, verbatim, or
      `"unknown"` if there wasn't a usable one. Nothing is invented or
      renamed here.
    * `submit_status_known` -- whether that string is one of the four
      documented outcomes above. An unrecognized status is reported as-is
      with this flag `False`, never quietly treated as `streaming`.
    * `submit` -- the whole acknowledgement, still verbatim.

    Only `streaming` means a reply is coming back on `WS /ws/events` for
    *this* message; the client is responsible for telling the user so for
    every other outcome.

    Errors are structured: 404 for an unknown stored id, 502 for any other
    Hermes failure -- never a bare 500.
    """
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
    """Body for `POST /api/projects/{project_id}/sessions/new` (P2-17).

    Distinct from `SessionFiling` (`api/projects.py`) on purpose: that body
    carries an *existing* Hermes stored id and that route never touches the
    adapter. This one carries no id at all -- there isn't one yet -- because
    it is what the "+ New session" pill sends to mint a session from scratch.

    `first_message` is required, not optional. Hermes does not persist a
    session until it has content (`docs/PROTOCOL_VERIFIED.md`, "A new session
    is not persisted until it has content"): filing a session immediately
    after `session.create` with nothing said would file a stored id that
    `session.resume` answers `[4007] session not found` for the moment the
    socket reconnects, and the app would open straight into a dead end.
    Requiring the first turn up front means the session this route hands
    back has always already been primed by the time it is filed.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    first_message: str = Field(min_length=1)
    #: Which Hermes profile to create this session on. `default`
    #: unless the caller names a connected one (`GET /api/profiles`) -- an
    #: unconnected name is a 503 from `resolve_profile_adapter`, same as
    #: every other session-scoped route, not a silent fall-back to default.
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

    Three steps, strictly in order, each only attempted once the previous one
    has actually succeeded:

    1. `session.create {title}` -- Hermes hands back both ids immediately,
       live handle included, before the session is durably saved.
    2. `prompt.submit` on that live handle with `first_message` -- the
       priming turn that makes Hermes actually persist the session (see
       `NewFiledSession`'s docstring). This is the point of no return: once
       it succeeds, a real Hermes session exists whether or not this route's
       remaining step does.
    3. `file_stored_session()` (`api/projects.py`) -- the exact DB-only
       filing core `POST /projects/{id}/sessions` uses, so this ends up with
       the same row a manual create-then-file two-step would leave, and the
       two routes cannot drift in what a "filed session" row looks like.

    The live handle is cached (`LiveHandleCache.put`) so the very next
    request for this session -- the app immediately opening the conversation
    it just created -- does not pay a redundant `session.resume` round trip.
    It is still never written to the database.

    Errors before step 2 succeeds are a plain 502 (nothing exists yet that
    could be named in a 404). A failure in step 3 (DB unreachable) after step
    2 succeeded is also a 502, but by then leaves a real, orphaned, unfiled
    Hermes session behind -- the same class of partial-failure edge case
    `docs/BUGS.md` already tracks elsewhere; it is recoverable by hand (the
    session shows up in "All sessions" moments later and can be filed the
    ordinary way) and corrupts nothing, so this is not a full two-phase
    commit.
    """
    project = _load_project(db, project_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, body.profile)
    cache: LiveHandleCache = resolve_live_handle_cache(request.app.state, body.profile)

    # Project instructions: if this project has a
    # `HERMES.md`, create the session with its `cwd` pointing at the project's
    # workspace folder. Hermes bakes that file into the session's system
    # prompt at creation, so
    # every profile's session that is filed here starts on the same
    # foundation. The check reads the file's presence, not its content, and
    # only ADDS a `cwd` -- a project with no instructions creates exactly the
    # session it did before this feature (Hermes's default cwd). A failure to
    # check never blocks creation: `instructions_file_exists` answers False
    # rather than raising, so the session is still made.
    settings = get_settings()
    create_kwargs: dict[str, Any] = {}
    # WHICH profile's store the session lands in. Honoured since Hermes 0.21.3
    # (0.20.5 accepted the key and silently ignored it, which is why this was
    # not sent before); without it a session asked for on `kimi25` is created
    # on `default` and the user's next message goes to the wrong agent.
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
