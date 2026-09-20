"""Research Gateway FastAPI app.

The app object, its lifespan (every `app.state` service is built by
`api/bootstrap.py`, in a documented order), `GET /health` and
`WS /ws/events`. Every other surface --
the session routes included (`api/sessions.py`) -- is a router under `api/`
included at the bottom of this module. The event fan-out is
`domain/event_stream.py`, the stored->live handle cache
`domain/live_handles.py`; both are re-exported here for existing callers.

Run with (see `README.md`):

    cd services/research-gateway
    uv run uvicorn api.main:app --reload                    # localhost only
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8124  # reachable on the LAN

Auth: every `/api/*` route and `WS /ws/events` require HTTP Basic against
`RESEARCH_GATEWAY_USERNAME`/`RESEARCH_GATEWAY_PASSWORD` (see `api/auth.py`).
`GET /health` stays unauthenticated as a liveness probe.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    WebSocket,
)
from fastapi.exceptions import RequestValidationError

from api.artifacts import artifacts_router
from api.attachments import attachment_serve_router, attachments_router
from api.audit import audit_router
from api.auth import require_basic_auth, websocket_client_is_authorized
from api.background import background_router
from api.bootstrap import _profile_adapter_factory, shutdown, startup  # noqa: F401
from api.chat import chat_router
from api.commands import commands_router
from api.config import config_router
from api.converse import converse_router
from api.events_ws import (  # noqa: F401  (re-exported: tests import _watch_upstream from here)
    _forward_events,
    _report_desynchronized,
    _report_upstream_loss,
    _watch_for_disconnect,
    _watch_upstream,
    stream_events,
)
from api.handoff import handoff_router
from api.instance import instance_router
from api.profile_admin import profile_admin_router
from api.projects import (
    projects_router,
)
from api.prompts import prompts_router, scrub_prompt_validation_errors
from api.rewrite import rewrite_router
from api.runs import runs_router
from api.sandbox import sandbox_router
from api.sandbox_text import sandbox_text_router
from api.sessions import sessions_router
from api.speak import speak_router
from api.transcribe import transcribe_router
from config.settings import get_settings

# The event fan-out, the frame budget and the stream-control frames live in
# `domain/event_stream.py`; the stored->live handle cache in
# `domain/live_handles.py` (CLEANUP_PLAN step 3.1). Re-exported here so every
# existing `from api.main import ...` keeps resolving; new code should import
# from the domain modules directly.
from domain.event_stream import (  # noqa: F401
    _DESYNCHRONIZED_MESSAGE,
    _MAX_CLIENT_FRAME_BYTES,
    _UPSTREAM_HEALTH_POLL_S,
    _UPSTREAM_LOST_MESSAGE,
    CONNECTION_GENERATION_FIELD,
    DEFAULT_PROFILE_NAME,
    KNOWN_SUBMIT_STATUSES,
    LIVE_SESSION_ID_FIELD,
    PROFILE_FIELD,
    STORED_SESSION_ID_FIELD,
    STREAM_DESYNCHRONIZED_EVENT_TYPE,
    STREAM_READY_EVENT_TYPE,
    STREAM_RESYNC_EVENT_TYPE,
    SUBMIT_STATUS_QUEUED,
    SUBMIT_STATUS_REDIRECTED,
    SUBMIT_STATUS_STEERED,
    SUBMIT_STATUS_STREAMING,
    SUBMIT_STATUS_UNKNOWN,
    EventBroadcaster,
    desynchronized_frame,
    resync_required_frame,
    stream_ready_frame,
)

# Re-exported deliberately. These moved to `api/hermes_runtime.py` (now
# `domain/hermes_runtime.py`, CLEANUP_PLAN step 3.5) so
# `api/projects.py` can reach Hermes without a second copy of the reconnect
# rules; importing them here keeps `api.main._ensure_connected` /
# `api.main._with_reconnect` / `api.main._validate_stored_session_id` resolving
# for every caller and test that already reads them from this module.
#
# The stored->live resolution helpers (`_resume_for_live_id`,
# `_with_live_handle`) and the Hermes-error mapping (`_is_session_not_found`,
# `_http_error_from_hermes`, and the `[4001]`/`[4007]` code table) moved there
# for the same reason when the human-in-the-loop routes arrived
# (`api/prompts.py`): answering an approval and interrupting a turn both need
# the exact same stored->live resolution as `POST /turns`, and a second copy of
# *that* is the project's recurring bug class (two id spaces). Their names
# still resolve on `api.main`.
from domain.hermes_runtime import (  # noqa: F401
    _HERMES_LIVE_SESSION_NOT_FOUND_CODE,
    _HERMES_SESSION_NOT_FOUND_CODES,
    _HERMES_SESSION_NOT_FOUND_TEXT,
    _HERMES_STORED_SESSION_NOT_FOUND_CODE,
    _ensure_connected,
    _http_error_from_hermes,
    _is_session_not_found,
    _resume_for_live_id,
    _rpc_error_code,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)
from domain.live_handles import LiveHandleCache  # noqa: F401  (re-exported; see above)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build every `app.state` service on startup, tear them down on shutdown.

    The construction order is load-bearing and lives, with its reasons, in
    `api/bootstrap.py` (`startup()` / `shutdown()`). Deliberately never calls
    `login()`/`connect()`: `HermesAdapter()` makes no network calls at
    construction time, and the gateway must not exercise real Hermes
    credentials just from the process starting up -- only an actual request
    that needs Hermes triggers `login()`/`connect()`, lazily, once.
    """
    runtime = startup(app.state, get_settings())
    try:
        yield
    finally:
        await shutdown(app.state, runtime)


app = FastAPI(title="Research Gateway", version="0.1.0", lifespan=lifespan)

# Pydantic puts the value it rejected on every validation error and FastAPI's
# default 422 handler serializes it into the response body. On
# `/api/prompts/*` that value can be the operator's sudo password or a secret the
# agent asked them to supply (`ARCHITECTURE.md` §14), so those two keys are
# stripped for that prefix -- see `api/prompts.py`. Every other path keeps the
# default shape, where echoing the input is genuinely useful for debugging.
app.add_exception_handler(RequestValidationError, scrub_prompt_validation_errors)

# Every authenticated route hangs off this router, so the Basic-auth
# dependency is declared exactly once and a route added later cannot silently
# ship unauthenticated. `GET /health` and `WS /ws/events` are registered on
# `app` directly -- health is deliberately public, and the WebSocket
# authenticates itself inside the handler (see `ws_events`).
api = APIRouter(prefix="/api", dependencies=[Depends(require_basic_auth)])


@app.get("/health")
async def health() -> dict[str, str]:
    """Unauthenticated liveness probe. Must not reveal anything but liveness."""
    return {"status": "ok"}


# The session routes (`api/sessions.py`, CLEANUP_PLAN step 3.2): list /
# resume / messages / one row / submit a turn / new-and-file. Included FIRST
# on `api`, where they were registered before the extraction, so route
# precedence is exactly what it was. Mounted on `api`, so they inherit the
# Basic-auth dependency declared once on that router.
api.include_router(sessions_router)

# Projects, and the filing index that maps a Hermes stored session id into one
# (`api/projects.py`). Mounted on `api`, so it inherits the Basic-auth
# dependency declared once on that router.
api.include_router(projects_router)

# Answering the four human-in-the-loop prompts, and cancelling a running turn
# (`api/prompts.py`, B-37). Mounted on `api` for the same reason: every one of
# these routes is authenticated by construction, including the two that carry a
# credential.
api.include_router(prompts_router)

# Background tasks: submit + ledger lists (`api/background.py`, P2-1/B-42).
# Mounted on `api` like the others, so every route -- including the one that
# starts real work on the operator's Hermes -- is authenticated by construction.
api.include_router(background_router)

# The run read path (`api/runs.py`, P2-2e): flat run list + per-run event
# timeline behind the restart-surviving `after_seq` cursor. Mounted on `api`
# so both reads are authenticated by construction.
api.include_router(runs_router)

# The audit surface: what actually happened on the host (audit/README.md).
# Answers 503 with a reason when no store is configured, which is the default.
api.include_router(audit_router)

# Slash commands (`api/commands.py`, P2-4): cached catalog + resolve +
# dispatch, shaped by the P2-0a verdict (dispatch is a skill-expansion
# lookup, never an execution). Mounted on `api` -- authenticated by
# construction like everything else.
api.include_router(commands_router)

# Tier-1 raw sandbox browse (`api/sandbox.py`, P3-1a): listing + streamed
# download proxied over Hermes's own undocumented `/api/files*` surface, with
# gateway-side path confinement and a health check on the measured shape.
# Mounted on `api` -- authenticated by construction like everything else.
api.include_router(sandbox_router)

# Editor text read/write (`api/sandbox_text.py`, P7 phase 1): Hermes's
# `/api/fs/read-text` + `/api/fs/write-text` behind the SAME gateway-side
# path confinement as the browse routes -- which, for these two upstream
# routes, is the only confinement there is.
# Hash-guarded writes (409 on mismatch). Mounted on `api` -- authenticated
# by construction like everything else.
api.include_router(sandbox_text_router)

# Tier-2 durable artifact library (`api/artifacts.py`, P3-1/P3-2): the
# artifact REST surface (project-scoped + global listings, content with Range
# serving from the gateway's own store, v1 preview) and the manual promotion
# route auto-ingestion shares its pipeline with. Mounted on `api` --
# authenticated by construction like everything else.
api.include_router(artifacts_router)

# Composer attachments (`api/attachments.py`, P3-3): upload + poll + list for
# the asynchronous two-turn attach. Mounted on `api` -- authenticated by
# construction like everything else.
api.include_router(attachments_router)

# Speech-to-text (`api/transcribe.py`, P5-2a): multipart audio upload ->
# transcription, provider-abstracted (local faster-whisper default, keyed
# openai/groq passthrough). Mounted on `api` -- authenticated by construction
# like everything else.
api.include_router(transcribe_router)

# Prose rewrite for speech (`api/rewrite.py`, P5-4a): text -> read-aloud text
# via an OpenAI-compatible endpoint the operator points at OpenRouter today and a
# LAN model later, with no code change. Mounted on `api` -- authenticated by
# construction like everything else.
api.include_router(rewrite_router)

# Continue in a new session (`api/handoff.py`, owner ask 2026-09-07): the last
# several messages of a conversation distilled into the opening prompt for a
# fresh one, by the same endpoint the rewrite uses. Reads the transcript,
# writes nothing. Mounted on `api` -- authenticated by construction.
api.include_router(handoff_router)

# Text-to-speech (`api/speak.py`, P5-10): text -> audio bytes, provider-
# abstracted (local keyless piper by default, keyless cloud edge as the
# alternative, six recognised-but-unimplemented names answering an honest
# 503), plus the voice list the app's picker is built from -- probed from the
# engine, never a hardcoded table. Mounted on `api` -- authenticated by
# construction like everything else.
api.include_router(speak_router)

# Conversation about the operator's own research (`api/converse.py`, G-12): a
# spoken question plus a scope -> a two-or-three-sentence answer with real
# citations, assembled from this gateway's own runs, run events and artifacts
# plus the session's transcript. Deliberately NOT routed through the Hermes
# agent: agent turns are 2.5-15 minutes and B-62 completion signals arrive 8-10
# minutes late, so a seconds-scale answer can only come from a fast model
# beside the agent. Mounted on `api` -- authenticated by construction like
# everything else.
api.include_router(converse_router)

# Instance-shaped Hermes passthroughs (`api/instance.py`, 2026-09-01 review):
# the profile lens, the instance's approval mode, and session rename. Mounted
# on `api` -- authenticated by construction like everything else.
api.include_router(instance_router)

# Agent (profile) model management (`api/profile_admin.py`, P6,
# `docs/AGENT_MODEL_DESIGN.md` §8): one agent's detail with model facts and
# measured stats, the merged model catalog, the per-profile model write
# (`profiles.configure`, verified live 2026-09-06), and create / rename /
# delete. `GET /profiles` above is untouched. Mounted on `api` --
# authenticated by construction like everything else.
api.include_router(profile_admin_router)

# Session snapshots -- the durable archive (`api/snapshots.py`, P6-3): take /
# list / read a complete copy of a session, archive-to-project (snapshot +
# flag + best-effort pin) and unarchive. Imported here rather than at the top
# so this addition is one contiguous hunk (`docs/SESSION_ARCHIVE_DESIGN.md`
# §6.0's seam for Stream A). Mounted on `api` -- authenticated by construction
# like everything else.
from api.compress import compress_router  # noqa: E402  (P6-4)
from api.snapshots import snapshots_router  # noqa: E402

api.include_router(snapshots_router)
# P6-4: compress the context in place (snapshot first). See api/compress.py.
api.include_router(compress_router)

# Runtime provider configuration (`api/config.py`, P5-9): read/write/reset the
# rewrite and STT provider settings, plus the `/models` discovery probe that
# lets the app offer a picker instead of a text field. The overlay it writes
# sits above `.env` and takes effect on the next request, so the operator can
# change which endpoint, model and key are in force from the phone without a
# file edit or a restart. Mounted on `api` -- authenticated by construction
# like everything else, which matters more here than anywhere: this router
# accepts API keys.
api.include_router(config_router)

# The snapshot sweep's two routes (`api/snapshot_sweep.py`, P6-3 Stream B):
# run a pass by hand, read the last one. Imported here rather than at the top
# so this addition is one contiguous hunk (`docs/SESSION_ARCHIVE_DESIGN.md`
# §6.0); the `SnapshotSweeper` the lifespan above constructs is
# `domain/snapshot_sweeper.py`. Mounted on `api` -- authenticated by
# construction like everything else.
from api.snapshot_sweep import snapshot_sweep_router  # noqa: E402

api.include_router(snapshot_sweep_router)

# Chat history (`api/chat.py`, P1/B-136): the durable transcript page, served
# from `ChatStore` rather than Hermes directly (D-1). Mounted on `api` --
# authenticated by construction like everything else.
api.include_router(chat_router)

app.include_router(api)

# The attachment capability serve route is DELIBERATELY on `app`, not `api`:
# the priming turn's `curl` runs on the Hermes host without the gateway's
# Basic-auth credential (credentials never enter prompt text), so this one
# route authenticates by unguessable token instead -- see
# `api/attachments.py`'s module docstring for the full argument.
app.include_router(attachment_serve_router)


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """Proxy normalized events from one active Hermes session to this client.

    Authenticated with the same HTTP Basic credential as `/api/*`: our client
    is a native `URLSessionWebSocketTask`, which *can* set `Authorization` on
    the upgrade request, so no ticket/query-param scheme is needed. An
    unauthenticated client is closed with 1008 (policy violation) before the
    socket is accepted and before a single event is streamed.

    One socket carries every profile's events: the default connection's
    stream through the broadcaster plus every other profile's frames injected
    by its pump (`EventBroadcaster.inject`), each tagged `_profile`. Frames are
    stamped with the stored session id the app keys on (`_stamp_session_identity`)
    and a per-subscriber `seq`; a client that falls too far behind is told to
    resync rather than fed a gap.

    Two things this handler must get right, both learned the hard way in a
    live simulator run:

    * **It has to read.** An ASGI WebSocket handler only discovers that the
      peer went away by consuming the client->server side. This endpoint
      only ever *sends*, so without the `_watch_for_disconnect` task below
      the handler survived every client that ever left -- eight sockets
      accepted, zero closed -- and each zombie kept consuming events.
    * **It must not consume the adapter's queue directly.** `EventBroadcaster`
      exists because concurrent `adapter.events()` iterations split the
      stream instead of duplicating it; see its docstring.

    Two frames on this socket are the gateway's own rather than Hermes's:

    * a `stream.ready` envelope sent immediately after subscribing, so the
      client can tell a live-but-quiet stream from one that never came back
. It carries `seq` 0 and never spends a replay-cursor number.
    * `{"type": "error", ...}` + close 1011, sent either when Hermes cannot
      be reached at connect time or when the upstream connection dies while
      this socket is open. The client's reconnect is what rebuilds the
      upstream, so that path self-heals without needing an HTTP request.
    """
    if not websocket_client_is_authorized(websocket):
        # Accept-then-immediately-close rather than refusing the upgrade
        # outright: closing *before* accept makes an ASGI server reject the
        # handshake with a bare HTTP 403, and the client never sees a close
        # code at all. Accepting first means the client gets a real 1008
        # (policy violation) close frame and can tell "your credential is
        # wrong" apart from "the gateway is down" -- verified against a live
        # uvicorn, not just the test client. Nothing is streamed either way:
        # the socket is closed on the next line, before `adapter.events()` is
        # ever touched. The reason string is deliberately vague so it cannot
        # confirm whether a username exists.
        await websocket.accept()
        await websocket.close(code=1008, reason="Invalid or missing credentials")
        return

    await stream_events(websocket, websocket.app.state)
