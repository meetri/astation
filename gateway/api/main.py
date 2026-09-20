"""Research Gateway FastAPI app."""

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

# Re-exported, not used here: other modules and tests import these names from this one.
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
    """Build every `app.state` service on startup, tear them down on shutdown."""
    # No login()/connect() here: only a request that needs Hermes opens the connection, lazily.
    runtime = startup(app.state, get_settings())
    try:
        yield
    finally:
        await shutdown(app.state, runtime)


app = FastAPI(title="Research Gateway", version="0.1.0", lifespan=lifespan)


# The default 422 body echoes the rejected value, which on /api/prompts/* can be a secret.
app.add_exception_handler(RequestValidationError, scrub_prompt_validation_errors)


# One dependency declared here is why a route added later cannot ship unauthenticated.
api = APIRouter(prefix="/api", dependencies=[Depends(require_basic_auth)])


@app.get("/health")
async def health() -> dict[str, str]:
    """Unauthenticated liveness probe. Must not reveal anything but liveness."""
    return {"status": "ok"}


# Included first: route precedence depends on this order.
api.include_router(sessions_router)

api.include_router(projects_router)

api.include_router(prompts_router)

api.include_router(background_router)

api.include_router(runs_router)

api.include_router(audit_router)

api.include_router(commands_router)

api.include_router(sandbox_router)

api.include_router(sandbox_text_router)

api.include_router(artifacts_router)

api.include_router(attachments_router)

api.include_router(transcribe_router)

api.include_router(rewrite_router)

api.include_router(handoff_router)

api.include_router(speak_router)

api.include_router(converse_router)

api.include_router(instance_router)

api.include_router(profile_admin_router)

from api.compress import compress_router  # noqa: E402  (P6-4)
from api.snapshots import snapshots_router  # noqa: E402

api.include_router(snapshots_router)
api.include_router(compress_router)

api.include_router(config_router)

from api.snapshot_sweep import snapshot_sweep_router  # noqa: E402

api.include_router(snapshot_sweep_router)

api.include_router(chat_router)

app.include_router(api)


# On `app`, not `api`: this caller has no Basic credential and authenticates by token instead.
app.include_router(attachment_serve_router)


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """Proxy normalized events from one active Hermes session to this client."""
    if not websocket_client_is_authorized(websocket):

        # Accept before closing: closing first makes the server send a bare 403 with no close code.
        await websocket.accept()
        await websocket.close(code=1008, reason="Invalid or missing credentials")
        return

    await stream_events(websocket, websocket.app.state)
