"""Chat-history routes (B-136, `docs/CHAT_HISTORY_DESIGN.md` §4).

One route: `GET /api/sessions/{stored_session_id}/chat` -- the app's
transcript page, read from the gateway's own durable `ChatStore` rather than
Hermes directly (D-1).

**`GET /api/profiles` is NOT added here.** `api/instance.py` already has one
(the 2026-09-01 review's "profile lens" -- `profiles.list` forwarded
verbatim), and duplicating the path would either collide (FastAPI registers
both; whichever router is mounted first wins, silently shadowing the other)
or need a different path than the design names. Extending that existing
route to report `ProfileConnectionManager` connection health (now built and
wired at `lifespan` time, `app.state.profile_connection_manager` --
`api/main.py`) is still left as follow-up integration work, but the staleness
hazard that used to block it is fixed: the manager is constructed with
`app_state=app.state` and re-resolves `app.state.hermes_adapter` at the point
of use rather than capturing it once (see
`domain/profile_connection.py`'s module docstring, "the base-adapter
staleness hazard"). What is NOT wired is the reconciliation timer that would
ever actually provision a non-default profile's connection --
`RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S` defaults to `0` pending
B-137 (`docs/BUGS.md`): the default profile's connection reuses
`app.state.hermes_adapter`, the same object `EventBroadcaster` already
drains for `/ws/events`, and running both pumps concurrently would split
that single-consumer queue (B-08's failure class, reintroduced).

The turn-submit route (`POST /api/sessions/{stored_session_id}/turns`) stays
in `api/main.py` (it already exists there and this change extends it in
place) -- splitting it out would be a bigger diff than the feature needs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from sqlalchemy.exc import OperationalError

from domain.chat_store import ChatStore
from domain.db import schema_missing_error
from domain.hermes_runtime import _validate_stored_session_id

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["chat"])

#: Default profile name used when a caller omits `?profile=` -- matches
#: today's single-connection behavior exactly (D-6's default profile reuses
#: the existing unified connection, so an unscoped call must keep working).
DEFAULT_PROFILE = "default"


@chat_router.get("/sessions/{stored_session_id}/chat")
async def session_chat(
    stored_session_id: str,
    request: Request,
    profile: str = Query(default=DEFAULT_PROFILE),
    before: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """The app's transcript page, served entirely from `ChatStore` (D-1).

    `before` pages backwards by `seq` ("reveal earlier",
    `docs/CHAT_HISTORY_DESIGN.md` §6); omitted, this returns the newest
    `limit` rows. Never touches Hermes -- this is durable gateway state, not
    a live transcript fetch, so it stays readable even while Hermes is down.

    A 503 (not a 500 or an empty list) if the `chat_messages` migration has
    not been applied yet -- same convention as `api/runs.py`'s
    `_runs_db`/`api/background.py`'s `_background_db`.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    chat_store: ChatStore = request.app.state.chat_store
    try:
        rows = chat_store.page(
            profile=profile, stored_session_id=stored_id, before_seq=before, limit=limit
        )
    except OperationalError as exc:
        raise schema_missing_error() from exc

    return {
        "stored_session_id": stored_id,
        "profile": profile,
        "before": before,
        "limit": limit,
        "messages": rows,
    }
