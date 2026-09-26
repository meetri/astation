"""Chat-history routes (B-136, `docs/CHAT_HISTORY_DESIGN.md` §4)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from sqlalchemy.exc import OperationalError

from domain.chat_store import ChatStore
from domain.db import schema_missing_error
from domain.hermes_runtime import _validate_stored_session_id

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["chat"])

DEFAULT_PROFILE = "default"


@chat_router.get("/sessions/{stored_session_id}/chat")
async def session_chat(
    stored_session_id: str,
    request: Request,
    profile: str = Query(default=DEFAULT_PROFILE),
    before: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """The app's transcript page, served entirely from `ChatStore` (D-1)."""
    stored_id = _validate_stored_session_id(stored_session_id)
    chat_store: ChatStore = request.app.state.chat_store
    try:
        # A blocking SQLite read; off the event loop so it cannot stall other requests.
        rows = await asyncio.to_thread(
            chat_store.page,
            profile=profile, stored_session_id=stored_id, before_seq=before, limit=limit,
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
