"""Delegated subagents: the roster, a child's live log, and the two controls."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from adapters.hermes import HermesAdapter, HermesError
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)

logger = logging.getLogger(__name__)

subagents_router = APIRouter(tags=["subagents"])

LIST_FIELDS = (
    "subagent_id",
    "parent_id",
    "depth",
    "goal",
    "delegation_id",
    "model",
    "started_at",
    "status",
    "tool_count",
    "last_tool",
    "accepting_steer",
)


class SteerBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)


def _subagent_id(raw: str) -> str:
    value = raw.strip()
    if not value or len(value) > 128 or any(ch.isspace() for ch in value):
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="subagent_id must be a single non-empty token")
    return value


async def _call(
    request: Request,
    stored_session_id: str,
    profile: str,
    operation_name: str,
    **params: Any,
) -> tuple[str, str, dict[str, Any]]:
    """`(stored_id, live_id, result)` for one subagent RPC on the parent session."""
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)
    operation = getattr(adapter, operation_name)
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: operation(live, **params),
                profile=profile,
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc
    return stored_id, live_id, result if isinstance(result, dict) else {}


@subagents_router.get("/sessions/{stored_session_id}/subagents")
async def list_subagents(
    stored_session_id: str, request: Request, profile: str = Query(default="default")
) -> dict:
    """The children currently running under this session's turn, live."""
    stored_id, live_id, result = await _call(request, stored_session_id, profile, "subagent_list")
    raw = result.get("subagents")
    rows = [
        {key: entry.get(key) for key in LIST_FIELDS}
        for entry in (raw if isinstance(raw, list) else [])
        if isinstance(entry, dict)
    ]
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "subagents": rows,
        "count": len(rows),
    }


@subagents_router.get("/sessions/{stored_session_id}/subagents/{subagent_id}/tail")
async def tail_subagent(
    stored_session_id: str,
    subagent_id: str,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """The last 16 KB of Hermes's live transcript for one child, verbatim."""
    child = _subagent_id(subagent_id)
    stored_id, live_id, result = await _call(
        request, stored_session_id, profile, "subagent_tail", subagent_id=child
    )
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "subagent_id": child,
        "available": bool(result.get("available")),
        "text": result.get("text") if isinstance(result.get("text"), str) else "",
        "truncated": bool(result.get("truncated")),
    }


@subagents_router.post("/sessions/{stored_session_id}/subagents/{subagent_id}/interrupt")
async def interrupt_subagent(
    stored_session_id: str,
    subagent_id: str,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Ask one child to stop at its next iteration boundary."""
    child = _subagent_id(subagent_id)
    stored_id, live_id, result = await _call(
        request, stored_session_id, profile, "subagent_interrupt", subagent_id=child
    )
    found = bool(result.get("found"))
    logger.info("subagent interrupt %s on %s -> found=%s", child, stored_id, found)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "subagent_id": child,
        "found": found,
    }


@subagents_router.post("/sessions/{stored_session_id}/subagents/{subagent_id}/steer")
async def steer_subagent(
    stored_session_id: str,
    subagent_id: str,
    body: SteerBody,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Queue steering text into a running child."""
    child = _subagent_id(subagent_id)
    stored_id, live_id, result = await _call(
        request, stored_session_id, profile, "subagent_steer", subagent_id=child, text=body.text
    )
    status = result.get("status") if isinstance(result.get("status"), str) else "unknown"
    logger.info("subagent steer %s on %s -> %s", child, stored_id, status)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "subagent_id": child,
        "steer_status": status,
        "queued": status == "queued",
    }
