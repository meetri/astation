"""Work a session left running after its turn: delegated children and background processes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from adapters.hermes import HermesAdapter, HermesError, HermesRPCError
from api.subagents import LIST_FIELDS
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _rpc_error_code,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)

logger = logging.getLogger(__name__)

background_work_router = APIRouter(tags=["background-work"])

SUBAGENT_FIELDS = tuple(field for field in LIST_FIELDS if field != "accepting_steer")

PROCESS_FIELDS = (
    "command",
    "cwd",
    "pid",
    "owner_task_id",
    "started_at",
    "uptime_seconds",
    "status",
    "exit_code",
    "output_tail",
    "notify_on_complete",
    "detached",
)

_OUTPUT_TAIL_FALLBACK_KEY = "output_preview"


_NO_SUCH_PROCESS_CODE = 4044


def _process_id(raw: str) -> str:
    value = raw.strip()
    if not value or len(value) > 128 or any(ch.isspace() for ch in value):
        raise HTTPException(status_code=422, detail="process_id must be a single non-empty token")
    return value


def _owned_children(result: Any, stored_id: str) -> list[dict[str, Any]]:
    raw = result.get("active") if isinstance(result, dict) else None
    rows = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict) or entry.get("owner_agent_session_id") != stored_id:
            continue
        rows.append({key: entry.get(key) for key in SUBAGENT_FIELDS})
    return rows


def _processes(result: Any) -> list[dict[str, Any]]:
    raw = result.get("processes") if isinstance(result, dict) else None
    rows = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("session_id"), str):
            continue
        row = {"process_id": entry["session_id"]}
        row.update({key: entry.get(key) for key in PROCESS_FIELDS})
        if row["output_tail"] is None:
            row["output_tail"] = entry.get(_OUTPUT_TAIL_FALLBACK_KEY)
        rows.append(row)
    return rows


@background_work_router.get("/sessions/{stored_session_id}/background-work")
async def background_work(
    stored_session_id: str, request: Request, profile: str = Query(default="default")
) -> dict:
    """Delegated children and background processes this session has running now."""
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)
    errors: dict[str, str | None] = {"subagents": None, "processes": None}

    subagents: list[dict[str, Any]] = []
    try:
        status = await _with_reconnect(request.app.state, adapter, adapter.delegation_status)
        subagents = _owned_children(status, stored_id)
    except HermesError as exc:
        logger.info("background-work: delegation.status for %s failed: %s", stored_id, exc)
        errors["subagents"] = str(exc)

    processes: list[dict[str, Any]] = []
    live_id: str | None = None
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter, cache, stored_id, adapter.process_list, profile=profile
            ),
        )
        processes = _processes(result)
    except HermesError as exc:
        logger.info("background-work: process.list for %s failed: %s", stored_id, exc)
        errors["processes"] = str(exc)

    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "subagents": subagents,
        "processes": processes,
        "errors": errors,
    }


@background_work_router.post("/sessions/{stored_session_id}/processes/{process_id}/kill")
async def kill_process(
    stored_session_id: str,
    process_id: str,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Kill one of this session's background processes."""
    stored_id = _validate_stored_session_id(stored_session_id)
    proc = _process_id(process_id)
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
                lambda live: adapter.process_kill(live, process_id=proc),
                profile=profile,
            ),
        )
    except HermesError as exc:
        if isinstance(exc, HermesRPCError) and _rpc_error_code(exc) == _NO_SUCH_PROCESS_CODE:
            raise HTTPException(
                status_code=404, detail=f"session {stored_id!r} owns no process {proc!r}"
            ) from exc
        raise _http_error_from_hermes(exc, stored_id) from exc
    result = result if isinstance(result, dict) else {}
    status = result.get("status") if isinstance(result.get("status"), str) else "unknown"
    logger.info("process kill %s on %s -> %s", proc, stored_id, status)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "process_id": proc,
        "status": status,
        "error": result.get("error") if isinstance(result.get("error"), str) else None,
    }
