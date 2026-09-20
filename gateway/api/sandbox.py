"""Tier-1 raw sandbox browse: a thin authenticated proxy over Hermes's own
file API (P3-1a).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from adapters.hermes import HermesAdapter, HermesError
from config.settings import get_settings
from domain.sandbox_fs import describe_backend, sandbox_fs_for
from domain.sandbox_paths import validate_sandbox_path

logger = logging.getLogger(__name__)

sandbox_router = APIRouter(tags=["sandbox"])

_STREAMED_STATUSES = frozenset({200, 206})

_PASSTHROUGH_CLIENT_ERRORS = frozenset({403, 404, 416})

_DOWNLOAD_HEADER_ALLOWLIST = (
    "content-type",
    "content-length",
    "content-range",
    "accept-ranges",
    "etag",
    "last-modified",
)

_ERROR_BODY_CAP = 4096

_HEALTH_MAX_AGE_S = 900.0


async def _upstream_detail(response: httpx.Response) -> str:
    """Best-effort human detail from an upstream error body, bounded."""
    try:
        raw = await response.aread()
    except httpx.HTTPError:
        return "(upstream error body unreadable)"
    finally:
        await response.aclose()
    snippet = raw[:_ERROR_BODY_CAP].decode("utf-8", errors="replace")
    try:
        body = json.loads(snippet)
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            return body["detail"]
    except ValueError:
        pass
    return snippet or f"(empty body, HTTP {response.status_code})"


async def _raise_for_upstream(
    response: httpx.Response,
    path: str,
    *,
    passthrough: frozenset[int] = _PASSTHROUGH_CLIENT_ERRORS,
) -> None:
    """Map a non-OK upstream response onto an HTTPException. Closes `response`."""
    status = response.status_code
    detail = await _upstream_detail(response)
    if status in passthrough:
        raise HTTPException(status_code=status, detail=f"Hermes files API: {detail}")
    raise HTTPException(
        status_code=502,
        detail=(
            f"Hermes files API answered HTTP {status} for {path!r}: {detail} "
            "-- this route is undocumented upstream; if this persists after a "
            "Hermes upgrade, see api/sandbox.py's fallback note"
        ),
    )


def _listing_shape_problems(body: Any) -> list[str]:
    """Deviations from the MEASURED listing shape (PV "Phase 3 probe")."""
    problems: list[str] = []
    if not isinstance(body, dict):
        return [f"listing body is {type(body).__name__}, not an object"]
    if not isinstance(body.get("path"), str):
        problems.append("missing/non-string 'path'")
    entries = body.get("entries")
    if not isinstance(entries, list):
        return [*problems, "missing/non-list 'entries'"]
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"entries[{index}] is not an object")
            continue
        for key, kind in (("name", str), ("path", str), ("is_directory", bool)):
            if not isinstance(entry.get(key), kind):
                problems.append(f"entries[{index}] missing/mis-typed {key!r}")
        if entry.get("is_directory") is False and not isinstance(entry.get("mime_type"), str):
            problems.append(f"entries[{index}] (a file) has no 'mime_type'")
    return problems


def _record_health(app_state: Any, ok: bool, detail: str) -> dict[str, Any]:
    record = {
        "ok": ok,
        "detail": detail,
        "checked_at_monotonic": time.monotonic(),
    }
    app_state.sandbox_files_health = record
    if not ok:
        logger.error(
            "SANDBOX FILES ROUTE HEALTH CHECK FAILED -- Hermes's undocumented "
            "/api/files* surface no longer answers in the measured shape "
            "(%s). Tier-1 sandbox browse (and Tier-2 ingestion, which shares "
            "the transport) may be broken. See api/sandbox.py for the "
            "documented chat-reply-base64 fallback plan.",
            detail,
        )
    return record


def _check_listing_shape(app_state: Any, body: Any) -> None:
    """The free, continuous half of the health check: every listing verifies
    the measured contract as a side effect and refreshes the health record."""
    problems = _listing_shape_problems(body)
    if problems:
        _record_health(app_state, False, "; ".join(problems))
    else:
        _record_health(app_state, True, "listing answered in the measured shape")


async def run_sandbox_health_check(app_state: Any, adapter: HermesAdapter) -> dict[str, Any]:
    """One explicit probe: list the sandbox root, verify the measured shape."""
    root = get_settings().hermes_sandbox_root
    try:
        response = await sandbox_fs_for(app_state, adapter).files_list(root)
    except HermesError as exc:
        return _record_health(app_state, False, f"listing request failed: {exc}")
    if response.status_code != 200:
        detail = await _upstream_detail(response)
        return _record_health(
            app_state,
            False,
            f"listing the sandbox root answered HTTP {response.status_code}: {detail}",
        )
    try:
        body = response.json()
    except ValueError:
        return _record_health(app_state, False, "listing response was not JSON")
    problems = _listing_shape_problems(body)
    if problems:
        return _record_health(app_state, False, "; ".join(problems))
    return _record_health(app_state, True, "listing answered in the measured shape")


@sandbox_router.get("/sandbox/files")
async def list_sandbox_files(
    request: Request,
    path: str | None = Query(default=None),
) -> dict:
    """List a sandbox directory, forwarded verbatim from Hermes."""
    settings = get_settings()
    target = validate_sandbox_path(
        path or settings.hermes_sandbox_root, settings.hermes_sandbox_root
    )
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        response = await sandbox_fs_for(request.app.state, adapter).files_list(target)
    except HermesError as exc:
        _record_health(request.app.state, False, f"listing request failed: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if response.status_code != 200:
        await _raise_for_upstream(response, target)
    try:
        body = response.json()
    except ValueError as exc:
        _record_health(request.app.state, False, "listing response was not JSON")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Hermes files API returned non-JSON for {target!r} -- the "
                "undocumented listing route may have changed shape"
            ),
        ) from exc
    _check_listing_shape(request.app.state, body)
    return body


@sandbox_router.get("/sandbox/download")
async def download_sandbox_file(
    request: Request,
    path: str = Query(...),
) -> StreamingResponse:
    """Stream one sandbox file's bytes, Range passthrough included."""
    settings = get_settings()
    target = validate_sandbox_path(path, settings.hermes_sandbox_root)
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        response = await adapter.files_download(target, range_header=request.headers.get("range"))
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if response.status_code not in _STREAMED_STATUSES:
        await _raise_for_upstream(response, target)

    headers = {
        name: response.headers[name]
        for name in _DOWNLOAD_HEADER_ALLOWLIST
        if name in response.headers
    }

    async def _forward_bytes() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()

    return StreamingResponse(
        _forward_bytes(),
        status_code=response.status_code,
        headers=headers,
    )


@sandbox_router.get("/sandbox/health")
async def sandbox_health(
    request: Request,
    refresh: bool = Query(default=False),
) -> dict:
    """On-demand health of the undocumented upstream files routes."""
    record = getattr(request.app.state, "sandbox_files_health", None)
    stale = (
        record is None or (time.monotonic() - record["checked_at_monotonic"]) > _HEALTH_MAX_AGE_S
    )
    if refresh or stale:
        adapter: HermesAdapter = request.app.state.hermes_adapter
        record = await run_sandbox_health_check(request.app.state, adapter)
    age_s = time.monotonic() - record["checked_at_monotonic"]
    return {
        "ok": record["ok"],
        "detail": record["detail"],
        "checked_seconds_ago": round(age_s, 3),
        "sandbox_root": get_settings().hermes_sandbox_root,
        **describe_backend(request.app.state),
    }
