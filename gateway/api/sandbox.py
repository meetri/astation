"""Tier-1 raw sandbox browse: a thin authenticated proxy over Hermes's own
file API (P3-1a).

Two routes, no DB writes, no ingestion -- the "Live Sandbox" tier from
TASKS.md Phase 3: what's on the Hermes sandbox right now, ephemeral, gone if
the sandbox is cleaned. The durable, checksummed artifact library is Tier 2
(P3-1/P3-2) and does not live here.

* `GET /api/sandbox/files?path=<dir>` -- directory listing, forwarded
  **verbatim** from Hermes's `GET /api/files?path=` (measured shape, PV
  "Phase 3 probe": `path`/`parent`/`entries`/`root`/`locked_root`/
  `can_change_path`, with `mime_type` free on every file entry). Verbatim on
  purpose, same B-34 lesson as the transcript routes: the upstream route is
  undocumented, so the gateway forwards what it got instead of normalizing
  through a shape that predates whatever Hermes ships next.
* `GET /api/sandbox/download?path=<abs>` -- raw file bytes from Hermes's
  `GET /api/files/download?path=`, **streamed, never buffered** (the
  streaming-vs-buffering decision is made here, per the task spec), with the
  client's `Range` header passed through untouched and Hermes's own
  `206`/`Content-Range` answer passed back -- Hermes supports Range natively
  (measured), so no range logic is reimplemented on this side.

Path validation happens on the gateway BEFORE any upstream call, against
`Settings.hermes_sandbox_root` (default `/opt/data`, the measured
`locked_root`). Hermes enforces its own confinement upstream (403 "Path
outside managed files root", measured both directions), but that guard is
never trusted alone: the upstream routes are undocumented, a Hermes upgrade
could loosen them silently, and defense in depth costs one `normpath`. The
validation is lexical (absolute path, no `..` escape after normalization);
symlinks inside the sandbox that point outside it can only be resolved
host-side, which is exactly what Hermes's own `locked_root` layer is for --
the two guards cover each other.

**Undocumented-route resilience** (the SKEPTIC's condition on P3-1a): these
upstream routes were found by probing, not in any Hermes doc, so a future
Hermes upgrade may change or remove them without notice. Mitigations here:

* every successful listing is shape-checked against the measured contract,
  for free, and drift is logged at ERROR with an unmistakable message;
* `GET /api/sandbox/health` runs an explicit on-demand probe (listing the
  sandbox root + the same shape check) and reports the result, so a deploy
  script or the app can ask "does Tier 1 still work?" without guessing;
* the first sandbox request after process start counts as the startup check
  (the lifespan deliberately makes no network calls at startup -- see
  `api.main.lifespan` -- so a literal boot-time probe would break that
  contract), and a stale health record older than `_HEALTH_MAX_AGE_S` is
  refreshed by the next health-route call or listing.

Documented-but-unbuilt fallback if these routes ever vanish: the
chat-reply-base64 path (PV "Phase 3 probe", Attempt 4) -- deliberately worse
(ties up a whole prompt turn, minutes of model latency, transcript
pollution), never exercised, recorded so the recovery plan is written down
rather than re-derived under pressure.

Auth: mounted on the authenticated `/api` router in `api.main`, so Basic
auth is inherited by construction. Upstream, the adapter's
`files_list`/`files_download` carry the same cookie jar `login()` fills,
with the 401 -> re-login -> retry-once pattern (see
`HermesAdapter._files_get`).
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

#: Upstream statuses forwarded to the client as-is on the download route.
#: 200/206 stream bytes; 416 is Hermes's own "unsatisfiable Range" answer and
#: belongs to the client that sent the Range header.
_STREAMED_STATUSES = frozenset({200, 206})

#: Upstream *client* errors whose status is preserved (with the upstream
#: detail) instead of collapsing into a 502: the request was understood and
#: refused, which the caller needs to distinguish from "Hermes is broken".
_PASSTHROUGH_CLIENT_ERRORS = frozenset({403, 404, 416})

#: Response headers copied from Hermes's download response. Everything a
#: streaming client (AVFoundation scrubbing, PDFKit) needs; nothing else, so
#: an upstream header this gateway has never seen cannot leak through.
_DOWNLOAD_HEADER_ALLOWLIST = (
    "content-type",
    "content-length",
    "content-range",
    "accept-ranges",
    "etag",
    "last-modified",
)

#: Most bytes of an upstream *error* body this gateway will read while
#: extracting a detail message. Error bodies are tiny JSON (measured); the
#: bound exists so a pathological upstream cannot make the error path buffer.
_ERROR_BODY_CAP = 4096

#: A health record older than this is stale; the next health-route call (or
#: any listing) refreshes it. 15 minutes: cheap enough to re-probe, long
#: enough that polling the health route does not hammer Hermes.
_HEALTH_MAX_AGE_S = 900.0


# ---------------------------------------------------------------------------
# Upstream error mapping
# ---------------------------------------------------------------------------


async def _upstream_detail(response: httpx.Response) -> str:
    """Best-effort human detail from an upstream error body, bounded.

    The measured error bodies are tiny JSON (`{"detail": "Path outside
    managed files root"}`, the 401's `{"error": "unauthenticated", ...}`);
    anything else degrades to a truncated text snippet, never a parse crash.
    """
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
    """Map a non-OK upstream response onto an HTTPException. Closes `response`.

    Client errors Hermes answered deliberately (403/404/416) keep their
    status and carry the upstream detail; anything else is an upstream
    problem and becomes a 502 -- never a bare 500 (same contract as
    `_http_error_from_hermes`).

    `passthrough` widens the preserved set for a caller whose upstream route
    has its own measured client errors (`api/sandbox_text.py`: 400 for a
    directory, 413 for an oversized write). The default is this module's
    own set, so the listing/download routes are unaffected.
    """
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


# ---------------------------------------------------------------------------
# Health: shape-check the undocumented upstream contract
# ---------------------------------------------------------------------------


def _listing_shape_problems(body: Any) -> list[str]:
    """Deviations from the MEASURED listing shape (PV "Phase 3 probe").

    Checks exactly what this gateway and the app depend on -- top-level
    `path` + `entries`, and per entry `name`/`path`/`is_directory` plus
    `mime_type` on files (P3-4a's viewers key off it). Nothing speculative:
    every requirement here traces to a verbatim measured response.
    """
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
        # LOUD by design: these upstream routes are undocumented and this is
        # the tripwire for a Hermes upgrade breaking Tier 1 silently.
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
    """One explicit probe: list the sandbox root, verify the measured shape.

    Cheap (one small GET), safe (read-only, root listing only), and the
    single source of truth for `GET /api/sandbox/health`. Never raises --
    the whole point is to report brokenness, not to be broken by it.
    """
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@sandbox_router.get("/sandbox/files")
async def list_sandbox_files(
    request: Request,
    path: str | None = Query(default=None),
) -> dict:
    """List a sandbox directory, forwarded verbatim from Hermes.

    `path` defaults to the sandbox root, so the app's browse screen can open
    with no arguments. The response body is Hermes's own listing object,
    untouched -- `entries[].mime_type` in particular is what P3-4a's viewers
    key off.
    """
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
    """Stream one sandbox file's bytes, Range passthrough included.

    The client's `Range` header (if any) goes upstream verbatim; Hermes's
    status (200 or 206) and its `Content-Type`/`Content-Length`/
    `Content-Range`/`Accept-Ranges` headers come back verbatim. The body is
    never buffered on the gateway: chunks are forwarded as they arrive and
    the upstream response is closed when the stream ends (including on a
    client disconnect -- Starlette closes the generator, which runs the
    `finally`).
    """
    settings = get_settings()
    target = validate_sandbox_path(path, settings.hermes_sandbox_root)
    adapter: HermesAdapter = request.app.state.hermes_adapter
    # The one file route that does NOT follow the backend choice in
    # `domain/sandbox_fs.py`. Hermes's download route already streams with
    # `Range` support and confines server-side, and it is a single round trip
    # rather than the thousands a walk costs -- so raw bytes keep the second
    # wall even when everything else has moved to direct I/O.
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
    """On-demand health of the undocumented upstream files routes.

    Serves the cached record when it is fresh (any recent listing refreshed
    it for free); probes Hermes when the record is missing, stale, or
    `refresh=true`. `ok: false` means Tier-1 browse -- and Tier-2 ingestion,
    which shares the transport -- should be presumed broken until a human
    looks; the ERROR log line has already fired by the time this returns.
    """
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
        # Which backend answered. The two differ in confinement and in whether
        # a save can be clobbered, and the fallback to HTTP is silent by
        # design -- so "which one am I on?" has to be askable from outside.
        **describe_backend(request.app.state),
    }
