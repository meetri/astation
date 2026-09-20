"""`POST /api/sessions/{stored}/compress` -- compress a session's context, in place (P6-4)."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError, HermesRPCError
from api.instance import _validate_stored_session_id_for_argv
from api.projects import workspace_db
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _rpc_error_code,
    _with_live_handle,
    _with_reconnect,
)
from domain.snapshot_builder import SnapshotStorageError, snapshot_row, take_snapshot

logger = logging.getLogger(__name__)

compress_router = APIRouter(tags=["compress"])

HERMES_SESSION_BUSY_CODE = 4009
COMPRESS_COMMAND_NAME = "/compress"
_NOOP_PREFIX = "No changes from compression"
_HEADLINE_RE = re.compile(r"(?:Compressed:\s*)?(\d[\d,]*)\s*(?:→|->)\s*(\d[\d,]*)\s*messages", re.I)
_NOOP_MESSAGES_RE = re.compile(r"No changes from compression:\s*(\d[\d,]*)\s*messages", re.I)
_TOKENS_RE = re.compile(r"~?(\d[\d,]*)\s*(?:→|->)\s*~?(\d[\d,]*)\s*tokens", re.I)
_TOKENS_UNCHANGED_RE = re.compile(r"~?(\d[\d,]*)\s*tokens\s*\(unchanged\)", re.I)


class CompressRequest(BaseModel):
    """`keep`: keep the last N exchanges verbatim (`/compress here N`). `focus`: a topic to
    prioritise (`/compress <focus>`; effect under LCM unmeasured). Neither -> plain compress."""

    model_config = ConfigDict(extra="forbid")

    keep: int | None = Field(default=None, ge=1, le=500)
    focus: str | None = Field(default=None, max_length=200)

    @field_validator("focus")
    @classmethod
    def _clean_focus(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        if not cleaned:
            return None
        if cleaned.startswith("-"):
            raise ValueError("focus must be a topic, not an option (no leading '-')")
        return cleaned


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.replace(",", ""))
    except ValueError:
        return None


def parse_dispatch_output(output: str | None) -> dict[str, Any]:
    """Normalise `command.dispatch`'s two-line text into the structured shape."""
    text = (output or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headline = lines[0] if lines else ""
    token_line = lines[1] if len(lines) > 1 else None
    noop = headline.startswith(_NOOP_PREFIX)
    before = after = None
    if match := _HEADLINE_RE.search(headline):
        before, after = _int(match.group(1)), _int(match.group(2))
    elif match := _NOOP_MESSAGES_RE.search(headline):
        before = after = _int(match.group(1))
    before_tokens = after_tokens = None
    if token_line:
        if match := _TOKENS_RE.search(token_line):
            before_tokens, after_tokens = _int(match.group(1)), _int(match.group(2))
        elif match := _TOKENS_UNCHANGED_RE.search(token_line):
            before_tokens = after_tokens = _int(match.group(1))
    removed = (before - after) if (before is not None and after is not None) else None
    return {
        "noop": noop,
        "headline": headline or None,
        "token_line": token_line,
        "removed": removed,
        "before_messages": before,
        "after_messages": after,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "usage": None,
    }


def normalise_compress_result(result: Any) -> dict[str, Any]:
    """Normalise `session.compress`'s structured answer into the same shape."""
    payload = result if isinstance(result, dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    noop = summary.get("noop")
    if not isinstance(noop, bool):
        removed = payload.get("removed")
        noop = isinstance(removed, int) and removed == 0
    return {
        "noop": noop,
        "headline": summary.get("headline") if isinstance(summary.get("headline"), str) else None,
        "token_line": summary.get("token_line")
        if isinstance(summary.get("token_line"), str)
        else None,
        "removed": payload.get("removed") if isinstance(payload.get("removed"), int) else None,
        "before_messages": payload.get("before_messages")
        if isinstance(payload.get("before_messages"), int)
        else None,
        "after_messages": payload.get("after_messages")
        if isinstance(payload.get("after_messages"), int)
        else None,
        "before_tokens": payload.get("before_tokens")
        if isinstance(payload.get("before_tokens"), int)
        else None,
        "after_tokens": payload.get("after_tokens")
        if isinstance(payload.get("after_tokens"), int)
        else None,
        "usage": _usage_subset(usage),
        "aborted": summary.get("aborted") if isinstance(summary.get("aborted"), bool) else None,
        "refused_would_grow": summary.get("refused_would_grow")
        if isinstance(summary.get("refused_would_grow"), bool)
        else None,
        "note": summary.get("note") if isinstance(summary.get("note"), str) else None,
    }


def _usage_subset(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not usage:
        return None
    keys = ("context_used", "context_max", "context_percent", "compressions")
    return {k: usage.get(k) for k in keys if k in usage}


def compress_args(body: CompressRequest) -> str | None:
    """The raw `/compress` argument for the partial/focus forms, or `None` for plain."""
    parts: list[str] = []
    if body.keep is not None:
        parts.append(f"here {body.keep}")
    if body.focus:
        parts.append(body.focus)
    return " ".join(parts) or None


def _compress_http_error(exc: HermesError, stored_id: str) -> HTTPException:
    if isinstance(exc, HermesRPCError) and _rpc_error_code(exc) == HERMES_SESSION_BUSY_CODE:
        return HTTPException(
            status_code=409,
            detail=(
                f"session {stored_id!r} is mid-turn; Hermes refuses to compress while a "
                f"turn runs -- wait for the reply to finish, or stop it (Hermes: {exc})"
            ),
        )
    return _http_error_from_hermes(exc, stored_id)


@compress_router.post("/sessions/{stored_session_id}/compress")
async def compress_session(
    stored_session_id: str,
    request: Request,
    body: CompressRequest | None = None,
    force: bool = Query(default=False),
    db: OrmSession = Depends(workspace_db),
) -> Any:
    """Snapshot, then compress in place. See the module docstring for the contract."""
    stored_id = _validate_stored_session_id_for_argv(stored_session_id)
    body = body or CompressRequest()
    args = compress_args(body)

    recorder = getattr(request.app.state, "run_recorder", None)
    if recorder is not None and recorder.has_open_run(stored_id):
        raise HTTPException(
            status_code=409,
            detail=f"session {stored_id!r} has a turn running through this gateway; wait for it to finish",
        )

    snapshot_id_row: dict[str, Any] | None = None
    if not force:
        try:
            snapshot = await take_snapshot(
                request.app.state, stored_id, reason="pre_compress", db=db
            )
        except HermesError as exc:
            raise _http_error_from_hermes(exc, stored_id) from exc
        except SnapshotStorageError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": (
                        f"the pre-compress snapshot of session {stored_id!r} failed, so nothing "
                        "was compressed; retry, or repeat with ?force=true to compress without a copy"
                    ),
                    "snapshot_error": str(exc),
                },
            )
        snapshot_id_row = snapshot_row(snapshot)

    adapter: HermesAdapter = request.app.state.hermes_adapter
    cache = request.app.state.live_handle_cache
    route: Literal["session.compress", "command.dispatch"]
    try:
        if args is None:
            route = "session.compress"
            _live, result = await _with_reconnect(
                request.app.state,
                adapter,
                lambda: _with_live_handle(adapter, cache, stored_id, adapter.session_compress),
            )
            normalised = normalise_compress_result(result)
        else:
            route = "command.dispatch"
            _live, result = await _with_reconnect(
                request.app.state,
                adapter,
                lambda: _with_live_handle(
                    adapter,
                    cache,
                    stored_id,
                    lambda live: adapter.command_dispatch(
                        COMPRESS_COMMAND_NAME, args=args, session_id=live
                    ),
                ),
            )
            payload = result if isinstance(result, dict) else {}
            normalised = parse_dispatch_output(
                payload.get("output") if isinstance(payload.get("output"), str) else None
            )
    except HermesError as exc:
        raise _compress_http_error(exc, stored_id) from exc

    logger.info(
        "compressed session %s via %s (%s): %s",
        stored_id,
        route,
        args or "plain",
        normalised.get("headline"),
    )
    return {
        "stored_session_id": stored_id,
        "mode": "in_place",
        "route": route,
        "args": args,
        "snapshot": snapshot_id_row,
        **normalised,
    }
