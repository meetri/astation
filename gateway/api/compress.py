"""`POST /api/sessions/{stored}/compress` -- compress a session's context, in place (P6-4).

Owner, 2026-09-04: *"Really I just want to compress the context."* Everything here rests on
the live measurement in `docs/PROTOCOL_VERIFIED.md`, "Compress + the LCM context engine --
verified live 2026-09-04":

* The instance's context engine is the **hermes-lcm** plugin. A compress is **in place**:
  same session, same stored id, same live handle, no continuation session. Under LCM only raw
  messages **outside the fresh tail** (32 messages / 24k tokens on the owner's host) are
  eligible, so a short session is an honest no-op and Hermes says so
  (`summary.noop`, headline "No changes from compression: N messages").
* Two upstream routes, both taking the LIVE handle: `session.compress` (plain; every argument
  is silently ignored) and `command.dispatch {name: "/compress", args: "here N"}` (the partial
  form -- keeps the last N exchanges verbatim; `args` is the field, `arg` is ignored). Dispatch
  answers `{"type": "exec", "output": "<headline>\\n<token line>"}`, `session.compress` answers
  a structured dict; this route normalises both into ONE response shape so the app does not
  care which ran.
* `[4009] session busy -- /interrupt the current turn before /compress` while a turn runs ->
  **409**. The gateway's own `RunRecorder.has_open_run()` is checked first for the same answer
  without a round trip (it only sees turns made through this gateway, so the upstream code is
  still mapped).
* **A `pre_compress` snapshot is taken first** (P6-3's fourth trigger). Under LCM the raw rows
  survive in `lcm.db` for the *agent*, but the transcript this app can read is the compacted
  one, so the snapshot is the reader's full copy. Snapshot failure -> 409, nothing compressed
  (the delete route's rule). `?force=true` skips it.
* Completion, for a client watching the event stream: `status.update kind=compacted` (measured
  order: `compressing` -> `compacting` -> `compacted` -> `session.info` -> `status` "ready").
  The RPC itself returns after the work on this build, but the app keys its transcript reload
  on the event, not on this response, so a future asynchronous compress needs no client change.

Not built: `--preview` (a `[-32603] internal error` on this build) and a focus topic (accepted
by dispatch, effect unmeasured -- `focus` is passed through when given, honestly labelled).
"""

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

#: Hermes: `session busy -- /interrupt the current turn before /compress` (measured 2026-09-04).
HERMES_SESSION_BUSY_CODE = 4009
#: The slash command `command.dispatch` executes. `args` carries the raw argument string.
COMPRESS_COMMAND_NAME = "/compress"
#: Hermes's own "nothing happened" headline prefix (both routes print it).
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
            # `--preview` is an internal error upstream and any `--flag` would
            # be parsed as an option, not a topic. A topic never starts with '-'.
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
    """Normalise `command.dispatch`'s two-line text into the structured shape.

    Measured lines (2026-09-04):
      "Compressed: 23 → 22 messages\\nApprox request size: ~66,322 → ~66,271 tokens"
      "No changes from compression: 22 messages\\nApprox request size: ~59,724 tokens (unchanged)"
    Anything else is passed through verbatim as `headline` with the numbers `None` --
    never invented.
    """
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
    """Snapshot, then compress in place. See the module docstring for the contract.

    Response (both upstream routes normalised):
    `{"stored_session_id", "mode": "in_place", "route": "session.compress" |
    "command.dispatch", "args": <str|null>, "snapshot": <row>|null, "noop", "headline",
    "token_line", "removed", "before_messages", "after_messages", "before_tokens",
    "after_tokens", "usage": {context_used, context_max, context_percent, compressions}|null}`.
    409 mid-turn (local `has_open_run` or Hermes `[4009]`) or when the pre-compress snapshot
    failed (`{"detail", "snapshot_error"}`); 404 unknown session; 422 bad id/body; 502 upstream.
    """
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
