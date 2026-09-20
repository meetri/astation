"""Pure normalization of raw Hermes JSON-RPC events into canonical events."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from .canonical import (
    CANCEL_RAW_EVENTS,
    EXPIRY_RAW_EVENTS,
    MERGED_RAW_EVENTS,
    RAW_TO_CANONICAL_TYPE,
    CanonicalEvent,
    EventContext,
)


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def raw_event_type(raw_event: dict[str, Any]) -> str | None:
    """Extract the event name from a raw Hermes event dict."""
    return raw_event.get("method") or raw_event.get("type")


def _raw_params(raw_event: dict[str, Any]) -> dict[str, Any]:
    """Extract the event body, tolerating a couple of plausible shapes."""
    params = raw_event.get("params", raw_event.get("payload"))
    if isinstance(params, dict):
        return params
    return {
        k: v
        for k, v in raw_event.items()
        if k not in ("method", "type", "jsonrpc", "id", "params", "payload")
    }


def normalize_event(
    raw_event: dict[str, Any],
    context: EventContext,
    *,
    now: datetime | None = None,
) -> CanonicalEvent | None:
    """Map one raw Hermes event dict to one `CanonicalEvent`."""
    method = raw_event_type(raw_event)
    if method is None or method not in RAW_TO_CANONICAL_TYPE:
        return None

    canonical_type = RAW_TO_CANONICAL_TYPE[method]
    payload = dict(_raw_params(raw_event))

    if method in MERGED_RAW_EVENTS:
        payload["_raw_type"] = method

    if method in EXPIRY_RAW_EVENTS:
        payload.setdefault("resolution", "expired")

    if method in CANCEL_RAW_EVENTS:
        reason = payload.get("reason")
        payload["resolution"] = reason if isinstance(reason, str) and reason else "cancelled"

    timestamp = now if now is not None else datetime.now(UTC)

    return CanonicalEvent(
        event_id=_new_event_id(),
        project_id=context.project_id,
        session_id=context.session_id,
        run_id=context.run_id,
        seq=context.seq,
        type=canonical_type,
        timestamp=timestamp,
        payload=payload,
    )
