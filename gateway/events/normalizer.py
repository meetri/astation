"""Pure normalization of raw Hermes JSON-RPC events into canonical events.

`normalize_event` does no I/O: no DB session, no network, no clock
dependency it doesn't accept as a parameter. It only shapes data. Persisting
the result is `events/persistence.py`'s job (see P0-3 in `TASKS.md`).
"""

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
    """Extract the event name from a raw Hermes event dict.

    The transport is "newline-delimited JSON-RPC 2.0" (`PROTOCOL_VERIFIED.md`),
    whose notification shape is `{"jsonrpc": "2.0", "method": ..., "params":
    ...}`. That exact envelope is not yet confirmed against a live capture
    for *events* specifically (only for method calls), so this also accepts
    a flatter `{"type": ...}` shape defensively -- whichever the real
    adapter turns out to hand us, one of these two keys will carry the name.

    Public because the caller needs the same answer this module used in
    order to say *which* type it just declined to normalize (B-14's
    once-per-unknown-type logging); re-deriving it at the call site would
    let the two disagree.
    """
    return raw_event.get("method") or raw_event.get("type")


def _raw_params(raw_event: dict[str, Any]) -> dict[str, Any]:
    """Extract the event body, tolerating a couple of plausible shapes."""
    params = raw_event.get("params", raw_event.get("payload"))
    if isinstance(params, dict):
        return params
    # Fall back to "everything except the envelope keys" so a flattened
    # event (name + fields at the top level, no nested params/payload) still
    # yields a usable payload instead of an empty one. This also covers the
    # case where `params`/`payload` is *present* but not a dict (e.g. an
    # explicit `"params": null`) -- those keys must be excluded here too, or
    # the literal non-dict value would leak into the payload alongside the
    # flattened fields.
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
    """Map one raw Hermes event dict to one `CanonicalEvent`.

    Returns `None` for any raw event name not in `RAW_TO_CANONICAL_TYPE`.
    Returning `None` rather than raising lets the caller pump an arbitrary
    Hermes event stream through this function without needing its own
    allow-list of "event types I know this handles".

    A `None` covers two *different* situations, and the caller is expected
    to tell them apart via `DELIBERATELY_DROPPED_RAW_EVENTS`: a name
    listed there was dropped on purpose, for the reason recorded next to it,
    while any other `None` is a type nobody has looked at yet and should be
    logged once so it becomes discoverable. Keeping the *decision* here and
    the *logging* at the call site is what keeps this function pure.

    Args:
        raw_event: one decoded JSON-RPC line from Hermes.
        context: the workspace-side identifiers Hermes itself doesn't carry
            (project/session/run id, this event's seq number, and the
            active turn id if any) -- see `EventContext`.
        now: injectable clock for tests; defaults to real UTC now.

    Returns:
        A `CanonicalEvent` ready for the broadcast path (which attributes it
        and, for the types in `events.persistence.PERSISTED_RUN_EVENT_TYPES`,
        persists it), or `None` if `raw_event`'s method/type isn't one this
        normalizer handles.
    """
    method = raw_event_type(raw_event)
    if method is None or method not in RAW_TO_CANONICAL_TYPE:
        return None

    canonical_type = RAW_TO_CANONICAL_TYPE[method]
    payload = dict(_raw_params(raw_event))

    if method in MERGED_RAW_EVENTS:
        # Several raw names share this canonical type (see MERGED_RAW_EVENTS).
        # Record which one produced this frame so the merge stays lossless.
        # Assigned, not `setdefault`: the `_`-prefixed namespace is the
        # gateway's (same convention as B-02's `_truncated` marker), so an
        # upstream key of the same name must not be able to lie about
        # provenance.
        payload["_raw_type"] = method

    if method in EXPIRY_RAW_EVENTS:
        # See canonical.py's module docstring: an expiry *is* a resolution,
        # just not one that went through the matching `*.respond` RPC.
        payload.setdefault("resolution", "expired")

    if method in CANCEL_RAW_EVENTS:
        # Hermes says WHY it withdrew the prompt (`timeout`,
        # `interrupted`, `shutdown`). Keep its word rather than flattening
        # every withdrawal to "expired" -- "the turn was interrupted" and
        # "you took too long" are different things to tell someone.
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
