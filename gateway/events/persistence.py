"""Write run-relevant canonical events into the `run_events` table (P2-2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from domain.models import RunEvent

PERSISTED_RUN_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "message.started",
        # message.interim carries each segment's full text, so no segment is lost.
        "message.interim",
        "message.completed",
        "tool.generating",
        "tool.started",
        "tool.progress",
        "tool.completed",
        # approval.resolved and clarify.resolved are gateway-authored; no raw producer.
        "approval.requested",
        "approval.resolved",
        "clarify.requested",
        "clarify.resolved",
        "sudo.requested",
        "sudo.resolved",
        "secret.requested",
        "secret.resolved",
        "status.update",
        "session.usage",
    }
)

NOT_PERSISTED_EVENT_TYPES: dict[str, str] = {
    "message.delta": (
        "token-scale (328 frames in one measured turn) and fully redundant: "
        "the finished text lands on the persisted message.interim/"
        "message.completed rows, and the authoritative transcript is "
        "Hermes's own, served verbatim (B-34). Persisting every token would "
        "bury the run timeline and triple the pump's write volume for data "
        "with no reader."
    ),
    "reasoning.delta": (
        "the highest-volume stream on the wire (945 frames vs 328 "
        "message.delta in one measured turn), display-only; the "
        "authoritative reasoning text for a turn arrives once on "
        "message.completed's payload (B-14/B-35) and is persisted there."
    ),
    "thinking.status": (
        "a transient TUI status label with replace-not-append semantics and "
        "no durable meaning ('( ͡° ͜ʖ ͡°) brainstorming...', then '' to "
        "clear it)."
    ),
    "session.updated": (
        "session lifecycle, not run activity: session.info fires on every "
        "resume and session.title on renames, neither belongs to a turn, and "
        "the durable home for session metadata is the sessions table / "
        "Hermes itself -- not a run's event log."
    ),
    "background.completed": (
        "the background-task LEDGER owns this event (P2-1/B-42: "
        "`background_tasks.result_text` is the durable copy, written by "
        "BackgroundLedger.handle_completed before fan-out). A background "
        "task is not a turn -- the wire is silent while it runs, so there is "
        "no run to attach it to, and a second copy in run_events would just "
        "be a divergence risk."
    ),
}


def is_persisted_event_type(event_type: str) -> bool:
    """Whether this canonical type earns a `run_events` row."""
    return event_type in PERSISTED_RUN_EVENT_TYPES


def persist_run_event(
    db: Session,
    *,
    run_id: str,
    seq: int,
    event_type: str,
    payload: dict[str, Any] | None,
    timestamp: datetime,
) -> RunEvent:
    """Write one `run_events` row."""
    # No commit or flush: the caller batches a whole forwarded frame into one commit.
    run_event = RunEvent(
        run_id=run_id,
        # The caller-allocated per-run cursor, never the broadcaster's global counter.
        seq=seq,
        event_type=event_type,
        payload_json=payload,
        timestamp=timestamp,
    )
    db.add(run_event)
    return run_event
