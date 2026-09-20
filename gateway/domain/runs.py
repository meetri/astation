"""Pure operations on the `runs` / `run_events` tables (P2-2)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from domain.models import Run, RunEvent, utcnow

logger = logging.getLogger(__name__)

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"

RUN_KIND_TURN = "turn"


def open_run(
    db: OrmSession,
    *,
    runtime_session_id: str,
    workspace_session_id: str | None,
    project_id: str | None,
    profile: str = "default",
    now: datetime | None = None,
) -> Run:
    """Create the Run row for a turn that just started producing frames."""
    run = Run(
        project_id=project_id,
        session_id=workspace_session_id,
        runtime_session_id=runtime_session_id,
        profile=profile,
        kind=RUN_KIND_TURN,
        status=STATUS_RUNNING,
        started_at=now if now is not None else utcnow(),
    )
    db.add(run)
    return run


def close_run(run: Run, *, now: datetime | None = None) -> Run:
    """`running -> completed`, on the turn's `message.completed`."""
    run.status = STATUS_COMPLETED
    run.ended_at = now if now is not None else utcnow()
    return run


STALE_CLOSE_NOTE_KEY = "stale_close_note"


def close_run_stale(run: Run, *, note: str, now: datetime | None = None) -> Run:
    """`running -> completed` for a run whose completion event never arrived."""
    run.status = STATUS_COMPLETED
    run.ended_at = now if now is not None else utcnow()
    detail = dict(run.command_json) if isinstance(run.command_json, dict) else {}
    detail[STALE_CLOSE_NOTE_KEY] = note
    run.command_json = detail
    return run


def stale_close_note(run: Run) -> str | None:
    """The reconciliation note recorded by `close_run_stale`, or None."""
    return _note(run, STALE_CLOSE_NOTE_KEY)


CLOSE_NOTE_KEY = "close_note"


def close_run_abnormal(run: Run, *, status: str, note: str, now: datetime | None = None) -> Run:
    """`running -> failed | interrupted`, as reported by the closing frame."""
    if status not in (STATUS_FAILED, STATUS_INTERRUPTED):
        raise ValueError(f"not an abnormal close status: {status!r}")
    run.status = status
    run.ended_at = now if now is not None else utcnow()
    detail = dict(run.command_json) if isinstance(run.command_json, dict) else {}
    detail[CLOSE_NOTE_KEY] = note
    run.command_json = detail
    return run


def close_note(run: Run) -> str | None:
    """The wire-reported close note (`close_run_abnormal`), or None."""
    return _note(run, CLOSE_NOTE_KEY)


def _note(run: Run, key: str) -> str | None:
    detail = run.command_json
    if not isinstance(detail, dict):
        return None
    note = detail.get(key)
    return note if isinstance(note, str) else None


def interrupt_all_running(db: OrmSession, *, now: datetime | None = None) -> list[Run]:
    """`running -> interrupted` for every running run; returns the rows touched."""
    ended_at = now if now is not None else utcnow()
    rows = list(db.scalars(select(Run).where(Run.status == STATUS_RUNNING)))
    for row in rows:
        row.status = STATUS_INTERRUPTED
        row.ended_at = ended_at
    return rows


def list_runs(
    db: OrmSession,
    *,
    runtime_session_id: str | None = None,
    project_id: str | None = None,
    limit: int = 100,
) -> list[Run]:
    """Newest-first run list, optionally filtered (P2-2e's GET /api/runs)."""
    stmt = select(Run)
    if runtime_session_id is not None:
        stmt = stmt.where(Run.runtime_session_id == runtime_session_id)
    if project_id is not None:
        stmt = stmt.where(Run.project_id == project_id)
    stmt = stmt.order_by(Run.started_at.desc(), Run.id.desc()).limit(limit)
    return list(db.scalars(stmt))


def events_after(
    db: OrmSession, run_id: str, *, after_seq: int = 0, limit: int = 1000
) -> list[RunEvent]:
    """This run's events with `seq > after_seq`, ascending -- the cursor read."""
    stmt = (
        select(RunEvent)
        .where(RunEvent.run_id == run_id, RunEvent.seq > after_seq)
        .order_by(RunEvent.seq)
        .limit(limit)
    )
    return list(db.scalars(stmt))


def last_seq_by_run(db: OrmSession, run_ids: list[str]) -> dict[str, int]:
    """`{run_id: max(seq)}` for the given runs, one query; absent = no events."""
    if not run_ids:
        return {}
    rows = db.execute(
        select(RunEvent.run_id, func.max(RunEvent.seq))
        .where(RunEvent.run_id.in_(run_ids))
        .group_by(RunEvent.run_id)
    ).all()
    return {run_id: int(max_seq) for run_id, max_seq in rows if max_seq is not None}


def payload_dict(event: RunEvent) -> dict[str, Any]:
    """The stored payload as a dict, tolerating legacy/odd JSON values."""
    payload = event.payload_json
    return payload if isinstance(payload, dict) else {}
