"""Pure state-machine operations on the `background_tasks` ledger (P2-1, B-42)."""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from domain.models import BackgroundTask, utcnow

logger = logging.getLogger(__name__)

STATE_RUNNING = "running"
STATE_FINISHED = "finished"
STATE_ORPHANED = "orphaned"


def record_submitted(
    db: OrmSession,
    *,
    task_id: str,
    stored_session_id: str,
    prompt_text: str,
    connection_generation: int | None,
    now: datetime | None = None,
) -> BackgroundTask:
    """Write the submit-side ledger row -- the task's only durable record."""
    existing = db.get(BackgroundTask, task_id)
    if existing is not None:
        logger.warning(
            "background task id %r already in the ledger (state %r, submitted %s); "
            "Hermes re-minted the id -- overwriting the stale row with the new submit",
            task_id,
            existing.state,
            existing.submitted_at,
        )
        db.delete(existing)
        db.flush()
    row = BackgroundTask(
        task_id=task_id,
        stored_session_id=stored_session_id,
        prompt_text=prompt_text,
        submitted_at=now if now is not None else utcnow(),
        state=STATE_RUNNING,
        connection_generation=connection_generation,
    )
    db.add(row)
    return row


def record_completed(
    db: OrmSession,
    *,
    task_id: str,
    result_text: str | None,
    stored_session_id_hint: str | None = None,
    now: datetime | None = None,
) -> BackgroundTask:
    """Apply one observed `background.complete` to the ledger."""
    finished_at = now if now is not None else utcnow()
    row = db.get(BackgroundTask, task_id)
    if row is None:
        logger.warning(
            "background.complete for task %r has no submit row in the ledger "
            "(submitted outside this gateway, or before the ledger existed); "
            "recording the completion so the result text is not lost",
            task_id,
        )
        row = BackgroundTask(
            task_id=task_id,
            stored_session_id=stored_session_id_hint,
            prompt_text=None,
            submitted_at=finished_at,
            state=STATE_FINISHED,
            result_text=result_text,
            finished_at=finished_at,
        )
        db.add(row)
        return row

    if row.state == STATE_FINISHED:
        logger.warning(
            "duplicate background.complete for task %r; keeping the first result",
            task_id,
        )
        return row

    row.state = STATE_FINISHED
    row.result_text = result_text
    row.finished_at = finished_at
    if row.stored_session_id is None and stored_session_id_hint:
        row.stored_session_id = stored_session_id_hint
    return row


def orphan_all_running(db: OrmSession) -> list[BackgroundTask]:
    """`running -> orphaned` for every running row; returns the rows touched."""
    rows = list(db.scalars(select(BackgroundTask).where(BackgroundTask.state == STATE_RUNNING)))
    for row in rows:
        row.state = STATE_ORPHANED
    return rows


def stored_session_ids_with_unfinished_tasks(db: OrmSession) -> list[str]:
    """Distinct stored session ids that have a running or orphaned task."""
    stmt = (
        select(BackgroundTask.stored_session_id)
        .where(BackgroundTask.state.in_((STATE_RUNNING, STATE_ORPHANED)))
        .where(BackgroundTask.stored_session_id.is_not(None))
        .distinct()
    )
    return [value for value in db.scalars(stmt) if value]


def tasks_for_session(db: OrmSession, stored_session_id: str) -> list[BackgroundTask]:
    """Every ledger row for one stored session, newest submit first."""
    stmt = (
        select(BackgroundTask)
        .where(BackgroundTask.stored_session_id == stored_session_id)
        .order_by(BackgroundTask.submitted_at.desc(), BackgroundTask.task_id)
    )
    return list(db.scalars(stmt))


def all_tasks(db: OrmSession, *, limit: int = 500) -> list[BackgroundTask]:
    """The global ledger, newest submit first, bounded."""
    stmt = (
        select(BackgroundTask)
        .order_by(BackgroundTask.submitted_at.desc(), BackgroundTask.task_id)
        .limit(limit)
    )
    return list(db.scalars(stmt))


def finished_results_for_session(db: OrmSession, stored_session_id: str) -> list[BackgroundTask]:
    """Finished rows with a result, oldest first -- the transcript-injection feed."""
    stmt = (
        select(BackgroundTask)
        .where(
            BackgroundTask.stored_session_id == stored_session_id,
            BackgroundTask.state == STATE_FINISHED,
            BackgroundTask.result_text.is_not(None),
        )
        .order_by(BackgroundTask.finished_at, BackgroundTask.task_id)
    )
    return list(db.scalars(stmt))
