"""Pure state-machine operations on the `background_tasks` ledger (P2-1, B-42).

Every function here takes an open SQLAlchemy session and does no I/O beyond
it -- no network, no clock it does not accept as a parameter, and **no
`commit()`**: the caller owns transaction boundaries, exactly like
`events/persistence.py`. That is what keeps these unit-testable against an
in-memory database and lets the event-pump hook batch its write with the
frame it is handling.

The state machine, and why it is shaped this way (PV "Phase 2 probe" +
"Phase 2a probes", both measured live 2026-08-30):

    running --(background.complete observed)------------> finished
    running --(gateway restart / Hermes reconnect)------> orphaned
    orphaned --(background.complete observed anyway)----> finished

* `running -> orphaned` on every gateway restart and on every
  Hermes-connection-generation change, because the probe proved the
  completion event does **not** survive a reconnect by default: it is
  delivered only to connections attached to the session at the instant it
  fires, never buffered, never replayed (3/3 lost on a bare reconnect).
  "Orphaned" is honest copy for "submitted before a reconnect; the work
  likely completed, but the outcome is unknown".
* `orphaned -> finished` is deliberately legal, because the same probe
  proved the loss is rescuable: re-resuming the session on the new
  connection *before* the task completes re-attaches it, and the completion
  then arrives normally (2/2). The rescue lives in
  `api/background.py`'s ledger orchestration; this module only has to
  accept the late good news.
* `finished` is terminal. A second completion for the same task id is
  logged and ignored rather than overwriting a result the owner may have
  already read.
"""

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
    """Write the submit-side ledger row -- the task's only durable record.

    Upserts rather than inserting blindly: `task_id` is `bg_` + 6 hex, a
    small space Hermes could re-mint across restarts, and a PK collision must
    not turn a successful submit into a 500 after the task is already
    running. A collision overwrites the stale row (it can no longer complete
    -- its announcement window is gone) and is logged loudly.
    """
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
        # The delete must land before the insert of the same PK in the same
        # flush, or SQLAlchemy's unit of work may order them insert-first.
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
    """Apply one observed `background.complete` to the ledger.

    `running -> finished` is the normal path; `orphaned -> finished` is the
    rescued-reconnect path (see module docstring). A completion with no
    matching submit row still gets a row -- the event is the only copy of the
    result anywhere, and discarding it because *we* never saw the submit
    would re-create B-42 for tasks submitted outside this gateway.
    `stored_session_id_hint` is the broadcaster's live-handle attribution for
    that case; the ledger's own submit-time record always wins, because the
    completion frame's live handle can be unmapped or re-minted by the time
    it fires.
    """
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
    """`running -> orphaned` for every running row; returns the rows touched.

    Called on gateway startup (a completion that fired while the process was
    down is unrecoverable) and on every Hermes connection-generation change
    (the completion is delivered only to attached connections -- P2-0b). The
    caller is expected to follow the generation-change case with the
    re-resume rescue, which can still upgrade these rows to `finished`.
    """
    rows = list(db.scalars(select(BackgroundTask).where(BackgroundTask.state == STATE_RUNNING)))
    for row in rows:
        row.state = STATE_ORPHANED
    return rows


def stored_session_ids_with_unfinished_tasks(db: OrmSession) -> list[str]:
    """Distinct stored session ids that have a running or orphaned task.

    This is the rescue's target list: re-resuming these sessions on the new
    connection is what lets a still-running task's completion arrive at all
    (PV "Phase 2a probes": delivered 2/2 with a pre-completion re-resume,
    0/3 without). Orphaned rows are included on purpose -- orphaning happens
    *at* the reconnect, so the rows the rescue exists for are already
    orphaned by the time it runs.
    """
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
    """Finished rows with a result, oldest first -- the transcript-injection feed.

    Oldest-first because these are appended to the end of a transcript in the
    order the results arrived.
    """
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
