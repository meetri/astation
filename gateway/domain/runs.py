"""Pure operations on the `runs` / `run_events` tables (P2-2).

Same contract as `domain/background_tasks.py`: every function takes an open
SQLAlchemy session and does no I/O beyond it -- no network, no clock it does
not accept as a parameter, and **no `commit()`**; the caller owns transaction
boundaries. That is what keeps these unit-testable against an in-memory
database and lets `api/runs.py::RunRecorder` batch all the rows for one
forwarded frame into one commit (the P2-2b batching decision).

The run state machine, one row per Hermes turn, flat (no tree -- P2-2a):

    running --(message.completed observed)----------------> completed
    running --(gateway restart / Hermes reconnect)--------> interrupted

* A run OPENS on the first turn-activity frame attributable to a stored
  session with no run already open (the wire has no `turn.started`; the
  first frame of a turn is usually `reasoning.delta` or `message.started`).
* `message.completed` is the turn's one end-of-turn signal on the wire
  (B-38: it carries `final_response`; there is exactly one per turn), so it
  closes the run.
* **`interrupted` is the honest state for a run whose stream died under
  it.** The wire has no run-failed signal at all (ROADMAP: "'Run failed' is
  dropped -- no such signal exists"), and after a gateway restart or a
  Hermes reconnect the gateway provably missed frames (events are neither
  buffered nor replayed -- P2-0b). The turn itself most likely finished on
  the Hermes side; what ended is this gateway's *view* of it. Same honesty
  rule as the background ledger's `orphaned`.

Per-run `seq` (P2-2c) is allocated by the caller (`RunRecorder` holds the
counter for each open run, starting at 1 on open); this module only stores
and queries it. It survives restarts *structurally*: a restart interrupts
every open run and any later activity opens a NEW run with a fresh counter,
so no counter ever has to be recovered -- and the rows already written keep
their numbers forever, which is what makes `after_seq` a durable cursor.
"""

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
#: The turn ended with Hermes reporting an error on its closing frame
#: (`message.completed` `status: "error"`, B-186). Measured 2026-09-17 on the
#: gateway's own ledger: 22 of 198 recorded closes were errors, all from the
#: provider ("Context length exceeded: max compression attempts (3) reached",
#: "Response truncated due to output length limit"), and every one had been
#: closed as `completed`.
STATUS_FAILED = "failed"

#: Every run this recorder writes is one Hermes turn (P2-2a: flat, no tree).
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
    """Create the Run row for a turn that just started producing frames.

    `runtime_session_id` is the Hermes STORED id and is always present;
    `workspace_session_id`/`project_id` are the filing lookup's answer and
    are both None for an unfiled session (P2-2d).

    `profile` is which Hermes connection the turn was observed on (B-136).
    Open time is the only moment it is known for certain -- see
    `Run.profile`'s own comment for why it cannot be recovered later.
    """
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


#: Key under `runs.command_json` marking a run closed by the B-84/B-62
#: staleness reconciliation, holding the honest note verbatim. The runs
#: schema has no dedicated detail/explanation column and this fix adds no
#: migration; `command_json` is the one JSON column on the row, never
#: written for `turn`-kind runs (the recorder writes no command), so a
#: reserved key here is a durable marker that cannot collide with real
#: command payloads. `run_row` surfaces it as `status_explanation`.
STALE_CLOSE_NOTE_KEY = "stale_close_note"


def close_run_stale(run: Run, *, note: str, now: datetime | None = None) -> Run:
    """`running -> completed` for a run whose completion event never arrived.

    The B-84/B-62 reconciliation path: Hermes reports the run's session idle
    (or no longer live at all) well after the grace period, so the turn is
    over -- only the `message.completed` frame went missing. Unlike
    `close_run` (a signal observed on the wire), this records WHY the run
    was closed, durably, so the row stays honest forever.
    """
    run.status = STATUS_COMPLETED
    run.ended_at = now if now is not None else utcnow()
    detail = dict(run.command_json) if isinstance(run.command_json, dict) else {}
    detail[STALE_CLOSE_NOTE_KEY] = note
    run.command_json = detail
    return run


def stale_close_note(run: Run) -> str | None:
    """The reconciliation note recorded by `close_run_stale`, or None."""
    return _note(run, STALE_CLOSE_NOTE_KEY)


#: Key under `runs.command_json` for a close the wire itself explained
#: (B-186): the closing frame said `error` (-> `failed`) or `interrupted`
#: (-> `interrupted`), and the note is what it said. Same reserved-key
#: mechanism as `STALE_CLOSE_NOTE_KEY`, same reason (no explanation column,
#: no migration).
CLOSE_NOTE_KEY = "close_note"


def close_run_abnormal(run: Run, *, status: str, note: str, now: datetime | None = None) -> Run:
    """`running -> failed | interrupted`, as reported by the closing frame.

    `close_run` is for a turn whose `message.completed` said `complete`.
    This is for the other two statuses the frame carries (measured live:
    `error` with an `error` text, and `interrupted`), recording WHY on the
    row so `run_row` can say it forever.
    """
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
    """`running -> interrupted` for every running run; returns the rows touched.

    Called on gateway startup (frames emitted while the process was down are
    gone -- nothing buffers them) and on every Hermes connection-generation
    change (same measurement, P2-0b). Terminal: unlike a background task,
    a run's identity is "one contiguous observed turn", so a post-reconnect
    frame belongs to a *new* run rather than resurrecting this one.
    """
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
