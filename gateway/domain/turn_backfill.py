"""Give pre-existing chat rows their run id, from timestamps alone.

## The gap

`chat_messages.turn_id` is the run a row belongs to (`runs.id` -- one run is
one turn, `domain/runs.py`). It has been written live only since 2026-09-18
(B-188: the recorder stamps `run_id` on the envelope before `ChatStore`
captures it; `ChatStore.attach_turn` links the submit route's user row when
the run opens; B-190 stamps the foreign-prompt row). Measured on the deployed
gateway the same day: ~106 sessions of rows captured since ~2026-09-05, all
with a null `turn_id` except the rows captured that day -- so the
run-structured chat (RUN_EVENT_UX_AUDIT §6) has no turn boundaries for any
history the store already holds.

Both tables carry the timestamps to recover the link: every row has
`created_at` (`utcnow()` at write time), every run has `started_at` and,
once closed, `ended_at`. A row captured off the live stream was captured
*while its run was open*, so the run whose window holds the row's timestamp
is its run.

## The rule (`assign_turn_ids`)

For one session -- one `(profile, stored_session_id)` -- given its runs
sorted by `started_at` and the rows with no turn yet:

* A run's **window** is `[started_at - SLACK, end + SLACK]`, where `end` is
  `ended_at`, else the next run's `started_at`, else +inf. `SLACK` (2 s)
  absorbs the gap between the recorder's clock (the frame's timestamp) and
  the store's (`utcnow()` at insert), which are not the same clock.
* A **non-user row** (assistant, tool, marker) belongs to the run whose
  window contains `created_at`. A row inside two windows (the previous run
  ended within SLACK of the next opening) goes to the run whose *own*
  span `[started_at, end]` holds it, else the earlier of the two: a
  `message.completed` row is written just after its run closes, so the
  gap after a run belongs to that run.
* A **user row from the submit route** (`source == "submit"`) is written
  BEFORE Hermes is called, so its run opens after it: it belongs to the
  next run starting at or after `created_at - SLACK`, unless a window
  already contains it (the usual case -- the first frame lands well within
  the slack -- and the queued-behind-a-running-turn case, where the spec
  keeps the containing run).
* A **user row read back from Hermes** (`source == "backfill"`) belongs to
  the containing run, else the nearest run starting after it.
* A **compaction marker** (`role == "marker"`, `compacted`) outside every
  window keeps null: a compaction between turns belongs to no turn, and the
  app renders it between groups. Every other unmatched row keeps null too.
* Runs recorded on **another profile** never match: `request_id`s, session
  ids and turns are all per-Hermes-process, and the store keys sessions on
  `(profile, stored_session_id)` for the same reason.

Naive datetimes are read as UTC (SQLite returns `DateTime(timezone=True)`
columns naive; both writers only ever write UTC -- `api/runs.py::
_run_age_seconds` makes the same call).

## The walker (`backfill_turn_ids`)

Every session with at least one null-turn row AND at least one run gets the
rule applied; only rows that received an id are written; one commit per
session, so a failure mid-way keeps what was already linked. Idempotent: a
second pass finds no null rows it can place and writes nothing. Rows that
already have a turn are never touched. Wired into the lifespan
(`api/bootstrap.py::run_turn_backfill`) after the chat store and recorder
exist and before any pump starts, best-effort: the report is logged and a
failure is logged and ignored -- a backfill can never keep the gateway from
starting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from domain.models import ChatMessage, Run

logger = logging.getLogger(__name__)

#: Clock slack either side of a run's span (module docstring).
SLACK = timedelta(seconds=2)

#: `ChatMessage.source` values with their own placement rule.
SOURCE_SUBMIT = "submit"
SOURCE_BACKFILL = "backfill"
ROLE_USER = "user"
ROLE_MARKER = "marker"


@dataclass(frozen=True)
class RunWindow:
    """What the rule needs from one run. `from_run` reads it off the ORM row."""

    id: str
    profile: str
    started_at: datetime
    ended_at: datetime | None = None

    @classmethod
    def from_run(cls, run: Run) -> RunWindow | None:
        """None for a run with no `started_at` -- it has no window to offer."""
        if run.started_at is None:
            return None
        return cls(
            id=run.id,
            profile=run.profile or "default",
            started_at=_as_utc(run.started_at),
            ended_at=_as_utc(run.ended_at) if run.ended_at is not None else None,
        )


@dataclass(frozen=True)
class RowToLink:
    """What the rule needs from one null-turn chat row."""

    id: str
    profile: str
    role: str
    source: str
    created_at: datetime
    compacted: bool = False

    @classmethod
    def from_row(cls, row: ChatMessage) -> RowToLink:
        return cls(
            id=row.id,
            profile=row.profile,
            role=row.role,
            source=row.source,
            created_at=_as_utc(row.created_at),
            compacted=bool(row.compacted),
        )


@dataclass
class BackfillReport:
    """What one `backfill_turn_ids` pass did."""

    sessions_scanned: int = 0
    rows_assigned: int = 0
    rows_left_null: int = 0
    #: Sessions whose pass raised (logged, rolled back, skipped). Not part
    #: of `sessions_scanned`.
    sessions_failed: int = 0


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True)
class _Span:
    run_id: str
    start: datetime
    end: datetime | None  # None: still open, and no later run bounds it

    def holds(self, when: datetime, slack: timedelta) -> bool:
        if when < self.start - slack:
            return False
        return self.end is None or when <= self.end + slack


def _spans(runs: Iterable[RunWindow]) -> list[_Span]:
    ordered = sorted(runs, key=lambda run: (run.started_at, run.id))
    spans: list[_Span] = []
    for index, run in enumerate(ordered):
        end = run.ended_at
        if end is None and index + 1 < len(ordered):
            end = ordered[index + 1].started_at
        spans.append(_Span(run_id=run.id, start=run.started_at, end=end))
    return spans


def _containing(spans: Sequence[_Span], when: datetime, slack: timedelta) -> str | None:
    """The run whose own span holds `when`, else the first whose slack window does."""
    for span in spans:
        if span.holds(when, timedelta(0)):
            return span.run_id
    for span in spans:
        if span.holds(when, slack):
            return span.run_id
    return None


def _next_starting_at_or_after(spans: Sequence[_Span], when: datetime) -> str | None:
    for span in spans:  # sorted by start
        if span.start >= when:
            return span.run_id
    return None


def assign_turn_ids(
    runs: Iterable[RunWindow], rows: Iterable[RowToLink], *, slack: timedelta = SLACK
) -> dict[str, str]:
    """`{row id: run id}` for every row the rule can place (module docstring).

    Pure: no I/O, no clock. Rows it cannot place are simply absent from the
    result -- the caller leaves their `turn_id` null. Runs are grouped by
    profile and a row only ever sees the runs of its own profile.
    """
    by_profile: dict[str, list[RunWindow]] = {}
    for run in runs:
        by_profile.setdefault(run.profile, []).append(run)
    spans_by_profile = {profile: _spans(group) for profile, group in by_profile.items()}

    assigned: dict[str, str] = {}
    for row in rows:
        spans = spans_by_profile.get(row.profile)
        if not spans:
            continue
        run_id = _containing(spans, row.created_at, slack)
        if run_id is None and row.role == ROLE_USER:
            if row.source == SOURCE_SUBMIT:
                run_id = _next_starting_at_or_after(spans, row.created_at - slack)
            elif row.source == SOURCE_BACKFILL:
                run_id = _next_starting_at_or_after(spans, row.created_at)
        if run_id is not None:
            assigned[row.id] = run_id
    return assigned


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


def _sessions_to_backfill(db: OrmSession) -> list[tuple[str, str]]:
    """Every `(profile, stored_session_id)` with a null-turn row and a run."""
    stmt = (
        select(ChatMessage.profile, ChatMessage.stored_session_id)
        .join(
            Run,
            (Run.runtime_session_id == ChatMessage.stored_session_id)
            & (Run.profile == ChatMessage.profile),
        )
        .where(ChatMessage.turn_id.is_(None))
        .distinct()
        .order_by(ChatMessage.profile, ChatMessage.stored_session_id)
    )
    return [(profile, stored_id) for profile, stored_id in db.execute(stmt).all()]


def _backfill_session(db: OrmSession, profile: str, stored_id: str) -> tuple[int, int]:
    """Apply the rule to one session; returns `(assigned, left null)`. No commit."""
    run_rows = db.scalars(
        select(Run)
        .where(Run.runtime_session_id == stored_id, Run.profile == profile)
        .order_by(Run.started_at, Run.id)
    ).all()
    windows = [w for w in (RunWindow.from_run(run) for run in run_rows) if w is not None]
    rows = db.scalars(
        select(ChatMessage)
        .where(
            ChatMessage.profile == profile,
            ChatMessage.stored_session_id == stored_id,
            ChatMessage.turn_id.is_(None),
        )
        .order_by(ChatMessage.seq)
    ).all()
    assigned = assign_turn_ids(windows, (RowToLink.from_row(row) for row in rows))
    for row in rows:
        run_id = assigned.get(row.id)
        if run_id is not None:
            row.turn_id = run_id
    return len(assigned), len(rows) - len(assigned)


def backfill_turn_ids(
    session_factory: sessionmaker[OrmSession], *, limit_sessions: int | None = None
) -> BackfillReport:
    """Link every null-turn chat row the rule can place, one commit per session.

    `limit_sessions` caps how many sessions one pass touches (None: all).
    Raises only if the session list itself cannot be read (an unmigrated
    DB); a failure inside one session is logged, rolled back and skipped,
    and the pass carries on -- the lifespan hook swallows either way.
    """
    report = BackfillReport()
    with session_factory() as db:
        sessions = _sessions_to_backfill(db)
    if limit_sessions is not None:
        sessions = sessions[:limit_sessions]
    for profile, stored_id in sessions:
        try:
            with session_factory() as db:
                assigned, left = _backfill_session(db, profile, stored_id)
                db.commit()
        except Exception:
            report.sessions_failed += 1
            logger.exception(
                "turn-id backfill failed for session %s (profile=%r); skipped",
                stored_id,
                profile,
            )
            continue
        report.sessions_scanned += 1
        report.rows_assigned += assigned
        report.rows_left_null += left
    return report


def run_turn_backfill(session_factory: Any) -> BackfillReport | None:
    """The lifespan's best-effort call: log the report, never raise."""
    try:
        report = backfill_turn_ids(session_factory)
    except Exception:
        logger.warning(
            "turn-id backfill skipped: could not read the chat/run tables "
            "(is the database migrated?)",
            exc_info=True,
        )
        return None
    logger.info(
        "turn-id backfill: %d session(s) scanned, %d row(s) linked, %d left without a run, "
        "%d session(s) failed",
        report.sessions_scanned,
        report.rows_assigned,
        report.rows_left_null,
        report.sessions_failed,
    )
    return report
