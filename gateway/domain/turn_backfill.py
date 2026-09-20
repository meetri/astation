"""Give pre-existing chat rows their run id, from timestamps alone."""

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

SLACK = timedelta(seconds=2)

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
    sessions_failed: int = 0


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True)
class _Span:
    run_id: str
    start: datetime
    end: datetime | None

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
    for span in spans:
        if span.start >= when:
            return span.run_id
    return None


def assign_turn_ids(
    runs: Iterable[RunWindow], rows: Iterable[RowToLink], *, slack: timedelta = SLACK
) -> dict[str, str]:
    """`{row id: run id}` for every row the rule can place (module docstring)."""
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
    """Link every null-turn chat row the rule can place, one commit per session."""
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
