"""Bound the growth of `run_events` without losing what the product reads back.

A busy workspace writes a row for every tool call and result. On the reference deployment 78% of the
table was three tool-detail types (`tool.completed`, `tool.started`, `subagent.tool`), while the
rows other features depend on were a small fraction of it: `session.usage` (the usage stats subtract
one run's cumulative total from the previous run's, so they need every run's), the messages,
approvals and clarifications.

So retention does not delete rows. For a run that has finished and is older than the window it
replaces the *payload* of the tool-detail events with a small stub. Row count, `seq` order, the
event types and everything the stats and transcript read are untouched; only the bulky tool inputs
and outputs of old runs are dropped. A run that is still open is never touched.

Freed pages go back to SQLite's free list and are reused, so the file stops growing; it does not
shrink on its own. To return the space to the filesystem run `VACUUM` once, with the gateway
stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import sessionmaker

from domain.models import Run, RunEvent, utcnow

logger = logging.getLogger(__name__)

# The bulky ones. Everything else a run records is kept verbatim.
PRUNABLE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "tool.started",
        "tool.progress",
        "tool.completed",
        "subagent.started",
        "subagent.tool",
        "subagent.completed",
    }
)

# What a pruned payload is replaced with. Short enough that `_STUB_LENGTH_FLOOR` can tell it apart
# from an unpruned payload without parsing, so a second pass skips what the first one did.
PRUNED_PAYLOAD: dict[str, bool] = {"_pruned": True}
_STUB_LENGTH_FLOOR = 40

# A run in one of these states may still be writing events.
OPEN_RUN_STATUSES: frozenset[str] = frozenset({"queued", "running"})

DEFAULT_BATCH_ROWS = 500


def prune_run_events(
    session_factory: sessionmaker,
    *,
    retention_days: int,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> int:
    """Stub the tool-detail payloads of finished runs older than `retention_days`.

    Returns how many events were pruned. Works in small transactions so a write never holds the
    database long enough to stall the event pipeline; safe to call repeatedly.
    """
    if retention_days <= 0:
        return 0
    cutoff = utcnow() - timedelta(days=retention_days)
    pruned = 0
    while True:
        with session_factory() as db:
            ids = db.execute(
                select(RunEvent.id)
                .join(Run, Run.id == RunEvent.run_id)
                .where(
                    RunEvent.event_type.in_(PRUNABLE_EVENT_TYPES),
                    RunEvent.timestamp < cutoff,
                    Run.status.not_in(OPEN_RUN_STATUSES),
                    func.length(RunEvent.payload_json) > _STUB_LENGTH_FLOOR,
                )
                .limit(batch_rows)
            ).scalars().all()
            if not ids:
                return pruned
            db.execute(
                update(RunEvent).where(RunEvent.id.in_(ids)).values(payload_json=PRUNED_PAYLOAD)
            )
            db.commit()
            pruned += len(ids)


class RunEventRetention:
    """Runs `prune_run_events` once a day for as long as the gateway is up."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        retention_days: int,
        interval_s: float = 86_400,
        startup_delay_s: float = 300,
        batch_rows: int = DEFAULT_BATCH_ROWS,
    ) -> None:
        self._session_factory = session_factory
        self._retention_days = retention_days
        self._interval_s = interval_s
        self._startup_delay_s = startup_delay_s
        self._batch_rows = batch_rows
        self._task: asyncio.Task[None] | None = None

    @classmethod
    def from_settings(cls, app_state: Any, settings: Any) -> RunEventRetention:
        return cls(
            app_state.db_sessions,
            retention_days=settings.research_gateway_run_event_retention_days,
            interval_s=settings.research_gateway_run_event_prune_interval_s,
            startup_delay_s=settings.research_gateway_run_event_prune_startup_delay_s,
        )

    @property
    def enabled(self) -> bool:
        return self._retention_days > 0

    def start(self) -> None:
        if not self.enabled:
            logger.info("run-event retention is off (retention days = 0); old tool detail is kept")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="run-event-retention")

    async def prune_once(self) -> int:
        """One pass, off the event loop."""
        return await asyncio.to_thread(
            prune_run_events,
            self._session_factory,
            retention_days=self._retention_days,
            batch_rows=self._batch_rows,
        )

    async def _run(self) -> None:
        await asyncio.sleep(self._startup_delay_s)
        while True:
            try:
                count = await self.prune_once()
                if count:
                    logger.info(
                        "run-event retention: pruned the tool detail of %d event(s) older than %d day(s)",
                        count,
                        self._retention_days,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("run-event retention pass failed; trying again next interval")
            await asyncio.sleep(self._interval_s)

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
