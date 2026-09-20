"""The snapshot sweep: what makes filed sessions durable without a tap (P6-3, Stream B)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError
from config.settings import Settings
from domain.coerce import int_or_none
from domain.hermes_runtime import _with_reconnect
from domain.models import Run, utcnow
from domain.snapshot_builder import (
    SnapshotStorageError,
    latest_snapshots_by_stored_id,
    snapshot_candidates,
    take_snapshot,
)
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

SWEEP_SCOPES: tuple[str, ...] = ("filed", "all")

ACTIVE_IDLE_STATUS = "idle"

SKIP_NOT_LISTED = "not_listed"
SKIP_CURRENT = "current"
SKIP_OPEN_RUN = "open_run"
SKIP_CAP = "cap"
SKIP_UNCHANGED = "unchanged"

ACTIVE_STATUS_UNKNOWN = "unknown"


def active_skip_reason(status: Any) -> str:
    """`active:<status>` for a non-idle `active_list` row, verbatim spelling."""
    text = status if isinstance(status, str) and status else ACTIVE_STATUS_UNKNOWN
    return f"active:{text}"


class SweepInProgress(Exception):
    """A pass is already running; the route answers 409."""


def _as_utc(value: datetime) -> datetime:
    """Aware UTC. SQLite hands `DateTime(timezone=True)` back naive; treat naive as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class _HandleRecorder:
    """The `LiveHandleCache` stand-in the sweep gives `take_snapshot`."""

    def __init__(self) -> None:
        self.live_id: str | None = None

    # Always answers "nothing cached": the sweep must never seed the shared handle cache.
    def get(self, stored_id: str) -> str | None:
        return None

    def put(self, stored_id: str, live_id: str) -> None:
        self.live_id = live_id

    def discard(self, stored_id: str) -> None:
        return None


class _SnapshotAppState:
    """`app.state` as `take_snapshot` sees it from the sweep."""

    def __init__(self, app_state: Any, recorder: _HandleRecorder) -> None:
        self._app_state = app_state
        self.live_handle_cache = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app_state, name)


class _HeldSession(OrmSession):
    """An ORM session whose `commit()` waits for the sweep's say-so."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commit_requested = False

    # Records the request only; the sweep commits or rolls back after the checksum test.
    def commit(self) -> None:  # type: ignore[override]
        self.commit_requested = True

    def commit_for_real(self) -> None:
        super().commit()


# active_list carries the stored id as session_key; its "id" is the live handle.
def _rows_by_stored_id(result: Any, key: str) -> dict[str, dict[str, Any]]:
    """`{stored_id: row}` for every dict row whose `key` is a non-empty string."""
    sessions = result.get("sessions") if isinstance(result, dict) else None
    if not isinstance(sessions, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in sessions:
        if not isinstance(row, dict):
            continue
        stored = row.get(key)
        if isinstance(stored, str) and stored:
            rows.setdefault(stored, row)
    return rows


def _newest_run_started_at(db: OrmSession, stored_ids: list[str]) -> dict[str, datetime]:
    """`{stored_id: max(runs.started_at)}` for the ids that have a dated run. One query."""
    if not stored_ids:
        return {}
    rows = db.execute(
        select(Run.runtime_session_id, func.max(Run.started_at))
        .where(Run.runtime_session_id.in_(stored_ids))
        .group_by(Run.runtime_session_id)
    ).all()
    newest: dict[str, datetime] = {}
    for stored, started in rows:
        if isinstance(stored, str) and isinstance(started, datetime):
            newest[stored] = _as_utc(started)
    return newest


class _Prior:
    """What the sweep keeps of a candidate's latest snapshot, as plain values."""

    __slots__ = ("content_checksum", "list_message_count", "taken_at")

    def __init__(
        self, taken_at: datetime, list_message_count: int | None, content_checksum: str
    ) -> None:
        self.taken_at = _as_utc(taken_at)
        self.list_message_count = list_message_count
        self.content_checksum = content_checksum


def _change_reason(
    prior: _Prior | None, list_row: dict[str, Any], newest_run: datetime | None
) -> str | None:
    """Why a listed candidate needs a snapshot, or None when its copy is current."""
    if prior is None:
        return "no_snapshot"
    # message_count is a change flag, not a size: test difference, never growth.
    if int_or_none(list_row.get("message_count")) != prior.list_message_count:
        return "message_count"
    if newest_run is not None and newest_run > prior.taken_at:
        return "run"
    return None


class SnapshotSweeper:
    """Owns the timer task, the one `asyncio.Lock`, and the last pass summary."""

    def __init__(
        self,
        app_state: Any,
        *,
        interval_s: int,
        max_per_pass: int,
        startup_delay_s: float,
        scope: str,
        connect_poll_s: float = 5.0,
    ) -> None:
        if scope not in SWEEP_SCOPES:
            raise ValueError(f"unknown sweep scope {scope!r}; expected one of {SWEEP_SCOPES}")
        if max_per_pass < 1:
            raise ValueError("max_per_pass must be at least 1")
        self._app_state = app_state
        self.interval_s = int(interval_s)
        self.max_per_pass = int(max_per_pass)
        self.startup_delay_s = float(startup_delay_s)
        self.scope = scope
        self.connect_poll_s = float(connect_poll_s)
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self.last: dict[str, Any] | None = None

    @classmethod
    def from_settings(cls, app_state: Any, settings: Settings) -> SnapshotSweeper:
        return cls(
            app_state,
            interval_s=settings.research_gateway_snapshot_sweep_interval_s,
            max_per_pass=settings.research_gateway_snapshot_sweep_max_per_pass,
            startup_delay_s=settings.research_gateway_snapshot_sweep_startup_delay_s,
            scope=settings.research_gateway_snapshot_sweep_scope,
        )


    @property
    def enabled(self) -> bool:
        """Whether the timer runs. `POST /api/snapshot-sweeps` works either way."""
        return self.interval_s > 0

    @property
    def running(self) -> bool:
        return self._lock.locked()

    @property
    def timer_task(self) -> asyncio.Task[None] | None:
        return self._task

    def start(self) -> None:
        """Create the timer task, or log that there is none (`interval_s == 0`)."""
        if self._task is not None:
            return
        if not self.enabled:
            logger.info(
                "snapshot sweep timer disabled (RESEARCH_GATEWAY_SNAPSHOT_SWEEP_INTERVAL_S=0); "
                "POST /api/snapshot-sweeps still runs a pass"
            )
            return
        self._task = asyncio.create_task(self._run_timer(), name="snapshot-sweep")
        logger.info(
            "snapshot sweep timer started: every %d s, scope %s, cap %d/pass, first pass after "
            "%.0f s and once Hermes is connected",
            self.interval_s,
            self.scope,
            self.max_per_pass,
            self.startup_delay_s,
        )

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _adapter(self) -> HermesAdapter:
        return self._app_state.hermes_adapter

    async def _run_timer(self) -> None:
        if self.startup_delay_s > 0:
            await asyncio.sleep(self.startup_delay_s)
        # Waits for a connection, never opens one: startup must not exercise credentials.
        while not self._adapter().is_connected:
            await asyncio.sleep(self.connect_poll_s)
        while True:
            try:
                await self.sweep()
            except SweepInProgress:
                pass
            except Exception:
                logger.exception(
                    "snapshot sweep pass failed; next attempt in %d s", self.interval_s
                )
            await asyncio.sleep(self.interval_s)


    async def sweep(self) -> dict[str, Any]:
        """Run one pass now and return its summary. Raises `SweepInProgress` if one is running."""
        if self._lock.locked():
            raise SweepInProgress("a snapshot sweep pass is already running")
        async with self._lock:
            summary = await self._pass()
            self.last = summary
            return summary

    async def _pass(self) -> dict[str, Any]:
        started_at = utcnow()
        app_state = self._app_state
        adapter = self._adapter()
        run_recorder = getattr(app_state, "run_recorder", None)

        list_result = await _with_reconnect(app_state, adapter, adapter.session_list)
        active_result = await _with_reconnect(app_state, adapter, adapter.session_active_list)
        listed = _rows_by_stored_id(list_result, "id")
        active = _rows_by_stored_id(active_result, "session_key")

        with app_state.db_sessions() as db:
            candidates = set(listed) if self.scope == "all" else snapshot_candidates(db)
            ordered = sorted(candidates, reverse=True)
            latest = latest_snapshots_by_stored_id(db, ordered)
            priors = {
                stored: _Prior(row.taken_at, row.list_message_count, row.content_checksum)
                for stored, row in latest.items()
            }
            newest_runs = _newest_run_started_at(db, ordered)

        snapshotted: list[str] = []
        skipped: dict[str, str] = {}
        errors: dict[str, str] = {}
        attempts = 0

        for stored in ordered:
            list_row = listed.get(stored)
            if list_row is None:
                skipped[stored] = SKIP_NOT_LISTED
                continue
            prior = priors.get(stored)
            why = _change_reason(prior, list_row, newest_runs.get(stored))
            if why is None:
                skipped[stored] = SKIP_CURRENT
                continue
            if run_recorder is not None and run_recorder.has_open_run(stored):
                skipped[stored] = SKIP_OPEN_RUN
                continue
            was_live = stored in active
            if was_live:
                status = active[stored].get("status")
                # Only an exact "idle" may proceed; a missing status counts as working.
                if status != ACTIVE_IDLE_STATUS:
                    skipped[stored] = active_skip_reason(status)
                    continue
            if attempts >= self.max_per_pass:
                skipped[stored] = SKIP_CAP
                continue
            attempts += 1
            try:
                outcome = await self._snapshot_one(stored, list_row, prior, was_live=was_live)
            except (HermesError, SnapshotStorageError) as exc:
                errors[stored] = f"{exc.__class__.__name__}: {exc}"
                logger.warning("snapshot sweep: %s failed for %s: %s", why, stored, exc)
            except Exception as exc:
                errors[stored] = f"{exc.__class__.__name__}: {exc}"
                logger.exception("snapshot sweep: unexpected failure snapshotting %s", stored)
            else:
                if outcome == SKIP_UNCHANGED:
                    skipped[stored] = SKIP_UNCHANGED
                else:
                    snapshotted.append(stored)

        finished_at = utcnow()
        summary = {
            "started_at": iso_z(started_at),
            "finished_at": iso_z(finished_at),
            "considered": len(ordered),
            "snapshotted": snapshotted,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info(
            "snapshot sweep (%s): considered %d, snapshotted %d, skipped %d (%s), errors %d in %.1f s",
            self.scope,
            len(ordered),
            len(snapshotted),
            len(skipped),
            _skip_histogram(skipped),
            len(errors),
            (finished_at - started_at).total_seconds(),
        )
        return summary

    async def _snapshot_one(
        self,
        stored: str,
        list_row: dict[str, Any],
        prior: _Prior | None,
        *,
        was_live: bool,
    ) -> str:
        """Snapshot one candidate; returns `"snapshotted"` or `SKIP_UNCHANGED`."""
        app_state = self._app_state
        cache = getattr(app_state, "live_handle_cache", None)
        had_cached = cache is not None and cache.get(stored) is not None
        recorder = _HandleRecorder()
        held = _HeldSession(**app_state.db_sessions.kw)
        try:
            row = await take_snapshot(
                _SnapshotAppState(app_state, recorder),
                stored,
                reason="sweep",
                list_row=list_row,
                db=held,
            )
            if prior is not None and prior.content_checksum == row.content_checksum:
                held.rollback()
                return SKIP_UNCHANGED
            held.commit_for_real()
            return "snapshotted"
        finally:
            held.close()
            minted = recorder.live_id
            # Only a handle this pass minted is closed; the app's own is left alone.
            if minted is not None and not was_live and not had_cached:
                await self._close_footprint(stored, minted)

    async def _close_footprint(self, stored: str, live_id: str) -> None:
        """`session.close` the handle this pass minted. Best-effort, never raises."""
        try:
            # No _with_reconnect: a handle dies with its socket, so a retry is pointless.
            result = await self._adapter().session_close(live_id)
        except Exception as exc:
            logger.warning("snapshot sweep: session.close for %s raised: %s", stored, exc)
            return
        closed = result.get("closed") if isinstance(result, dict) else None
        if closed is not True:
            logger.info("snapshot sweep: session.close for %s did not confirm: %r", stored, result)


def _skip_histogram(skipped: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    for reason in skipped.values():
        counts[reason] = counts.get(reason, 0) + 1
    return ", ".join(f"{reason} {count}" for reason, count in sorted(counts.items())) or "none"
