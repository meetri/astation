"""The snapshot sweep: what makes filed sessions durable without a tap (P6-3, Stream B).

Moved out of `api/snapshot_sweep.py` (CLEANUP_PLAN step 3.5); the two routes
stay there and re-export every name here.

**What this is.** `api/snapshots.py` gives the gateway a complete copy of a
session on demand -- manual archive, pre-delete. This module takes those copies
*unprompted*: an `asyncio` task started by `api.main.lifespan` that, on a
timer, re-snapshots every filed session (or every session Hermes lists, by
scope) whose transcript has moved since its last copy. Stored ids only, one
`session.list` and one `session.active_list` per pass, one session at a time,
and a hard cap per pass. `POST /api/snapshot-sweeps` runs the same pass by
hand; `GET /api/snapshot-sweeps/latest` reports the last one. The contract is
`docs/SESSION_ARCHIVE_DESIGN.md` §6.5 (routes §6.4, Stream B rows).

**Measured facts this rests on** (`docs/PROTOCOL_VERIFIED.md`, "Session
archive probe" and "Three-eyes follow-up", both 2026-09-03):

* **`session.list.message_count` is a change flag, never a size.** One
  session read 104 on the list, 1,379 rows on resume and 1,622 on history.
  It *moves* when a turn lands (2 -> 4 on the spike) and was stable for 7
  sessions across ~80 quiet minutes. So criterion (b) compares "differs from
  the `list_message_count` recorded with the last snapshot" -- never
  "greater", never against a row count.
* **`session.active_list` statuses are `"idle"`, `"starting"` and
  `"working"`.** `starting` is transient right after a resume (6 of 10
  resumed sessions showed it; all were `idle` ~60 s later); `working` is a
  real turn. Anything but exactly `idle` waits for the next pass
  (`active:<status>`); a session absent from the list is not live in the
  Hermes process at all and is fair game.
* **A resume leaves a footprint for the life of the connection.**
  `active_list` was empty at connect, grew to 12 after 12 resumes, and every
  entry was still there at the end. A sweep that resumed 130 sessions would
  therefore leave 130 live handles in Hermes's process until the socket
  dropped. Hence the footprint rule: when a candidate was **not** in
  `active_list` before the pass and `LiveHandleCache` holds no handle for it,
  the handle the sweep minted is `session.close`d (`{"closed": true}`,
  measured) after the copy, and the sweep never seeds the shared cache. When
  it **was** live, the sweep leaves it alone -- that handle may be the app's.
* **A cold resume does not stall another session's stream** (1.84 s for the
  3.96 MB session; 95 deltas landed inside a concurrent resume window; max
  inter-delta gap 0.101 s). The open-run guard (`RunRecorder.has_open_run`)
  and the `active_list` guard are belt-and-braces on top of that, and they
  cannot see turns driven from other Hermes clients (the TUI, another front-end) -- a
  mid-turn snapshot is tolerated and superseded by the next pass.

**How "nothing is written" works for an unchanged transcript.** The contract
says: compare the new `content_checksum` with the latest snapshot's before
inserting; equal -> `skipped[stored] = "unchanged"` and no index row. But
`take_snapshot` writes the bytes first and commits its own row last, and this
module must not edit it (§6.0). The seam is its `db` parameter, which exists
so a caller can keep "snapshot then something" in one connection: the sweep
hands it a `_HeldSession` whose `commit()` only *records* that a commit was
asked for. Once `take_snapshot` returns, the sweep compares checksums and
either commits for real or rolls back, so an unchanged transcript leaves no
row -- only content-addressed gzip bytes nobody references, which is exactly
what §6.5 says happens. No snapshot is ever deleted here.

**What this deliberately does not do.** It never persists a live handle (the
handle it mints goes to a private recorder, not the shared cache, and is
closed or forgotten). It never forces a connection into being from the
timer: the first pass waits for `adapter.is_connected` -- a lifespan must not
exercise Hermes credentials on its own -- and later passes go through
`_with_reconnect` exactly like every route. It never dedups a `manual` or
`pre_delete` snapshot (those are not taken here). Per-session failures are
recorded and the pass continues; every pass logs its summary at INFO.
"""

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

#: The two candidate scopes. `filed` = `snapshot_candidates(db)` (every filed
#: session plus every session that already has a snapshot); `all` = every id
#: in the pass's `session.list`.
SWEEP_SCOPES: tuple[str, ...] = ("filed", "all")

#: The one `session.active_list.status` a snapshot may proceed under.
ACTIVE_IDLE_STATUS = "idle"

#: Skip reasons, as they appear in `summary["skipped"]`. `active:<status>` is
#: built by `active_skip_reason()`.
SKIP_NOT_LISTED = "not_listed"
SKIP_CURRENT = "current"
SKIP_OPEN_RUN = "open_run"
SKIP_CAP = "cap"
SKIP_UNCHANGED = "unchanged"

#: What `active_skip_reason()` says when the `active_list` row carries no
#: string `status` at all. A row without a status is "possibly working", never
#: "idle" (the B-34 forwarding rule applied to a guard).
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


# ---------------------------------------------------------------------------
# The pieces the sweep hands `take_snapshot` instead of the shared ones
# ---------------------------------------------------------------------------


class _HandleRecorder:
    """The `LiveHandleCache` stand-in the sweep gives `take_snapshot`.

    `_resume_for_live_id` writes the handle it minted into whatever cache it
    is handed. The sweep needs that handle -- to `session.close` it under the
    footprint rule -- but must never seed the shared cache with it (§6.5), so
    it records the handle privately and answers every lookup with "nothing
    cached". Never persisted, never returned to a caller.
    """

    def __init__(self) -> None:
        self.live_id: str | None = None

    def get(self, stored_id: str) -> str | None:
        return None

    def put(self, stored_id: str, live_id: str) -> None:
        self.live_id = live_id

    def discard(self, stored_id: str) -> None:
        return None


class _SnapshotAppState:
    """`app.state` as `take_snapshot` sees it from the sweep.

    Everything resolves to the real application state -- the adapter, the
    connect lock (`_ensure_connected` needs the *same* lock as every route,
    B-22), the artifact store, the sessionmaker -- except `live_handle_cache`,
    which is the private `_HandleRecorder` above. `__getattr__` only runs for
    names not found on the instance, so the override is the one attribute set
    here and nothing else is shadowed.
    """

    def __init__(self, app_state: Any, recorder: _HandleRecorder) -> None:
        self._app_state = app_state
        self.live_handle_cache = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app_state, name)


class _HeldSession(OrmSession):
    """An ORM session whose `commit()` waits for the sweep's say-so.

    `take_snapshot(db=...)` adds its index row and calls `commit()` on the
    session it was given. Here that call only sets `commit_requested`; the
    sweep then compares `content_checksum` against the latest snapshot's and
    calls `commit_for_real()` (a new copy) or `rollback()` (an unchanged
    transcript -- the pending row is discarded and nothing reaches the
    database). Every read `take_snapshot` makes on the way -- the filing row,
    the ledger, the runs -- is unaffected.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commit_requested = False

    def commit(self) -> None:  # type: ignore[override]
        self.commit_requested = True

    def commit_for_real(self) -> None:
        super().commit()


# ---------------------------------------------------------------------------
# Reading Hermes's two lists
# ---------------------------------------------------------------------------


def _rows_by_stored_id(result: Any, key: str) -> dict[str, dict[str, Any]]:
    """`{stored_id: row}` for every dict row whose `key` is a non-empty string.

    `session.list` keys its rows on `id` (the STORED id there); `active_list`
    carries the stored id as `session_key` and the LIVE handle as `id`. The
    caller says which. Anything that is not a dict, or has no such key, is
    dropped rather than guessed at.
    """
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
    """What the sweep keeps of a candidate's latest snapshot, as plain values.

    Copied out of the ORM row inside the read session so the loop below,
    which runs after that session is closed and across awaits, never touches
    a detached object.
    """

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
    """Why a listed candidate needs a snapshot, or None when its copy is current.

    (a) no snapshot at all; (b) the list row's `message_count` *differs* from
    the one recorded with the latest snapshot (a change flag, never a size);
    (c) a `runs` row started after the latest snapshot was taken.
    """
    if prior is None:
        return "no_snapshot"
    if int_or_none(list_row.get("message_count")) != prior.list_message_count:
        return "message_count"
    if newest_run is not None and newest_run > prior.taken_at:
        return "run"
    return None


# ---------------------------------------------------------------------------
# The sweeper
# ---------------------------------------------------------------------------


class SnapshotSweeper:
    """Owns the timer task, the one `asyncio.Lock`, and the last pass summary.

    Construct with `from_settings()` in the lifespan; `start()` creates the
    timer task (none when `interval_s == 0`); `close()` cancels it. `sweep()`
    is one pass and is what both the timer and `POST /api/snapshot-sweeps`
    call. Reads `app_state.hermes_adapter` / `run_recorder` /
    `live_handle_cache` at pass time, not at construction, so a test that
    swaps the adapter after startup (the `snap_client` pattern) is honoured.
    """

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
        #: How often the timer re-checks `adapter.is_connected` before its
        #: first pass. Overridable so a test does not wait 5 s.
        self.connect_poll_s = float(connect_poll_s)
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        #: The last *finished* pass, in the §6.4 summary shape, or None.
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

    # -- lifecycle ----------------------------------------------------------

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
        # Wait, never connect: the lifespan must not exercise credentials on
        # its own. The first real request does, and then this proceeds.
        while not self._adapter().is_connected:
            await asyncio.sleep(self.connect_poll_s)
        while True:
            try:
                await self.sweep()
            except SweepInProgress:
                # A POST is mid-pass; this tick is redundant, not an error.
                pass
            except Exception:
                logger.exception(
                    "snapshot sweep pass failed; next attempt in %d s", self.interval_s
                )
            await asyncio.sleep(self.interval_s)

    # -- one pass -------------------------------------------------------------

    async def sweep(self) -> dict[str, Any]:
        """Run one pass now and return its summary. Raises `SweepInProgress` if one is running.

        Raises `HermesError` when the two list calls fail (nothing can be
        decided without them); `self.last` is then left as it was.
        """
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

        # Exactly one of each per pass. Both go through `_with_reconnect` like
        # every route: a socket that died since the last pass is rebuilt once.
        list_result = await _with_reconnect(app_state, adapter, adapter.session_list)
        active_result = await _with_reconnect(app_state, adapter, adapter.session_active_list)
        listed = _rows_by_stored_id(list_result, "id")
        active = _rows_by_stored_id(active_result, "session_key")

        with app_state.db_sessions() as db:
            candidates = set(listed) if self.scope == "all" else snapshot_candidates(db)
            # Newest first by the stored id's date prefix (`YYYYMMDD_HHMMSS_...`),
            # so a cap-bound pass protects the sessions being worked on before
            # it gets to last month's.
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
        """Snapshot one candidate; returns `"snapshotted"` or `SKIP_UNCHANGED`.

        The footprint rule runs in the `finally`: the handle `take_snapshot`
        minted is closed even when the copy itself failed after the resume,
        so a storage error does not leave a stray live session behind.
        """
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
            if minted is not None and not was_live and not had_cached:
                await self._close_footprint(stored, minted)

    async def _close_footprint(self, stored: str, live_id: str) -> None:
        """`session.close` the handle this pass minted. Best-effort, never raises.

        Called directly, not through `_with_reconnect`: a live handle dies
        with its connection, so if the socket dropped since the resume there
        is nothing left to close and reconnecting to try would be pointless.
        """
        try:
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
