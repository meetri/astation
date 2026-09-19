"""Run attribution, event persistence, and the run read path (P2-2).

This is the phase's biggest gateway build, scoped honestly: before P2-2,
**nothing persisted events** -- `events/persistence.py` had zero live call
sites and the broadcaster stamped Phase-0 placeholder attribution ids on
every envelope. Two cooperating pieces, mirroring the P2-1 layout (`api/background.py`);
the first lives in `domain/run_recorder.py` now (CLEANUP_PLAN step 3.5) and is
re-exported here with its constants:

1. **`RunRecorder`** -- the event-side orchestration, wired into
   `EventBroadcaster` by `api.main.lifespan` as its `on_canonical_event`
   hook. For every forwarded frame it (a) attributes the canonical envelope
   for real, and (b) persists the run-relevant ones. Its per-frame decisions:

   * **Attribution (P2-2a).** The frame's `_stored_session_id` (stamped by
     `_stamp_session_identity`, B-29) names the Hermes session; the filing
     lookup (sessions table, done once per run open, cached on the open-run
     record) yields the workspace `session_id`/`project_id` -- or None for
     an unfiled session (P2-2d). One `Run` row per turn, **flat**: the wire
     produces no child runs (PV "Phase 2 probe"), so the §9 tree is
     explicitly cut and `parent_run_id` is never written. A frame that
     belongs to no run (unattributable, or a non-run type like
     `session.updated`) carries explicit nulls -- the honest answer, and
     the app already degrades a null envelope field ("").

   * **Turn boundaries.** There is no `turn.started` on the wire. A run
     opens on the first turn-activity frame for a stored session with no
     open run (usually `reasoning.delta` -- the first ~30s of a real turn
     are reasoning only, B-14) and closes on `message.completed`, the
     turn's single end-of-turn signal (B-38: one per turn, carrying
     `final_response`).

   * **Persistence (P2-2b) -- the sync-SQLite-in-async-pump decision.** The
     DB layer is deliberately synchronous (`domain/db.py`); the pump is
     async. That is fine *because of the selective policy*: after
     `PERSISTED_RUN_EVENT_TYPES` filters out the token-scale streams
     (`message.delta`/`reasoning.delta`/`thinking.status` -- see
     `events/persistence.py::NOT_PERSISTED_EVENT_TYPES` for every reason),
     what remains is ~a dozen coarse rows per multi-tool turn, and a
     single-row SQLite insert against a local file is microseconds --
     measured well under the per-frame budget the pump already spends on
     `json.dumps` size-checking (B-02). **Batching boundary: one forwarded
     frame = one session = one commit** (run-open + event row + run-close
     share the transaction when they coincide). Rejected alternatives:
     per-turn batching (a crash loses the whole turn's timeline, and an
     `approval.requested` row must be durable when the P2-3 notify relay
     wants it, not at end-of-turn) and a background flusher task (a second
     silently-dying pump is the exact B-26 failure class). **A DB error is
     logged and never fatal to the stream**: every DB touch is wrapped, a
     failed write costs that row only, and fan-out proceeds regardless --
     same contract as the P2-1 ledger hook.

   * **Per-run cursor (P2-2c).** `run_events.seq` starts at 1 per run and
     is allocated at persist time from the open-run record's in-memory
     counter. It survives restarts *structurally*, not by recovery: a
     restart (or Hermes reconnect) interrupts every open run and later
     activity opens a NEW run with a fresh counter -- so no counter ever
     spans a restart, written rows keep their numbers forever, and
     `after_seq` is durable. The broadcaster's global `seq` (which resets
     with the process) is never used for this.

2. **Routes** (`runs_router`, mounted on the authenticated `/api` router):
   `GET /api/runs` (filters: `session` = Hermes STORED id, `project`) and
   `GET /api/runs/{run_id}/events?after_seq=` (P2-2e). Served entirely from
   the gateway's own tables -- no Hermes round trip, so the run history
   stays readable while Hermes is down. 503 (not 500) when the P2-2
   migration has not run, same pattern as `api/background.py`.

"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesError
from domain import runs as runs_ops
from domain.db import columns_present, schema_checked_db
from domain.hermes_runtime import _with_reconnect, resolve_profile_adapter
from domain.models import Run
from domain.run_recorder import (  # noqa: F401  (re-exported; see module docstring)
    _STORED_SESSION_ID_FIELD,
    DEFAULT_RUN_PROFILE,
    FAILED_WITHOUT_REASON_EXPLANATION,
    HERMES_INTERRUPTED_EXPLANATION,
    NON_RUN_TYPES,
    RUN_ATTACH_ONLY_TYPES,
    RUN_CLOSING_TYPE,
    RUN_OPENING_TYPES,
    RunRecorder,
    _OpenRun,
)
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

runs_router = APIRouter(tags=["runs"])

#: Honest copy for an interrupted run (same convention as the background
#: ledger's ORPHANED_EXPLANATION -- the client should not invent its own).
INTERRUPTED_EXPLANATION = (
    "the gateway restarted or lost its Hermes connection while this turn was "
    "streaming; events after that point were not delivered to the gateway "
    "(nothing buffers or replays them), so this timeline is truncated and "
    "the turn's real outcome is unknown -- the transcript has the final text "
    "if the turn finished"
)


#: Honest copy for a run closed by the B-84/B-62 staleness reconciliation
#: (same convention as INTERRUPTED_EXPLANATION above). Stored verbatim on the
#: row by `runs_ops.close_run_stale` so the answer survives restarts.
STALE_CLOSED_EXPLANATION = (
    "closed by staleness reconciliation: Hermes reports this session idle "
    "and the completion event never arrived (B-62)"
)

#: A `running` run younger than this is never a staleness candidate: a
#: just-started turn can race the `session.active_list` snapshot (the first
#: ~30s of a real turn are reasoning-only, B-14, and Hermes could plausibly
#: report the session idle in the submit/stream gap). 60s is far past any
#: such race and far short of the ~8-10 min B-62 delivery lag being fixed.
STALE_GRACE_SECONDS = 60.0

#: The one `session.active_list` per-session status observed live
#: (2026-09-01, PV "session.active_list & session.status"): every quiet
#: session reports `"idle"`. The non-idle value was NOT observed, so ONLY
#: this exact string (or the session being absent from the list entirely)
#: may close a stale run -- anything else means "possibly still working".
HERMES_IDLE_STATUS = "idle"


def run_row(run: Run, last_seq: int) -> dict[str, Any]:
    """One run as a JSON row. `last_seq` is the cursor high-water mark."""
    row: dict[str, Any] = {
        "id": run.id,
        "kind": run.kind,
        "status": run.status,
        # Hermes STORED session id -- always present, the filter key.
        "runtime_session_id": run.runtime_session_id,
        # Which connection the turn was observed on (B-136). Always a real
        # name, never null: pre-column rows read as "default", which is what
        # they were.
        "profile": run.profile or DEFAULT_RUN_PROFILE,
        # Workspace filing (P2-2d): both null for an unfiled session.
        "session_id": run.session_id,
        "project_id": run.project_id,
        "started_at": iso_z(run.started_at),
        "ended_at": iso_z(run.ended_at),
        "last_seq": last_seq,
    }
    # B-186: a close the wire explained (failed / Hermes-reported interrupted)
    # carries its own note; an interruption with no note is this gateway's
    # own (restart / reconnect); a stale close carries the reconciliation note.
    wire_note = runs_ops.close_note(run)
    if wire_note is not None:
        row["status_explanation"] = wire_note
    elif run.status == runs_ops.STATUS_INTERRUPTED:
        row["status_explanation"] = INTERRUPTED_EXPLANATION
    else:
        stale_note = runs_ops.stale_close_note(run)
        if stale_note is not None:
            row["status_explanation"] = stale_note
    return row


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------


#: A DB session, with "you never ran the P2-2 migration" turned into a 503
#: (`domain.db.schema_checked_db`). The `runs` table itself has existed since
#: the initial schema, so `has_table` proves nothing here; the check is for
#: the column this feature added (`runtime_session_id`).
_runs_db = schema_checked_db(
    "runs_schema_verified", lambda engine: columns_present(engine, "runs", "runtime_session_id")
)


# ---------------------------------------------------------------------------
# Routes (P2-2e)
# ---------------------------------------------------------------------------


def _run_age_seconds(run: Run, now: datetime) -> float | None:
    """Seconds since the run started, or None if it never recorded a start.

    `started_at` comes back naive from SQLite; the recorder only ever writes
    UTC (same convention `_iso` relies on), so a naive value is read as UTC.
    """
    started = run.started_at
    if started is None:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (now - started).total_seconds()


async def _reconcile_stale_running(request: Request, db: OrmSession, rows: list[Run]) -> None:
    """Close `running` rows whose session Hermes says is over (B-84/B-62).

    **Scoped per profile (B-136).** Each stale row is checked against the
    connection it was RECORDED on (`Run.profile`, written at open time),
    never against whichever connection happens to be the default. Asking the
    wrong one is not a near-miss: a `kimi25` session is simply absent from
    `default`'s `session.active_list`, and absent is exactly the answer this
    function reads as "the turn is over", so a single cross-profile check
    would close a genuinely-running turn as stale. That is why `Run` carries
    its own `profile` column rather than reading one off `Run.session_id` --
    which is NULL for every unfiled session, i.e. most of them.

    A profile whose connection is not available right now is SKIPPED, not
    guessed at: its rows stay `running` until its own event stream resolves
    them. That is the honest degradation the previous, default-only version
    of this function claimed to make but did not.

    B-62, measured live: a turn's `message.completed` can arrive ~8-10
    minutes late -- or only when the session is next touched -- so the
    recorder's run row stays `running` long after the turn is visibly done,
    and the app's "Run in progress" pill stays lit. This is the read-path
    reconciliation: when the list being served contains a `running` row
    older than `STALE_GRACE_SECONDS`, ONE `session.active_list` call (the
    same `_with_reconnect` machinery every Hermes-touching route uses)
    answers whether each such session is actually still working.

    * session absent from the list, or present with `status == "idle"` ->
      the turn is over and the event went missing: `close_run_stale`
      (`completed`, `ended_at = now`, the honest note stored on the row).
    * present with ANY other status -> possibly still working; untouched
      (the non-idle value has never been observed live -- PV 2026-09-01).
    * the Hermes call fails -> rows are served unchanged, logged at info.
      The run list must stay readable while Hermes is down (P2-2e), so
      reconciliation is strictly best-effort.

    The common case (no stale candidate) returns before any Hermes round
    trip -- the route stays DB-only. If the late `message.completed` does
    arrive afterwards, the recorder's ordinary close simply re-affirms
    `completed` with the wire timestamp; nothing conflicts.
    """
    now = datetime.now(UTC)
    stale = [
        run
        for run in rows
        if run.status == runs_ops.STATUS_RUNNING
        and (age := _run_age_seconds(run, now)) is not None
        and age > STALE_GRACE_SECONDS
    ]
    if not stale:
        return

    by_profile: dict[str, list[Run]] = {}
    for run in stale:
        by_profile.setdefault(run.profile or DEFAULT_RUN_PROFILE, []).append(run)

    closed: list[str] = []
    for profile, runs_for_profile in by_profile.items():
        try:
            adapter = resolve_profile_adapter(request.app.state, profile)
        except Exception as exc:
            # Not connected right now (`resolve_profile_adapter` raises a 503
            # naming it). Leaving these rows `running` is the correct answer:
            # the only alternative is asking a connection that has never
            # heard of them, which reads as "over".
            logger.info(
                "staleness reconciliation skipped for profile %r "
                "(%d candidate(s) left running): %s",
                profile,
                len(runs_for_profile),
                exc,
            )
            continue
        try:
            result = await _with_reconnect(request.app.state, adapter, adapter.session_active_list)
        except HermesError as exc:
            logger.info(
                "staleness reconciliation skipped for profile %r "
                "(%d candidate(s) left running): session.active_list failed: %s",
                profile,
                len(runs_for_profile),
                exc,
            )
            continue

        sessions = result.get("sessions") if isinstance(result, dict) else None
        statuses: dict[str, Any] = {}
        for entry in sessions if isinstance(sessions, list) else []:
            if isinstance(entry, dict):
                key = entry.get("session_key")  # the STORED id (PV 2026-09-01)
                if isinstance(key, str) and key:
                    statuses[key] = entry.get("status")

        for run in runs_for_profile:
            if (
                run.runtime_session_id in statuses
                and statuses[run.runtime_session_id] != HERMES_IDLE_STATUS
            ):
                continue  # possibly still working: leave it alone
            runs_ops.close_run_stale(run, note=STALE_CLOSED_EXPLANATION, now=now)
            closed.append(run.id)
    if closed:
        db.commit()
        logger.info(
            "staleness reconciliation closed %d run(s) whose completion event "
            "never arrived (B-62): %s",
            len(closed),
            closed,
        )


@runs_router.get("/runs")
async def list_runs(
    request: Request,
    session: str | None = Query(default=None),
    project: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: OrmSession = Depends(_runs_db),
) -> dict:
    """Newest-first run list, filterable by Hermes stored session id / project.

    `session` is the **STORED / durable** Hermes id (`20260829_...`) -- the
    id the app holds for every session, filed or not -- never a live handle
    and never a workspace `sess_...` id. Served from the gateway's own
    tables -- with one exception: a `running` row past the staleness grace
    triggers the B-84/B-62 reconciliation above, which is best-effort and
    never fails or blocks the response (the run history stays readable while
    Hermes is unreachable).
    """
    rows = runs_ops.list_runs(db, runtime_session_id=session, project_id=project, limit=limit)
    try:
        await _reconcile_stale_running(request, db, rows)
    except Exception:
        db.rollback()
        logger.exception("staleness reconciliation failed unexpectedly; serving runs unchanged")
    last_seqs = runs_ops.last_seq_by_run(db, [run.id for run in rows])
    return {"runs": [run_row(run, last_seqs.get(run.id, 0)) for run in rows]}


@runs_router.get("/runs/{run_id}/events")
async def list_run_events(
    request: Request,
    run_id: str,
    after_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
    db: OrmSession = Depends(_runs_db),
) -> dict:
    """This run's persisted timeline after a cursor (P2-2c/e).

    `after_seq` is the per-run cursor: pass the last `seq` already seen (0
    for everything) and events with `seq > after_seq` come back ascending.
    The value is durable across gateway restarts -- it lives on the rows,
    not in any process counter -- so a client may hold it indefinitely.
    `last_seq` in the response is the high-water mark to poll with next.

    The `run` in the response gets the SAME B-84/B-62 staleness
    reconciliation the list route runs, on this one row (B-99). This route
    is reachable on its own: an artifact's "producing run" opens the run
    detail from a run id alone, so it never passes through a reconciling
    list read -- and the detail view then polls here every 3 s, which used
    to mean a finished turn showed `Running` forever behind a live Stop
    button. The guarantees are the helper's, unchanged: nothing younger
    than the grace is a candidate, a run that is not stale costs zero
    Hermes round trips, and a Hermes failure serves the row untouched.
    """
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id!r}")
    try:
        await _reconcile_stale_running(request, db, [run])
    except Exception:
        db.rollback()
        logger.exception("staleness reconciliation failed unexpectedly; serving run unchanged")
    events = runs_ops.events_after(db, run_id, after_seq=after_seq, limit=limit)
    last_seqs = runs_ops.last_seq_by_run(db, [run_id])
    return {
        "run": run_row(run, last_seqs.get(run_id, 0)),
        "after_seq": after_seq,
        "events": [
            {
                "seq": event.seq,
                "type": event.event_type,
                "timestamp": iso_z(event.timestamp),
                "payload": runs_ops.payload_dict(event),
            }
            for event in events
        ],
    }
