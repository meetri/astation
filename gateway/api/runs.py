"""Run attribution, event persistence, and the run read path (P2-2)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesError
from domain import runs as runs_ops
from domain.db import columns_present, schema_checked_db
from domain.hermes_runtime import (
    _with_reconnect,
    profile_is_observable,
    resolve_profile_adapter,
)
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

INTERRUPTED_EXPLANATION = (
    "the gateway restarted or lost its Hermes connection while this turn was "
    "streaming; events after that point were not delivered to the gateway "
    "(nothing buffers or replays them), so this timeline is truncated and "
    "the turn's real outcome is unknown -- the transcript has the final text "
    "if the turn finished"
)


STALE_CLOSED_EXPLANATION = (
    "closed by staleness reconciliation: Hermes reports this session idle "
    "and the completion event never arrived (B-62)"
)

STALE_GRACE_SECONDS = 60.0

HERMES_IDLE_STATUS = "idle"


def run_row(run: Run, last_seq: int) -> dict[str, Any]:
    """One run as a JSON row. `last_seq` is the cursor high-water mark."""
    row: dict[str, Any] = {
        "id": run.id,
        "kind": run.kind,
        "status": run.status,
        "runtime_session_id": run.runtime_session_id,
        "profile": run.profile or DEFAULT_RUN_PROFILE,
        "session_id": run.session_id,
        "project_id": run.project_id,
        "started_at": iso_z(run.started_at),
        "ended_at": iso_z(run.ended_at),
        "last_seq": last_seq,
    }
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


_runs_db = schema_checked_db(
    "runs_schema_verified", lambda engine: columns_present(engine, "runs", "runtime_session_id")
)


def _run_age_seconds(run: Run, now: datetime) -> float | None:
    """Seconds since the run started, or None if it never recorded a start."""
    started = run.started_at
    if started is None:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (now - started).total_seconds()


async def _reconcile_stale_running(request: Request, db: OrmSession, rows: list[Run]) -> None:
    """Close `running` rows whose session Hermes says is over."""
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
            if not profile_is_observable(request.app.state, profile):
                logger.info(
                    "staleness reconciliation skipped for profile %r "
                    "(no connection that can observe it)",
                    profile,
                )
                continue
            adapter = resolve_profile_adapter(request.app.state, profile)
        except Exception as exc:
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
                key = entry.get("session_key")
                if isinstance(key, str) and key:
                    statuses[key] = entry.get("status")

        for run in runs_for_profile:
            if (
                run.runtime_session_id in statuses
                and statuses[run.runtime_session_id] != HERMES_IDLE_STATUS
            ):
                continue
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
    """Newest-first run list, filterable by Hermes stored session id / project."""
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
    """This run's persisted timeline after a cursor (P2-2c/e)."""
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
