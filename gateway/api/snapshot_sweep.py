"""The snapshot sweep routes (P6-3, Stream B): run a pass by hand, read the last one.

`SnapshotSweeper` itself -- the timer, the candidate rules, the footprint
rule -- is `domain/snapshot_sweeper.py` (CLEANUP_PLAN step 3.5), re-exported
here with its constants. `POST /api/snapshot-sweeps` runs the same pass by
hand; `GET /api/snapshot-sweeps/latest` reports the last one. The contract is
`docs/SESSION_ARCHIVE_DESIGN.md` §6.5 (routes §6.4, Stream B rows).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from adapters.hermes import HermesError
from domain.snapshot_sweeper import (  # noqa: F401  (re-exported; see module docstring)
    ACTIVE_IDLE_STATUS,
    ACTIVE_STATUS_UNKNOWN,
    SKIP_CAP,
    SKIP_CURRENT,
    SKIP_NOT_LISTED,
    SKIP_OPEN_RUN,
    SKIP_UNCHANGED,
    SWEEP_SCOPES,
    SnapshotSweeper,
    SweepInProgress,
    _as_utc,
    active_skip_reason,
)

logger = logging.getLogger(__name__)

snapshot_sweep_router = APIRouter(tags=["snapshot-sweeps"])

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _sweeper(request: Request) -> SnapshotSweeper:
    sweeper = getattr(request.app.state, "snapshot_sweeper", None)
    if sweeper is None:  # pragma: no cover - lifespan always sets it
        raise HTTPException(status_code=503, detail="the snapshot sweeper is not running")
    return sweeper


@snapshot_sweep_router.post("/snapshot-sweeps", status_code=202)
async def run_snapshot_sweep(request: Request) -> dict:
    """Run one sweep pass now and return its summary once it has finished.

    **202** with `{"started_at", "finished_at", "considered", "snapshotted",
    "skipped", "errors"}`; **409** while a pass (timer or another POST) is
    running; **502** when Hermes could not answer the two list calls, in which
    case nothing was decided and `latest` is unchanged.
    """
    sweeper = _sweeper(request)
    if sweeper.running:
        raise HTTPException(status_code=409, detail="a snapshot sweep pass is already running")
    try:
        return await sweeper.sweep()
    except SweepInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@snapshot_sweep_router.get("/snapshot-sweeps/latest")
async def latest_snapshot_sweep(request: Request) -> dict:
    """`{"last": <summary>|null, "interval_s", "scope", "enabled", "running"}`. Hermes is not called."""
    sweeper = _sweeper(request)
    return {
        "last": sweeper.last,
        "interval_s": sweeper.interval_s,
        "scope": sweeper.scope,
        "enabled": sweeper.enabled,
        "running": sweeper.running,
    }


__all__ = [
    "ACTIVE_IDLE_STATUS",
    "SKIP_CAP",
    "SKIP_CURRENT",
    "SKIP_NOT_LISTED",
    "SKIP_OPEN_RUN",
    "SKIP_UNCHANGED",
    "SWEEP_SCOPES",
    "SnapshotSweeper",
    "SweepInProgress",
    "active_skip_reason",
    "snapshot_sweep_router",
]
