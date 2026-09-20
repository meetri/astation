"""Session snapshots: the gateway's own durable, complete copies of Hermes sessions (P6-3)."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError
from api.instance import (
    HERMES_SESSIONS_PIN_ARGV,
    MAX_TITLE_CHARS,
    _validate_stored_session_id_for_argv,
)
from api.projects import (
    HERMES_RUNTIME,
    _find_filing,
    _load_project,
    file_stored_session,
    workspace_db,
)
from domain.artifact_store import ArtifactStore
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _with_reconnect,
    resolve_profile_adapter,
)
from domain.models import SessionSnapshot, utcnow
from domain.snapshot_builder import (  # noqa: F401  (re-exported; see module docstring)
    PRODUCER_GIT_SHA,
    SNAPSHOT_FORMAT,
    SNAPSHOT_PRODUCER_SERVICE,
    SNAPSHOT_REASONS,
    SNAPSHOT_SOURCE_RESUME,
    SnapshotStorageError,
    _newest_first,
    is_archived,
    latest_snapshot_summary,
    latest_snapshots_by_stored_id,
    read_snapshot_document,
    snapshot_candidates,
    snapshot_counts_by_stored_id,
    snapshot_only_stored_ids,
    snapshot_row,
    take_snapshot,
)
from domain.timeutil import iso_z
from domain.transcript import TRANSCRIPT_DETAIL_LIGHT, _project_transcript

logger = logging.getLogger(__name__)

snapshots_router = APIRouter(tags=["snapshots"])


def _load_snapshot(db: OrmSession, snapshot_id: str) -> SessionSnapshot:
    snapshot = db.get(SessionSnapshot, snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"no snapshot with id {snapshot_id!r}")
    return snapshot


class SnapshotRequest(BaseModel):
    """Body for `POST /api/sessions/{stored}/snapshots`. `{}` or `{"reason": "manual"}`."""

    model_config = ConfigDict(extra="forbid")

    reason: Literal["manual"] = "manual"


class ArchiveRequest(BaseModel):
    """Body for `POST /api/sessions/{stored}/archive`. Optional entirely."""

    model_config = ConfigDict(extra="forbid")

    project_id: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)

    @field_validator("project_id")
    @classmethod
    def _strip_project_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("project_id must not be blank")
        return cleaned


def _raise_for_snapshot_failure(exc: Exception, stored_id: str) -> None:
    if isinstance(exc, HermesError):
        raise _http_error_from_hermes(exc, stored_id) from exc
    if isinstance(exc, SnapshotStorageError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    raise exc


@snapshots_router.post("/sessions/{stored_session_id}/snapshots", status_code=201)
async def create_snapshot(
    stored_session_id: str,
    request: Request,
    body: SnapshotRequest | None = None,
    profile: str | None = Query(default=None),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """Take a snapshot of this session now. **201** `{"snapshot": <row>}`."""
    stored_id = _validate_stored_session_id_for_argv(
        stored_session_id
    )
    reason = body.reason if body is not None else "manual"
    try:
        snapshot = await take_snapshot(
            request.app.state, stored_id, reason=reason, db=db, profile=profile
        )
    except (HermesError, SnapshotStorageError) as exc:
        _raise_for_snapshot_failure(exc, stored_id)
    return {"snapshot": snapshot_row(snapshot)}


@snapshots_router.get("/sessions/{stored_session_id}/snapshots")
async def list_session_snapshots(
    stored_session_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """Every snapshot of one session, newest first. Workspace-only; Hermes is not called."""
    stored_id = _validate_stored_session_id_for_argv(
        stored_session_id
    )
    rows = db.execute(
        _newest_first(
            select(SessionSnapshot).where(SessionSnapshot.stored_session_id == stored_id)
        ).limit(limit)
    ).scalars()
    return {"stored_session_id": stored_id, "snapshots": [snapshot_row(s) for s in rows]}


@snapshots_router.get("/snapshots")
async def list_snapshots(
    project_id: str | None = Query(default=None),
    unfiled: bool | None = Query(default=None),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """Snapshots by project (`?project_id=`) or with none (`?unfiled=true`). Exactly one."""
    wants_project = project_id is not None and project_id.strip() != ""
    wants_unfiled = unfiled is True
    if wants_project == wants_unfiled:
        raise HTTPException(
            status_code=422,
            detail="pass exactly one of `project_id` or `unfiled=true`",
        )
    stmt = select(SessionSnapshot)
    if wants_project:
        stmt = stmt.where(SessionSnapshot.project_id == project_id.strip())
    else:
        stmt = stmt.where(SessionSnapshot.project_id.is_(None))
    rows = db.execute(_newest_first(stmt)).scalars()
    return {"snapshots": [snapshot_row(s) for s in rows]}


@snapshots_router.get("/snapshots/{snapshot_id}")
async def get_snapshot(
    snapshot_id: str,
    request: Request,
    detail: Literal["full", "light"] = Query(default="full"),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """The manifest row plus the transcript, in `GET /messages`'s shape."""
    snapshot = _load_snapshot(db, snapshot_id)
    store: ArtifactStore = request.app.state.artifact_store
    document = read_snapshot_document(store, snapshot)
    messages = document.get("messages")
    rows = list(messages) if isinstance(messages, list) else []
    background = document.get("background_results")
    if isinstance(background, list):
        rows.extend(background)
    return {
        "snapshot": snapshot_row(snapshot),
        "stored_session_id": snapshot.stored_session_id,
        "count": len(rows),
        "messages": _project_transcript(rows, light=detail == TRANSCRIPT_DETAIL_LIGHT),
        "pending_approval": document.get("pending_approval"),
        "pending_clarify": document.get("pending_clarify"),
        "archived": True,
    }


@snapshots_router.post("/sessions/{stored_session_id}/archive")
async def archive_session(
    stored_session_id: str,
    request: Request,
    body: ArchiveRequest | None = None,
    profile: str = Query(default="default"),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """Archive to project: file (if asked) -> snapshot -> flag -> best-effort pin."""
    stored_id = _validate_stored_session_id_for_argv(stored_session_id)
    body = body if body is not None else ArchiveRequest()

    filing = _find_filing(db, HERMES_RUNTIME, stored_id)
    if filing is None:
        if body.project_id is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"session {stored_id!r} is not filed in any project; pass "
                    "`project_id` to file it and archive it there"
                ),
            )
        project = _load_project(db, body.project_id)
        file_stored_session(db, project, stored_id, body.title)
        filing = _find_filing(db, HERMES_RUNTIME, stored_id)
        if filing is None:  # pragma: no cover - file_stored_session just committed it
            raise HTTPException(status_code=500, detail="filing row vanished after filing")
    elif body.project_id is not None and body.project_id != filing.project_id:
        project = _load_project(db, body.project_id)
        file_stored_session(db, project, stored_id, body.title)

    session_profile = filing.profile or profile if filing is not None else profile

    try:
        snapshot = await take_snapshot(
            request.app.state, stored_id, reason="manual", db=db, profile=session_profile
        )
    except (HermesError, SnapshotStorageError) as exc:
        _raise_for_snapshot_failure(exc, stored_id)

    filing = _find_filing(db, HERMES_RUNTIME, stored_id)
    archived_at = utcnow()
    if filing is not None:
        filing.archived_at = archived_at
        db.commit()

    pinned, pin_detail = await _pin_best_effort(request.app.state, stored_id, session_profile)
    return {
        "stored_session_id": stored_id,
        "archived_at": iso_z(archived_at),
        "snapshot": snapshot_row(snapshot),
        "project_id": filing.project_id if filing is not None else None,
        "pinned": pinned,
        "pin_detail": pin_detail,
    }


async def _pin_best_effort(
    app_state: Any, stored_id: str, profile: str = "default"
) -> tuple[bool | None, str | None]:
    """`hermes sessions pin <stored>`; tri-state result, never raises."""
    try:
        adapter: HermesAdapter = resolve_profile_adapter(app_state, profile)
    except Exception as exc:
        logger.info("no connection for profile %r; archive proceeds unpinned: %s", profile, exc)
        return None, str(exc)
    argv = [*HERMES_SESSIONS_PIN_ARGV, stored_id]
    try:
        result = await _with_reconnect(app_state, adapter, lambda: adapter.cli_exec(argv))
    except Exception as exc:
        logger.info("sessions pin for %s raised; archive proceeds unpinned: %s", stored_id, exc)
        return None, str(exc)
    payload = result if isinstance(result, dict) else {}
    code = payload.get("code")
    output = payload.get("output")
    output_text = output if isinstance(output, str) else None
    ok = (
        payload.get("blocked") is not True
        and isinstance(code, int)
        and not isinstance(code, bool)
        and code == 0
    )
    if ok:
        return True, output_text
    hint = payload.get("hint")
    return False, output_text or (str(hint) if hint else None)


@snapshots_router.post("/sessions/{stored_session_id}/unarchive")
async def unarchive_session(stored_session_id: str, db: OrmSession = Depends(workspace_db)) -> dict:
    """Clear the flag. Does NOT unpin, does not touch Hermes, deletes nothing."""
    stored_id = _validate_stored_session_id_for_argv(stored_session_id)
    filing = _find_filing(db, HERMES_RUNTIME, stored_id)
    if filing is None:
        raise HTTPException(
            status_code=404,
            detail=f"session {stored_id!r} is not filed in any project, so it is not archived",
        )
    filing.archived_at = None
    db.commit()
    return {"stored_session_id": stored_id, "archived_at": None}


__all__ = [
    "PRODUCER_GIT_SHA",
    "SNAPSHOT_FORMAT",
    "SNAPSHOT_REASONS",
    "SnapshotStorageError",
    "is_archived",
    "latest_snapshot_summary",
    "latest_snapshots_by_stored_id",
    "read_snapshot_document",
    "snapshot_candidates",
    "snapshot_counts_by_stored_id",
    "snapshot_only_stored_ids",
    "snapshot_row",
    "snapshots_router",
    "take_snapshot",
]
