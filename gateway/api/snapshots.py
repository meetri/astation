"""Session snapshots: the gateway's own durable, complete copies of Hermes sessions (P6-3).

**What this is.** Until this module the gateway kept no copy of any transcript:
the `messages` table has 0 rows by design (P2-2), `run_events` only sees turns
made *through this gateway while connected*, and Hermes's `~/.hermes/state.db`
is one file on one host, auto-prunable, and mutable under us (compress, branch,
delete, archive). `DELETE /api/sessions/{id}`'s own docstring said it: "there
is no undo, and there is no copy in this gateway." A **snapshot** is that copy
-- a point-in-time, full-detail transcript plus Hermes's metadata and the
gateway's own background results, gzipped into the content-addressed
`ArtifactStore` and indexed by one `session_snapshots` row. "Archive" in the
app is a snapshot filed into a project, plus a user-intent flag
(`sessions.archived_at`) that moves the row between the project's Active and
Archived sections. Nothing is hidden anywhere and Hermes is never told. See
`docs/SESSION_ARCHIVE_DESIGN.md` (§6 is the frozen contract this implements).

**What this deliberately does not do.** It never accepts or persists a live
handle (stored ids in URLs, resolved per call; the handle `_resume_for_live_id`
mints is cached for the connection like every other route's). It never
deletes a snapshot -- the `ON DELETE SET NULL` FKs on `session_snapshots`
mean unfiling, project deletion and session deletion all leave the row and
the bytes. It never touches the Hermes session beyond the read (`resume`,
`list`) and the best-effort pin. It does not sweep -- `api/snapshot_sweep.py`
(Stream B) owns the timer.

The write path (`take_snapshot`), the index queries and
`read_snapshot_document` live in `domain/snapshot_builder.py` (CLEANUP_PLAN
step 3.5) and are re-exported here; this module keeps the routes.
"""

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

# ---------------------------------------------------------------------------
# Reading a document back
# ---------------------------------------------------------------------------


def _load_snapshot(db: OrmSession, snapshot_id: str) -> SessionSnapshot:
    snapshot = db.get(SessionSnapshot, snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"no snapshot with id {snapshot_id!r}")
    return snapshot


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class SnapshotRequest(BaseModel):
    """Body for `POST /api/sessions/{stored}/snapshots`. `{}` or `{"reason": "manual"}`.

    Only `manual` is accepted from a client: `pre_delete`, `sweep` and
    `pre_compress` describe things the *gateway* did, and a client that could
    label its snapshot "sweep" would be forging the sweep's own record.
    """

    model_config = ConfigDict(extra="forbid")

    reason: Literal["manual"] = "manual"


class ArchiveRequest(BaseModel):
    """Body for `POST /api/sessions/{stored}/archive`. Optional entirely.

    `project_id` is required only when the session is unfiled (409 otherwise
    -- an archive lives *in* a project). `title` is the filing fallback title,
    same meaning as `SessionFiling.title`.
    """

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


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
    """Take a snapshot of this session now. **201** `{"snapshot": <row>}`.

    The session stays exactly as it was on Hermes. 404 for a stored id Hermes
    cannot resume, 502 for any other upstream failure or a local storage one.
    """
    stored_id = _validate_stored_session_id_for_argv(
        stored_session_id
    )  # B-120: the id a snapshot records is the one delete/pin later put into argv
    reason = body.reason if body is not None else "manual"
    try:
        # `profile=None` lets `take_snapshot` read it off the filing row --
        # see its own docstring.
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
    )  # B-120: the id a snapshot records is the one delete/pin later put into argv
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
    """The manifest row plus the transcript, in `GET /messages`'s shape.

    `messages` is the document's verbatim `messages` with its
    `background_results` appended, passed through the live routes' own
    `_project_transcript()` so B-86 applies here exactly as it does there
    (duplicate `reasoning_content` dropped; `?detail=light` omits the bodies).
    **There is no `live_session_id` key at all** -- a snapshot is not a live
    session and the response must not let a client believe otherwise;
    `archived: true` says what this is.

    404 unknown id; 502 naming `storage_key` when the file is missing, not
    gzip, fails its checksum, or is not JSON.
    """
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
    """Archive to project: file (if asked) -> snapshot -> flag -> best-effort pin.

    Strictly in that order, and the flag is set **only after the snapshot row
    is committed** -- `archived_at` must never point at a copy that does not
    exist. The id passes `_validate_stored_session_id_for_argv` first, before
    any Hermes call, because it ends up in `cli.exec`'s argv for the pin.

    * 409 when the session is unfiled and no `project_id` was given (an
      archive lives in a project); 404 for an unknown `project_id` or a stored
      id Hermes cannot resume; 422 for a malformed id; 502 upstream/storage.
    * `pinned`: `true` only for `blocked: false` and `code == 0`; `false` for
      a non-zero code (`pin_detail` = the CLI's output); `null` if the call
      raised. Pin failure never fails the archive.

    The Hermes session is never hidden, archived or otherwise touched.
    """
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
        # Reuses the filing path's own 409 ("already filed in ...; PATCH it to
        # move it") rather than silently re-homing the session.
        file_stored_session(db, project, stored_id, body.title)

    # B-147: the filing row is the honest source now that it exists (it is
    # written with the session's profile), and by here the session is always
    # filed. `?profile=` is the fallback for a row that predates the column.
    session_profile = filing.profile or profile if filing is not None else profile

    try:
        snapshot = await take_snapshot(
            request.app.state, stored_id, reason="manual", db=db, profile=session_profile
        )
    except (HermesError, SnapshotStorageError) as exc:
        _raise_for_snapshot_failure(exc, stored_id)

    # Only now -- the snapshot row is committed.
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
    """`hermes sessions pin <stored>`; tri-state result, never raises.

    B-147: on the session's own connection. Best-effort either way, but
    pinning against the wrong profile's Hermes is a guaranteed no-op.
    """
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
    """Clear the flag. Does NOT unpin, does not touch Hermes, deletes nothing.

    The pin flag is shared with Hermes Desktop's Pinned sidebar and the
    gateway cannot know who set it. 404 when the session has no filing row.
    """
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
