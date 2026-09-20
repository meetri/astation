"""Projects, and the index that files a Hermes session into one."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError
from config.settings import get_settings
from domain.artifact_store import _artifact_json
from domain.db import schema_checked_db, schema_is_present
from domain.filing import find_filing
from domain.hermes_runtime import (
    _validate_stored_session_id,
    _with_reconnect,
    resolve_profile_adapter,
)
from domain.models import HERMES_RUNTIME, Artifact, Project, Run, Session, utcnow
from domain.project_workspace import ensure_instructions_file, instructions_path
from domain.run_recorder import DEFAULT_RUN_PROFILE as DEFAULT_PROFILE
from domain.sandbox_fs import sandbox_fs_for
from domain.sandbox_paths import validate_sandbox_path
from domain.snapshot_builder import (
    is_archived,
    latest_snapshot_summary,
    latest_snapshots_by_stored_id,
    snapshot_counts_by_stored_id,
    snapshot_only_stored_ids,
)
from domain.tag_store import attach as tag_attach
from domain.tag_store import detach as tag_detach
from domain.tag_store import names_for as tag_names_for
from domain.tag_store import names_of as tag_names_of
from domain.tag_store import owner_ids_with_all as tag_owner_ids_with_all
from domain.tags import TagNameError, normalize_tag_names
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

_MAX_TITLE_CHARS = 200
_MAX_DESCRIPTION_CHARS = 4000

projects_router = APIRouter(tags=["projects"])


class ProjectCreate(BaseModel):
    """Body for `POST /api/projects`."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=_MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION_CHARS)
    folder_path: str | None = Field(default=None, max_length=4096)

    @field_validator("title")
    @classmethod
    def _require_real_title(cls, value: str) -> str:
        """A title of spaces is an untitled project with extra steps."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title must contain non-whitespace characters")
        return cleaned

    @field_validator("description")
    @classmethod
    def _normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("folder_path")
    @classmethod
    def _normalize_folder_path(cls, value: str | None) -> str | None:
        return _normalized_folder_path(value)


class ProjectUpdate(BaseModel):
    """Body for `PATCH /api/projects/{id}` -- rename, re-describe, or both."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=_MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION_CHARS)
    folder_path: str | None = Field(default=None, max_length=4096)

    @field_validator("title")
    @classmethod
    def _require_real_title(cls, value: str | None) -> str | None:
        if value is None:
            raise ValueError("title cannot be null; omit it to leave it unchanged")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title must contain non-whitespace characters")
        return cleaned

    @field_validator("description")
    @classmethod
    def _normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("folder_path")
    @classmethod
    def _normalize_folder_path(cls, value: str | None) -> str | None:
        return _normalized_folder_path(value)

    @model_validator(mode="after")
    def _require_something_to_do(self) -> ProjectUpdate:
        if not self.model_fields_set:
            raise ValueError("provide at least one of: title, description, folder_path")
        return self


def _normalized_folder_path(value: str | None) -> str | None:
    """Trim a folder-path bookmark; blank becomes null (= no bookmark)."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


class SessionFiling(BaseModel):
    """Body for `POST /api/projects/{id}/sessions` -- file a Hermes session."""

    model_config = ConfigDict(extra="forbid")

    stored_session_id: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=_MAX_TITLE_CHARS)
    profile: str = Field(default="default")


class SessionMove(BaseModel):
    """Body for `PATCH /api/projects/{id}/sessions/{stored_session_id}`."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1)


_find_filing = find_filing


workspace_db = schema_checked_db("db_schema_verified", schema_is_present)


def _project_row(
    project: Project,
    session_count: int,
    tags: list[str] | None = None,
    *,
    pinned: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "tags": tags or [],
        "pinned_artifact": pinned,
        "id": project.id,
        "title": project.title,
        "description": project.description,
        "folder_path": project.folder_path,
        "instructions_path": instructions_path(get_settings(), project.id),
        "created_at": iso_z(project.created_at),
        "updated_at": iso_z(project.updated_at),
        "session_count": session_count,
    }


def _session_counts(db: OrmSession, project_ids: list[str]) -> dict[str, int]:
    """`{project_id: ACTIVE filed session count}` in one query, zeros included."""
    counts = dict.fromkeys(project_ids, 0)
    if not project_ids:
        return counts
    rows = db.execute(
        select(Session.project_id, func.count(Session.id))
        .where(Session.project_id.in_(project_ids), Session.archived_at.is_(None))
        .group_by(Session.project_id)
    ).all()
    for project_id, count in rows:
        counts[project_id] = count
    return counts


def _load_project(db: OrmSession, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project with id {project_id!r}")
    return project


def filed_project_index(
    db: OrmSession, stored_session_ids: list[str], runtime: str = HERMES_RUNTIME
) -> dict[str, dict[str, Any]]:
    """`{stored_session_id: {project_id, project_title, workspace_session_id, archived_at}}`."""
    if not stored_session_ids:
        return {}
    rows = db.execute(
        select(
            Session.runtime_session_id,
            Session.id,
            Project.id,
            Project.title,
            Session.archived_at,
        )
        .join(Project, Project.id == Session.project_id)
        .where(
            Session.runtime == runtime,
            Session.runtime_session_id.in_(stored_session_ids),
        )
    ).all()
    return {
        stored_id: {
            "workspace_session_id": workspace_id,
            "project_id": project_id,
            "project_title": project_title,
            "archived_at": iso_z(archived_at),
        }
        for stored_id, workspace_id, project_id, project_title, archived_at in rows
    }


async def _hermes_session_index(
    request: Request, profiles: Iterable[str] | None = None
) -> tuple[dict[str, Any] | None, set[str], str | None]:
    """`({stored_id: metadata}, profiles actually checked, error)` -- read-only."""
    wanted = sorted({(p or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE for p in (profiles or ())})
    if not wanted:
        wanted = [DEFAULT_PROFILE]

    known = await _known_profile_names(request)
    index: dict[str, Any] = {}
    checked: set[str] = set()
    last_error: str | None = None
    for profile in wanted:
        if known is not None and profile not in known:
            logger.info(
                "session listing: profile %r is not one Hermes knows; "
                "its rows report missing=null rather than a loss",
                profile,
            )
            continue
        listed, error = await _hermes_sessions_for_profile(request, profile)
        if listed is None:
            last_error = error
            continue
        checked.add(profile)
        index.update(listed)
    if not checked:
        return None, checked, last_error
    return index, checked, None


async def _known_profile_names(request: Request) -> set[str] | None:
    """Every profile Hermes currently has, or `None` when it cannot be asked."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        result = await _with_reconnect(request.app.state, adapter, adapter.profiles_list)
    except Exception as exc:
        logger.info("could not list Hermes profiles while enriching sessions: %s", exc)
        return None
    rows = result.get("profiles") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return None
    names = {
        row["name"]
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"]
    }
    return names or None


async def _hermes_sessions_for_profile(
    request: Request, profile: str
) -> tuple[dict[str, Any] | None, str | None]:
    """One profile's `session.list`, indexed by stored id. `(None, why)` on failure."""
    try:
        adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    except Exception as exc:  # pragma: no cover - resolver no longer refuses
        return None, str(exc)
    extra = {"profile": profile} if profile and profile != DEFAULT_PROFILE else {}
    try:
        result = await _with_reconnect(
            request.app.state, adapter, lambda: adapter.session_list(**extra)
        )
    except HermesError as exc:
        logger.info(
            "Hermes unreachable while enriching profile %r's session list: %s", profile, exc
        )
        return None, str(exc)
    except Exception as exc:
        logger.warning(
            "unexpected failure listing Hermes sessions for profile %r", profile, exc_info=True
        )
        return None, str(exc)

    sessions = result.get("sessions") if isinstance(result, dict) else None
    if not isinstance(sessions, list):
        return {}, None
    index: dict[str, Any] = {}
    for row in sessions:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]:
            index[row["id"]] = row
    return index, None


def _filed_session_row(
    session: Session,
    project: Project,
    hermes: dict[str, Any] | None,
    *,
    runtime_available: bool,
    latest_snapshot: Any = None,
    snapshot_count: int = 0,
) -> dict[str, Any]:
    """One element of `GET /api/projects/{id}/sessions` for a FILED session."""
    known = hermes if isinstance(hermes, dict) else {}
    if runtime_available:
        missing: bool | None = hermes is None
    else:
        missing = None
    return {
        "id": session.runtime_session_id,
        "title": known.get("title") if known else session.title,
        "preview": known.get("preview"),
        "started_at": known.get("started_at"),
        "message_count": known.get("message_count"),
        "source": known.get("source"),
        "missing": missing,
        "filed": True,
        "workspace_session_id": session.id,
        "project_id": project.id,
        "project_title": project.title,
        "runtime": session.runtime,
        "profile": session.profile,
        "status": session.status,
        "filed_at": iso_z(session.created_at),
        "row_key": session.id,
        "archived_at": iso_z(session.archived_at),
        "snapshot_count": snapshot_count,
        "latest_snapshot": latest_snapshot_summary(latest_snapshot),
        "archived": is_archived(
            archived_at=session.archived_at,
            has_filing=True,
            missing=missing,
            snapshot_count=snapshot_count,
        ),
    }


def _snapshot_only_row(
    stored_id: str,
    project: Project,
    hermes: dict[str, Any] | None,
    *,
    runtime_available: bool,
    latest_snapshot: Any,
    snapshot_count: int,
) -> dict[str, Any]:
    """A project row for a session that has a snapshot here but no filing row."""
    meta = latest_snapshot.hermes_meta_json if latest_snapshot is not None else None
    meta = meta if isinstance(meta, dict) else {}
    if runtime_available:
        missing: bool | None = hermes is None
    else:
        missing = None
    title = latest_snapshot.title if latest_snapshot is not None else None
    if not title:
        meta_title = meta.get("title")
        title = meta_title if isinstance(meta_title, str) and meta_title else None
    return {
        "id": stored_id,
        "title": title,
        "preview": meta.get("preview"),
        "started_at": meta.get("started_at"),
        "message_count": meta.get("message_count"),
        "source": meta.get("source"),
        "missing": missing,
        "filed": False,
        "workspace_session_id": None,
        "project_id": project.id,
        "project_title": project.title,
        "runtime": HERMES_RUNTIME,
        "status": "deleted"
        if missing is True
        or (missing is None and getattr(latest_snapshot, "reason", None) == "pre_delete")
        else "unfiled",
        "filed_at": None,
        "row_key": f"snapshot:{latest_snapshot.id}" if latest_snapshot is not None else None,
        "archived_at": None,
        "snapshot_count": snapshot_count,
        "latest_snapshot": latest_snapshot_summary(latest_snapshot),
        "archived": is_archived(
            archived_at=None,
            has_filing=False,
            missing=missing,
            snapshot_count=snapshot_count,
        ),
    }


def _normalized_tag_list(names: list[str]) -> list[str]:
    """Normalize a `?tag=` filter, turning a bad name into a 422 that says
    which rule it broke."""
    try:
        return normalize_tag_names(names)
    except TagNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class PinnedArtifactRequest(BaseModel):
    """Body for `PUT /api/projects/{id}/pinned-artifact`."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)


@projects_router.put("/projects/{project_id}/pinned-artifact")
async def pin_artifact(
    project_id: str, body: PinnedArtifactRequest, db: OrmSession = Depends(workspace_db)
) -> dict:
    """Pin the one artifact this project is currently ABOUT."""
    project = _load_project(db, project_id)
    artifact = db.get(Artifact, body.artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"no artifact with id {body.artifact_id!r}")
    project.pinned_artifact_id = artifact.id
    project.updated_at = utcnow()
    db.commit()
    return _project_row_with_pin(db, project)


@projects_router.delete("/projects/{project_id}/pinned-artifact")
async def unpin_artifact(project_id: str, db: OrmSession = Depends(workspace_db)) -> dict:
    """Un-pin. Idempotent: a project with nothing pinned is already where the
    caller wants it."""
    project = _load_project(db, project_id)
    project.pinned_artifact_id = None
    project.updated_at = utcnow()
    db.commit()
    return _project_row_with_pin(db, project)


def _project_row_with_pin(db: OrmSession, project: Project) -> dict[str, Any]:
    return _project_row(
        project,
        _session_counts(db, [project.id])[project.id],
        tag_names_of(db, "project", project.id),
        pinned=_pinned_artifact_json(db, project),
    )


def _pinned_artifact_json(db: OrmSession, project: Project) -> dict[str, Any] | None:
    """The pinned row, or `null`."""
    if not project.pinned_artifact_id:
        return None
    artifact = db.get(Artifact, project.pinned_artifact_id)
    return _artifact_json(artifact) if artifact is not None else None


def _pinned_artifacts_for(db: OrmSession, projects: list[Project]) -> dict[str, dict[str, Any]]:
    """Every project's pinned row, by project id, in ONE query."""
    wanted = {p.pinned_artifact_id for p in projects if p.pinned_artifact_id}
    if not wanted:
        return {}
    rows = {
        artifact.id: _artifact_json(artifact)
        for artifact in db.execute(select(Artifact).where(Artifact.id.in_(wanted))).scalars()
    }
    return {
        project.id: rows[project.pinned_artifact_id]
        for project in projects
        # The pin is not a foreign key, so it can dangle; a dangling pin reads as unpinned.
        if project.pinned_artifact_id in rows
    }


@projects_router.put("/projects/{project_id}/tags/{name}")
async def tag_project(project_id: str, name: str, db: OrmSession = Depends(workspace_db)) -> dict:
    """Give a project a tag, from the SAME vocabulary artifacts use."""
    project = _load_project(db, project_id)
    tag_attach(db, "project", project.id, _normalized_tag_list([name])[0])
    db.commit()
    return _project_row_with_pin(db, project)


@projects_router.delete("/projects/{project_id}/tags/{name}")
async def untag_project(project_id: str, name: str, db: OrmSession = Depends(workspace_db)) -> dict:
    """Take a tag off a project. Idempotent; the tag itself survives."""
    project = _load_project(db, project_id)
    tag_detach(db, "project", project.id, _normalized_tag_list([name])[0])
    db.commit()
    return _project_row_with_pin(db, project)


@projects_router.get("/projects")
async def list_projects(
    tag: list[str] = Query(default_factory=list), db: OrmSession = Depends(workspace_db)
) -> dict:
    """Every project, newest first, each with how many sessions are filed in it."""
    wanted = _normalized_tag_list(tag)
    projects = list(
        db.execute(select(Project).order_by(Project.created_at.desc(), Project.id)).scalars()
    )
    if wanted:
        matching = set(tag_owner_ids_with_all(db, "project", wanted))
        projects = [project for project in projects if project.id in matching]
    counts = _session_counts(db, [project.id for project in projects])
    tags_by_project = tag_names_for(db, "project", [project.id for project in projects])
    pins = _pinned_artifacts_for(db, projects)
    return {
        "tags": wanted,
        "projects": [
            _project_row(
                project,
                counts.get(project.id, 0),
                tags_by_project.get(project.id, []),
                pinned=pins.get(project.id),
            )
            for project in projects
        ],
    }


@projects_router.post("/projects", status_code=201)
async def create_project(body: ProjectCreate, db: OrmSession = Depends(workspace_db)) -> dict:
    """Create a project. Title required, description optional."""
    folder_path = _validated_folder_path(body.folder_path)
    project = Project(title=body.title, description=body.description, folder_path=folder_path)
    db.add(project)
    db.commit()
    return _project_row(project, 0)


def _validated_folder_path(raw: str | None) -> str | None:
    """The bookmark's confinement check, shared by create and update."""
    if raw is None:
        return None
    return validate_sandbox_path(raw, get_settings().hermes_sandbox_root)


@projects_router.get("/projects/{project_id}")
async def get_project(project_id: str, db: OrmSession = Depends(workspace_db)) -> dict:
    return _project_row_with_pin(db, _load_project(db, project_id))


@projects_router.post("/projects/{project_id}/instructions/ensure")
async def ensure_project_instructions(
    project_id: str, request: Request, db: OrmSession = Depends(workspace_db)
) -> dict:
    """Make sure the project's `HERMES.md` exists, and say where it is."""
    project = _load_project(db, project_id)
    settings = get_settings()
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, "default")
    try:
        path = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: ensure_instructions_file(
                sandbox_fs_for(request.app.state, adapter), settings, project.id
            ),
        )
    except (HermesError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"project_id": project.id, "instructions_path": path}


@projects_router.patch("/projects/{project_id}")
async def update_project(
    project_id: str, body: ProjectUpdate, db: OrmSession = Depends(workspace_db)
) -> dict:
    """Rename a project and/or change its description."""
    project = _load_project(db, project_id)
    fields = body.model_fields_set
    if "title" in fields and body.title is not None:
        project.title = body.title
    if "description" in fields:
        project.description = body.description
    if "folder_path" in fields:
        project.folder_path = _validated_folder_path(body.folder_path)
    # Set explicitly: onupdate only fires when a column value actually changed.
    project.updated_at = utcnow()
    db.commit()
    return _project_row_with_pin(db, project)


@projects_router.delete("/projects/{project_id}")
async def delete_project(project_id: str, db: OrmSession = Depends(workspace_db)) -> dict:
    """Delete a project. **This deletes only our own rows. Never a Hermes session.**"""
    project = _load_project(db, project_id)
    unfiled = list(
        db.execute(
            select(Session.runtime_session_id).where(Session.project_id == project.id)
        ).scalars()
    )
    # Runs must be detached before the delete; the FKs are enforced and would refuse it otherwise.
    db.execute(
        update(Run).where(Run.project_id == project.id).values(session_id=None, project_id=None)
    )
    # Artifacts are unfiled here, never deleted: nothing in this system deletes an artifact.
    db.execute(update(Artifact).where(Artifact.project_id == project.id).values(project_id=None))
    filing_ids = list(
        db.execute(select(Session.id).where(Session.project_id == project.id)).scalars()
    )
    if filing_ids:
        db.execute(update(Run).where(Run.session_id.in_(filing_ids)).values(session_id=None))
    db.execute(delete(Session).where(Session.project_id == project.id))
    db.delete(project)
    db.commit()
    return {
        "id": project_id,
        "deleted": True,
        "unfiled_stored_session_ids": [stored for stored in unfiled if stored],
        "unfiled_session_count": len(unfiled),
        "hermes_sessions_deleted": 0,
    }


@projects_router.get("/projects/{project_id}/sessions")
async def list_project_sessions(
    project_id: str, request: Request, db: OrmSession = Depends(workspace_db)
) -> dict:
    """The sessions filed into this project, enriched with Hermes's own metadata."""
    project = _load_project(db, project_id)
    rows = list(
        db.execute(
            select(Session)
            .where(Session.project_id == project.id)
            .order_by(Session.created_at.desc(), Session.id)
        ).scalars()
    )
    filed_ids = [row.runtime_session_id for row in rows if row.runtime_session_id]
    orphan_ids = snapshot_only_stored_ids(db, project.id, set(filed_ids))
    all_ids = filed_ids + orphan_ids
    latest = latest_snapshots_by_stored_id(db, all_ids, project_id=project.id)
    counts = snapshot_counts_by_stored_id(db, all_ids, project_id=project.id)

    index, checked_profiles, runtime_error = await _hermes_session_index(
        request, {row.profile for row in rows}
    )
    runtime_available = index is not None
    lookup = index or {}

    def _checked(profile: str | None) -> bool:
        """Whether THIS row's profile was actually listed."""
        return runtime_available and (profile or DEFAULT_PROFILE) in checked_profiles

    sessions = [
        _filed_session_row(
            row,
            project,
            lookup.get(row.runtime_session_id) if row.runtime_session_id else None,
            runtime_available=_checked(row.profile),
            latest_snapshot=latest.get(row.runtime_session_id) if row.runtime_session_id else None,
            snapshot_count=counts.get(row.runtime_session_id, 0) if row.runtime_session_id else 0,
        )
        for row in rows
    ]
    # session.list returns rows already ranked by last activity; their dict order IS that rank.
    hermes_rank = {stored_id: rank for rank, stored_id in enumerate(lookup.keys())}
    sessions.sort(key=lambda row: hermes_rank.get(row["id"], len(hermes_rank)))
    orphans = [
        _snapshot_only_row(
            stored_id,
            project,
            lookup.get(stored_id),
            runtime_available=runtime_available,
            latest_snapshot=latest.get(stored_id),
            snapshot_count=counts.get(stored_id, 0),
        )
        for stored_id in orphan_ids
        if latest.get(stored_id) is not None
    ]
    # Sorts on the datetime, not the ISO string: the string form does not order correctly.
    orphans.sort(key=lambda row: latest[row["id"]].taken_at, reverse=True)
    sessions.extend(orphans)
    active_filed = sum(1 for row in rows if row.archived_at is None)
    return {
        "project": _project_row(
            project,
            active_filed,
            tag_names_of(db, "project", project.id),
            pinned=_pinned_artifact_json(db, project),
        ),
        "sessions": sessions,
        "archived_count": sum(1 for row in sessions if row["archived"] is True),
        "runtime_available": runtime_available,
        "runtime_error": runtime_error,
    }


def file_stored_session(
    db: OrmSession,
    project: Project,
    stored_id: str,
    title: str | None,
    profile: str = "default",
) -> tuple[dict[str, Any], bool]:
    """The DB-only core of filing a stored session into a project."""
    runtime = HERMES_RUNTIME
    existing = _find_filing(db, runtime, stored_id, profile=profile)
    if existing is not None:
        if existing.project_id == project.id:
            return _filing_row(existing), False
        raise _already_filed(db, existing, stored_id)

    session = Session(
        project_id=project.id,
        runtime=runtime,
        profile=profile,
        runtime_session_id=stored_id,
        # Never persist the live handle: it is process-local and re-minted on every reconnect.
        runtime_live_session_id=None,
        title=(title or "").strip() or None,
        status="active",
    )
    db.add(session)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        winner = _find_filing(db, runtime, stored_id, profile=profile)
        if winner is None:  # pragma: no cover - only a non-unique-constraint failure
            raise
        if winner.project_id != project.id:
            raise _already_filed(db, winner, stored_id) from None
        return _filing_row(winner), False

    return _filing_row(session), True


@projects_router.post("/projects/{project_id}/sessions", status_code=201)
async def file_session(
    project_id: str,
    body: SessionFiling,
    response: Response,
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """File a Hermes session into this project. Additive, idempotent, reversible."""
    project = _load_project(db, project_id)
    stored_id = _validate_stored_session_id(body.stored_session_id)

    filing_row, created = file_stored_session(
        db, project, stored_id, body.title, profile=body.profile
    )
    if not created:
        response.status_code = 200
    return {"created": created, **filing_row}


@projects_router.patch("/projects/{project_id}/sessions/{stored_session_id}")
async def move_filed_session(
    project_id: str,
    stored_session_id: str,
    body: SessionMove,
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """Move a filed session from this project into another one."""
    source = _load_project(db, project_id)
    stored_id = _validate_stored_session_id(stored_session_id)
    session = _find_filing(db, HERMES_RUNTIME, stored_id)
    if session is None or session.project_id != source.id:
        raise HTTPException(
            status_code=404,
            detail=(f"session {stored_id!r} is not filed in project {project_id!r}"),
        )
    target = _load_project(db, body.project_id)
    session.project_id = target.id
    db.commit()
    return _filing_row(session)


@projects_router.delete("/projects/{project_id}/sessions/{stored_session_id}")
async def unfile_session(
    project_id: str, stored_session_id: str, db: OrmSession = Depends(workspace_db)
) -> dict:
    """Unfile a session: delete OUR row. The Hermes session is untouched."""
    project = _load_project(db, project_id)
    stored_id = _validate_stored_session_id(stored_session_id)
    session = _find_filing(db, HERMES_RUNTIME, stored_id)
    if session is None or session.project_id != project.id:
        raise HTTPException(
            status_code=404,
            detail=f"session {stored_id!r} is not filed in project {project_id!r}",
        )
    workspace_id = session.id
    db.execute(update(Run).where(Run.session_id == workspace_id).values(session_id=None))
    db.delete(session)
    db.commit()
    return {
        "workspace_session_id": workspace_id,
        "stored_session_id": stored_id,
        "project_id": project.id,
        "unfiled": True,
        "hermes_sessions_deleted": 0,
    }


def _filing_row(session: Session) -> dict[str, Any]:
    """The workspace row itself, without Hermes enrichment."""
    return {
        "workspace_session_id": session.id,
        "id": session.runtime_session_id,
        "stored_session_id": session.runtime_session_id,
        "project_id": session.project_id,
        "runtime": session.runtime,
        "profile": session.profile,
        "title": session.title,
        "status": session.status,
        "filed": True,
        "filed_at": iso_z(session.created_at),
    }


def _already_filed(db: OrmSession, existing: Session, stored_id: str) -> HTTPException:
    """409 naming the project a session is currently filed in."""
    current = db.get(Project, existing.project_id)
    current_title = current.title if current is not None else None
    return HTTPException(
        status_code=409,
        detail=(
            f"session {stored_id!r} is already filed in project "
            f"{existing.project_id!r}"
            + (f" ({current_title!r})" if current_title else "")
            + "; PATCH it to move it"
        ),
    )


__all__ = [
    "HERMES_RUNTIME",
    "filed_project_index",
    "projects_router",
    "workspace_db",
]
