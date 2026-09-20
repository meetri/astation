"""Projects, and the index that files a Hermes session into one.

This is the whole architectural thesis of the product in one module
(`docs/ARCHITECTURE.md` §5.1): **a project is the durable intellectual
workspace and a session is one replaceable path through it.** A project holds
many sessions; sessions come and go; the project, its artifacts, its notes and
its context outlive any particular conversation. Without that separation a
long-lived research project is hostage to a single chat history, which is the
thing this application exists to stop.

Three rules govern everything below, and they are rules rather than
preferences:

**1. Hermes owns sessions. We own an index over them.**
Filing a session creates a row here that *references* Hermes's stored session
id. It does not copy the session, move it, retitle it, or tell Hermes anything
at all. Nothing in *this module's own routes* calls a Hermes RPC that mutates
a session -- `session.list` is the only upstream call in the file, it is
read-only, and it is used purely to decorate our rows with the
title/preview/message-count the UI already renders.

**Exception, by construction, not by omission (P2-17):** `POST
/projects/{id}/sessions/new` (`api/main.py`) genuinely mints a new Hermes
session (`session.create` + one priming `prompt.submit`) before filing it --
that route owns the adapter, not this module. It calls back into this file's
`file_stored_session()` only *after* Hermes has confirmed the session exists,
handing it an already-resolved stored id exactly like `file_session` gets one
from its request body. This file still never dials Hermes itself.

**2. Deleting a project deletes OUR rows and nothing else.**
`DELETE /api/projects/{id}` removes the project and the filing rows that point
into it. The Hermes sessions themselves are untouched and stay exactly where
they were -- they simply become unfiled and continue to appear in
`GET /api/sessions`, the All-sessions view. This is asserted in
`tests/test_projects.py` against an adapter that raises on *any* RPC, because
"we did not mean to" is not a guarantee and a user must be able to delete a
project without wondering whether their research is going with it. Since
P6-3 the same is true of the project's **session snapshots**
(`session_snapshots`, `api/snapshots.py`): both of their FKs are `ON DELETE
SET NULL`, so deleting a project or unfiling a session detaches the copies and
never deletes them -- no cleanup code here, by construction
(`tests/test_snapshots.py` proves it).

**3. Filing is optional, gradual, and additive.**
An unfiled session is not a second-class session; it is the normal state. The
owner had 47 sessions the day projects arrived and every one of them is still
reachable, filed or not. Nothing here may ever make a session unreachable.

**Neither Hermes id space is a primary key** (`ARCHITECTURE.md` §21). The
workspace `Session` row keyed `sess_...` is the primary key;
`runtime_session_id` holds the **stored/durable** id (`20260829_182532_991e3f`)
as the durable mapping, and the **live handle** is never written here at all --
it is process-local, re-minted on every reconnect, and lives only in
`LiveHandleCache`. See `docs/PROTOCOL_VERIFIED.md`, "Session resume & the two
id spaces".
"""

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

#: Longest title/description we will store. Not a security boundary -- the
#: surface is authenticated and single-user -- but an unbounded string in a
#: title column ends up rendered in a list row on a phone.
_MAX_TITLE_CHARS = 200
_MAX_DESCRIPTION_CHARS = 4000

projects_router = APIRouter(tags=["projects"])


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    """Body for `POST /api/projects`.

    `extra="forbid"` throughout this module for the same reason
    `TurnSubmission` uses it: a closed schema means the set of things a client
    can put on the wire is exactly the set of things listed here, and a field
    added to the model later cannot be set by an older client by accident.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=_MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION_CHARS)
    #: The file-browser bookmark. An absolute sandbox
    #: path, or null. Its confinement to the sandbox root is checked in the
    #: route, not here, because the root lives in settings; the validator only
    #: normalizes the string. Deliberately unrelated to the project's
    #: instructions (a `HERMES.md` file, `domain/project_workspace.py`).
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
    """Body for `PATCH /api/projects/{id}` -- rename, re-describe, or both.

    A PATCH must be able to say three different things about `description`:
    leave it alone, set it, and *clear* it. So absence and an explicit `null`
    are deliberately different here, and `model_fields_set` is what tells them
    apart -- reading `description is None` alone would silently wipe a
    description on every rename.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=_MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION_CHARS)
    #: Absence, explicit null, and a value are three different intents for the
    #: bookmark exactly as for `description` (see the class docstring):
    #: leave it, clear it, set it. `model_fields_set` in the route tells them
    #: apart; a `null` clears the bookmark (the browser reverts to the root).
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
    """Trim a folder-path bookmark; blank becomes null (= no bookmark).

    Only the shape is settled here -- that the confined value is inside the
    sandbox root is the route's job (`validate_sandbox_path`), because the
    root is a runtime setting rather than a property of the string.
    """
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


class SessionFiling(BaseModel):
    """Body for `POST /api/projects/{id}/sessions` -- file a Hermes session.

    `stored_session_id` is the runtime's **STORED / durable** id, the `id` from
    `GET /api/sessions`. A live handle sent here would be a bug on the client
    side that this gateway cannot detect (the two id spaces are both opaque
    strings), which is why the field is named for the space it belongs to
    rather than a bare `session_id`.

    `title` is optional and is only a **fallback**: Hermes owns the title, and
    `GET /api/projects/{id}/sessions` shows Hermes's. What we store is what the
    UI can still render if that session ever vanishes upstream.

    There is deliberately **no `runtime` field.** The column exists and every
    row is written `"hermes"`, because the schema is built for a second runtime
    later -- but letting a client choose one today would let it write a row that
    `PATCH`/`DELETE` (which look sessions up as Hermes sessions) could never
    find again. An unmovable, unfileable orphan is a worse thing to ship than a
    knob nobody can use yet. The field goes in when a second adapter does, along
    with the routes that can address it.
    """

    model_config = ConfigDict(extra="forbid")

    stored_session_id: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=_MAX_TITLE_CHARS)
    #: Which Hermes profile `stored_session_id` belongs to -- e.g.
    #: filing in a session found by browsing `GET /api/sessions?profile=
    #: kimi25`. `default` unless named, matching every row filed before this
    #: field existed. Not validated against `GET /api/profiles` here: unlike
    #: creating a *new* session, filing an existing one makes no Hermes call
    #: at all (see the module docstring), so there is nothing to 503 against
    #: -- an operator who mistypes the name simply files it under the wrong
    #: label, discoverable by trying to open it.
    profile: str = Field(default="default")


class SessionMove(BaseModel):
    """Body for `PATCH /api/projects/{id}/sessions/{stored_session_id}`."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Dependencies and small helpers
# ---------------------------------------------------------------------------

#: `domain.filing.find_filing` under the name this module's routes (and
#: `api/instance.py`, `api/snapshots.py`) have always used.
_find_filing = find_filing


#: A DB session, with "you never ran the migration" turned into a 503 -- see
#: `domain.db.schema_checked_db`. Verifies the Phase 1 tables; every router
#: whose feature added a later migration checks its own (`api/runs.py`,
#: `api/artifacts.py`, ...) under its own `app.state` flag.
workspace_db = schema_checked_db("db_schema_verified", schema_is_present)


def _project_row(
    project: Project,
    session_count: int,
    tags: list[str] | None = None,
    *,
    pinned: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        # Always present, `[]` when there are none, so a client never has to
        # treat absence as a special case. Sorted by name.
        "tags": tags or [],
        # The one artifact this project is currently ABOUT, rendered rather
        # than a bare id so the project card does not need a second request.
        # Always present, `null` when nothing is pinned -- same rule as tags.
        "pinned_artifact": pinned,
        "id": project.id,
        "title": project.title,
        "description": project.description,
        # The file-browser bookmark, and where the project's instructions file
        # lives. `instructions_path` is PURE (project id + sandbox root), so it
        # is always present -- the app opens the editor there via the
        # `instructions/ensure` route, which creates the file on first open.
        # It is NOT a claim the file exists yet.
        "folder_path": project.folder_path,
        "instructions_path": instructions_path(get_settings(), project.id),
        "created_at": iso_z(project.created_at),
        "updated_at": iso_z(project.updated_at),
        "session_count": session_count,
    }


def _session_counts(db: OrmSession, project_ids: list[str]) -> dict[str, int]:
    """`{project_id: ACTIVE filed session count}` in one query, zeros included.

    Active = `archived_at IS NULL` (P6-3, `docs/SESSION_ARCHIVE_DESIGN.md`
    §6.4): `session_count` is what the Project Library card and the project's
    "Sessions" header show, and an archived session lives in the Archived
    section instead, counted by `archived_count` on the project session list.
    """
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
    """`{stored_session_id: {project_id, project_title, workspace_session_id, archived_at}}`.

    One query for the whole All-sessions view (`GET /api/sessions`), so showing
    "in project X" on 47 rows costs one join rather than 47 lookups or a second
    HTTP round trip from the phone. `archived_at` (iso `Z` or null) rides along
    since P6-3 so the All-sessions row can say "archived" without a second
    request -- and the session is still *in* that list, which is the
    All-sessions guarantee the archive flag must not break.
    """
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
    """`({stored_id: metadata}, profiles actually checked, error)` -- read-only.

    Returns `(None, "why")` rather than raising when Hermes cannot be reached.
    A project's session list is *workspace* state and must survive the runtime
    being down (`ARCHITECTURE.md` §19: "show session runtime offline rather than
    making the whole application unavailable"). The rows still render; they just
    render without the upstream title/preview/message count.

    Never indexes into a row it has not checked the shape of (B-34): a session
    entry that is not a dict, or has no string `id`, is skipped rather than
    trusted.

    **One call per profile (B-200).** `session.list` is scoped to ONE Hermes
    profile, so listing only `default` meant a filed session on any other
    profile was absent from the index -- and absent from a reachable Hermes is
    how `_filed_session_row` computes `missing: true`, which the app renders
    as "No longer on Hermes". The owner's `gpt-sol` and `ornith` sessions were
    being reported as gone while Hermes still had every one of them. This was
    carried as an accepted degradation ("the row still renders using the
    workspace's own stored title") on the belief that it only cost
    enrichment; it cost more than that.

    `profiles` names the profiles actually represented in the caller's rows,
    so an instance with nine profiles still makes one call when a project's
    sessions all live on one. A profile whose own call fails is skipped with a
    log rather than failing the whole listing -- its rows then render with
    `missing: null`, "we could not check", which is the honest answer and not
    a claimed loss.
    """
    wanted = sorted({(p or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE for p in (profiles or ())})
    if not wanted:
        wanted = [DEFAULT_PROFILE]

    known = await _known_profile_names(request)
    index: dict[str, Any] = {}
    checked: set[str] = set()
    last_error: str | None = None
    for profile in wanted:
        if known is not None and profile not in known:
            # B-200, second half: this row names a profile Hermes does not
            # have -- renamed, removed, or misspelled at filing time. Asking
            # for it returns an empty list, and reading that as "the session
            # is gone" is a false claim about a session we never checked.
            # Leaving it out of `checked` makes the row answer `null`.
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
    """Every profile Hermes currently has, or `None` when it cannot be asked.

    `None` means "do not filter": a failed `profiles.list` must not turn every
    row into an unchecked one, which would hide a real loss just as surely as
    the bug this guards against claimed a false one.
    """
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
    """One element of `GET /api/projects/{id}/sessions` for a FILED session.

    Shaped like an element of `GET /api/sessions` on purpose -- same `id`
    (the STORED session id), same `title` / `preview` / `started_at` /
    `message_count` / `source` -- so the UI renders a project's sessions with
    the code it already has for the All-sessions list, and so a session looks
    like the same object in both places.

    `missing` is three-valued, and the three values are genuinely different
    answers:

    * `false` -- Hermes listed this session; the metadata here is Hermes's.
    * `true`  -- Hermes was reachable and did **not** list it. The session has
      gone from the runtime. The row still renders, from what we saved when it
      was filed, because a filed session disappearing upstream must not blank
      out (or 500) the whole project.
    * `null`  -- we could not reach Hermes, so we do not know. Reporting `true`
      here would tell the user their sessions are gone every time the Wi-Fi
      drops; reporting `false` would hide a real loss. Neither is honest.

    P6-3 adds `row_key` (the filing row's id -- the app's list identity),
    `archived_at`, `snapshot_count`, `latest_snapshot` and `archived`
    (`api.snapshots.is_archived`: the user flag, or a missing session that
    has a copy). Snapshot-only rows are built by `_snapshot_only_row` below.
    """
    known = hermes if isinstance(hermes, dict) else {}
    if runtime_available:
        missing: bool | None = hermes is None
    else:
        missing = None
    return {
        # The STORED id, the same key the All-sessions list uses. Not a
        # workspace primary key -- `workspace_session_id` is that.
        "id": session.runtime_session_id,
        "title": known.get("title") if known else session.title,
        "preview": known.get("preview"),
        "started_at": known.get("started_at"),
        "message_count": known.get("message_count"),
        "source": known.get("source"),
        "missing": missing,
        # Workspace fields. `filed` is constant `true` on this route and is
        # included anyway so one decoder serves both session lists.
        "filed": True,
        "workspace_session_id": session.id,
        "project_id": project.id,
        "project_title": project.title,
        "runtime": session.runtime,
        "profile": session.profile,
        "status": session.status,
        "filed_at": iso_z(session.created_at),
        # P6-3.
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
    """A project row for a session that has a snapshot here but no filing row.

    This is what the Archived section shows for a session that was **deleted**
    (the pre-delete snapshot stayed, the filing row went with the delete) or
    **removed from the project** (unfiled; the snapshot's `project_id` still
    names this project). The two are told apart by `status`: `"deleted"` when
    Hermes no longer lists it (`missing == true`), else `"unfiled"` -- a
    session alive on Hermes is not deleted, whatever happened to its filing.

    `row_key` is `"snapshot:" + <latest snapshot id>` because the shipped
    decoder keyed rows on `workspace_session_id`, which this row has none of.
    Title/preview/etc. come from the snapshot's recorded `session.list` row
    (`hermes_meta`), so the row renders the same with Hermes up or down.
    """
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
        # `deleted` when Hermes no longer lists it -- or, when Hermes cannot be
        # asked, when this gateway's own last act on the session was the
        # pre-delete snapshot (it knows; a Wi-Fi drop should not relabel a
        # session it deleted a minute ago as "removed from project").
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


# ---------------------------------------------------------------------------
# Projects CRUD
# ---------------------------------------------------------------------------


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
    """Pin the one artifact this project is currently ABOUT.

    One, not a list: a project with five deliverables has none. Pinning a
    second replaces the first rather than erroring, because "this is the
    important one now" is the whole gesture.

    The artifact does not have to belong to this project. A report that lives
    in a shared folder is still the thing this project produced, and refusing
    would push the owner to file a copy just to pin it.
    """
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
    """The pinned row, or `null`.

    Rendered rather than returned as a bare id: the project home shows the
    card, and a client that had to fetch it separately would render the
    project before knowing what it is about.
    """
    if not project.pinned_artifact_id:
        return None
    artifact = db.get(Artifact, project.pinned_artifact_id)
    return _artifact_json(artifact) if artifact is not None else None


def _pinned_artifacts_for(db: OrmSession, projects: list[Project]) -> dict[str, dict[str, Any]]:
    """Every project's pinned row, by project id, in ONE query.

    The library screen shows a dozen projects; a per-row `db.get` would be a
    dozen round trips for the same cards `list_projects` already promises to
    deliver in a single request.
    """
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
        # A pin whose artifact is gone reads as unpinned rather than as a
        # broken card: the id is not a foreign key (the circular
        # projects<->artifacts reference SQLite cannot create), so a dangling
        # one is a state this has to answer for.
        if project.pinned_artifact_id in rows
    }


@projects_router.put("/projects/{project_id}/tags/{name}")
async def tag_project(project_id: str, name: str, db: OrmSession = Depends(workspace_db)) -> dict:
    """Give a project a tag, from the SAME vocabulary artifacts use.

    One vocabulary on purpose: this workspace is one project holding 95% of
    everything, and a tag meaning different things on a project and on a file
    would be two vocabularies wearing one name.
    """
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
    """Every project, newest first, each with how many sessions are filed in it.

    The count comes from one grouped query rather than a per-project lookup, so
    the Project Library screen is a single round trip regardless of how many
    projects exist.
    """
    wanted = _normalized_tag_list(tag)
    projects = list(
        db.execute(select(Project).order_by(Project.created_at.desc(), Project.id)).scalars()
    )
    if wanted:
        # AND, like the artifact listing: tags narrow, and OR would hand back
        # more projects the more precisely the operator asked.
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
    """Create a project. Title required, description optional.

    Titles are deliberately **not** unique. Two projects can legitimately be
    called "Battery degradation" -- they are distinguished by their id, and
    refusing the second one would be the workspace telling the user how to
    organize their own research.
    """
    folder_path = _validated_folder_path(body.folder_path)
    project = Project(title=body.title, description=body.description, folder_path=folder_path)
    db.add(project)
    db.commit()
    return _project_row(project, 0)


def _validated_folder_path(raw: str | None) -> str | None:
    """The bookmark's confinement check, shared by create and update.

    `None` passes straight through (no bookmark / clear it). A value must be a
    well-formed absolute path inside the sandbox root; `validate_sandbox_path`
    answers 422 for a malformed one and 403 for one outside the root, the same
    statuses the file routes give, and returns it normalized (`..` resolved).
    """
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
    """Make sure the project's `HERMES.md` exists, and say where it is.

    The app's "Instructions" door calls this, then opens the file in the same
    editor every sandbox text file uses (`PUT /api/sandbox/text`). That editor
    cannot create a file or its parent directory, so this does both -- once,
    idempotently: `files_mkdir` is `mkdir -p`, and an existing `HERMES.md` is
    never overwritten (`domain/project_workspace.ensure_instructions_file`).

    Runs on the DEFAULT connection: the instructions folder is one place per
    project, not per profile, and the file it seeds is profile-agnostic -- it
    becomes a session's system prompt through that session's own `cwd`
    whichever profile created it.

    404 if the project does not exist; 502 if the workspace could not be
    created upstream (a real filesystem/permission failure, not a routine
    "already there").
    """
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
    """Rename a project and/or change its description.

    Only the fields actually present in the request body are written, so a
    rename cannot clear a description as a side effect. Sending
    `{"description": null}` explicitly *does* clear it -- that is the one way to
    say "remove this", and it is distinguishable from omission.
    """
    project = _load_project(db, project_id)
    fields = body.model_fields_set
    if "title" in fields and body.title is not None:
        project.title = body.title
    if "description" in fields:
        project.description = body.description
    if "folder_path" in fields:
        # Absent leaves it, explicit null clears it, a value sets it -- and a
        # value must be inside the sandbox root (403 otherwise), the same wall
        # every file route enforces. Clearing reverts the browser to the root.
        project.folder_path = _validated_folder_path(body.folder_path)
    # `onupdate=utcnow` only fires when a column actually changed; a PATCH that
    # sets a field to the value it already held should still count as a touch.
    project.updated_at = utcnow()
    db.commit()
    return _project_row_with_pin(db, project)


@projects_router.delete("/projects/{project_id}")
async def delete_project(project_id: str, db: OrmSession = Depends(workspace_db)) -> dict:
    """Delete a project. **This deletes only our own rows. Never a Hermes session.**

    Read that literally. Deleting a project removes:

      * the `projects` row, and
      * the `sessions` rows that filed a Hermes session into it -- which are
        *references*, not sessions.

    It does **not** call `session.close`, `session.interrupt`, `prompt.submit`
    or any other Hermes RPC. It does not call Hermes at all: this function has
    no adapter, takes no `Request`, and could not reach the runtime if it tried.
    That is deliberate and it is the point of the feature. Hermes owns the
    sessions; we own an index over them, and destroying an index must never
    destroy the thing it indexes.

    So every session that was filed here is simply **unfiled**. It keeps its
    entire transcript, it stays on the Hermes instance, and it goes on appearing
    in `GET /api/sessions` -- the All-sessions view -- with `filed: false`. It
    can be filed into another project immediately.

    `tests/test_projects.py::test_deleting_a_project_never_calls_hermes` proves
    this against an adapter whose every method raises, and additionally checks
    the sessions are still listed afterwards. Anyone adding a Hermes call to
    this path will fail that test, which is exactly what should happen.
    """
    project = _load_project(db, project_id)
    unfiled = list(
        db.execute(
            select(Session.runtime_session_id).where(Session.project_id == project.id)
        ).scalars()
    )
    # B-90: `runs.session_id` is ALSO a FK to `sessions.id`, and with
    # `PRAGMA foreign_keys=ON` (domain/db.py) SQLite refuses to delete a
    # filing row that any run still points at -- so deleting a project whose
    # sessions have ever run a turn raised `IntegrityError` -> 500, and the
    # app swallowed it as "the swipe does nothing" (owner-reported
    # 2026-09-01). Detach first, exactly as `api/instance.py`'s session
    # delete does: `session_id = NULL` is the "unfiled" state P2-2d already
    # defines, and `runtime_session_id`/`project_id` stay on the run so the
    # history remains attributable.
    # `runs` FKs the project TWICE -- `runs.session_id -> sessions.id` and
    # `runs.project_id -> projects.id` -- so both must be released. Detaching
    # by `project_id` also catches runs recorded against the project that were
    # never filed to one of its sessions.
    db.execute(
        update(Run).where(Run.project_id == project.id).values(session_id=None, project_id=None)
    )
    # `artifacts.project_id` is a THIRD FK to this project. Artifacts are
    # content-addressed and expensive to re-derive, so they are UNFILED, never
    # deleted -- they remain reachable via `GET /api/artifacts?unfiled=true`
    # and keep their `producing_run_id`.
    db.execute(update(Artifact).where(Artifact.project_id == project.id).values(project_id=None))
    filing_ids = list(
        db.execute(select(Session.id).where(Session.project_id == project.id)).scalars()
    )
    if filing_ids:
        db.execute(update(Run).where(Run.session_id.in_(filing_ids)).values(session_id=None))
    # Delete the filing rows before the project: `sessions.project_id` is a
    # non-nullable FK, and `PRAGMA foreign_keys=ON` (domain/db.py) means the
    # database enforces it rather than leaving orphans behind.
    db.execute(delete(Session).where(Session.project_id == project.id))
    db.delete(project)
    db.commit()
    return {
        "id": project_id,
        "deleted": True,
        # The stored ids that just became unfiled. They are still on Hermes and
        # still in `GET /api/sessions`; naming them lets a client refresh its
        # filing badges without re-fetching everything.
        "unfiled_stored_session_ids": [stored for stored in unfiled if stored],
        "unfiled_session_count": len(unfiled),
        # Stated explicitly in the response because it is the guarantee, not an
        # implementation detail.
        "hermes_sessions_deleted": 0,
    }


# ---------------------------------------------------------------------------
# Filing: workspace Session rows become real
# ---------------------------------------------------------------------------


@projects_router.get("/projects/{project_id}/sessions")
async def list_project_sessions(
    project_id: str, request: Request, db: OrmSession = Depends(workspace_db)
) -> dict:
    """The sessions filed into this project, enriched with Hermes's own metadata.

    One read-only `session.list` upstream, joined against our filing rows. Each
    element is shaped like an element of `GET /api/sessions` (see
    `_filed_session_row`) so the UI renders both lists with one decoder.

    **A filed session whose Hermes session has vanished still renders.** It
    comes back with `missing: true` and whatever title we saved at filing time,
    and it does not disturb any other row. The alternative -- 404 or 500 on the
    whole list because one referenced session is gone -- would mean one deleted
    session upstream could black out a project the user has been building for
    weeks.

    If Hermes cannot be reached at all, the list is still served: every row gets
    `missing: null` (unknown, not gone) and the response says
    `runtime_available: false` with the reason. Workspace state does not depend
    on the runtime being up.

    **P6-3 (`docs/SESSION_ARCHIVE_DESIGN.md` §6.4): every row gains
    `row_key`, `archived_at`, `snapshot_count`, `latest_snapshot`, `archived`;
    snapshot-only rows are appended after the filed rows** (stored ids with a
    snapshot under this project and no filing row -- deleted or unfiled
    sessions whose copy remains), ordered by `latest_snapshot.taken_at` desc.
    Top level gains `archived_count`. `project.session_count` counts **active
    filed rows only**. Nothing a client already decoded is removed or renamed.
    Still exactly one upstream call (`session.list`), still no Hermes write.
    """
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
    # Scoped to THIS project: a session snapshotted here, moved elsewhere and
    # snapshotted again must not make this project's row open the other copy.
    latest = latest_snapshots_by_stored_id(db, all_ids, project_id=project.id)
    counts = snapshot_counts_by_stored_id(db, all_ids, project_id=project.id)

    # B-200: one `session.list` per profile these rows actually live on --
    # listing only `default` reported every other profile's session as gone.
    index, checked_profiles, runtime_error = await _hermes_session_index(
        request, {row.profile for row in rows}
    )
    runtime_available = index is not None
    lookup = index or {}

    def _checked(profile: str | None) -> bool:
        """Whether THIS row's profile was actually listed (B-200).

        Global availability is not enough: one project can hold rows on a
        profile Hermes has and rows on one it does not, and only the first
        kind can honestly be called missing.
        """
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
    # Owner request 2026-09-04: sort the FILED rows by Hermes's own
    # last-active order, not filing time. `lookup` (== `index`, above) is
    # built by `_hermes_session_index` iterating `session.list`'s response
    # into a plain dict IN ORDER, and that order is not creation order --
    # verified live against the real instance: Hermes's `session.list` rows
    # are NOT sorted by `started_at` (a session can rank above another with
    # a strictly newer `started_at`), because Hermes's own query underneath
    # (`db.list_sessions_rich(..., order_by_last_active=True)`, upstream
    # source `tui_gateway/methods_session.py`) already orders by real
    # last-activity server-side -- the row shape just never carried the raw
    # timestamp over the wire, only the correct rank. A session Hermes
    # doesn't currently list (runtime unreachable, or a stored id it no
    # longer knows -- `missing: true` rows) has no rank and sorts after
    # every ranked one; Python's stable sort keeps such rows in their
    # original filing-time relative order among themselves. Orphans
    # (snapshot-only rows, below) are UNCHANGED by this -- they keep their
    # own documented `latest_snapshot.taken_at desc` order, appended after.
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
    # Sort on the datetime, not its ISO string: `_iso` drops the fractional
    # part on a whole second and 'Z' > '.', which mis-ordered rows.
    orphans.sort(key=lambda row: latest[row["id"]].taken_at, reverse=True)
    sessions.extend(orphans)
    active_filed = sum(1 for row in rows if row.archived_at is None)
    return {
        "project": _project_row(
            project,
            active_filed,
            tag_names_of(db, "project", project.id),
            # The project home renders the pinned card above the sessions, so
            # this row has to carry it too -- see `_pinned_artifact_json`.
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
    """The DB-only core of filing a stored session into a project.

    Pulled out of the `file_session` route so `POST
    /projects/{id}/sessions/new` (`api/main.py`, P2-17) can share the exact
    same idempotent-file/already-filed-409/race-retry logic after it creates
    a brand-new Hermes session, instead of a second copy drifting out of sync
    with this one. **Still does not call Hermes** -- `stored_id` is handed in
    already resolved, so the "session.list is the only upstream call in this
    file" invariant in the module docstring holds for every call *originating
    in this file*; the caller in `api/main.py` is the one with the adapter.

    `profile` (B-136) is the Hermes profile `stored_id` belongs to -- a
    stored id is only unique *within* a profile, so idempotency and the
    already-filed check below are both scoped by it too (`_find_filing`).

    Returns `(filing_row, created)`, the same shape `file_session` used to
    build its response from inline.
    """
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
        # Never the live handle. See the callers' docstrings, and B-06.
        runtime_live_session_id=None,
        title=(title or "").strip() or None,
        status="active",
    )
    db.add(session)
    try:
        db.commit()
    except IntegrityError:
        # Two filings of the same stored id raced. The unique constraint is the
        # arbiter; whoever lost re-reads the winner and reports success, which
        # is the same answer the idempotent path above gives.
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
    """File a Hermes session into this project. Additive, idempotent, reversible.

    The body carries the runtime's **STORED / durable** session id. What is
    created is one workspace `Session` row whose primary key is our own
    `sess_...` id and whose `runtime_session_id` column *references* Hermes's.
    Nothing is sent to Hermes -- this route has no adapter either. The session
    is not moved, copied, retitled or claimed; it is indexed.

    **Idempotent.** Filing the same stored id into the same project twice
    returns the existing row with `201 -> 200` and `created: false`. It does not
    duplicate, and it does not error: a phone that retries a request whose
    response was lost must not end up with two rows, and must not be shown a
    failure for an operation that already succeeded.

    **Already filed somewhere else -> 409, never a silent move.** The schema's
    `UNIQUE (runtime, runtime_session_id)` says a stored session belongs to at
    most one project, which is what makes "which project is this in?" a question
    with one answer. Quietly relocating someone's session because they tapped
    the wrong row is not a thing this should do; the 409 names the project it is
    currently in, and `PATCH` is the explicit move.

    The live handle is deliberately **not** stored (`runtime_live_session_id`
    stays NULL). It is process-local, it is re-minted on every gateway<->Hermes
    reconnect, and persisting one is this project's recurring bug class (B-06).
    """
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
    """Move a filed session from this project into another one.

    A single-row update of `sessions.project_id` -- the workspace `Session` row
    keeps its identity, so anything that later hangs off it (runs, artifacts,
    notes) survives the move. Hermes is not involved and is not called.

    `{project_id}` in the path is where the session is filed *now*; a mismatch
    is a 404 rather than a silent success, so a stale UI cannot move a session
    it was not actually looking at. Moving into the project it is already in is
    a no-op and succeeds.
    """
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
    """Unfile a session: delete OUR row. The Hermes session is untouched.

    Same guarantee as `delete_project`, at single-session granularity, and for
    the same reason -- this route has no adapter and makes no upstream call. The
    session keeps its transcript, stays on the instance, and reappears in
    `GET /api/sessions` with `filed: false` the moment this returns.
    """
    project = _load_project(db, project_id)
    stored_id = _validate_stored_session_id(stored_session_id)
    session = _find_filing(db, HERMES_RUNTIME, stored_id)
    if session is None or session.project_id != project.id:
        raise HTTPException(
            status_code=404,
            detail=f"session {stored_id!r} is not filed in project {project_id!r}",
        )
    workspace_id = session.id
    # B-90, same trap as the project delete above: any run pointing at this
    # filing row blocks the delete under `PRAGMA foreign_keys=ON`. Unfiling a
    # session that has run a turn 500'd, which the app showed as a swipe that
    # silently did nothing.
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
    """The workspace row itself, without Hermes enrichment.

    Returned by the write routes, which deliberately do not call Hermes -- a
    filing is a local fact and confirming it should not depend on the runtime
    being reachable. `GET /api/projects/{id}/sessions` is where enrichment
    happens.
    """
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
