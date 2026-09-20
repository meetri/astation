"""Tier-2 durable artifact library: the artifact REST API (P3-1 / P3-2)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session as OrmSession

from config.settings import get_settings
from domain.artifact_ingest import (  # noqa: F401  (re-exported; see module docstring)
    _DESYNCHRONIZED_TYPE,
    AUTO_INGEST_TOOL_NAMES,
    DIFF_SCAN_MAX_DEPTH,
    DIFF_SCAN_MAX_DIRS,
    MAX_PENDING_DIFF_RUNS,
    SANDBOX_DIFF_DENYLIST_DIRS,
    SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS,
    ArtifactIngestor,
    IngestOutcome,
    MediaTagIngestor,
    SandboxDiffDenylist,
    SandboxDiffIngestor,
    extract_media_tag_paths,
)
from domain.artifact_kinds import KINDS as ARTIFACT_KINDS
from domain.artifact_kinds import folder_components, is_valid_kind, is_within
from domain.artifact_store import (  # noqa: F401  (re-exported; see module docstring)
    MAX_INGEST_BYTES,
    STATUS_AVAILABLE,
    STATUS_UNAVAILABLE,
    ArtifactStore,
    ArtifactTooLargeError,
    _artifact_json,
)
from domain.bookmark_store import UNFILED_SCOPE
from domain.bookmark_store import scoped_ids as bookmark_scoped_ids
from domain.bookmark_store import scopes_of as bookmark_scopes_of
from domain.bookmark_store import set_bookmark as bookmark_set
from domain.bookmark_store import starred_at as bookmark_starred_at
from domain.collection_store import (
    MAX_COLLECTION_DESCRIPTION_CHARS,
    MAX_COLLECTION_NAME_CHARS,
    CollectionNameError,
    collection_json,
    normalize_collection_name,
)
from domain.collection_store import add as collection_add
from domain.collection_store import containing as collections_containing
from domain.collection_store import create as collection_create
from domain.collection_store import delete_collection as collection_delete
from domain.collection_store import listing as collection_listing
from domain.collection_store import member_ids as collection_member_ids
from domain.collection_store import remove as collection_remove
from domain.collection_store import reorder as collection_reorder
from domain.db import columns_present, schema_checked_db
from domain.models import Artifact, Collection, Project, Run, utcnow
from domain.sandbox_paths import validate_sandbox_path
from domain.tag_store import apply_bulk as tag_apply_bulk
from domain.tag_store import attach as tag_attach
from domain.tag_store import catalog as tag_catalog
from domain.tag_store import delete_tag as tag_delete
from domain.tag_store import detach as tag_detach
from domain.tag_store import get_or_create as tag_get_or_create
from domain.tag_store import names_for as tag_names_for
from domain.tag_store import names_of as tag_names_of
from domain.tag_store import owner_ids_with_all as tag_owner_ids_with_all
from domain.tag_store import rename as tag_rename
from domain.tags import TagNameError, normalize_tag_name, normalize_tag_names
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

artifacts_router = APIRouter(tags=["artifacts"])

_SERVE_CHUNK_BYTES = 64 * 1024

PREVIEW_MAX_BYTES = 512 * 1024


_artifacts_db = schema_checked_db(
    "artifacts_schema_verified",
    lambda engine: columns_present(engine, "artifacts", "status", "source_path"),
)


class _Unsatisfiable:
    """Sentinel: the Range header was valid `bytes=` syntax but names nothing
    inside the file -> 416."""


UNSATISFIABLE = _Unsatisfiable()


# A malformed or multi-range header is ignored and the whole body served, per RFC 9110.
def parse_range_header(header: str | None, size: int):
    """`None` (serve everything), `(start, end)` inclusive (206), or
    `UNSATISFIABLE` (416).
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes=") :].strip()
    if "," in spec or not spec:
        return None
    start_s, sep, end_s = spec.partition("-")
    if not sep:
        return None
    start_s = start_s.strip()
    end_s = end_s.strip()
    try:
        if start_s == "":
            n = int(end_s)
            if n <= 0 or size == 0:
                return UNSATISFIABLE
            return (max(size - n, 0), size - 1)
        start = int(start_s)
        if start < 0:
            return None
        if start >= size:
            return UNSATISFIABLE
        end = int(end_s) if end_s else size - 1
        if end < start:
            return UNSATISFIABLE
        return (start, min(end, size - 1))
    except ValueError:
        return None


def _file_slice(path: Path, start: int, end: int) -> Iterator[bytes]:
    """Yield `[start, end]` (inclusive) of `path` in bounded chunks."""
    remaining = end - start + 1
    with open(path, "rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(_SERVE_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


# Callers must pass rows newest-first; the first row seen for a path is kept.
def _collapse_to_latest(rows) -> list[dict]:
    """B-184: one row per `source_path`, the newest, plus how many rows it
    stands for as `versions`.
    """
    latest: dict[str, dict] = {}
    ordered: list[dict] = []
    for artifact in rows:
        key = artifact.source_path
        if key is not None and key in latest:
            latest[key]["versions"] += 1
            continue
        entry = _artifact_json(artifact)
        entry["versions"] = 1
        if key is not None:
            latest[key] = entry
        ordered.append(entry)
    return ordered


# Not a SQL filter: kind is derived from mime and extension, so limit counts unfiltered.
def _filter_by_kind(rows: list[dict], kind: str | None) -> list[dict]:
    """Keep only rows of one kind."""
    if not kind:
        return rows
    return [row for row in rows if row.get("kind") == kind]


def _validated_kind(kind: str | None) -> str | None:
    """422 on an unknown kind rather than silently returning nothing."""
    if kind is None or kind == "":
        return None
    if not is_valid_kind(kind):
        raise HTTPException(
            status_code=422,
            detail=f"unknown kind {kind!r}; expected one of {', '.join(ARTIFACT_KINDS)}",
        )
    return kind


def _bookmark_shelf(
    db: OrmSession, *, scope: str | None, limit: int, kind: str | None
) -> list[dict]:
    """One scope's starred artifacts, newest star first."""
    ordered = bookmark_scoped_ids(db, scope)
    if not ordered:
        return []
    by_id = {
        artifact.id: artifact
        for artifact in db.execute(
            select(Artifact)
            .where(Artifact.id.in_(ordered), Artifact.archived_at.is_(None))
        ).scalars()
    }
    rows = [_artifact_json(by_id[a]) for a in ordered if a in by_id][:limit]
    return _with_bookmarks(db, rows, scope=scope)


def _listing(db: OrmSession, query, *, latest: bool, limit: int) -> list[dict]:
    """The rows of one listing query: capped at `limit` either way, but with
    `latest=True` the cap applies AFTER collapsing versions, so the page is
    `limit` files rather than `limit` saves."""
    if not latest:
        return [_artifact_json(a) for a in db.execute(query.limit(limit)).scalars()]
    return _collapse_to_latest(db.execute(query).scalars())[:limit]


ArchivedFilter = Literal["false", "true", "all"]

MAX_BULK_IDS = 200

_MIN_QUERY_CHARS = 2
_MAX_QUERY_CHARS = 80


def _validated_prefix(prefix: str | None) -> str | None:
    """A sandbox directory to scope a listing to, or `None`."""
    if prefix is None:
        return None
    cleaned = prefix.strip().rstrip("/")
    if not cleaned:
        return None
    return validate_sandbox_path(cleaned, get_settings().hermes_sandbox_root)


def _validated_query(q: str | None) -> str | None:
    if q is None:
        return None
    cleaned = q.strip()
    if not cleaned:
        return None
    if len(cleaned) < _MIN_QUERY_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"`q` must be at least {_MIN_QUERY_CHARS} characters: one character "
                "matches most of the library and costs a full scan to say so"
            ),
        )
    if len(cleaned) > _MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=422, detail=f"`q` must be at most {_MAX_QUERY_CHARS} characters"
        )
    return cleaned


def _tag_filter(db: OrmSession, query, tags: list[str]) -> tuple[list[str], Any]:
    """`(normalized names, narrowed query)`."""
    if not tags:
        return [], query
    wanted = _normalized_list_or_422(tags)
    matching = tag_owner_ids_with_all(db, "artifact", wanted)
    return wanted, query.where(Artifact.id.in_(matching))


def _browse_filters(query, *, prefix: str | None, q: str | None, archived: ArchivedFilter):
    """Apply the scope-independent filters both listings share."""
    if prefix is not None:
        # The trailing slash matters: a bare prefix puts /opt/data-old inside /opt/data.
        query = query.where(
            or_(
                Artifact.source_path == prefix,
                Artifact.source_path.startswith(prefix + "/"),
            )
        )
    if q is not None:
        like = f"%{q}%"
        query = query.where(or_(Artifact.title.ilike(like), Artifact.source_path.ilike(like)))
    if archived == "false":
        query = query.where(Artifact.archived_at.is_(None))
    elif archived == "true":
        query = query.where(Artifact.archived_at.is_not(None))
    return query


def _decoded_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    """`<created_at iso>|<id>` -> the keyset position, or `None`."""
    if cursor is None or not cursor.strip():
        return None
    raw_at, _, raw_id = cursor.partition("|")
    try:
        at = datetime.fromisoformat(raw_at)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="`cursor` must be `<created_at iso>|<artifact id>`"
        ) from exc
    if not raw_id:
        raise HTTPException(
            status_code=422, detail="`cursor` must be `<created_at iso>|<artifact id>`"
        )
    return at, raw_id


def _after_cursor(query, position: tuple[datetime, str] | None):
    """Keyset, never offset."""
    if position is None:
        return query
    at, artifact_id = position
    return query.where(
        or_(
            Artifact.created_at < at,
            and_(Artifact.created_at == at, Artifact.id > artifact_id),
        )
    )


def _page(
    db: OrmSession, query, *, latest: bool, limit: int, position: tuple[datetime, str] | None
) -> tuple[list[dict], str | None]:
    """One page of rows, and the cursor for the next one (`None` when last)."""
    if latest:
        # The cursor is applied after the collapse; in SQL older saves would re-collapse.
        collapsed = _collapse_to_latest(db.execute(query).scalars())
        start = 0
        if position is not None:
            at, artifact_id = position
            for index, row in enumerate(collapsed):
                if (row["created_at"], row["id"]) == (iso_z(at), artifact_id):
                    start = index + 1
                    break
        window = collapsed[start:]
        page, more = window[:limit], len(window) > limit
    else:
        rows = list(db.execute(_after_cursor(query, position).limit(limit + 1)).scalars())
        more = len(rows) > limit
        page = [_artifact_json(row) for row in rows[:limit]]
    if not page or not more:
        return page, None
    last = page[-1]
    return page, f"{last['created_at']}|{last['id']}"


def _artifact_json_full(db: OrmSession, artifact: Artifact) -> dict:
    """One row with everything a DETAIL screen shows: tags and collections."""
    row = _artifact_json_with_tags(db, artifact)
    row["collections"] = collections_containing(db, artifact.id)
    return row


def _artifact_json_with_tags(db: OrmSession, artifact: Artifact) -> dict:
    """One row plus its tags."""
    row = _artifact_json(artifact)
    row["tags"] = tag_names_of(db, "artifact", artifact.id)
    return row


def _with_tags(db: OrmSession, rows: list[dict]) -> list[dict]:
    """Attach tags to a whole page in ONE query."""
    if not rows:
        return rows
    by_owner = tag_names_for(db, "artifact", [row["id"] for row in rows])
    for row in rows:
        row["tags"] = by_owner.get(row["id"], [])
    return rows


def _star_scope(db: OrmSession, *, project: str | None, unfiled: bool) -> str | None:
    """Which scope's stars a request is asking about."""
    if project is not None:
        if db.get(Project, project) is None:
            raise HTTPException(status_code=404, detail=f"no project with id {project!r}")
        return project
    return UNFILED_SCOPE if unfiled else None


def _with_bookmarks(db: OrmSession, rows: list[dict], *, scope: str | None) -> list[dict]:
    """Say whether each row is starred IN THIS SCOPE, in one query."""
    if not rows:
        return rows
    starred = bookmark_starred_at(db, [row["id"] for row in rows], scope)
    for row in rows:
        at = starred.get(row["id"])
        row["bookmarked"] = at is not None
        row["bookmarked_at"] = iso_z(at) if at else None
    return rows


def _load_artifact(db: OrmSession, artifact_id: str) -> Artifact:
    artifact = db.get(Artifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"no artifact with id {artifact_id!r}")
    return artifact


@artifacts_router.get("/projects/{project_id}/artifacts")
async def list_project_artifacts(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    latest: bool = Query(default=False),
    kind: str | None = Query(default=None),
    bookmarks: bool = Query(default=True),
    tag: list[str] = Query(default_factory=list),
    prefix: str | None = Query(default=None),
    q: str | None = Query(default=None),
    archived: ArchivedFilter = Query(default="false"),
    cursor: str | None = Query(default=None),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """This project's artifacts, newest first (P3-2, project-scoped)."""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project with id {project_id!r}")
    wanted = _validated_kind(kind)
    scoped_prefix = _validated_prefix(prefix)
    search = _validated_query(q)
    query = (
        select(Artifact)
        .where(Artifact.project_id == project_id)
        .order_by(Artifact.created_at.desc(), Artifact.id)
    )
    query = _browse_filters(query, prefix=scoped_prefix, q=search, archived=archived)
    wanted_tags, query = _tag_filter(db, query, tag)
    rows, next_cursor = _page(
        db, query, latest=latest, limit=limit, position=_decoded_cursor(cursor)
    )
    return {
        "project_id": project_id,
        "tags": wanted_tags,
        "latest_only": latest,
        "kind": wanted,
        "prefix": scoped_prefix,
        "q": search,
        "archived_filter": archived,
        "next_cursor": next_cursor,
        "artifacts": _with_bookmarks(
            db, _with_tags(db, _filter_by_kind(rows, wanted)), scope=project_id
        ),
        "bookmarked": (
            _bookmark_shelf(db, scope=project_id, limit=limit, kind=wanted) if bookmarks else []
        ),
    }


@artifacts_router.get("/artifacts")
async def list_artifacts(
    session: str | None = Query(default=None),
    unfiled: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    latest: bool = Query(default=False),
    kind: str | None = Query(default=None),
    bookmarked: bool = Query(default=False),
    project: str | None = Query(default=None),
    tag: list[str] = Query(default_factory=list),
    prefix: str | None = Query(default=None),
    q: str | None = Query(default=None),
    archived: ArchivedFilter = Query(default="false"),
    cursor: str | None = Query(default=None),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """Global artifact listing, newest first, with two mutually exclusive filters."""
    if session is not None and unfiled:
        raise HTTPException(
            status_code=422,
            detail=(
                "`session` and `unfiled` cannot be combined: one asks which "
                "conversation produced an artifact, the other which project it "
                "was filed into. Pick one."
            ),
        )
    query = select(Artifact).order_by(Artifact.created_at.desc(), Artifact.id)
    stored_session_id: str | None = None
    if session is not None:
        stored_session_id = session.strip()
        if not stored_session_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "`session` must be a non-empty Hermes STORED session id "
                    "(e.g. 20260829_182532_991e3f), not a live handle"
                ),
            )
        # The join drops artifacts with no producing run; run_attributed_only says so.
        query = query.join(Run, Artifact.producing_run_id == Run.id).where(
            Run.runtime_session_id == stored_session_id
        )
    if unfiled:
        query = query.where(Artifact.project_id.is_(None))
    star_scope = _star_scope(db, project=project, unfiled=unfiled)
    if bookmarked:
        starred = bookmark_scoped_ids(db, star_scope)
        query = query.where(Artifact.id.in_(starred))
    wanted = _validated_kind(kind)
    scoped_prefix = _validated_prefix(prefix)
    search = _validated_query(q)
    query = _browse_filters(query, prefix=scoped_prefix, q=search, archived=archived)
    wanted_tags, query = _tag_filter(db, query, tag)
    rows, next_cursor = _page(
        db, query, latest=latest, limit=limit, position=_decoded_cursor(cursor)
    )
    return {
        "session": stored_session_id,
        "tags": wanted_tags,
        "run_attributed_only": stored_session_id is not None,
        "unfiled_only": unfiled,
        "latest_only": latest,
        "kind": wanted,
        "bookmarked_only": bookmarked,
        "bookmark_scope": (star_scope or None) if star_scope is not None else None,
        "prefix": scoped_prefix,
        "q": search,
        "archived_filter": archived,
        "next_cursor": next_cursor,
        "artifacts": _with_bookmarks(
            db, _with_tags(db, _filter_by_kind(rows, wanted)), scope=star_scope
        ),
    }


def _folder_tree(
    rows: list[tuple[str | None, Any]], *, prefix: str | None
) -> tuple[list[dict], int]:
    """`(folders, files directly in prefix)` for one level of the tree."""
    depth = len([part for part in (prefix or "").strip("/").split("/") if part])
    direct = 0
    buckets: dict[str, dict[str, Any]] = {}
    for source_path, created_at in rows:
        components = folder_components(source_path)
        if len(components) == depth:
            direct += 1
            continue
        name = components[depth]
        bucket = buckets.setdefault(name, {"files": 0, "latest_at": None, "children": set()})
        bucket["files"] += 1
        if bucket["latest_at"] is None or (created_at and created_at > bucket["latest_at"]):
            bucket["latest_at"] = created_at
        bucket["children"].add(components[depth + 1] if len(components) > depth + 1 else None)

    base = (prefix or "").rstrip("/")
    folders: list[dict] = []
    for name, bucket in buckets.items():
        path = f"{base}/{name}"
        children = bucket["children"]
        # None in children marks a file directly here, which is what blocks a collapse.
        while len(children) == 1 and None not in children:
            only = next(iter(children))
            path = f"{path}/{only}"
            deeper = {
                folder_components(sp)[len(path.strip("/").split("/"))]
                if len(folder_components(sp)) > len(path.strip("/").split("/"))
                else None
                for sp, _ in rows
                if is_within(sp, path)
            }
            children = deeper or {None}
        folders.append(
            {
                "path": path,
                "name": path[len(base) + 1 :] if path.startswith(base + "/") else name,
                "files": bucket["files"],
                "latest_at": iso_z(bucket["latest_at"]) if bucket["latest_at"] else None,
            }
        )
    folders.sort(key=lambda entry: (-entry["files"], entry["path"]))
    return folders, direct


def _folder_rows(db: OrmSession, query) -> list[tuple[str | None, Any]]:
    """One `(source_path, created_at)` per distinct path, newest kept."""
    newest: dict[str, Any] = {}
    for source_path, created_at in db.execute(query):
        if not source_path:
            continue
        current = newest.get(source_path)
        if current is None or (created_at and current and created_at > current):
            newest[source_path] = created_at
    return list(newest.items())


@artifacts_router.get("/projects/{project_id}/artifacts/folders")
async def list_project_artifact_folders(
    project_id: str,
    prefix: str | None = Query(default=None),
    archived: ArchivedFilter = Query(default="false"),
    q: str | None = Query(default=None),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """The folder level under `prefix` for one project's artifacts."""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project with id {project_id!r}")
    scoped_prefix = _validated_prefix(prefix)
    search = _validated_query(q)
    query = select(Artifact.source_path, Artifact.created_at).where(
        Artifact.project_id == project_id
    )
    query = _browse_filters(query, prefix=scoped_prefix, q=search, archived=archived)
    rows = _folder_rows(db, query)
    folders, direct = _folder_tree(rows, prefix=scoped_prefix)
    return {
        "project_id": project_id,
        "prefix": scoped_prefix,
        "folders": folders,
        "files": direct,
        "total_files": len(rows),
    }


@artifacts_router.get("/artifacts/folders")
async def list_artifact_folders(
    session: str | None = Query(default=None),
    unfiled: bool = Query(default=False),
    bookmarked: bool = Query(default=False),
    project: str | None = Query(default=None),
    tag: list[str] = Query(default_factory=list),
    prefix: str | None = Query(default=None),
    archived: ArchivedFilter = Query(default="false"),
    q: str | None = Query(default=None),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """The folder level under `prefix` for a session, the unfiled rows, or all."""
    if session is not None and unfiled:
        raise HTTPException(
            status_code=422,
            detail=(
                "`session` and `unfiled` cannot be combined: one asks which "
                "conversation produced an artifact, the other which project it "
                "was filed into. Pick one."
            ),
        )
    scoped_prefix = _validated_prefix(prefix)
    search = _validated_query(q)
    query = select(Artifact.source_path, Artifact.created_at)
    stored_session_id: str | None = None
    if session is not None:
        stored_session_id = session.strip()
        if not stored_session_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "`session` must be a non-empty Hermes STORED session id "
                    "(e.g. 20260829_182532_991e3f), not a live handle"
                ),
            )
        query = query.join(Run, Artifact.producing_run_id == Run.id).where(
            Run.runtime_session_id == stored_session_id
        )
    if unfiled:
        query = query.where(Artifact.project_id.is_(None))
    if bookmarked:
        query = query.where(
            Artifact.id.in_(
                bookmark_scoped_ids(db, _star_scope(db, project=project, unfiled=unfiled))
            )
        )
    query = _browse_filters(query, prefix=scoped_prefix, q=search, archived=archived)
    rows = _folder_rows(db, query)
    folders, direct = _folder_tree(rows, prefix=scoped_prefix)
    return {
        "session": stored_session_id,
        "unfiled_only": unfiled,
        "prefix": scoped_prefix,
        "folders": folders,
        "files": direct,
        "total_files": len(rows),
    }


class TagNameRequest(BaseModel):
    """Body for creating or renaming a tag. Closed schema, same rule as every
    other inbound body here."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)


class TagBulkRequest(BaseModel):
    """Body for `POST /api/artifacts/tags`: apply tags across a selection."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=MAX_BULK_IDS)
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _something_to_do(self) -> TagBulkRequest:
        if not self.add and not self.remove:
            raise ValueError("send at least one tag to `add` or to `remove`")
        return self


def _normalized_or_422(name: str) -> str:
    """Normalize, turning the rule that was broken into the 422 detail."""
    try:
        return normalize_tag_name(name)
    except TagNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _normalized_list_or_422(names: list[str]) -> list[str]:
    try:
        return normalize_tag_names(names)
    except TagNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@artifacts_router.get("/tags")
async def list_tags(
    project: str | None = Query(default=None),
    unfiled: bool = Query(default=False),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """The vocabulary, most used first, ties alphabetical."""
    if project is not None and db.get(Project, project) is None:
        raise HTTPException(status_code=404, detail=f"no project with id {project!r}")
    return {
        "project_id": project,
        "unfiled_only": unfiled,
        "tags": tag_catalog(db, project_id=project, unfiled=unfiled),
    }


@artifacts_router.post("/tags")
async def create_tag(body: TagNameRequest, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Add a name to the vocabulary. **Idempotent** -- an existing name is a
    200 with that tag, not a 409: the caller's intent ("this name should
    exist") is already true, and the unique index is what stops a second row
    rendering identically to the first."""
    name = _normalized_or_422(body.name)
    tag = tag_get_or_create(db, name)
    db.commit()
    return {"id": tag.id, "name": tag.name}


@artifacts_router.patch("/tags/{tag_id}")
async def rename_tag(
    tag_id: str, body: TagNameRequest, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Rename a tag, **merging** when the new name already exists."""
    _normalized_or_422(body.name)
    tag = tag_rename(db, tag_id, body.name)
    if tag is None:
        raise HTTPException(status_code=404, detail=f"no tag with id {tag_id!r}")
    db.commit()
    return {"id": tag.id, "name": tag.name}


@artifacts_router.delete("/tags/{tag_id}")
async def delete_tag_route(tag_id: str, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Remove a tag from the vocabulary. **Never deletes an artifact or a
    project** -- the cascade only clears the joins, and the response says how
    many owners lost it so a mistaken delete is visible immediately."""
    deleted, detached = tag_delete(db, tag_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no tag with id {tag_id!r}")
    db.commit()
    return {"deleted": True, "detached": detached}


@artifacts_router.put("/artifacts/{artifact_id}/tags/{name}")
async def tag_artifact(
    artifact_id: str, name: str, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Give an artifact a tag, creating the tag if it is new. Idempotent."""
    artifact = _load_artifact(db, artifact_id)
    tag_attach(db, "artifact", artifact.id, _normalized_or_422(name))
    db.commit()
    db.refresh(artifact)
    return _artifact_json_with_tags(db, artifact)


@artifacts_router.delete("/artifacts/{artifact_id}/tags/{name}")
async def untag_artifact(
    artifact_id: str, name: str, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Take a tag off. Idempotent, and the TAG survives -- it is a vocabulary
    entry the operator typed, and deleting it because its last artifact lost it
    would make the tag list flicker with their own work."""
    artifact = _load_artifact(db, artifact_id)
    tag_detach(db, "artifact", artifact.id, _normalized_or_422(name))
    db.commit()
    db.refresh(artifact)
    return _artifact_json_with_tags(db, artifact)


@artifacts_router.post("/artifacts/tags")
async def tag_artifacts(body: TagBulkRequest, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Apply tags across a selection, in one transaction."""
    add = _normalized_list_or_422(body.add)
    remove = _normalized_list_or_422(body.remove)
    known = [
        artifact_id
        for (artifact_id,) in db.execute(select(Artifact.id).where(Artifact.id.in_(body.ids)))
    ]
    changed = tag_apply_bulk(db, "artifact", known, add=add, remove=remove)
    db.commit()
    return {"updated": changed}


class CollectionRequest(BaseModel):
    """Body for creating or editing a collection."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=MAX_COLLECTION_NAME_CHARS)
    description: str | None = Field(default=None, max_length=MAX_COLLECTION_DESCRIPTION_CHARS)


class CollectionOrderRequest(BaseModel):
    """Body for `PUT /api/collections/{id}/order`: the WHOLE membership."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1)


def _named_or_422(name: str) -> str:
    try:
        return normalize_collection_name(name)
    except CollectionNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _load_collection(db: OrmSession, collection_id: str) -> Collection:
    collection = db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail=f"no collection with id {collection_id!r}")
    return collection


@artifacts_router.get("/collections")
async def list_collections(db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Every collection, most recently touched first, each with its first
    member so a list row can show what it is without a second request."""
    return {"collections": collection_listing(db, _artifact_json)}


@artifacts_router.post("/collections")
async def create_collection(
    body: CollectionRequest, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Make a collection. **Idempotent on the name**, like creating a tag: the
    caller's intent is already true when one exists, and 409ing would just
    move the race to the client."""
    _named_or_422(body.name)
    collection = collection_create(db, body.name, body.description)
    db.commit()
    db.refresh(collection)
    return collection_json(collection)


@artifacts_router.patch("/collections/{collection_id}")
async def update_collection(
    collection_id: str, body: CollectionRequest, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    collection = _load_collection(db, collection_id)
    collection.name = _named_or_422(body.name)
    collection.description = (body.description or "").strip() or None
    collection.updated_at = utcnow()
    db.commit()
    db.refresh(collection)
    return collection_json(collection)


@artifacts_router.delete("/collections/{collection_id}")
async def delete_collection_route(
    collection_id: str, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Remove a collection. **The artifacts are untouched** -- the response
    says how many memberships went with it, never how many files."""
    deleted, members = collection_delete(db, collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no collection with id {collection_id!r}")
    db.commit()
    return {"deleted": True, "detached": members}


@artifacts_router.get("/collections/{collection_id}/artifacts")
async def list_collection_artifacts(
    collection_id: str, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """The members, in the curated order."""
    collection = _load_collection(db, collection_id)
    ordered = collection_member_ids(db, collection.id)
    rows = {
        artifact.id: artifact
        for artifact in db.execute(select(Artifact).where(Artifact.id.in_(ordered))).scalars()
    }
    artifacts = [_artifact_json(rows[a]) for a in ordered if a in rows]
    return {
        "collection": collection_json(collection, count=len(artifacts)),
        "artifacts": _with_tags(db, artifacts),
    }


@artifacts_router.put("/collections/{collection_id}/artifacts/{artifact_id}")
async def add_to_collection(
    collection_id: str, artifact_id: str, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Append one artifact. Idempotent, and an existing member does NOT move:
    the order is the operator's decision, and re-adding should not send a row to
    the bottom of a list they arranged."""
    collection = _load_collection(db, collection_id)
    _load_artifact(db, artifact_id)
    added = collection_add(db, collection, [artifact_id])
    db.commit()
    return {"added": added}


@artifacts_router.delete("/collections/{collection_id}/artifacts/{artifact_id}")
async def remove_from_collection(
    collection_id: str, artifact_id: str, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Drop one member. Idempotent. **Never deletes the artifact.**"""
    collection = _load_collection(db, collection_id)
    removed = collection_remove(db, collection, artifact_id)
    db.commit()
    return {"removed": removed}


@artifacts_router.post("/collections/{collection_id}/artifacts")
async def add_many_to_collection(
    collection_id: str, body: ArtifactIdsRequest, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Append a selection, in the order it was given."""
    collection = _load_collection(db, collection_id)
    added = collection_add(db, collection, body.ids)
    db.commit()
    return {"added": added}


@artifacts_router.put("/collections/{collection_id}/order")
async def reorder_collection(
    collection_id: str, body: CollectionOrderRequest, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Rewrite the order. **409 unless `ids` is exactly the membership.**"""
    collection = _load_collection(db, collection_id)
    if not collection_reorder(db, collection, body.ids):
        raise HTTPException(
            status_code=409,
            detail=(
                "`ids` must be exactly this collection's members: a partial order "
                "would leave the rows it does not name in an ambiguous position"
            ),
        )
    db.commit()
    return {"reordered": True}


@artifacts_router.get("/artifacts/kinds")
async def list_artifact_kinds() -> dict:
    """The filter vocabulary, so the app's chips come from the server rather
    than a copy that can drift out of step with the classifier."""
    return {"kinds": list(ARTIFACT_KINDS)}


@artifacts_router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    project: str | None = Query(default=None),
    unfiled: bool = Query(default=False),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """One row, with everything a detail screen shows."""
    artifact = _load_artifact(db, artifact_id)
    row = _artifact_json_full(db, artifact)
    scope = _write_scope(db, artifact, project, unfiled)
    row["bookmarks"] = bookmark_scopes_of(db, artifact.id)
    return _with_bookmarks(db, [row], scope=scope)[0]


def _write_scope(
    db: OrmSession, artifact: Artifact, project: str | None, unfiled: bool = False
) -> str:
    """Where a star written right now belongs."""
    if unfiled:
        return UNFILED_SCOPE
    if project is not None:
        if db.get(Project, project) is None:
            raise HTTPException(status_code=404, detail=f"no project with id {project!r}")
        return project
    return artifact.project_id or UNFILED_SCOPE


@artifacts_router.put("/artifacts/{artifact_id}/bookmark")
async def bookmark_artifact(
    artifact_id: str,
    project: str | None = Query(default=None),
    unfiled: bool = Query(default=False),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """Star an artifact **in one project**."""
    artifact = _load_artifact(db, artifact_id)
    scope = _write_scope(db, artifact, project, unfiled)
    bookmark_set(db, artifact.id, scope, starred=True)
    db.commit()
    return _with_bookmarks(db, [_artifact_json(artifact)], scope=scope)[0]


@artifacts_router.delete("/artifacts/{artifact_id}/bookmark")
async def unbookmark_artifact(
    artifact_id: str,
    project: str | None = Query(default=None),
    unfiled: bool = Query(default=False),
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """Un-star an artifact in one scope. **Only that scope** -- a star made in
    another project is that project's, and this must not reach into it.
    """
    artifact = _load_artifact(db, artifact_id)
    scope = _write_scope(db, artifact, project, unfiled)
    bookmark_set(db, artifact.id, scope, starred=False)
    db.commit()
    return _with_bookmarks(db, [_artifact_json(artifact)], scope=scope)[0]


@artifacts_router.put("/artifacts/{artifact_id}/archive")
async def archive_artifact(artifact_id: str, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Hide an artifact from the library's default listings."""
    artifact = _load_artifact(db, artifact_id)
    artifact.archived_at = utcnow()
    db.commit()
    db.refresh(artifact)
    return _artifact_json(artifact)


@artifacts_router.delete("/artifacts/{artifact_id}/archive")
async def unarchive_artifact(artifact_id: str, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Put an archived artifact back. Idempotent, and a 200 for a row that was
    never archived: the caller's intent ("this should be visible") is already
    true, exactly as `DELETE .../bookmark` treats an unbookmarked row."""
    artifact = _load_artifact(db, artifact_id)
    artifact.archived_at = None
    db.commit()
    db.refresh(artifact)
    return _artifact_json(artifact)


class ArtifactIdsRequest(BaseModel):
    """Body for the bulk archive routes. Closed schema, same rule as every
    other inbound body here."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=MAX_BULK_IDS)


def _bulk_set_archived(db: OrmSession, ids: list[str], *, archived: bool) -> int:
    """Set (or clear) `archived_at` on many rows in one transaction."""
    rows = list(db.execute(select(Artifact).where(Artifact.id.in_(ids))).scalars())
    stamp = utcnow() if archived else None
    changed = 0
    for row in rows:
        if (row.archived_at is None) == archived:
            changed += 1
        row.archived_at = stamp
    db.commit()
    return changed


@artifacts_router.post("/artifacts/archive")
async def archive_artifacts(
    body: ArtifactIdsRequest, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Archive up to `MAX_BULK_IDS` artifacts in one request."""
    return {"archived": _bulk_set_archived(db, body.ids, archived=True)}


@artifacts_router.post("/artifacts/unarchive")
async def unarchive_artifacts(
    body: ArtifactIdsRequest, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    return {"unarchived": _bulk_set_archived(db, body.ids, archived=False)}


@artifacts_router.post("/artifacts/archive-ignored")
async def archive_ignored_artifacts(
    request: Request, db: OrmSession = Depends(_artifacts_db)
) -> dict:
    """Archive every visible row the current ignore rules would refuse today."""
    denylist = getattr(request.app.state, "artifact_ignore_rules", None)
    if denylist is None:
        ingestor = getattr(request.app.state, "artifact_ingestor", None)
        denylist = getattr(ingestor, "_denylist", None)
    if denylist is None:
        raise HTTPException(
            status_code=503,
            detail="artifact ingest is not running, so its ignore rules cannot be read",
        )

    rows = list(
        db.execute(
            select(Artifact).where(
                Artifact.archived_at.is_(None), Artifact.source_path.is_not(None)
            )
        ).scalars()
    )
    stamp = utcnow()
    archived = 0
    for row in rows:
        if row.source_path and denylist.denies_file(row.source_path):
            row.archived_at = stamp
            archived += 1
    db.commit()
    logger.info("archive-ignored swept %d of %d visible rows", archived, len(rows))
    return {"archived": archived, "scanned": len(rows)}


@artifacts_router.get("/artifacts/{artifact_id}/content")
async def artifact_content(
    artifact_id: str,
    request: Request,
    db: OrmSession = Depends(_artifacts_db),
):
    """Stream one artifact's stored bytes, with standard Range support."""
    artifact = _load_artifact(db, artifact_id)
    if artifact.status != STATUS_AVAILABLE or not artifact.storage_key:
        raise HTTPException(
            status_code=409,
            detail=(
                f"artifact {artifact_id!r} is {artifact.status}: its bytes were "
                "never fetched from the sandbox (see the row's metadata.ingest_error; "
                "POST /api/sandbox/promote can retry the fetch)"
            ),
        )
    store: ArtifactStore = request.app.state.artifact_store
    try:
        path = store.path_for_key(artifact.storage_key)
    except ValueError as exc:
        logger.error("artifact %s has a corrupt storage key: %s", artifact_id, exc)
        raise HTTPException(
            status_code=500, detail=f"artifact {artifact_id!r} has a corrupt storage key"
        ) from exc
    if not path.is_file():
        logger.error(
            "artifact %s is marked available but its stored file %s is missing (store/DB drift)",
            artifact_id,
            path,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"artifact {artifact_id!r} is marked available but its stored "
                "bytes are missing from the artifact store; re-promote its "
                "source_path to re-fetch"
            ),
        )
    size = path.stat().st_size
    filename = (artifact.title or artifact.id).replace('"', "")
    base_headers = {
        "accept-ranges": "bytes",
        "content-disposition": f'inline; filename="{filename}"',
    }
    if artifact.checksum:
        base_headers["etag"] = f'"{artifact.checksum}"'

    span = parse_range_header(request.headers.get("range"), size)
    if span is UNSATISFIABLE:
        raise HTTPException(
            status_code=416,
            detail="requested range not satisfiable",
            headers={"content-range": f"bytes */{size}"},
        )
    if span is None:
        return StreamingResponse(
            _file_slice(path, 0, size - 1) if size else iter(()),
            media_type=artifact.mime_type,
            headers={**base_headers, "content-length": str(size)},
        )
    start, end = span
    return StreamingResponse(
        _file_slice(path, start, end),
        status_code=206,
        media_type=artifact.mime_type,
        headers={
            **base_headers,
            "content-length": str(end - start + 1),
            "content-range": f"bytes {start}-{end}/{size}",
        },
    )


@artifacts_router.get("/artifacts/{artifact_id}/preview")
async def artifact_preview(
    artifact_id: str,
    request: Request,
    db: OrmSession = Depends(_artifacts_db),
):
    """v1 preview (P3-2): a small image answers with its own bytes; everything
    else is 204 and the client renders a type glyph. Real thumbnailing (and a
    PDF first page) are deliberately deferred."""
    artifact = _load_artifact(db, artifact_id)
    if (
        artifact.status != STATUS_AVAILABLE
        or not artifact.storage_key
        or not artifact.mime_type.startswith("image/")
        or artifact.size_bytes is None
        or artifact.size_bytes > PREVIEW_MAX_BYTES
    ):
        return Response(status_code=204)
    store: ArtifactStore = request.app.state.artifact_store
    try:
        path = store.path_for_key(artifact.storage_key)
    except ValueError:
        return Response(status_code=204)
    if not path.is_file():
        return Response(status_code=204)
    return Response(content=path.read_bytes(), media_type=artifact.mime_type)


class PromoteRequest(BaseModel):
    """Body for `POST /api/sandbox/promote`. Closed schema, same rule as every
    other inbound body (`TurnSubmission`): nothing but these keys crosses."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    project_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    collection: str | None = Field(default=None, max_length=MAX_COLLECTION_NAME_CHARS)


@artifacts_router.post("/sandbox/promote")
async def promote_sandbox_file(
    body: PromoteRequest,
    request: Request,
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """Promote one sandbox file into the durable library, on demand."""
    if body.project_id is not None and db.get(Project, body.project_id) is None:
        raise HTTPException(status_code=404, detail=f"no project with id {body.project_id!r}")
    ingestor: ArtifactIngestor = request.app.state.artifact_ingestor
    outcome = await ingestor.ingest(body.path, project_id=body.project_id)
    if not outcome.ok:
        status = outcome.upstream_status if outcome.upstream_status in (403, 404) else 502
        raise HTTPException(
            status_code=status,
            detail=(
                f"could not fetch {body.path!r} from the sandbox ({outcome.error}); "
                f"recorded artifact {outcome.artifact.get('id')!r} as unavailable"
            ),
        )
    artifact_id = outcome.artifact.get("id") if isinstance(outcome.artifact, dict) else None
    if artifact_id and (body.tags or body.collection):
        for name in _normalized_list_or_422(body.tags):
            tag_attach(db, "artifact", artifact_id, name)
        if body.collection:
            collection = collection_create(db, _named_or_422(body.collection), None)
            collection_add(db, collection, [artifact_id])
        db.commit()
        artifact = db.get(Artifact, artifact_id)
        if artifact is not None:
            return {
                "artifact": _artifact_json_full(db, artifact),
                "deduplicated": outcome.deduplicated,
            }
    return {"artifact": outcome.artifact, "deduplicated": outcome.deduplicated}
