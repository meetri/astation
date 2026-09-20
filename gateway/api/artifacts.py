"""Tier-2 durable artifact library: the artifact REST API (P3-1 / P3-2).

The store (`ArtifactStore`, the PINNED on-disk layout) is
`domain/artifact_store.py`; the three ingesting subscribers
(`ArtifactIngestor`, `SandboxDiffIngestor` + its B-61 denylist,
`MediaTagIngestor`) are `domain/artifact_ingest.py` -- CLEANUP_PLAN step 3.4.
Both are re-exported here for existing importers. This module keeps the
routes, the HTTP Range helpers and the DB dependency.

Tier 1 (`api/sandbox.py`) is a live, ephemeral window onto the Hermes
sandbox; this module is the durable half: bytes are pulled off the sandbox,
checksummed, stored under the gateway's own `RESEARCH_GATEWAY_ARTIFACT_ROOT`,
and recorded as `artifacts` rows with provenance -- so an audio file or PDF
the agent produced is still playable after the sandbox is cleaned.

## REST (P3-2, §11.1)

* `GET  /api/projects/{id}/artifacts` -- project-scoped listing (404 for an
  unknown project; one project's artifacts are never visible through
  another's listing).
* `GET  /api/artifacts` -- global listing; `?unfiled=true` restricts to
  `project_id IS NULL`; `?session=<stored id>` restricts to the artifacts
  produced by that Hermes session's runs (see `list_artifacts` for the join
  and for what it can never contain).
* `GET  /api/artifacts/{id}` -- one row.
* `GET  /api/artifacts/{id}/content` -- bytes streamed **from the gateway's
  own store** (the content is local now, so standard single-range HTTP Range
  serving is implemented here -- 206/`Content-Range`, 416 with
  `bytes */<size>` when unsatisfiable; a malformed or multi-range header is
  ignored per RFC 9110 and the full body served). No arbitrary filesystem
  paths: the only thing addressable is an artifact id, and the row's
  `storage_key` is re-confined to the artifact root before any open (§14).
* `GET  /api/artifacts/{id}/preview` -- v1 per the OPTIMIST note: a small
  image answers with its own bytes; everything else is 204 and the client
  renders a type glyph. Real thumbnailing is deferred.
* `POST /api/sandbox/promote` -- the manual-promotion entrypoint: same
  validate+fetch+store+row path as auto-ingestion, on demand.

All mounted on the authenticated `/api` router in `api.main`, so Basic auth
is inherited by construction. 503 with the migration hint when the P3-1.0
schema (`artifacts.status`) has not been applied, same pattern as
`api/runs.py::_runs_db`.
"""

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

#: Streaming chunk size for serving stored content.
_SERVE_CHUNK_BYTES = 64 * 1024

#: Preview policy (P3-2 v1): an image at/below this size answers the preview
#: route with its own bytes; everything else is 204 and the client renders a
#: type glyph. Real thumbnailing is deferred deliberately.
PREVIEW_MAX_BYTES = 512 * 1024

# ---------------------------------------------------------------------------
# DB dependency (503 until the P3-1.0 migration has run)
# ---------------------------------------------------------------------------


#: A DB session, with "you never ran the P3-1.0 migration" as a 503
#: (`domain.db.schema_checked_db`). The `artifacts` table has existed since
#: the initial schema, so `has_table` proves nothing; the check is for the
#: columns this phase added (`status`, `source_path`).
_artifacts_db = schema_checked_db(
    "artifacts_schema_verified",
    lambda engine: columns_present(engine, "artifacts", "status", "source_path"),
)


# ---------------------------------------------------------------------------
# HTTP Range serving (P3-2: the content is local, so the gateway implements
# standard single-range serving itself -- nothing to pass through anymore)
# ---------------------------------------------------------------------------


class _Unsatisfiable:
    """Sentinel: the Range header was valid `bytes=` syntax but names nothing
    inside the file -> 416."""


UNSATISFIABLE = _Unsatisfiable()


def parse_range_header(header: str | None, size: int):
    """`None` (serve everything), `(start, end)` inclusive (206), or
    `UNSATISFIABLE` (416).

    RFC 9110 semantics, deliberately minimal: only single `bytes=` ranges are
    honored (`a-b`, `a-`, `-n`). A malformed header, a non-bytes unit, or a
    multi-range list is *ignored* -- the RFC allows a server to serve 200 for
    anything it does not care to satisfy, and AVFoundation/PDFKit send simple
    single ranges.
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes=") :].strip()
    if "," in spec or not spec:
        return None  # multi-range / empty: ignored, serve 200
    start_s, sep, end_s = spec.partition("-")
    if not sep:
        return None
    start_s = start_s.strip()
    end_s = end_s.strip()
    try:
        if start_s == "":
            # Suffix range: last n bytes.
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
        return None  # malformed: ignored, serve 200


def _file_slice(path: Path, start: int, end: int) -> Iterator[bytes]:
    """Yield `[start, end]` (inclusive) of `path` in bounded chunks.

    A plain sync generator: Starlette iterates it in a worker thread, and the
    reads are chunk-sized against a local file.
    """
    remaining = end - start + 1
    with open(path, "rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(_SERVE_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _collapse_to_latest(rows) -> list[dict]:
    """B-184: one row per `source_path`, the newest, plus how many rows it
    stands for as `versions`.

    `rows` must arrive newest-first (every listing orders that way). The
    P3-1 dedup rule is `(source_path, checksum)`, so every time the agent
    rewrites a file the library gains a row: measured live 2026-09-13, one
    project's listing was 500 rows for 242 distinct paths (`STATE.md` alone
    35 times). The owner reads that as duplicates, and for a library it is
    -- a file, not each of its saves, is the unit. A row with no
    `source_path` (nothing to collapse on) passes through untouched. The
    newest row wins even when it is `unavailable`: that is the file's
    current state, and hiding it behind an older fetch would be the silent
    substitution P3-1 forbids.
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


def _filter_by_kind(rows: list[dict], kind: str | None) -> list[dict]:
    """Keep only rows of one kind.

    Applied to the SERIALIZED rows rather than as SQL, because `kind` is
    derived from mime type and extension (`domain/artifact_kinds.py`) and is
    deliberately not a column — so improving the classification applies to
    every existing row without a migration. The cost is that `limit` counts
    rows before filtering, which is why the routes fetch the cap and then
    filter rather than the other way round.
    """
    if not kind:
        return rows
    return [row for row in rows if row.get("kind") == kind]


def _validated_kind(kind: str | None) -> str | None:
    """422 on an unknown kind rather than silently returning nothing.

    An empty list for a typo'd filter reads as "there are none of those",
    which is a different and wrong answer.
    """
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
    """One scope's starred artifacts, newest star first.

    **Scoped since 2026-09-19.** A star used to live on the artifact row, so
    there was one shelf and every project showed it -- the owner opened a
    brand-new project onto forty files it had nothing to do with. A star is
    now a decision taken inside a scope (`domain/bookmark_store.py`), and this
    reads the scope it was asked for; `None` means "starred anywhere".

    Returned as its own field rather than mixed into a project's list, so the
    route's "only this project's artifacts" guarantee still holds for
    `artifacts` -- a star made HERE may sit on a file that lives elsewhere.
    """
    ordered = bookmark_scoped_ids(db, scope)
    if not ordered:
        return []
    by_id = {
        artifact.id: artifact
        for artifact in db.execute(
            select(Artifact)
            # An archived artifact is one the operator put away; leaving it on a
            # shelf would make "archive" mean nothing on exactly the rows they
            # cared enough about to star. The star itself survives, so
            # un-archiving brings it back.
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


# ---------------------------------------------------------------------------
# Browsing: prefix, search, archive and keyset paging
#
# Measured on the deploy host 2026-09-19: 5,226 rows, 2,443 distinct files,
# 4,955 of them in ONE project, against a listing capped at 500 rows with no
# paging and a type filter that ran on the phone over whatever page it had.
# Roughly three quarters of the operator's largest project could not be reached
# from the app at all. These four parameters are what make the corpus
# navigable; the folders route below is what gives it shape.
# ---------------------------------------------------------------------------

#: The three answers to "should archived rows be in this listing?". `all`
#: exists because "show me everything" and "show me only what I put away" are
#: different questions and a boolean can only ask one of them.
ArchivedFilter = Literal["false", "true", "all"]

#: Ceiling on a bulk archive/unarchive request. The app's Select mode works on
#: a page it has loaded, so a route with no ceiling invites "select all 5,226"
#: from a client that has seen 500 of them.
MAX_BULK_IDS = 200

#: Minimum length for `?q=`. One character matches most of the corpus and
#: costs a full scan to say so.
_MIN_QUERY_CHARS = 2
_MAX_QUERY_CHARS = 80


def _validated_prefix(prefix: str | None) -> str | None:
    """A sandbox directory to scope a listing to, or `None`.

    Validated through the same `validate_sandbox_path` every read route uses,
    so a listing cannot be pointed outside the sandbox root or walked out of
    it with `..` -- a filter is not a read, but a filter that accepts a path
    the rest of the system would refuse is a seam worth not having.
    """
    if prefix is None:
        return None
    cleaned = prefix.strip().rstrip("/")
    if not cleaned:
        return None
    # Raises the same 422/403 every read route raises for the same offense.
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
    """`(normalized names, narrowed query)`.

    **AND, not OR.** Tags narrow: someone filtering by `l328` and `results` is
    asking for the intersection, and OR would hand back more rows the more
    precisely they asked. An empty intersection narrows to nothing rather than
    being ignored, because silently dropping a filter is how a listing claims
    to have answered a question it did not.
    """
    if not tags:
        return [], query
    wanted = _normalized_list_or_422(tags)
    matching = tag_owner_ids_with_all(db, "artifact", wanted)
    return wanted, query.where(Artifact.id.in_(matching))


def _browse_filters(query, *, prefix: str | None, q: str | None, archived: ArchivedFilter):
    """Apply the scope-independent filters both listings share."""
    if prefix is not None:
        # Component-wise containment, not a bare prefix match: `startswith`
        # would put `/opt/data-old/x` inside `/opt/data` (`is_within`).
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
    """`<created_at iso>|<id>` -> the keyset position, or `None`.

    A malformed cursor is a 422 rather than a silent first page: a client that
    pages with a broken cursor would otherwise loop over page one forever and
    look like a server that never advances.
    """
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
    """Keyset, never offset.

    This listing is ordered newest-first and grows at the FRONT -- the agent
    files artifacts while the owner is scrolling -- so an offset page would
    skip rows it had already passed and repeat others. A keyset position is
    stable against inserts anywhere.
    """
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
    """One page of rows, and the cursor for the next one (`None` when last).

    With `latest=true` the collapse runs over everything after the cursor
    before the page is cut, because the version COUNT on a row is only right
    if every version of that path was seen -- the same reason `limit` has
    always applied after the collapse (B-184).
    """
    if latest:
        # The cursor is applied AFTER the collapse, not in SQL. A collapsed
        # row stands for every save of its path, and older saves of a path
        # already shown sort after the cursor -- filtering in SQL would let
        # them collapse again on the next page and hand the reader the same
        # file twice. Collapsing reads the whole scope either way, so
        # this costs nothing that was not already being paid.
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
    """One row with everything a DETAIL screen shows: tags and collections.

    The listings deliberately do not carry collections: a page of 200 rows
    would need a second join for something only the detail screen renders.
    """
    row = _artifact_json_with_tags(db, artifact)
    row["collections"] = collections_containing(db, artifact.id)
    return row


def _artifact_json_with_tags(db: OrmSession, artifact: Artifact) -> dict:
    """One row plus its tags.

    Separate from `_artifact_json` (which knows nothing about the database)
    rather than folding a query into it: that function renders rows in a loop
    over a whole page, and a per-row query there is how a 200-row listing
    becomes 200 queries.
    """
    row = _artifact_json(artifact)
    row["tags"] = tag_names_of(db, "artifact", artifact.id)
    return row


def _with_tags(db: OrmSession, rows: list[dict]) -> list[dict]:
    """Attach tags to a whole page in ONE query.

    `tags` is always present, `[]` when there are none, so a client never has
    to treat absence as a special case (B-34).
    """
    if not rows:
        return rows
    by_owner = tag_names_for(db, "artifact", [row["id"] for row in rows])
    for row in rows:
        row["tags"] = by_owner.get(row["id"], [])
    return rows


def _star_scope(db: OrmSession, *, project: str | None, unfiled: bool) -> str | None:
    """Which scope's stars a request is asking about.

    A named project (404 if it does not exist -- a typo must not silently
    answer "nothing is starred"), the unfiled scope for `?unfiled=true`, or
    `None` for "anywhere", which is what a request that names no scope at all
    is asking.
    """
    if project is not None:
        if db.get(Project, project) is None:
            raise HTTPException(status_code=404, detail=f"no project with id {project!r}")
        return project
    return UNFILED_SCOPE if unfiled else None


def _with_bookmarks(db: OrmSession, rows: list[dict], *, scope: str | None) -> list[dict]:
    """Say whether each row is starred IN THIS SCOPE, in one query.

    The pair to `_with_tags`, and for the same reason: a 200-row page that
    asked per row would be 200 queries for a glyph.

    `scope` is the scope the request named -- a project id, the unfiled
    sentinel, or `None` for "starred anywhere". The rule the whole feature
    rests on: **the scope in the request is the scope the stars are read in**,
    so the star a row shows is the one the owner would be toggling if they
    tapped it here.
    """
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
    """This project's artifacts, newest first (P3-2, project-scoped).

    404 for an unknown project -- and the filter is strict equality on
    `project_id`, so another project's (or an unfiled) artifact can never
    appear here (§14 access control at the only boundary this single-user
    gateway has).

    `?latest=true` (B-184): one row per `source_path`, the newest, each
    carrying `versions` = how many saves it stands for; `limit` then counts
    files, not saves. See `_collapse_to_latest`.
    """
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
        # THIS project's shelf. A star is per project since 2026-09-19, so the
        # scope is the project in the path; it stays a SEPARATE field because
        # a star made here may sit on a file that lives elsewhere, and this
        # route's "only this project's artifacts" guarantee holds for
        # `artifacts`.
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
    """Global artifact listing, newest first, with two mutually exclusive filters.

    `?latest=true` (B-184) collapses to the newest row per `source_path`
    with a `versions` count, and `limit` then counts files -- the same
    rule as the project listing (`_collapse_to_latest`).

    * `?unfiled=true` -- the rows with no project (P3-2's global/unfiled
      listing; NULL means unfiled, same convention as runs, never a synthetic
      project).
    * `?session=<stored id>` -- **"the files from this chat"** (2026-09-01
      review, ask #5). Joins `artifacts.producing_run_id -> runs.id ->
      runs.runtime_session_id`, so it answers for filed and unfiled sessions
      alike.

    Passing both is a 422: they answer different questions ("which project"
    vs "which conversation") and silently intersecting them would quietly
    return an empty list for a perfectly good session.

    **The filter key is `runtime_session_id`, never `session_id`.**
    `runs.session_id` is the *workspace* `sess_...` id, which is NULL for
    every unfiled session (43 of 151 measured), so filtering on it would
    return nothing for every spike/unfiled conversation. `runtime_session_id`
    is the Hermes STORED id -- the one the app holds for every session --
    which is exactly what `GET /api/runs?session=` already takes.

    **What this listing can never contain, by construction:** an artifact
    whose `producing_run_id` is NULL. That is every manual promotion
    (`POST /api/sandbox/promote` files by project, not by run) and every row
    whose producing run could not be resolved. Measured on the live library:
    386 artifacts, 385 of them join (99.7%). So the response says
    `run_attributed_only: true` for a session query rather than implying the
    set is complete -- a client must not present it as "everything this
    session produced".

    An unknown/typo'd session id is an empty list, not a 404: this gateway's
    artifact tables are not a session registry, and "no artifacts recorded
    for that session" is the honest answer for both cases.
    """
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
        query = query.join(Run, Artifact.producing_run_id == Run.id).where(
            Run.runtime_session_id == stored_session_id
        )
    if unfiled:
        query = query.where(Artifact.project_id.is_(None))
    # **The scope in the request is the scope the stars are read in.**
    # `?project=` names it explicitly (a chat's library passes the project it
    # was opened from); `?unfiled=true` means the unfiled scope; naming
    # neither asks about stars anywhere.
    star_scope = _star_scope(db, project=project, unfiled=unfiled)
    if bookmarked:
        starred = bookmark_scoped_ids(db, star_scope)
        # `in_([])` is valid SQL and matches nothing, which is the right
        # answer for a scope with no stars yet -- a brand-new project.
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
        # True only for a session-scoped query: see the docstring -- artifacts
        # with no producing run (manual promotions) can never appear there.
        "run_attributed_only": stored_session_id is not None,
        "unfiled_only": unfiled,
        "latest_only": latest,
        "kind": wanted,
        "bookmarked_only": bookmarked,
        #: Which scope's stars these rows report. `null` means "anywhere".
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
    """`(folders, files directly in prefix)` for one level of the tree.

    `rows` is `(source_path, created_at)` for every visible artifact in scope,
    already deduplicated by path -- folder counts are DISTINCT FILES, not
    rows, because the library's default view is one row per file (B-184) and a
    folder claiming 612 when it shows 242 is a folder nobody trusts.

    **A chain of single-child directories is collapsed into one entry.** Every
    path on the owner's instance starts `/opt/data/...`, so an uncollapsed
    top level is a single "opt" row hiding everything behind two taps that ask
    nothing. `a/b/c` with no content of its own in `a` or `b` shows as
    `a/b/c`, which is what the owner would have named the folder anyway.
    """
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
        # Remember whether this subtree has content of its own at this level,
        # which is what decides a collapse below.
        bucket["children"].add(components[depth + 1] if len(components) > depth + 1 else None)

    base = (prefix or "").rstrip("/")
    folders: list[dict] = []
    for name, bucket in buckets.items():
        path = f"{base}/{name}"
        children = bucket["children"]
        # Collapse while this level holds exactly one child directory and no
        # file of its own.
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
    """The folder level under `prefix` for one project's artifacts.

    The agent already organizes what it writes into directories that mean
    something; until this route existed the library flattened all of it into
    one newest-first list. Folders are derived from `source_path` and never
    stored, so a file that moves in the sandbox moves here on the next ingest.
    """
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
    """The folder level under `prefix` for a session, the unfiled rows, or all.

    Scoped exactly like `GET /api/artifacts`, so the app can switch to the
    Folders view without changing which artifacts it is looking at.
    """
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


# ---------------------------------------------------------------------------
# Tags (Phase B, `docs/ARTIFACT_ORGANIZATION_PLAN.md` §5.2)
#
# One vocabulary shared by artifacts and projects. Every write normalizes the
# name through `domain/tags.py`, because the whole value of a tag vocabulary
# is that `Results` and `results` are one tag -- two rows that render
# identically in a chip and filter to different sets is the failure this is
# built to prevent.
# ---------------------------------------------------------------------------


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
    """Normalize, turning the rule that was broken into the 422 detail.

    The message names the rule rather than saying "invalid tag", because the
    owner is typing into a field and needs to know what to type instead.
    """
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
    """The vocabulary, most used first, ties alphabetical.

    Counts are derived from the joins rather than stored, so they cannot drift
    from the rows they describe.

    `?project=` (or `?unfiled=true`) narrows to the tags that scope's
    artifacts actually carry, with that scope's counts -- what the filter menu
    asks for, since a brand-new project offering forty tags that match nothing
    in it is a menu of dead ends. The vocabulary itself is still shared; this
    only changes which part of it a scope is shown. 404 for an unknown
    project, never an empty list: a typo must not read as "no tags here".
    """
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
    """Rename a tag, **merging** when the new name already exists.

    Merging rather than refusing: renaming `fig` to `figure` when `figure`
    exists means "these are the same thing", and a 409 would leave the owner
    to redo it by hand on every artifact.
    """
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
    entry the owner typed, and deleting it because its last artifact lost it
    would make the tag list flicker with their own work."""
    artifact = _load_artifact(db, artifact_id)
    tag_detach(db, "artifact", artifact.id, _normalized_or_422(name))
    db.commit()
    db.refresh(artifact)
    return _artifact_json_with_tags(db, artifact)


@artifacts_router.post("/artifacts/tags")
async def tag_artifacts(body: TagBulkRequest, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Apply tags across a selection, in one transaction.

    Unknown ids are skipped rather than failing the batch, for the same reason
    bulk archive skips them: the selection came from a page fetched a moment
    ago, and one row deleted underneath it must not lose the other 199.
    """
    add = _normalized_list_or_422(body.add)
    remove = _normalized_list_or_422(body.remove)
    known = [
        artifact_id
        for (artifact_id,) in db.execute(select(Artifact.id).where(Artifact.id.in_(body.ids)))
    ]
    changed = tag_apply_bulk(db, "artifact", known, add=add, remove=remove)
    db.commit()
    return {"updated": changed}


# ---------------------------------------------------------------------------
# Collections (Phase C, §5.3)
#
# A collection is a named, ORDERED, cross-project set the operator curates.
# Ordered is the whole difference from a tag: "figures for the L328 paper" has
# a figure 1 and a figure 2.
# ---------------------------------------------------------------------------


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
    """The members, in the curated order.

    Not paged: a collection is something a person assembled by hand, so it is
    tens of rows and not thousands. If one ever outgrows a screenful, paging
    it would also have to answer what a cursor means under a reorder, which is
    a question worth not having.
    """
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
    the order is the owner's decision, and re-adding should not send a row to
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
    """Rewrite the order. **409 unless `ids` is exactly the membership.**

    A partial reorder leaves the positions it does not mention ambiguous, and
    the two readings -- "leave them" and "push them to the end" -- give
    different lists from the same request. Refusing is the only answer that
    cannot silently scramble a list someone arranged by hand.
    """
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
    """One row, with everything a detail screen shows.

    `?project=` names the scope whose star `bookmarked` reports -- the screen
    was opened from somewhere, and the star it shows has to be the one a tap
    would toggle. Without it the scope is the artifact's own project.
    `bookmarks` lists every scope holding a star, so the screen can say "also
    starred in two other projects" rather than quietly showing one of them.
    """
    artifact = _load_artifact(db, artifact_id)
    row = _artifact_json_full(db, artifact)
    scope = _write_scope(db, artifact, project, unfiled)
    row["bookmarks"] = bookmark_scopes_of(db, artifact.id)
    return _with_bookmarks(db, [row], scope=scope)[0]


def _write_scope(
    db: OrmSession, artifact: Artifact, project: str | None, unfiled: bool = False
) -> str:
    """Where a star written right now belongs.

    The scope the caller names -- `?project=` for a project, `?unfiled=true`
    for the unfiled view -- else **the artifact's own project**, which is
    where someone starring from a project listing is standing and the only
    default that cannot put a star somewhere the owner was not looking.

    `unfiled` is explicit rather than implied by an absent project because the
    two are different asks: a client browsing the unfiled view is standing in
    the unfiled scope even when the row it taps belongs to a project.
    """
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
    """Star an artifact **in one project**.

    Scoped since 2026-09-19 (owner: *"I just created a new project and I see
    bookmarked artifacts from other projects"*). `?project=` names the scope;
    without it the star lands in the artifact's own project, or in the
    unfiled scope for an artifact that has none.

    The scope does NOT have to be the artifact's project: starring a shared
    report while standing in the project that cites it is a legitimate thing
    to want, and the same rule already governs the pinned deliverable.

    Idempotent, but re-starring DOES move it to the top of that scope's
    shelf: the timestamp records when the owner last said this matters.
    """
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

    A no-op on an artifact that was not starred here, not a 404: the caller's
    intent ("this should not be starred here") is already true.
    """
    artifact = _load_artifact(db, artifact_id)
    scope = _write_scope(db, artifact, project, unfiled)
    bookmark_set(db, artifact.id, scope, starred=False)
    db.commit()
    return _with_bookmarks(db, [_artifact_json(artifact)], scope=scope)[0]


@artifacts_router.put("/artifacts/{artifact_id}/archive")
async def archive_artifact(artifact_id: str, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Hide an artifact from the library's default listings.

    **Not a delete.** The bytes stay in the store, the provenance stays on the
    row, and every transcript chip that opens this artifact keeps working --
    a link the owner followed yesterday must not break because they tidied up
    today. Archived rows come back under `?archived=true` (or `all`), and are
    kept off the bookmark shelf and out of folder counts meanwhile.

    Idempotent, but re-archiving refreshes the timestamp: it orders "recently
    archived", the same convention a star's timestamp follows.
    """
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
    """Set (or clear) `archived_at` on many rows in one transaction.

    Unknown ids are skipped rather than failing the batch: bulk work runs
    against a page the client fetched a moment ago, and one row deleted
    underneath it must not lose the other 199 the owner selected. The count
    returned is what actually changed, so a caller can tell.
    """
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
    """Archive up to `MAX_BULK_IDS` artifacts in one request.

    The cap is the point: the app's Select mode operates on a loaded page, and
    a bulk route with no ceiling invites "select all 5,226" from a client that
    has only ever seen 500 of them.
    """
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
    """Archive every visible row the current ignore rules would refuse today.

    The rules only gate NEW ingests; the library already holds what was filed
    before they existed. Measured on the owner's instance 2026-09-19: npm
    logs, curator backup blobs and profile internals, all operational trees
    nobody asked to keep.

    Archives rather than deletes, like everything else here, so a rule that
    turns out to be too broad costs one unarchive and not a lost file.
    Idempotent: a second call finds nothing left to do.
    """
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
    """Stream one artifact's stored bytes, with standard Range support.

    Serves from the gateway's own store -- Hermes is not consulted, so the
    library keeps working after the sandbox is cleaned or Hermes is down.
    409 for an `unavailable` row (the bytes were never fetched; the row's
    `metadata.ingest_error` says why -- re-promote to retry).
    """
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


# ---------------------------------------------------------------------------
# Manual promotion (P3-1's PROMOTION entrypoint; P3-4a's "Save to project")
# ---------------------------------------------------------------------------


class PromoteRequest(BaseModel):
    """Body for `POST /api/sandbox/promote`. Closed schema, same rule as every
    other inbound body (`TurnSubmission`): nothing but these keys crosses."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    project_id: str | None = None
    #: C3: file it at the same time. The producer knows what a file IS at the
    #: moment it writes it; asking a person to classify it later, from a list
    #: of 2,443, is the expensive way to learn the same fact.
    tags: list[str] = Field(default_factory=list)
    collection: str | None = Field(default=None, max_length=MAX_COLLECTION_NAME_CHARS)


@artifacts_router.post("/sandbox/promote")
async def promote_sandbox_file(
    body: PromoteRequest,
    request: Request,
    db: OrmSession = Depends(_artifacts_db),
) -> dict:
    """Promote one sandbox file into the durable library, on demand.

    The same fetch+store+row path as auto-ingestion -- this is the answer for
    `terminal`-produced files (no structured path on the wire, P3-1) and the
    retry path for an `unavailable` row. `producing_run_id` is NULL here:
    a manual promotion is not a tool completion, and inventing a run would be
    fake provenance. 422/403 for a bad path (gateway confinement, before any
    upstream call); a fetch failure records the unavailable row AND reports
    the failure (404/403 passthrough, else 502) -- honest twice over.
    """
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
    # C3: file it in the same call. Tags and a collection applied here mean
    # the agent can keep a file AND say what it is in one step, which is the
    # only moment anyone knows that for free.
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
