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
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

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
from domain.artifact_kinds import is_valid_kind
from domain.artifact_store import (  # noqa: F401  (re-exported; see module docstring)
    MAX_INGEST_BYTES,
    STATUS_AVAILABLE,
    STATUS_UNAVAILABLE,
    ArtifactStore,
    ArtifactTooLargeError,
    _artifact_json,
)
from domain.db import columns_present, schema_checked_db
from domain.models import Artifact, Project, Run, utcnow

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


def _bookmark_shelf(db: OrmSession, *, limit: int, kind: str | None) -> list[dict]:
    """Every bookmarked artifact, newest bookmark first.

    Deliberately NOT scoped to a project: the owner's ask is that a bookmarked
    artifact is reachable from wherever they are, including a project that did
    not produce it. It is returned as its own field rather than mixed into the
    project's own list, so the existing project-scoped contract still holds and
    the app can render a separate shelf.
    """
    query = (
        select(Artifact)
        .where(Artifact.bookmarked_at.is_not(None))
        .order_by(Artifact.bookmarked_at.desc(), Artifact.id)
    )
    rows = [_artifact_json(a) for a in db.execute(query.limit(limit)).scalars()]
    return _filter_by_kind(rows, kind)


def _listing(db: OrmSession, query, *, latest: bool, limit: int) -> list[dict]:
    """The rows of one listing query: capped at `limit` either way, but with
    `latest=True` the cap applies AFTER collapsing versions, so the page is
    `limit` files rather than `limit` saves."""
    if not latest:
        return [_artifact_json(a) for a in db.execute(query.limit(limit)).scalars()]
    return _collapse_to_latest(db.execute(query).scalars())[:limit]


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
    query = (
        select(Artifact)
        .where(Artifact.project_id == project_id)
        .order_by(Artifact.created_at.desc(), Artifact.id)
    )
    wanted = _validated_kind(kind)
    return {
        "project_id": project_id,
        "latest_only": latest,
        "kind": wanted,
        "artifacts": _filter_by_kind(_listing(db, query, latest=latest, limit=limit), wanted),
        # The bookmark shelf is global by design -- see `_bookmark_shelf`. It
        # is a SEPARATE field so this route's "only this project's artifacts"
        # guarantee still holds for `artifacts`.
        "bookmarked": _bookmark_shelf(db, limit=limit, kind=wanted) if bookmarks else [],
    }


@artifacts_router.get("/artifacts")
async def list_artifacts(
    session: str | None = Query(default=None),
    unfiled: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    latest: bool = Query(default=False),
    kind: str | None = Query(default=None),
    bookmarked: bool = Query(default=False),
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
    if bookmarked:
        query = query.where(Artifact.bookmarked_at.is_not(None))
    wanted = _validated_kind(kind)
    return {
        "session": stored_session_id,
        # True only for a session-scoped query: see the docstring -- artifacts
        # with no producing run (manual promotions) can never appear there.
        "run_attributed_only": stored_session_id is not None,
        "unfiled_only": unfiled,
        "latest_only": latest,
        "kind": wanted,
        "bookmarked_only": bookmarked,
        "artifacts": _filter_by_kind(_listing(db, query, latest=latest, limit=limit), wanted),
    }


@artifacts_router.get("/artifacts/kinds")
async def list_artifact_kinds() -> dict:
    """The filter vocabulary, so the app's chips come from the server rather
    than a copy that can drift out of step with the classifier."""
    return {"kinds": list(ARTIFACT_KINDS)}


@artifacts_router.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str, db: OrmSession = Depends(_artifacts_db)) -> dict:
    return _artifact_json(_load_artifact(db, artifact_id))


@artifacts_router.put("/artifacts/{artifact_id}/bookmark")
async def bookmark_artifact(artifact_id: str, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Bookmark an artifact so it is reachable from every project.

    Idempotent, but re-bookmarking DOES move it to the top of the shelf: the
    timestamp records when the user last said this matters, which is the order
    they expect to find it in.
    """
    artifact = _load_artifact(db, artifact_id)
    artifact.bookmarked_at = utcnow()
    db.commit()
    db.refresh(artifact)
    return _artifact_json(artifact)


@artifacts_router.delete("/artifacts/{artifact_id}/bookmark")
async def unbookmark_artifact(artifact_id: str, db: OrmSession = Depends(_artifacts_db)) -> dict:
    """Remove a bookmark. A no-op on an artifact that has none, not a 404 --
    the caller's intent ("this should not be bookmarked") is already true."""
    artifact = _load_artifact(db, artifact_id)
    artifact.bookmarked_at = None
    db.commit()
    db.refresh(artifact)
    return _artifact_json(artifact)


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
    return {"artifact": outcome.artifact, "deduplicated": outcome.deduplicated}
