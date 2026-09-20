"""Building, indexing and reading back session snapshots (P6-3).

The write path (`take_snapshot` and the document it builds), the index
queries the project listings and the sweep read, and `read_snapshot_document`.
Moved out of `api/snapshots.py` (CLEANUP_PLAN step 3.5), which keeps the
routes and re-exports every name here; the contract is
`docs/SESSION_ARCHIVE_DESIGN.md` §6 and the concept is explained in that
module's docstring.

**Measured facts this rests on**:

* Rows come from **one real `session.resume`**, the same call `POST /resume`
  makes, via `_resume_for_live_id`. Not `session.history` -- on a cache miss
  that would download the transcript twice, and it carries neither
  `messages_omitted`, `info` nor `pending_*`. Resume result keys measured on
  the 3.96 MB B-86 session: `inflight, info, message_count,
  messages, messages_omitted, resumed, running, session_id, session_key,
  started_at, status`. Cold resume of that session: 1.84 s, and it did not
  stall a concurrent stream.
* **A transcript row has exactly one guaranteed key, `role`** (B-34, measured
  over all 16,572 rows on the instance). Every access into a row here is a
  `.get()`; no key is required and **no row is reshaped at rest** -- the
  document keeps the `reasoning_content` duplicates that B-86 drops on the
  wire (41% of the payload, but gzip folds an exact copy to almost nothing,
  and an archive is storage, not bandwidth). Only `GET /snapshots/{id}`
  applies B-86, through the very same `_project_transcript()` the live
  routes use, so the app's decoder renders a snapshot unchanged.
* **`session.list.message_count` is a change flag, not a size** (104 vs 1,379
  resume rows vs 1,622 history count for one session). It is recorded as
  `list_message_count` so the sweep can compare "differs"; `message_rows` is
  `len(messages) + len(background_results)` and is the positional truth.
* `hermes sessions pin <id>` exists on the deployed CLI and exempts a session
  from the stale sweep (round-tripped on a spike). It is attempted
  best-effort on archive and **never on unarchive**: the flag is shared with
  Hermes Desktop's Pinned sidebar and the gateway cannot know who set it.
* `cli.exec` output is capped at 48,000 bytes with `code: 0`, which is why
  `sessions export` is NOT the transcript path and the wire is.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError
from config.settings import get_settings
from domain import background_tasks as ledger_ops
from domain.artifact_store import MAX_INGEST_BYTES, ArtifactStore, ArtifactTooLargeError
from domain.background_ledger import synthesized_transcript_row, task_row
from domain.coerce import int_or_none
from domain.filing import find_filing
from domain.hermes_runtime import (
    _resume_for_live_id,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)
from domain.models import HERMES_RUNTIME, Project, Run, Session, SessionSnapshot, utcnow
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

#: The document format tag. Bump only for an incompatible change of shape.
SNAPSHOT_FORMAT = "trg-session-snapshot/1"

#: Who wrote the document -- so a reader of a file found on disk in a year
#: knows what produced it without the database.
SNAPSHOT_PRODUCER_SERVICE = "astation/research-gateway"

#: Why a snapshot was taken. `manual` is the only one a client may request;
#: the others are minted by the gateway itself (`DELETE`, the sweep, compress).
SNAPSHOT_REASONS: tuple[str, ...] = ("manual", "pre_delete", "sweep", "pre_compress")

#: Where the transcript rows came from. One value today; recorded per document
#: so a future `session.history`- or file-export-sourced snapshot is honest
#: about it.
SNAPSHOT_SOURCE_RESUME = "session.resume"


class SnapshotStorageError(Exception):
    """A snapshot could not be serialized or written locally.

    Distinct from `HermesError` on purpose: `DELETE /api/sessions/{id}` maps
    "Hermes has no such session" to 404 and *everything else* that stops the
    pre-delete snapshot to 409, and the route needs to tell a local disk/DB
    failure from an upstream one without string-matching.
    """


def _git_sha() -> str | None:
    """`git rev-parse --short HEAD` at import time, or None.

    Best-effort provenance for the document's `producer` block. Null in a
    Docker image without `.git`, on a machine without `git`, or if the command
    fails for any reason -- never an exception at import.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


#: Resolved once per process. A snapshot names the code that wrote it.
PRODUCER_GIT_SHA: str | None = _git_sha()


def _hermes_endpoint() -> str:
    settings = get_settings()
    return f"{settings.hermes_host}:{settings.hermes_port}"


# ---------------------------------------------------------------------------
# Writing a snapshot
# ---------------------------------------------------------------------------


def _list_row_for(list_result: Any, stored_id: str) -> dict[str, Any] | None:
    """The `session.list` row whose `id` is `stored_id`, verbatim, or None."""
    sessions = list_result.get("sessions") if isinstance(list_result, dict) else None
    if not isinstance(sessions, list):
        return None
    for row in sessions:
        if isinstance(row, dict) and row.get("id") == stored_id:
            return row
    return None


def _runs_for(db: OrmSession, stored_id: str) -> list[dict[str, Any]]:
    rows = db.execute(
        select(Run).where(Run.runtime_session_id == stored_id).order_by(Run.started_at, Run.id)
    ).scalars()
    return [
        {
            "id": run.id,
            "started_at": iso_z(run.started_at),
            "ended_at": iso_z(run.ended_at),
            "status": run.status,
            "kind": run.kind,
        }
        for run in rows
    ]


def _parent_stored_id(db: OrmSession, filing: Session | None) -> str | None:
    """`branch_parent_id` -> that filing row's stored id, when both exist."""
    if filing is None or not filing.branch_parent_id:
        return None
    parent = db.get(Session, filing.branch_parent_id)
    if parent is None or not parent.runtime_session_id:
        return None
    return parent.runtime_session_id


def _build_document(
    *,
    stored_id: str,
    reason: str,
    taken_at: datetime,
    resume_result: Any,
    list_row: dict[str, Any] | None,
    filing: Session | None,
    project: Project | None,
    parent_stored_id: str | None,
    runs: list[dict[str, Any]],
    background_results: list[dict[str, Any]],
    background_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """The §6.2 document. Every read of a Hermes payload is a `.get()`."""
    result = resume_result if isinstance(resume_result, dict) else {}
    messages = result.get("messages")
    info = result.get("info")
    rows_returned = len(messages) if isinstance(messages, list) else 0
    status = result.get("status")
    running = result.get("running")
    return {
        "format": SNAPSHOT_FORMAT,
        "producer": {
            "service": SNAPSHOT_PRODUCER_SERVICE,
            "git_sha": PRODUCER_GIT_SHA,
            "hermes": _hermes_endpoint(),
        },
        "taken_at": iso_z(taken_at),
        "reason": reason,
        "stored_session_id": stored_id,
        "workspace": {
            "workspace_session_id": filing.id if filing is not None else None,
            "project_id": project.id if project is not None else None,
            "project_title": project.title if project is not None else None,
            "filed_at": iso_z(filing.created_at) if filing is not None else None,
            "archived_at": iso_z(filing.archived_at) if filing is not None else None,
        },
        "hermes": {
            "list_row": list_row,
            "info": info if isinstance(info, dict) else None,
            "resume_meta": {
                "message_count": int_or_none(result.get("message_count")),
                "status": status if isinstance(status, str) else None,
                "running": running if isinstance(running, bool) else None,
                "started_at": result.get("started_at"),
                "session_key": result.get("session_key"),
            },
        },
        "completeness": {
            "messages_omitted": result.get("messages_omitted"),
            "rows_returned": rows_returned,
            "source": SNAPSHOT_SOURCE_RESUME,
        },
        "lineage": {
            "parent_stored_session_id": parent_stored_id,
            "source": "workspace" if parent_stored_id is not None else None,
        },
        "runs": runs,
        # VERBATIM. No B-86 dedup, no light projection, no reshaping. A
        # non-list here is stored as Hermes sent it (B-34's forwarding rule).
        "messages": messages if messages is not None else [],
        "background_results": background_results,
        "background_tasks": background_tasks,
        "pending_approval": result.get("pending_approval"),
        "pending_clarify": result.get("pending_clarify"),
    }


def _content_checksum(messages: Any, background_results: list[dict[str, Any]]) -> str:
    rows = list(messages) if isinstance(messages, list) else []
    payload = json.dumps(
        rows + background_results, sort_keys=True
    )  # exactly the §6.1 recipe
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _one_chunk(data: bytes):
    yield data


async def _store_document(store: ArtifactStore, document_bytes: bytes) -> tuple[str, int]:
    """gzip + write; returns `(storage_key, size_bytes)`. Local failures -> SnapshotStorageError."""
    if len(document_bytes) > MAX_INGEST_BYTES:
        raise SnapshotStorageError(
            f"snapshot document is {len(document_bytes)} B, over the {MAX_INGEST_BYTES} B store cap"
        )
    # mtime=0 so identical documents gzip to identical bytes -- the store is
    # content-addressed and a timestamp in the header would defeat that.
    compressed = gzip.compress(document_bytes, mtime=0)
    try:
        _checksum, size, storage_key = await store.write_stream(_one_chunk(compressed))
    except ArtifactTooLargeError as exc:
        raise SnapshotStorageError(str(exc)) from exc
    except OSError as exc:
        raise SnapshotStorageError(
            f"could not write the snapshot to the artifact store: {exc}"
        ) from exc
    return storage_key, size


def _profile_for(stored_id: str, db: OrmSession | None) -> str:
    """Which profile a session belongs to, from its filing row.

    `default` when the session is unfiled or the lookup cannot run: an unfiled
    session has no row to record a profile on, and that is exactly the case
    where the caller (a route) is the one that knows and should pass it
    explicitly. Never raises -- a snapshot must not fail because a metadata
    lookup did.
    """
    if db is None:
        return "default"
    try:
        row = db.execute(
            select(Session.profile).where(
                Session.runtime == HERMES_RUNTIME,
                Session.runtime_session_id == stored_id,
            )
        ).first()
    except SQLAlchemyError:
        return "default"
    if row is None or not row[0]:
        return "default"
    return str(row[0])


async def take_snapshot(
    app_state: Any,
    stored_id: str,
    *,
    reason: str,
    list_row: dict[str, Any] | None = None,
    db: OrmSession | None = None,
    profile: str | None = None,
) -> SessionSnapshot:
    """Take one snapshot of `stored_id` now and return its committed index row.

    Order, and why: **one real `session.resume`** (through `_with_reconnect`
    and `_resume_for_live_id`, exactly like `POST /resume`, so the handle it
    mints is cached for this connection and no second transcript download
    follows) -> `session.list` only when the caller did not hand a row in (the
    sweep passes its one list per pass) -> read the workspace (filing row,
    project, lineage, runs, background ledger) -> build the document -> write
    the gzipped bytes to the store **first** -> insert the index row **last**.
    A failure anywhere before the insert leaves no row at all; a failure at
    the insert leaves content-addressed bytes nobody references, which is
    harmless. There is never a row without its bytes.

    Raises `HermesError` (upstream: unknown session, unreachable, protocol) or
    `SnapshotStorageError` (local: serialize/write/insert). `ValueError` for a
    reason outside `SNAPSHOT_REASONS` -- a programming error, not a client one.

    `db`: when None this function opens and commits its own session; when
    given, the row is added and committed on the caller's so a route can keep
    "snapshot then flag" in one connection. Either way the returned object is
    fully loaded (`expire_on_commit=False`).
    """
    if reason not in SNAPSHOT_REASONS:
        raise ValueError(f"unknown snapshot reason {reason!r}; expected one of {SNAPSHOT_REASONS}")

    # on the session's OWN connection, not whichever one is default.
    # A snapshot begins with a real `session.resume`, and resuming a
    # `kimi25`/`qwen38-flash` session against the default connection answers
    # `[4007] session not found` -- which surfaced as archiving simply failing
    # for every non-default session (owner-reported, caught live).
    #
    # `profile=None` means "look it up": a filed session records its profile
    # on its `Session` row, which is the honest source and needs no caller to
    # remember. That is what fixes the callers that cannot know it -- the
    # snapshot sweep, the pre-delete snapshot, pre-compress -- rather than
    # only the routes that were told.
    resolved_profile = profile if profile is not None else _profile_for(stored_id, db)
    adapter: HermesAdapter = resolve_profile_adapter(app_state, resolved_profile)
    cache = resolve_live_handle_cache(app_state, resolved_profile)
    store: ArtifactStore = app_state.artifact_store

    _live_id, resume_result = await _with_reconnect(
        app_state, adapter, lambda: _resume_for_live_id(adapter, stored_id, cache)
    )
    if list_row is None:
        try:
            list_result = await _with_reconnect(app_state, adapter, adapter.session_list)
        except HermesError as exc:
            # The transcript is the durable thing and it is already in hand;
            # the list row is metadata (title, source, change flag). Losing it
            # on a flaky second call must not lose the copy.
            logger.warning(
                "session.list failed while snapshotting %s; recording no list row (%s)",
                stored_id,
                exc,
            )
            list_result = None
        list_row = _list_row_for(list_result, stored_id)

    taken_at = utcnow()

    def _write(session: OrmSession) -> tuple[dict[str, Any], SessionSnapshot]:
        filing = find_filing(session, HERMES_RUNTIME, stored_id)
        project = session.get(Project, filing.project_id) if filing is not None else None
        finished = ledger_ops.finished_results_for_session(session, stored_id)
        background_results = [synthesized_transcript_row(task) for task in finished]
        background_tasks = [
            task_row(task) for task in ledger_ops.tasks_for_session(session, stored_id)
        ]
        document = _build_document(
            stored_id=stored_id,
            reason=reason,
            taken_at=taken_at,
            resume_result=resume_result,
            list_row=list_row,
            filing=filing,
            project=project,
            parent_stored_id=_parent_stored_id(session, filing),
            runs=_runs_for(session, stored_id),
            background_results=background_results,
            background_tasks=background_tasks,
        )
        messages = document["messages"]
        message_rows = (len(messages) if isinstance(messages, list) else 0) + len(
            background_results
        )
        list_title = list_row.get("title") if isinstance(list_row, dict) else None
        title = (
            list_title
            if isinstance(list_title, str) and list_title
            else (filing.title if filing is not None else None)
        )
        list_count = (
            int_or_none(list_row.get("message_count")) if isinstance(list_row, dict) else None
        )
        omitted = document["completeness"]["messages_omitted"]
        row = SessionSnapshot(
            project_id=project.id if project is not None else None,
            workspace_session_id=filing.id if filing is not None else None,
            stored_session_id=stored_id,
            taken_at=taken_at,
            reason=reason,
            title=title,
            message_rows=message_rows,
            list_message_count=list_count,
            messages_omitted=omitted if isinstance(omitted, bool) else None,
            content_checksum=_content_checksum(messages, background_results),
            hermes_meta_json=list_row,
            # Filled after the bytes are written.
            raw_bytes=0,
            size_bytes=0,
            checksum="",
            storage_key="",
        )
        return document, row

    async def _finish(session: OrmSession) -> SessionSnapshot:
        document, row = _write(session)
        try:
            document_bytes = json.dumps(document, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SnapshotStorageError(
                f"snapshot document is not JSON-serializable: {exc}"
            ) from exc
        row.raw_bytes = len(document_bytes)
        row.checksum = hashlib.sha256(document_bytes).hexdigest()
        row.storage_key, row.size_bytes = await _store_document(store, document_bytes)
        try:
            session.add(row)
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            raise SnapshotStorageError(f"could not record the snapshot row: {exc}") from exc
        logger.info(
            "snapshot %s of session %s (%s): %d rows, %d B raw, %d B gz",
            row.id,
            stored_id,
            reason,
            row.message_rows,
            row.raw_bytes,
            row.size_bytes,
        )
        return row

    if db is not None:
        return await _finish(db)
    with app_state.db_sessions() as own:
        return await _finish(own)


# ---------------------------------------------------------------------------
# Reading the index
# ---------------------------------------------------------------------------


def snapshot_row(snapshot: SessionSnapshot) -> dict[str, Any]:
    """The `<row>` serializer of §6.4. `hermes_meta`, not `hermes_meta_json`."""
    return {
        "id": snapshot.id,
        "stored_session_id": snapshot.stored_session_id,
        "project_id": snapshot.project_id,
        "workspace_session_id": snapshot.workspace_session_id,
        "taken_at": iso_z(snapshot.taken_at),
        "reason": snapshot.reason,
        "title": snapshot.title,
        "message_rows": snapshot.message_rows,
        "list_message_count": snapshot.list_message_count,
        "messages_omitted": snapshot.messages_omitted,
        "raw_bytes": snapshot.raw_bytes,
        "size_bytes": snapshot.size_bytes,
        "checksum": snapshot.checksum,
        "content_checksum": snapshot.content_checksum,
        "storage_key": snapshot.storage_key,
        "hermes_meta": snapshot.hermes_meta_json,
    }


def latest_snapshot_summary(snapshot: SessionSnapshot | None) -> dict[str, Any] | None:
    """The `latest_snapshot` sub-object on a project session row, or None."""
    if snapshot is None:
        return None
    return {
        "id": snapshot.id,
        "taken_at": iso_z(snapshot.taken_at),
        "message_rows": snapshot.message_rows,
        "reason": snapshot.reason,
    }


def _newest_first(stmt):
    return stmt.order_by(SessionSnapshot.taken_at.desc(), SessionSnapshot.id.desc())


def latest_snapshots_by_stored_id(
    db: OrmSession, stored_ids: list[str], *, project_id: str | None = None
) -> dict[str, SessionSnapshot]:
    """`{stored_id: its newest snapshot}` for the ids that have one. One query.

    `project_id` scopes the answer to snapshots taken under that project. The
    project-sessions route passes it so a project's Archived row names *its*
    copy: a session snapshotted in A, moved to B and snapshotted again must not
    make A's row open B's transcript (found by the Stream A tester, 2026-09-03).
    The sweep calls this unscoped -- it wants the newest copy anywhere.
    """
    if not stored_ids:
        return {}
    stmt = select(SessionSnapshot).where(SessionSnapshot.stored_session_id.in_(stored_ids))
    if project_id is not None:
        stmt = stmt.where(SessionSnapshot.project_id == project_id)
    rows = db.execute(_newest_first(stmt)).scalars()
    latest: dict[str, SessionSnapshot] = {}
    for snapshot in rows:
        latest.setdefault(snapshot.stored_session_id, snapshot)
    return latest


def snapshot_counts_by_stored_id(
    db: OrmSession, stored_ids: list[str], *, project_id: str | None = None
) -> dict[str, int]:
    """`{stored_id: snapshot count}`, zeros included. One grouped query.
    `project_id` scopes it exactly as in `latest_snapshots_by_stored_id`."""
    counts = dict.fromkeys(stored_ids, 0)
    if not stored_ids:
        return counts
    stmt = select(SessionSnapshot.stored_session_id, func.count(SessionSnapshot.id)).where(
        SessionSnapshot.stored_session_id.in_(stored_ids)
    )
    if project_id is not None:
        stmt = stmt.where(SessionSnapshot.project_id == project_id)
    rows = db.execute(stmt.group_by(SessionSnapshot.stored_session_id)).all()
    for stored_id, count in rows:
        counts[stored_id] = count
    return counts


def snapshot_only_stored_ids(db: OrmSession, project_id: str, exclude: set[str]) -> list[str]:
    """Stored ids with a snapshot under `project_id` and no filing row (§6.4)."""
    rows = db.execute(
        select(SessionSnapshot.stored_session_id)
        .where(SessionSnapshot.project_id == project_id)
        .distinct()
    ).scalars()
    return [stored_id for stored_id in rows if stored_id not in exclude]


def snapshot_candidates(db: OrmSession) -> set[str]:
    """`sessions.runtime_session_id` ∪ `session_snapshots.stored_session_id` -- the sweep's `filed` scope."""
    filed = db.execute(
        select(Session.runtime_session_id).where(Session.runtime_session_id.is_not(None))
    ).scalars()
    snapshotted = db.execute(select(SessionSnapshot.stored_session_id).distinct()).scalars()
    return {stored for stored in filed if stored} | {stored for stored in snapshotted if stored}


def is_archived(
    *, archived_at: Any, has_filing: bool, missing: bool | None, snapshot_count: int
) -> bool:
    """The one predicate both the gateway and the app use for the Archived section.

    Archived when the user said so (`archived_at`), when the row is
    snapshot-only (no filing row -- the session was removed from the project
    or deleted, and only the copy remains), or when Hermes no longer lists a
    filed session that has a copy (`missing is True` -- a `None` is "Hermes
    unreachable", which is not "gone").
    """
    return archived_at is not None or not has_filing or (missing is True and snapshot_count > 0)


def read_snapshot_document(store: ArtifactStore, snapshot: SessionSnapshot) -> dict[str, Any]:
    """gunzip + parse + verify the sha256 the row recorded. 502 naming the key otherwise."""

    def _corrupt(why: str) -> HTTPException:
        return HTTPException(
            status_code=502,
            detail=(
                f"snapshot {snapshot.id} is unreadable at storage_key "
                f"{snapshot.storage_key!r}: {why}"
            ),
        )

    try:
        path = store.path_for_key(snapshot.storage_key)
    except ValueError as exc:
        raise _corrupt(str(exc)) from exc
    try:
        compressed = path.read_bytes()
    except OSError as exc:
        raise _corrupt(f"file missing or unreadable ({exc.__class__.__name__})") from exc
    try:
        document_bytes = gzip.decompress(compressed)
    except (OSError, EOFError, ValueError) as exc:
        raise _corrupt(f"not a valid gzip stream ({exc})") from exc
    digest = hashlib.sha256(document_bytes).hexdigest()
    if digest != snapshot.checksum:
        raise _corrupt(f"checksum mismatch (row {snapshot.checksum}, file {digest})")
    try:
        document = json.loads(document_bytes)
    except ValueError as exc:
        raise _corrupt(f"not valid JSON ({exc})") from exc
    if not isinstance(document, dict):
        raise _corrupt("document is not a JSON object")
    return document
