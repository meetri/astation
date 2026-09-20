"""Building, indexing and reading back session snapshots (P6-3)."""

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

SNAPSHOT_FORMAT = "trg-session-snapshot/1"

SNAPSHOT_PRODUCER_SERVICE = "astation/research-gateway"

SNAPSHOT_REASONS: tuple[str, ...] = ("manual", "pre_delete", "sweep", "pre_compress")

SNAPSHOT_SOURCE_RESUME = "session.resume"


class SnapshotStorageError(Exception):
    """A snapshot could not be serialized or written locally."""


def _git_sha() -> str | None:
    """`git rev-parse --short HEAD` at import time, or None."""
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


PRODUCER_GIT_SHA: str | None = _git_sha()


def _hermes_endpoint() -> str:
    settings = get_settings()
    return f"{settings.hermes_host}:{settings.hermes_port}"


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
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _one_chunk(data: bytes):
    yield data


async def _store_document(store: ArtifactStore, document_bytes: bytes) -> tuple[str, int]:
    """gzip + write; returns `(storage_key, size_bytes)`. Local failures -> SnapshotStorageError."""
    if len(document_bytes) > MAX_INGEST_BYTES:
        raise SnapshotStorageError(
            f"snapshot document is {len(document_bytes)} B, over the {MAX_INGEST_BYTES} B store cap"
        )
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
    """Which profile a session belongs to, from its filing row."""
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
    """Take one snapshot of `stored_id` now and return its committed index row."""
    if reason not in SNAPSHOT_REASONS:
        raise ValueError(f"unknown snapshot reason {reason!r}; expected one of {SNAPSHOT_REASONS}")

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
    """`{stored_id: its newest snapshot}` for the ids that have one. One query."""
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
    """The one predicate both the gateway and the app use for the Archived section."""
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
