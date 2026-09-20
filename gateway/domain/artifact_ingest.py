"""Artifact ingestion (P3-1 / P3-9 / P3-10): the three broadcaster subscribers."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import logging
import posixpath
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from adapters.hermes import HermesAdapter, HermesError
from domain.artifact_filing import filing_tags
from domain.artifact_store import (
    STATUS_AVAILABLE,
    STATUS_UNAVAILABLE,
    ArtifactStore,
    ArtifactTooLargeError,
    _artifact_json,
    _guess_mime,
)
from domain.event_stream import STREAM_DESYNCHRONIZED_EVENT_TYPE
from domain.models import Artifact, utcnow
from domain.sandbox_fs import HermesHttpSandboxFS, SandboxFS
from domain.sandbox_paths import validate_sandbox_path
from domain.tag_store import attach as tag_attach
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)


# `terminal` is absent deliberately: its completion carries no path field to ingest from.
AUTO_INGEST_TOOL_NAMES: frozenset[str] = frozenset({"write_file", "patch"})

_TOOL_COMPLETED_TYPE = "tool.completed"

_DESYNCHRONIZED_TYPE = STREAM_DESYNCHRONIZED_EVENT_TYPE


@dataclass
class IngestOutcome:
    """What one ingestion attempt did -- success and failure both have a row."""

    artifact: dict[str, Any]
    deduplicated: bool
    error: str | None = None
    upstream_status: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ArtifactIngestor:
    """Fetches sandbox files into the durable library and records the rows."""

    def __init__(
        self,
        adapter: HermesAdapter,
        session_factory: sessionmaker,
        store: ArtifactStore,
        *,
        sandbox_root: str,
        denylist: SandboxDiffDenylist | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self._adapter = adapter
        self._session_factory = session_factory
        self._store = store
        self._sandbox_root = sandbox_root
        self._workspace_root = workspace_root
        self._denylist = denylist or SandboxDiffDenylist(sandbox_root)
        self.ignored_paths = 0
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()


    async def run(self, broadcaster: Any) -> None:
        """Consume the broadcaster's stream forever; never raises past itself."""
        while True:
            try:
                async with broadcaster.subscribe() as queue:
                    while True:
                        frame = await queue.get()
                        if isinstance(frame, dict) and frame.get("type") == _DESYNCHRONIZED_TYPE:
                            logger.warning(
                                "artifact ingestor fell behind the event stream "
                                "and was desynchronized; resubscribing (tool "
                                "completions in the gap were not auto-ingested)"
                            )
                            break
                        self.observe(frame)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("artifact ingestor subscriber failed; resubscribing")
            await asyncio.sleep(1.0)

    def observe(self, frame: Any) -> None:
        """Look at one forwarded frame; maybe schedule one ingestion."""
        signal = self.ingest_signal(frame)
        if signal is None:
            return
        source_path, producing_run_id, project_id = signal
        if self._denylist.denies_file(source_path):
            self.ignored_paths += 1
            logger.info(
                "artifact auto-ingest: ignoring %r (operational path, not user output)",
                source_path,
            )
            return
        try:
            task = asyncio.create_task(
                self._ingest_from_event(source_path, producing_run_id, project_id)
            )
        except RuntimeError:  # pragma: no cover - no running loop (sync tests)
            logger.warning("no event loop to schedule an artifact ingestion on")
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    def ingest_signal(frame: Any) -> tuple[str, str | None, str | None] | None:
        """`(source_path, producing_run_id, project_id)` if this frame should
        auto-ingest, else None.
        """
        if not isinstance(frame, dict) or frame.get("type") != _TOOL_COMPLETED_TYPE:
            return None
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            return None
        if payload.get("_truncated") is True:
            logger.warning(
                "a tool.completed frame arrived truncated (B-02); its file, if "
                "any, was not auto-ingested -- promote it manually via "
                "POST /api/sandbox/promote"
            )
            return None
        if payload.get("name") not in AUTO_INGEST_TOOL_NAMES:
            return None
        args = payload.get("args")
        path = args.get("path") if isinstance(args, dict) else None
        if not isinstance(path, str) or not path:
            logger.warning(
                "a %r tool.completed frame carried no usable args.path; skipping auto-ingest",
                payload.get("name"),
            )
            return None
        run_id = frame.get("run_id")
        project_id = frame.get("project_id")
        return (
            path,
            run_id if isinstance(run_id, str) else None,
            project_id if isinstance(project_id, str) else None,
        )

    async def _ingest_from_event(
        self, source_path: str, producing_run_id: str | None, project_id: str | None
    ) -> None:
        """One auto-ingestion, with every failure contained and recorded."""
        try:
            outcome = await self.ingest(
                source_path,
                producing_run_id=producing_run_id,
                project_id=project_id,
            )
        except HTTPException as exc:
            self._record_failure_row(
                source_path,
                producing_run_id,
                project_id,
                f"path validation failed: {exc.detail}",
            )
            return
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "artifact auto-ingest crashed for %r; no row was written",
                source_path,
            )
            return
        if outcome.ok:
            logger.info(
                "auto-ingested %r -> artifact %s (%s)",
                source_path,
                outcome.artifact.get("id"),
                "deduplicated" if outcome.deduplicated else "new",
            )
        else:
            logger.warning(
                "auto-ingest of %r could not fetch the bytes (%s); recorded "
                "artifact %s as unavailable",
                source_path,
                outcome.error,
                outcome.artifact.get("id"),
            )


    async def ingest(
        self,
        source_path: str,
        *,
        producing_run_id: str | None = None,
        project_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestOutcome:
        """Validate -> fetch -> store -> row. The whole P3-1 pipeline, once."""
        target = validate_sandbox_path(source_path, self._sandbox_root)
        async with self._lock:
            error: str | None = None
            upstream_status: int | None = None
            checksum = size = storage_key = None  # type: ignore[assignment]
            mime_type = _guess_mime(target)
            try:
                response = await self._adapter.files_download(target)
            except HermesError as exc:
                error = f"download request failed: {exc}"
            else:
                if response.status_code != 200:
                    upstream_status = response.status_code
                    error = f"Hermes answered HTTP {response.status_code}"
                    await response.aclose()
                else:
                    header_mime = response.headers.get("content-type", "")
                    if header_mime:
                        mime_type = header_mime.split(";")[0].strip() or mime_type
                    try:
                        checksum, size, storage_key = await self._store.write_stream(
                            response.aiter_bytes()
                        )
                    except ArtifactTooLargeError as exc:
                        error = str(exc)
                    except Exception as exc:
                        error = f"fetch/store failed: {exc.__class__.__name__}: {exc}"
                    finally:
                        await response.aclose()

            if error is not None:
                row = self._record_failure_row(
                    target,
                    producing_run_id,
                    project_id,
                    error,
                    mime_type=mime_type,
                    extra_metadata=extra_metadata,
                )
                return IngestOutcome(
                    artifact=row,
                    deduplicated=False,
                    error=error,
                    upstream_status=upstream_status,
                )

            row, deduplicated = self._record_success_row(
                target,
                producing_run_id,
                project_id,
                checksum=checksum,
                size=size,
                storage_key=storage_key,
                mime_type=mime_type,
                extra_metadata=extra_metadata,
            )
            return IngestOutcome(artifact=row, deduplicated=deduplicated)


    def _record_success_row(
        self,
        source_path: str,
        producing_run_id: str | None,
        project_id: str | None,
        *,
        checksum: str,
        size: int,
        storage_key: str,
        mime_type: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """`_write_success_row`, with dangling provenance degraded, not fatal."""
        try:
            return self._write_success_row(
                source_path,
                producing_run_id,
                project_id,
                checksum=checksum,
                size=size,
                storage_key=storage_key,
                mime_type=mime_type,
                extra_metadata=extra_metadata,
            )
        except IntegrityError:
            logger.warning(
                "artifact provenance (run %r / project %r) violated a foreign "
                "key; recording %r with NULL provenance instead",
                producing_run_id,
                project_id,
                source_path,
            )
            return self._write_success_row(
                source_path,
                None,
                None,
                checksum=checksum,
                size=size,
                storage_key=storage_key,
                mime_type=mime_type,
                extra_metadata=extra_metadata,
            )

    def _write_success_row(
        self,
        source_path: str,
        producing_run_id: str | None,
        project_id: str | None,
        *,
        checksum: str,
        size: int,
        storage_key: str,
        mime_type: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Upsert per the P3-1 dedup rule (module docstring)."""
        with self._session_factory() as db:
            rows = self._rows_for_path(db, source_path)
            existing = next((r for r in rows if r.checksum == checksum), None)
            pending = next(
                (r for r in rows if r.status == STATUS_UNAVAILABLE and r.checksum is None),
                None,
            )
            deduplicated = existing is not None
            artifact = existing or pending
            is_new = artifact is None
            if artifact is None:
                artifact = Artifact(
                    project_id=project_id,
                    producing_run_id=producing_run_id,
                    title=posixpath.basename(source_path),
                    mime_type=mime_type,
                    source_path=source_path,
                )
                db.add(artifact)
            artifact.status = STATUS_AVAILABLE
            artifact.checksum = checksum
            artifact.size_bytes = size
            artifact.storage_key = storage_key
            artifact.mime_type = mime_type
            kept = {
                k: v
                for k, v in (artifact.metadata_json or {}).items()
                if k not in ("ingest_error", "failed_at")
            }
            kept.update(extra_metadata or {})
            artifact.metadata_json = kept or None
            if artifact.producing_run_id is None:
                artifact.producing_run_id = producing_run_id
            if artifact.project_id is None:
                artifact.project_id = project_id
            if is_new:

                # Tag on create only: re-tagging on a later touch would restore a removed tag.
                db.flush()
                self._apply_filing_tags(db, artifact)
            db.commit()
            return _artifact_json(artifact), deduplicated

    def _apply_filing_tags(self, db: OrmSession, artifact: Artifact) -> None:
        """Tag a newly-filed artifact from where it was written."""
        if not self._workspace_root:
            return
        try:
            for name in filing_tags(artifact.source_path, self._workspace_root):
                tag_attach(db, "artifact", artifact.id, name)
        except Exception:  # pragma: no cover - defensive
            logger.exception("could not apply filing tags to %s", artifact.source_path)

    def _record_failure_row(
        self,
        source_path: str,
        producing_run_id: str | None,
        project_id: str | None,
        error: str,
        *,
        mime_type: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The never-silent failure row (P3-1). Best-effort: a DB failure here
        is logged, not raised -- the ingestor's contract is the notify relay's
        (nothing it does may crash its caller). Dangling provenance degrades
        to NULLs the same way `_record_success_row`'s does."""
        try:
            try:
                return self._write_failure_row(
                    source_path,
                    producing_run_id,
                    project_id,
                    error,
                    mime_type=mime_type,
                    extra_metadata=extra_metadata,
                )
            except IntegrityError:
                logger.warning(
                    "artifact provenance (run %r / project %r) violated a foreign "
                    "key; recording the unavailable row for %r with NULL provenance",
                    producing_run_id,
                    project_id,
                    source_path,
                )
                return self._write_failure_row(
                    source_path,
                    None,
                    None,
                    error,
                    mime_type=mime_type,
                    extra_metadata=extra_metadata,
                )
        except Exception:
            logger.exception(
                "could not record the unavailable-artifact row for %r (is the "
                "database migrated?); the failure (%s) is otherwise unrecorded",
                source_path,
                error,
            )
            return {"id": None, "source_path": source_path, "status": STATUS_UNAVAILABLE}

    def _write_failure_row(
        self,
        source_path: str,
        producing_run_id: str | None,
        project_id: str | None,
        error: str,
        *,
        mime_type: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._session_factory() as db:
            rows = self._rows_for_path(db, source_path)
            artifact = next(
                (r for r in rows if r.status == STATUS_UNAVAILABLE and r.checksum is None),
                None,
            )
            if artifact is None:
                artifact = Artifact(
                    project_id=project_id,
                    producing_run_id=producing_run_id,
                    title=posixpath.basename(source_path),
                    mime_type=mime_type or _guess_mime(source_path),
                    status=STATUS_UNAVAILABLE,
                    source_path=source_path,
                )
                db.add(artifact)
            artifact.status = STATUS_UNAVAILABLE
            artifact.metadata_json = {
                **(extra_metadata or {}),
                "ingest_error": error,
                "failed_at": iso_z(utcnow()),
            }
            if artifact.producing_run_id is None:
                artifact.producing_run_id = producing_run_id
            if artifact.project_id is None:
                artifact.project_id = project_id
            db.commit()
            return _artifact_json(artifact)

    @staticmethod
    def _rows_for_path(db: OrmSession, source_path: str) -> list[Artifact]:
        return list(
            db.execute(
                select(Artifact)
                .where(Artifact.source_path == source_path)
                .order_by(Artifact.created_at.desc())
            ).scalars()
        )


_MESSAGE_STARTED_TYPE = "message.started"
_MESSAGE_COMPLETED_TYPE = "message.completed"


# Bounds one turn's walk so a deep or wide tree cannot cost unbounded upstream listing calls.
DIFF_SCAN_MAX_DIRS = 64
DIFF_SCAN_MAX_DEPTH = 4


# Bounds before-snapshots whose turn never completes; the oldest pairing is evicted.
MAX_PENDING_DIFF_RUNS = 64


# The sandbox root is Hermes's whole data root; these trees churn every turn and are never output.
SANDBOX_DIFF_DENYLIST_DIRS: frozenset[str] = frozenset({"logs", "cron", "state", "cache"})

ARTIFACT_IGNORE_COMPONENTS: frozenset[str] = frozenset(
    {
        ".git",
        ".npm",
        ".cache",
        ".curator_backups",
        ".venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)

ARTIFACT_IGNORE_GLOBS: tuple[str, ...] = ("*.log", "*.pyc", "*.tmp", "*.swp", ".DS_Store")


SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS: tuple[str, ...] = (
    "*.db",
    "*.db-wal",
    "*.db-shm",
    "*.lock",
    "*heartbeat*",
    "*_last_success",
    "channel_directory.json",
)
DEFAULT_IGNORE_GLOBS: tuple[str, ...] = SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS + ARTIFACT_IGNORE_GLOBS


def _runtime_ignore(settings: Any, key: str) -> str:
    """One extra ignore list: the editable overlay, else the env value."""
    try:
        from config.runtime_config import read_overlay

        overlay = read_overlay(settings.research_gateway_runtime_config_path)
        value = (overlay.values or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value
    except Exception:  # pragma: no cover - a missing overlay is not an error
        pass
    return getattr(settings, key, "") or ""


def _denylist_csv(raw: str) -> tuple[str, ...]:
    """A comma-separated env override -> a tuple of entries (empty -> empty)."""
    return tuple(item.strip().strip("/") for item in raw.split(",") if item.strip().strip("/"))


class SandboxDiffDenylist:
    """Decides which sandbox paths the diff ingestor must never promote."""

    def __init__(
        self,
        root: str,
        dirs: frozenset[str] | set[str] | tuple[str, ...] = SANDBOX_DIFF_DENYLIST_DIRS,
        filename_globs: tuple[str, ...] = DEFAULT_IGNORE_GLOBS,
        components: frozenset[str] | set[str] | tuple[str, ...] = ARTIFACT_IGNORE_COMPONENTS,
    ) -> None:
        self._root = posixpath.normpath(root)
        self._dirs = frozenset(dirs)
        self._globs = tuple(filename_globs)
        self._components = frozenset(components)

    @classmethod
    def from_settings(cls, settings: Any) -> SandboxDiffDenylist:
        """The deployed construction (api.main): built-in defaults unless the
        corresponding env override is non-empty, in which case the override
        REPLACES that default wholesale (a layout change is a config edit)."""
        # An env override replaces a built-in list; the runtime additions below only add to it.
        dirs = _denylist_csv(settings.hermes_sandbox_denylist_dirs) or SANDBOX_DIFF_DENYLIST_DIRS
        globs = (
            _denylist_csv(settings.hermes_sandbox_denylist_globs)
            or SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS
        )
        extra_dirs = _denylist_csv(_runtime_ignore(settings, "artifact_ignore_dirs"))
        extra_globs = _denylist_csv(_runtime_ignore(settings, "artifact_ignore_globs"))
        return cls(
            settings.hermes_sandbox_root,
            frozenset(dirs),
            tuple(globs) + tuple(ARTIFACT_IGNORE_GLOBS) + tuple(extra_globs),
            frozenset(ARTIFACT_IGNORE_COMPONENTS) | frozenset(extra_dirs),
        )

    def _top_level_component(self, path: str) -> str | None:
        """The first path component below the root, or None (the root itself,
        or a path not under the root at all -- the latter is someone else's
        problem: `validate_sandbox_path` already refuses it downstream)."""
        rel = posixpath.relpath(posixpath.normpath(path), self._root)
        if rel == "." or rel.startswith(".."):
            return None
        return rel.split("/", 1)[0]

    def denies_dir(self, path: str) -> bool:
        """True for a denied top-level directory, a denied component anywhere
        in the path, or anything inside either."""
        if self._top_level_component(path) in self._dirs:
            return True
        rel = posixpath.relpath(posixpath.normpath(path), self._root)
        if rel == "." or rel.startswith(".."):
            return False
        return any(part in self._components for part in rel.split("/"))

    def denies_file(self, path: str) -> bool:
        """True when this path must not be auto-ingested, by EITHER path."""
        if self.denies_dir(path):
            return True
        name = posixpath.basename(path)
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self._globs)


class SandboxDiffIngestor:
    """Snapshots the sandbox tree at each turn's start/end; auto-ingests the diff."""

    def __init__(
        self,
        adapter: HermesAdapter,
        ingestor: ArtifactIngestor,
        *,
        sandbox_root: str,
        max_scan_dirs: int = DIFF_SCAN_MAX_DIRS,
        max_scan_depth: int = DIFF_SCAN_MAX_DEPTH,
        max_pending_runs: int = MAX_PENDING_DIFF_RUNS,
        denylist: SandboxDiffDenylist | None = None,
        sandbox_fs: SandboxFS | None = None,
    ) -> None:
        self._adapter = adapter
        self._sandbox_fs: SandboxFS = (
            sandbox_fs if sandbox_fs is not None else HermesHttpSandboxFS(adapter)
        )
        self._ingestor = ingestor
        self._sandbox_root = sandbox_root
        self._max_scan_dirs = max_scan_dirs
        self._max_scan_depth = max_scan_depth
        self._max_pending_runs = max_pending_runs
        self._denylist = denylist or SandboxDiffDenylist(sandbox_root)
        self._pending: dict[str, dict[str, tuple[Any, Any]] | None] = {}
        self._pending_order: deque[str] = deque()
        self._start_tasks: dict[str, asyncio.Task[None]] = {}
        self._tasks: set[asyncio.Task[None]] = set()


    async def run(self, broadcaster: Any) -> None:
        """Consume the broadcaster's stream forever; never raises past itself."""
        while True:
            try:
                async with broadcaster.subscribe() as queue:
                    while True:
                        frame = await queue.get()
                        if isinstance(frame, dict) and frame.get("type") == _DESYNCHRONIZED_TYPE:
                            logger.warning(
                                "sandbox diff ingestor fell behind the event stream "
                                "and was desynchronized; resubscribing (turn "
                                "boundaries in the gap will not be diffed)"
                            )
                            break
                        self.observe(frame)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("sandbox diff ingestor subscriber failed; resubscribing")
            await asyncio.sleep(1.0)

    def observe(self, frame: Any) -> None:
        """Look at one forwarded frame; maybe schedule a snapshot/diff."""
        signal = self.turn_boundary_signal(frame)
        if signal is None:
            return
        frame_type, run_id, project_id = signal
        try:
            if frame_type == _MESSAGE_STARTED_TYPE:
                task = asyncio.create_task(self._handle_started(run_id))
                self._start_tasks[run_id] = task
            else:
                task = asyncio.create_task(self._handle_completed(run_id, project_id))
        except RuntimeError:  # pragma: no cover - no running loop (sync tests)
            logger.warning("no event loop to schedule a sandbox diff snapshot on")
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    def turn_boundary_signal(frame: Any) -> tuple[str, str, str | None] | None:
        """`(frame_type, run_id, project_id)` for a pairable turn boundary, else None."""
        if not isinstance(frame, dict):
            return None
        frame_type = frame.get("type")
        if frame_type not in (_MESSAGE_STARTED_TYPE, _MESSAGE_COMPLETED_TYPE):
            return None
        run_id = frame.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return None
        project_id = frame.get("project_id")
        return (frame_type, run_id, project_id if isinstance(project_id, str) else None)


    async def _handle_started(self, run_id: str) -> None:
        """Take the before-snapshot for one turn and remember it by run_id."""
        snapshot = await self._snapshot(self._sandbox_root)
        self._pending[run_id] = snapshot
        self._pending_order.append(run_id)
        while len(self._pending_order) > self._max_pending_runs:
            stale = self._pending_order.popleft()
            self._pending.pop(stale, None)
            self._start_tasks.pop(stale, None)

    async def _handle_completed(self, run_id: str, project_id: str | None) -> None:
        """Take the after-snapshot, diff against the paired before, ingest."""
        start_task = self._start_tasks.pop(run_id, None)
        # Both handlers are independent tasks: the before-snapshot must land before it is read.
        if start_task is not None and not start_task.done():
            with contextlib.suppress(Exception):
                await start_task
        if run_id not in self._pending:
            return
        before = self._pending.pop(run_id)
        with contextlib.suppress(ValueError):
            self._pending_order.remove(run_id)
        if before is None:
            logger.warning(
                "sandbox diff auto-ingest: the before-listing for run %s failed; "
                "skipping the diff for this turn (any new files it produced are "
                "not auto-promoted -- POST /api/sandbox/promote can pick them up "
                "manually)",
                run_id,
            )
            return
        after = await self._snapshot(self._sandbox_root)
        if after is None:
            logger.warning(
                "sandbox diff auto-ingest: the after-listing for run %s failed; "
                "skipping the diff for this turn",
                run_id,
            )
            return
        new_or_changed = self._new_or_changed_paths(before, after)
        denied = [p for p in new_or_changed if self._denylist.denies_file(p)]
        if denied:
            logger.info(
                "sandbox diff auto-ingest: skipped %d Hermes-internal path(s) "
                "for run %s (B-61 denylist): %s",
                len(denied),
                run_id,
                ", ".join(sorted(denied)),
            )
        for path in new_or_changed:
            if self._denylist.denies_file(path):
                continue
            await self._ingest_one(path, run_id, project_id)

    @staticmethod
    def _new_or_changed_paths(
        before: dict[str, tuple[Any, Any]], after: dict[str, tuple[Any, Any]]
    ) -> list[str]:
        """Every path in `after` that is new, or whose `(size, mtime)` tuple
        differs from `before` -- decision 3, module docstring: both cases are
        handed to the identical `ingest()` call and let its checksum-dedup
        decide whether the bytes actually changed."""
        return [path for path, meta in after.items() if before.get(path) != meta]

    async def _ingest_one(self, path: str, run_id: str, project_id: str | None) -> None:
        """One diffed-in path through the shared ingestion pipeline."""
        try:
            outcome = await self._ingestor.ingest(
                path, producing_run_id=run_id, project_id=project_id
            )
        except Exception:  # pragma: no cover - defensive, same contract as ArtifactIngestor
            logger.exception(
                "sandbox diff auto-ingest crashed for %r (run %s); no row was written",
                path,
                run_id,
            )
            return
        if outcome.ok:
            logger.info(
                "sandbox diff auto-ingested %r -> artifact %s (%s)",
                path,
                outcome.artifact.get("id"),
                "deduplicated" if outcome.deduplicated else "new",
            )
        else:
            logger.warning(
                "sandbox diff auto-ingest of %r could not fetch the bytes (%s); "
                "recorded artifact %s as unavailable",
                path,
                outcome.error,
                outcome.artifact.get("id"),
            )


    async def _snapshot(self, root: str) -> dict[str, tuple[Any, Any]] | None:
        """One full listing of the tree under `root`: `path -> (size, mtime)`."""
        snapshot: dict[str, tuple[Any, Any]] = {}
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        visited_dirs = 0
        while queue:
            if visited_dirs >= self._max_scan_dirs:
                break
            path, depth = queue.popleft()
            entries = await self._list_one(path)
            visited_dirs += 1
            if entries is None:
                if path == root:
                    return None
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_path = entry.get("path")
                if not isinstance(entry_path, str) or not entry_path:
                    continue
                if entry.get("is_directory"):
                    if self._denylist.denies_dir(entry_path):
                        continue
                    if depth < self._max_scan_depth:
                        queue.append((entry_path, depth + 1))
                    continue
                snapshot[entry_path] = (entry.get("size"), entry.get("mtime"))
        return snapshot

    async def _list_one(self, path: str) -> list[Any] | None:
        """One `files_list` call -> its `entries` list, or `None` on failure."""
        try:
            response = await self._sandbox_fs.files_list(path)
        except HermesError as exc:
            logger.warning("sandbox diff snapshot: files_list(%r) failed: %s", path, exc)
            return None
        if response.status_code != 200:
            logger.warning(
                "sandbox diff snapshot: files_list(%r) answered HTTP %d",
                path,
                response.status_code,
            )
            return None
        try:
            body = response.json()
        except ValueError:
            logger.warning("sandbox diff snapshot: files_list(%r) returned non-JSON", path)
            return None
        entries = body.get("entries") if isinstance(body, dict) else None
        if not isinstance(entries, list):
            logger.warning(
                "sandbox diff snapshot: files_list(%r) had no usable 'entries' list", path
            )
            return None
        return entries


# Both whole-segment frames are scanned: a tool-using turn's completion carries only the last one.
_MESSAGE_INTERIM_TYPE = "message.interim"


# Line-anchored so prose cannot match; the app's own tag parser mirrors these rules exactly.
_MEDIA_TAG_RE = re.compile(r"^(?:\*\*|__)?\s*media:\s*(?:\*\*|__)?\s*(.*)$", re.IGNORECASE)

_LIST_MARKER_RE = re.compile(r"^(?:[-*+•]|\d+[.)])\s+")

_MARKDOWN_LINK_RE = re.compile(r"^\[[^\]]*\]\(\s*(.*?)\s*\)$")

_WRAPPERS = ("`", "**", "__")

_TRAILING_PUNCTUATION = ".,;:!?"


# A bare path line counts as a delivery only with one of these, matching the app's viewer.
MEDIA_PATH_EXTENSIONS: frozenset[str] = frozenset({
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus",
    "pdf", "md", "markdown", "html", "htm", "txt", "log", "csv", "json",
    "png", "jpg", "jpeg", "gif", "webp", "heic",
    "mp4", "m4v", "mov", "webm",
})  # fmt: skip

_EXTENSION_BOUNDARY_RE = re.compile(r"\.([A-Za-z0-9]+)(?=$|\s|[,;:!?)|])")

_DESCRIPTION_LEADERS = "(-—–:,;|[)"


@dataclass(frozen=True)
class MediaReference:
    """One file the model pointed the operator at inside a message text."""

    path: str
    explicit: bool


def _unwrap_pairs(line: str) -> str:
    """Strip up to two layers of wrapping `\\``/`**`/`__` pairs."""
    for _ in range(2):
        for wrapper in _WRAPPERS:
            width = len(wrapper)
            if len(line) > 2 * width and line.startswith(wrapper) and line.endswith(wrapper):
                line = line[width:-width].strip()
                break
        else:
            return line
    return line


def _normalize_line(raw_line: str) -> str:
    line = raw_line.strip()
    line = _LIST_MARKER_RE.sub("", line, count=1)
    return _unwrap_pairs(line)


def has_media_extension(path: str) -> bool:
    """True when `path`'s extension is one the app can route to a viewer."""
    name = posixpath.basename(path)
    _, dot, ext = name.rpartition(".")
    return bool(dot) and ext.lower() in MEDIA_PATH_EXTENSIONS


def _split_at_extension(path: str) -> tuple[str, str]:
    """`("/x.ogg", " (overview)")` -- the path up to its first
    viewer-openable extension boundary, and whatever followed it."""
    for match in _EXTENSION_BOUNDARY_RE.finditer(path):
        if match.group(1).lower() in MEDIA_PATH_EXTENSIONS:
            return path[: match.end()], path[match.end() :]
    return path, ""


def _path_from_remainder(remainder: str, *, explicit: bool) -> str | None:
    """The absolute path a tag remainder / bare line carries, or None."""
    candidate = _unwrap_pairs(remainder.strip())
    link = _MARKDOWN_LINK_RE.match(candidate)
    if link:
        candidate = link.group(1)
    if len(candidate) > 2 and candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1].strip()
    candidate = candidate.rstrip(_TRAILING_PUNCTUATION)
    if candidate.count(")") > candidate.count("("):
        candidate = candidate.rstrip(")").rstrip(_TRAILING_PUNCTUATION)
    if not candidate.startswith("/") or "\x00" in candidate:
        return None
    path, rest = _split_at_extension(candidate)
    if explicit:
        return path
    if not has_media_extension(path):
        return None
    rest = rest.lstrip()
    if rest and rest[0] not in _DESCRIPTION_LEADERS:
        return None
    return path


def extract_media_references(text: str) -> tuple[list[MediaReference], list[str]]:
    """`(references, malformed_lines)` from one message text."""
    references: list[MediaReference] = []
    malformed: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        tag = _MEDIA_TAG_RE.match(line)
        if tag:
            path = _path_from_remainder(tag.group(1), explicit=True)
            if path is None:
                malformed.append(raw_line)
                continue
            explicit = True
        else:
            if not line.startswith("/") and not line.startswith("["):
                continue
            path = _path_from_remainder(line, explicit=False)
            if path is None:
                continue
            explicit = False
        if path not in seen:
            seen.add(path)
            references.append(MediaReference(path=path, explicit=explicit))
        elif explicit:
            for i, ref in enumerate(references):
                if ref.path == path and not ref.explicit:
                    references[i] = MediaReference(path=path, explicit=True)
    return references, malformed


def extract_media_tag_paths(text: str) -> tuple[list[str], list[str]]:
    """`(paths, malformed_lines)` from one message text -- the path-only
    view of `extract_media_references` (tagged and bare alike)."""
    references, malformed = extract_media_references(text)
    return [ref.path for ref in references], malformed


class MediaTagIngestor:
    """Watches message text for `MEDIA:/abs/path` tags; ingests each file."""

    def __init__(
        self,
        ingestor: ArtifactIngestor,
        *,
        sandbox_root: str,
        denylist: SandboxDiffDenylist | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._sandbox_root = sandbox_root
        self._denylist = denylist or SandboxDiffDenylist(sandbox_root)
        self._tasks: set[asyncio.Task[None]] = set()


    async def run(self, broadcaster: Any) -> None:
        """Consume the broadcaster's stream forever; never raises past itself."""
        while True:
            try:
                async with broadcaster.subscribe() as queue:
                    while True:
                        frame = await queue.get()
                        if isinstance(frame, dict) and frame.get("type") == _DESYNCHRONIZED_TYPE:
                            logger.warning(
                                "media tag ingestor fell behind the event stream "
                                "and was desynchronized; resubscribing (MEDIA: "
                                "tags in the gap were not auto-ingested)"
                            )
                            break
                        self.observe(frame)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("media tag ingestor subscriber failed; resubscribing")
            await asyncio.sleep(1.0)

    def observe(self, frame: Any) -> None:
        """Look at one forwarded frame; maybe schedule ingestions."""
        signal = self.media_reference_signal(frame)
        if signal is None:
            return
        references, run_id, project_id = signal
        for ref in references:
            try:
                task = asyncio.create_task(
                    self._ingest_one(ref.path, run_id, project_id, explicit=ref.explicit)
                )
            except RuntimeError:  # pragma: no cover - no running loop (sync tests)
                logger.warning("no event loop to schedule a media-tag ingestion on")
                return
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    @classmethod
    def media_tag_signal(cls, frame: Any) -> tuple[list[str], str | None, str | None] | None:
        """`(paths, run_id, project_id)` if this frame carries MEDIA tags or
        bare media paths, else None. The path-only view of
        `media_reference_signal`, kept for callers that do not care whether
        a reference was an explicit tag."""
        signal = cls.media_reference_signal(frame)
        if signal is None:
            return None
        references, run_id, project_id = signal
        return [ref.path for ref in references], run_id, project_id

    @staticmethod
    def media_reference_signal(
        frame: Any,
    ) -> tuple[list[MediaReference], str | None, str | None] | None:
        """`(references, run_id, project_id)` if this frame carries MEDIA
        tags or bare media paths, else None.
        """
        if not isinstance(frame, dict) or frame.get("type") not in (
            _MESSAGE_COMPLETED_TYPE,
            _MESSAGE_INTERIM_TYPE,
        ):
            return None
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            return None
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            if payload.get("_truncated") is True:
                logger.warning(
                    "a %s frame arrived truncated (B-02) with no text; any "
                    "MEDIA: tags in it were not detected -- the P3-9 diff or "
                    "POST /api/sandbox/promote can pick the files up",
                    frame.get("type"),
                )
            return None
        references, malformed = extract_media_references(text)
        for line in malformed:
            logger.warning(
                "ignoring a malformed MEDIA: tag line (no absolute path): %r",
                line,
            )
        if not references:
            return None
        run_id = frame.get("run_id")
        project_id = frame.get("project_id")
        return (
            references,
            run_id if isinstance(run_id, str) else None,
            project_id if isinstance(project_id, str) else None,
        )


    async def _ingest_one(
        self,
        path: str,
        run_id: str | None,
        project_id: str | None,
        *,
        explicit: bool = True,
    ) -> None:
        """One MEDIA-tagged (or bare, B-182) path through `ArtifactIngestor.ingest()`."""
        source = "media_tag" if explicit else "media_path"
        if self._denylist.denies_file(path):
            logger.info(
                "MEDIA: tag names %r, which matches the B-61 sandbox-diff "
                "denylist -- ingesting anyway (an explicit tag outranks the "
                "churn filter), but the model tagging a Hermes-internal file "
                "is odd",
                path,
            )
        try:
            outcome = await self._ingestor.ingest(
                path,
                producing_run_id=run_id,
                project_id=project_id,
                extra_metadata={"source": source},
            )
        except HTTPException as exc:
            if not explicit:
                logger.info(
                    "bare media path %r is outside the sandbox (%s); treated as a "
                    "mention, not ingested",
                    path,
                    exc.detail,
                )
                return
            logger.warning(
                "MEDIA: tag path %r rejected by sandbox confinement (%s); not ingested",
                path,
                exc.detail,
            )
            self._ingestor._record_failure_row(
                path,
                run_id,
                project_id,
                f"path validation failed: {exc.detail}",
                extra_metadata={"source": source},
            )
            return
        except Exception:  # pragma: no cover - defensive, must never crash
            logger.exception("media-tag auto-ingest crashed for %r; no row was written", path)
            return
        if outcome.ok:
            logger.info(
                "media-tag auto-ingested %r -> artifact %s (%s)",
                path,
                outcome.artifact.get("id"),
                "deduplicated" if outcome.deduplicated else "new",
            )
        else:
            logger.warning(
                "media-tag auto-ingest of %r could not fetch the bytes (%s); "
                "recorded artifact %s as unavailable",
                path,
                outcome.error,
                outcome.artifact.get("id"),
            )
