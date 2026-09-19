"""Artifact ingestion (P3-1 / P3-9 / P3-10): the three broadcaster subscribers.

Moved out of `api/artifacts.py` (CLEANUP_PLAN step 3.4): `ArtifactIngestor`
(structured-path auto-ingest + the shared fetch/store/row pipeline manual
promotion reuses), `SandboxDiffDenylist` + `SandboxDiffIngestor` (before/after
sandbox-listing diff, B-61 denylist) and `MediaTagIngestor` (`MEDIA:` tags).
The store they write to is `domain/artifact_store.py`; the routes are still
`api/artifacts.py`.

## What auto-ingests, and what deliberately does not (P3-1)

The `ArtifactIngestor` is an **internal subscriber to the broadcaster**,
NOT a hook inside
`_forward_one`'s hot path. It watches forwarded `tool.completed` frames and
auto-ingests only tools with a **structured path signal**: `write_file` and
`patch`, whose `args.path` is machine-readable (PV "File-edit wire shape").
`terminal`-produced files are explicitly NOT auto-ingested: the P3-0a probe
measured that `terminal`'s completion carries only `command` + `output` --
no path field, no bytes -- and a stdout-parsing heuristic would be fragile.
The answer for those is manual promotion (`POST /api/sandbox/promote`, the
route the app's "Save to project" calls -- P3-4a).

Known, accepted gap: the subscriber sees the same bounded frames as every
client, so a `write_file` completion whose payload blew the B-02 frame
budget (`args.content` *is* the whole file) arrives `_truncated` with its
`args` dropped -- no path signal, no auto-ingest. Logged when seen; manual
promotion covers it.

Attribution rides in on the envelope itself: the `RunRecorder` hook stamps
`run_id` / `project_id` onto every forwarded frame *before* fan-out
(P2-2a), so `producing_run_id` is taken from `frame["run_id"]` -- the
current run when resolvable, honestly NULL otherwise (unfiled/unattributed
sessions, P3-1.0's nullable-project rationale in `domain/models.py`).

## Failure honesty (P3-1)

A fetch failure (upstream 403/404, transport error, the ingest size cap) is
**never silent**: the row is still written, `status: unavailable`,
`storage_key`/`checksum` NULL, with the error recorded in `metadata_json` --
same pattern as Phase 2a's orphaned ledger. An `available` row is never
flipped to `unavailable` (its stored bytes remain good); repeated failures
for one path touch a single unavailable row rather than piling up new ones.

## Dedup (P3-1)

Same `source_path` + same checksum => the existing row is updated/touched,
never duplicated. A *different* checksum for the same path is a new version
of the file and gets its own row (the old row's bytes are still in the
content-addressed store and still correct). A successful fetch first fills
in a pending `unavailable` row for the same path, if one exists, instead of
leaving it dangling beside a fresh `available` row.

## Before/after sandbox-listing diff auto-promotion (P3-9, TASKS.md, B-57)

`ArtifactIngestor` above only auto-ingests `write_file`/`patch` -- the two
tools with a structured `args.path`. Everything a `terminal` script
produces (audio, PDFs -- anything binary, since `write_file`'s `content` is
JSON text) has no such signal on the wire (P3-0a probe, PV "Phase 3 probe")
and previously required a manual "Save to project" tap. `SandboxDiffIngestor`
is the fix: a **sibling subscriber**, same broadcaster pattern, that snapshots
the sandbox directory tree at the start of a turn and again at the end, and
runs anything new-or-changed through the same `ArtifactIngestor.ingest()`
pipeline used everywhere else.

Four decisions this closes (TASKS.md left them open):

1. **Which directory.** `settings.hermes_sandbox_root` (`/opt/data`), same as
   P3-1a's Tier-1 browse -- always correct, and it does not depend on a
   per-session `cwd`. (Correction, 2026-09-01 review: `session.info.cwd` IS
   on the wire and this comment used to say it was unreadable. Measured, it
   reports `/opt/data` -- the root this walk already starts at -- so the
   decision below is unchanged; only the reason was wrong.) Not a single
   flat listing,
   though: the exact real-world case that produced the P3-0a probe evidence
   (a `terminal` script writing into `/opt/data/probe_scratch/...`, a
   *subdirectory* of the root) would be invisible to a non-recursive
   root-only listing. `_snapshot()` instead walks the tree breadth-first via
   repeated `files_list()` calls, bounded by `DIFF_SCAN_MAX_DIRS` (total
   directories listed) and `DIFF_SCAN_MAX_DEPTH` (levels below root) so a
   pathological tree cannot make one turn boundary cost an unbounded number
   of upstream calls.
2. **When.** `message.started` takes the before-listing, `message.completed`
   takes the after-listing and diffs -- exactly the spec's two extra
   `files_list` calls per turn (per directory visited; one directory in the
   common case). The existing write_file/patch structured-path ingestion is
   **not** changed to skip itself when the diff would also catch the same
   file: both routes end in the same `ArtifactIngestor.ingest()` call, which
   is already serialized on one `asyncio.Lock` and already dedups by
   `(source_path, checksum)` (`_write_success_row`) -- whichever ingestion
   wins the race creates the row, the other's identical checksum touches it
   instead of duplicating it (this is exactly what
   `test_same_path_same_checksum_touches_not_duplicates` already proves for
   two ordinary `ingest()` calls; the diff path is just a third caller of the
   same method). Verified, not assumed. Separately: the transcript **chip**
   the app renders for `write_file`/`patch` (`ConversationModel.artifactChip`)
   is derived entirely client-side from that tool row's own `args.path` --
   it has no dependency on the artifact row or this ingestor at all, so
   there is no "double chip" risk here regardless of ingestion race outcome.
3. **New vs. modified.** A path present after but absent before is new,
   ingested. A path present in **both** snapshots whose `(size, mtime)`
   differs is *also* re-ingested through the identical `ingest()` call,
   rather than being treated specially: the checksum-dedup path above
   already answers "is this actually different bytes" correctly -- same
   checksum touches the existing row (a `touch`/mtime bump with unchanged
   content costs nothing extra), a different checksum creates a new version
   row (`test_changed_content_is_a_new_version_row`'s existing contract).
   Reusing the proven rule beats inventing a second one.
4. **Concurrency.** Pairing is keyed by `run_id`, not a single global
   before/after or a session id. `run_id` is the id
   `api/runs.py::RunRecorder` already assigns per turn and stamps onto
   *every* frame of that turn -- including `message.started` and
   `message.completed` -- before any subscriber (this one included) ever
   sees it (P2-2a). Two turns, even concurrent ones in different sessions,
   get different `run_id`s for free; a `message.started`/`message.completed`
   pair for one turn always carries the same `run_id`. A frame with no
   `run_id` (unattributed) cannot be safely paired and is skipped rather than
   guessed. The one same-`run_id` race that IS possible -- the `complete`
   handler reading `_pending` before the `started` handler has finished
   writing its snapshot into it, since both are independently scheduled
   asyncio tasks -- is closed by having `_handle_completed` `await` the
   in-flight start task (if any) before reading `_pending`, the same
   "track the in-flight task, await it, never race the dict" shape the
   background ledger and `ArtifactIngestor`'s own `self._lock` already use
   elsewhere in this codebase.

A fifth decision arrived later, the hard way (B-61): `/opt/data` is Hermes's
ENTIRE operational data root, so the raw diff swept up Hermes's own churn
(logs, SQLite WALs, heartbeats) on every turn -- multi-MB, differently
checksummed each time, misattributed, unbounded. `SandboxDiffDenylist`
(constants + full rationale at its definition below) filters the diff BEFORE
ingestion; the denylist applies to this diff path ONLY, never to manual
promotion or write_file/patch structured-path ingestion.

## `MEDIA:` tag detection (P3-10, B-59)

Hermes has a native convention for "this file is for the user": the model
emits a line `MEDIA:/abs/path` inside its ordinary message text (confirmed
real over our TUI Gateway connection, unprompted, in owner use -- B-59 /
P3-10a: two tags in one message, `.../CL20_REPORT.md` and
`.../CL20_overview.ogg`). `MediaTagIngestor` is the FOURTH broadcaster
subscriber: it scans the text of canonical `message.completed` frames (and
`message.interim` frames -- B-38: `message.complete` carries only the final
segment of a multi-segment turn, so a tag emitted in commentary alongside a
tool call would otherwise be lost) for line-anchored `MEDIA:` tags and hands
each referenced path to the same `ArtifactIngestor.ingest()` pipeline, with
`{"source": "media_tag"}` merged into the row's metadata so provenance is
honest. Runs ALONGSIDE the P3-9 diff (tag = precise, intentional signal when
present; diff = catch-all when absent); the shared `(source_path, checksum)`
dedup means a file both routes see gets one row, never two. The B-61
denylist does NOT apply here -- an explicit MEDIA: tag is an intentional
signal and wins over the denylist (logged when it would have matched, since
the model flagging Hermes-internal churn would be odd).

B-182 (2026-09-13): the model is inconsistent about the tag -- `Media:` in
mixed case, `MEDIA: /path` with a space, `**MEDIA:** /path` in bold, a list
bullet in front, and most often the bare absolute path dropped on its own
line with no tag at all. `extract_media_references` tolerates every shape
seen and treats a bare path line as a reference when its extension is one
the app can open (`MEDIA_PATH_EXTENSIONS`). A bare path is `explicit=False`:
ingested like a tag inside the sandbox root, but merely logged (no
`unavailable` row) when it points outside it -- a mention, not a delivery.
The app's `MediaTag.swift` mirrors every rule; the two must move together.
"""

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
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

#: The two tools with a machine-readable path in `args` (PV "File-edit wire
#: shape"). `terminal` is deliberately absent -- see module docstring.
AUTO_INGEST_TOOL_NAMES: frozenset[str] = frozenset({"write_file", "patch"})

#: Canonical type the ingestor triggers on (events/canonical.py maps Hermes's
#: raw `tool.complete` to this).
_TOOL_COMPLETED_TYPE = "tool.completed"

#: B-33 cut-off frame type -- the subscriber resubscribes on it (the
#: broadcaster's own constant; this used to be a by-literal copy to dodge an
#: import cycle with `api.main`).
_DESYNCHRONIZED_TYPE = STREAM_DESYNCHRONIZED_EVENT_TYPE


# ---------------------------------------------------------------------------
# Ingestion (P3-1): fetch + store + row, shared by auto-ingest and promotion
# ---------------------------------------------------------------------------


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
    """Fetches sandbox files into the durable library and records the rows.

    One instance per app, built by `api.main.lifespan`. Two entry points:

    * `run(broadcaster)` -- the broadcaster-subscriber loop: watches `tool.completed` frames for the structured-path
      tools and schedules `ingest()` for each, off the consume path.
    * `ingest(source_path, ...)` -- one full validate -> fetch -> store ->
      row cycle, also called directly by `POST /api/sandbox/promote`.

    All ingestions are serialized on one internal lock: two `patch`
    completions for the same file land in order, and the dedup lookup can
    never race a concurrent insert for the same path.
    """

    def __init__(
        self,
        adapter: HermesAdapter,
        session_factory: sessionmaker,
        store: ArtifactStore,
        *,
        sandbox_root: str,
    ) -> None:
        self._adapter = adapter
        self._session_factory = session_factory
        self._store = store
        self._sandbox_root = sandbox_root
        self._lock = asyncio.Lock()
        # In-flight ingest tasks, held so they are not garbage-collected
        # mid-fetch; each removes itself on completion (held-task-set pattern).
        self._tasks: set[asyncio.Task[None]] = set()

    # -- the subscriber loop ----------------------------------------------

    async def run(self, broadcaster: Any) -> None:
        """Consume the broadcaster's stream forever; never raises past itself.

        Started by `api.main.lifespan`, cancelled at shutdown. Falling behind
        (`stream.desynchronized`) or any internal failure resubscribes for a
        fresh queue; completions lost in the gap are lost auto-ingests --
        recoverable by manual promotion, never a crashed task.
        """
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
        """Look at one forwarded frame; maybe schedule one ingestion.

        Synchronous and I/O-free (the consume-path contract every internal
        subscriber keeps): the fetch happens on a separate task.
        """
        signal = self.ingest_signal(frame)
        if signal is None:
            return
        source_path, producing_run_id, project_id = signal
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

        The structured-path contract, measured (PV "File-edit wire shape"):
        `write_file`/`patch` completions carry `args.path`. Attribution comes
        off the envelope keys the RunRecorder already stamped -- honest nulls
        when the run/project could not be resolved.
        """
        if not isinstance(frame, dict) or frame.get("type") != _TOOL_COMPLETED_TYPE:
            return None
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            return None
        if payload.get("_truncated") is True:
            # B-02 degraded frame: `args` (and its path) were dropped with the
            # payload. Say so -- the file exists but cannot be auto-ingested.
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
            # Validation refused the path before any fetch (outside the
            # sandbox root / malformed). Never silent (P3-1): record the
            # unavailable row here, since ingest() had nothing to record.
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

    # -- the shared ingestion cycle ---------------------------------------

    async def ingest(
        self,
        source_path: str,
        *,
        producing_run_id: str | None = None,
        project_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestOutcome:
        """Validate -> fetch -> store -> row. The whole P3-1 pipeline, once.

        Raises `HTTPException` (422/403) only for a path that fails the
        gateway-side sandbox confinement -- the same rule as
        `api/sandbox.py`, applied BEFORE any upstream call. Every failure
        *after* validation is not an exception: it is an `IngestOutcome`
        whose row has `status: unavailable` and whose `error` says why.

        `extra_metadata` (P3-10): honest-provenance keys merged into the
        row's `metadata_json` -- e.g. `{"source": "media_tag"}` when the
        model explicitly flagged the file with a `MEDIA:` line. Merged, not
        assigned: a later touch by an ingestion path with no extra metadata
        (the diff, a manual promote) never erases what an earlier explicit
        signal recorded.
        """
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
                    except Exception as exc:  # transport died mid-stream, disk full
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

    # -- row writing (sync SQLite, same policy as domain/db.py) ------------

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
        """`_write_success_row`, with dangling provenance degraded, not fatal.

        `producing_run_id`/`project_id` come off a forwarded frame; if either
        names a row that does not exist (a run whose open-insert failed, a
        project deleted mid-turn), the FK would reject the whole artifact row
        -- and losing the row over its *decoration* would be the silent-loss
        P3-1 forbids. Retried once with NULL provenance, logged.
        """
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
            # A success clears any recorded error, but keeps (and merges in)
            # honest-provenance keys: a `source: media_tag` recorded by an
            # earlier explicit MEDIA: signal survives a later diff/promote
            # touch, and vice versa (P3-10).
            kept = {
                k: v
                for k, v in (artifact.metadata_json or {}).items()
                if k not in ("ingest_error", "failed_at")
            }
            kept.update(extra_metadata or {})
            artifact.metadata_json = kept or None
            # Provenance/filing fill in when absent but are never stolen: a
            # re-touch by a later run keeps the original producer.
            if artifact.producing_run_id is None:
                artifact.producing_run_id = producing_run_id
            if artifact.project_id is None:
                artifact.project_id = project_id
            db.commit()
            return _artifact_json(artifact), deduplicated

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
            # Touch the one pending unavailable row rather than piling up;
            # never flip an available row (its stored bytes are still good).
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


# ---------------------------------------------------------------------------
# Before/after sandbox-listing diff auto-promotion (P3-9)
# ---------------------------------------------------------------------------

#: Turn boundary types this subscriber pairs on (events/canonical.py's
#: RAW_TO_CANONICAL_TYPE: message.start -> message.started,
#: message.complete -> message.completed).
_MESSAGE_STARTED_TYPE = "message.started"
_MESSAGE_COMPLETED_TYPE = "message.completed"

#: Bound on one snapshot walk: total directories `files_list`ed, and levels
#: below the root a discovered subdirectory will still be descended into.
#: Sized to comfortably cover a scratch dir or two of nested terminal
#: output, not an arbitrarily deep or wide tree -- see
#: `SandboxDiffIngestor._snapshot`.
DIFF_SCAN_MAX_DIRS = 64
DIFF_SCAN_MAX_DEPTH = 4

#: Bound on in-flight paired snapshots, keyed by `run_id`. A turn whose
#: `message.completed` never arrives (a crash, a `stream.desynchronized`
#: gap) would otherwise leak its before-snapshot for the life of the
#: process; oldest-evicted (bounded memory).
MAX_PENDING_DIFF_RUNS = 64

# -- B-61: the Hermes-internal churn denylist --------------------------------
#
# `hermes_sandbox_root` (`/opt/data`) is not a user-scratch-only tree -- it is
# Hermes's ENTIRE live operational data root. Measured live (B-61, 2026-08-31,
# 3/3 turns across two fresh sessions): every single before/after diff swept
# up Hermes's own churn -- `logs/agent.log` (~4MB), `logs/errors.log`
# (~1.3MB), `logs/gui.log`, `state.db-wal` (~10MB), `lcm.db-wal` (~4.6MB),
# `kanban.db-wal`, `cron/ticker_heartbeat`, `cron/ticker_last_success`,
# `cron/.tick.lock`, `state/gateway.heartbeat`, `channel_directory.json`,
# `cache/local_endpoint_probes.json` -- none of it produced by, or relevant
# to, the user's turn. Each mutates every turn, so checksum-dedup never
# collapses it: multi-MB of freshly-checksummed rows per turn, misattributed
# to whatever project was active, growing the store unbounded.
#
# The diff ingestor therefore refuses these paths BEFORE ingestion, on two
# axes:
#
# * `SANDBOX_DIFF_DENYLIST_DIRS` -- top-level directories under the sandbox
#   root that are Hermes's own operational trees, never user output. The
#   snapshot walk does not even descend into them (saves `files_list` calls
#   and the `DIFF_SCAN_MAX_DIRS` budget for real user directories).
# * `SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS` -- filenames that are operational
#   state wherever they live: SQLite databases and their WAL/shm siblings,
#   lockfiles, heartbeat/last-success markers, Hermes's channel directory.
#   These catch the root-level churn (`state.db-wal` et al.) and survive a
#   Hermes layout shuffle that moves them into new directories.
#
# Extending it: add entries to the constants below, or -- for a Hermes layout
# change that should not wait for a code edit -- set the
# `HERMES_SANDBOX_DENYLIST_DIRS` / `HERMES_SANDBOX_DENYLIST_GLOBS` env
# overrides (settings.py; comma-separated; a non-empty override REPLACES the
# corresponding built-in default, it does not append).
#
# Denylist, not allowlist -- considered and decided (B-61): the alternative
# is an allowlist of known user-facing subtrees (`scratch/`, `probe_scratch/`,
# `attachments/`). Rejected for now because `docs/PROTOCOL_VERIFIED.md`
# documents no stable user-facing subtree, and terminal scripts write
# wherever the agent chooses (measured: `/opt/data/scratch/`,
# `/opt/data/probe_scratch/`, root-level files). An allowlist fails CLOSED on
# anything unanticipated -- a real user artifact in a new location silently
# never auto-promotes (invisible loss, the exact silence P3-1 forbids); this
# denylist fails OPEN -- worst case is a junk row that is visible in the
# library, manually deletable, and self-documenting. Revisit the allowlist if
# PV ever documents a guaranteed user-output subtree.
#
# Scope: the DIFF ingestor only. Manual promotion (`POST /api/sandbox/promote`)
# and `write_file`/`patch` structured-path auto-ingest are deliberately
# untouched -- an explicit user request to promote `/opt/data/logs/agent.log`,
# or an agent explicitly `write_file`-ing into one of these trees, is a real
# signal and must still work.

#: Top-level directory names under the sandbox root the diff never enters.
SANDBOX_DIFF_DENYLIST_DIRS: frozenset[str] = frozenset({"logs", "cron", "state", "cache"})

#: Filename patterns (fnmatch, against the basename, case-sensitive) the diff
#: never promotes, wherever they live under the root.
SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS: tuple[str, ...] = (
    "*.db",  # SQLite databases (state.db -- one already 413'd, B-61)
    "*.db-wal",  # SQLite write-ahead logs (state.db-wal, lcm.db-wal, kanban.db-wal)
    "*.db-shm",  # SQLite shared-memory siblings
    "*.lock",  # lockfiles (cron/.tick.lock)
    "*heartbeat*",  # heartbeat markers (ticker_heartbeat, gateway.heartbeat)
    "*_last_success",  # cron success markers (ticker_last_success)
    "channel_directory.json",  # Hermes's live channel directory
)


def _denylist_csv(raw: str) -> tuple[str, ...]:
    """A comma-separated env override -> a tuple of entries (empty -> empty).

    Directory entries tolerate a trailing slash (`logs/` == `logs`)."""
    return tuple(item.strip().strip("/") for item in raw.split(",") if item.strip().strip("/"))


class SandboxDiffDenylist:
    """Decides which sandbox paths the diff ingestor must never promote (B-61).

    Built from the module defaults above, or from the settings overrides via
    `from_settings()`. Purely lexical: paths are compared against the sandbox
    root with `posixpath` semantics (the same alphabet
    `validate_sandbox_path` uses), no filesystem access.
    """

    def __init__(
        self,
        root: str,
        dirs: frozenset[str] | set[str] | tuple[str, ...] = SANDBOX_DIFF_DENYLIST_DIRS,
        filename_globs: tuple[str, ...] = SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS,
    ) -> None:
        self._root = posixpath.normpath(root)
        self._dirs = frozenset(dirs)
        self._globs = tuple(filename_globs)

    @classmethod
    def from_settings(cls, settings: Any) -> SandboxDiffDenylist:
        """The deployed construction (api.main): built-in defaults unless the
        corresponding env override is non-empty, in which case the override
        REPLACES that default wholesale (a layout change is a config edit)."""
        dirs = _denylist_csv(settings.hermes_sandbox_denylist_dirs) or SANDBOX_DIFF_DENYLIST_DIRS
        globs = (
            _denylist_csv(settings.hermes_sandbox_denylist_globs)
            or SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS
        )
        return cls(settings.hermes_sandbox_root, frozenset(dirs), tuple(globs))

    def _top_level_component(self, path: str) -> str | None:
        """The first path component below the root, or None (the root itself,
        or a path not under the root at all -- the latter is someone else's
        problem: `validate_sandbox_path` already refuses it downstream)."""
        rel = posixpath.relpath(posixpath.normpath(path), self._root)
        if rel == "." or rel.startswith(".."):
            return None
        return rel.split("/", 1)[0]

    def denies_dir(self, path: str) -> bool:
        """True for a denylisted top-level directory or anything inside one."""
        return self._top_level_component(path) in self._dirs

    def denies_file(self, path: str) -> bool:
        """True when this file path must not be auto-promoted by the diff."""
        if self.denies_dir(path):
            return True
        name = posixpath.basename(path)
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self._globs)


class SandboxDiffIngestor:
    """Snapshots the sandbox tree at each turn's start/end; auto-ingests the diff.

    See the module docstring's "Before/after sandbox-listing diff
    auto-promotion (P3-9)" section for the four design decisions. Wired by
    `api.main.lifespan` as a second internal broadcaster subscriber,
    alongside `ArtifactIngestor` -- it does not fetch or store bytes itself;
    every diffed-in path is handed to an existing `ArtifactIngestor`'s
    `ingest()`, which is where the fetch/store/row/dedup pipeline actually
    lives (reused, not duplicated).
    """

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
        # The walk is the hottest filesystem path in the gateway -- hundreds
        # of listings per turn -- so it is the one that gains most from direct
        # I/O when the gateway runs inside Hermes. Defaults to the adapter's
        # HTTP routes, which is what the standalone sidecar has and must keep:
        # it does not mount the sandbox at all.
        self._sandbox_fs: SandboxFS = (
            sandbox_fs if sandbox_fs is not None else HermesHttpSandboxFS(adapter)
        )
        self._ingestor = ingestor
        self._sandbox_root = sandbox_root
        self._max_scan_dirs = max_scan_dirs
        self._max_scan_depth = max_scan_depth
        self._max_pending_runs = max_pending_runs
        # B-61: Hermes-internal churn is filtered out of every diff. None
        # (the unit-test/default construction) means the built-in defaults
        # against this root; api.main passes `from_settings(...)` so the env
        # overrides apply in the deployed process.
        self._denylist = denylist or SandboxDiffDenylist(sandbox_root)
        # run_id -> before-snapshot; None means "the before-listing failed"
        # (a real, remembered outcome -- distinct from "no entry", which
        # means no message.started was ever paired for this run_id).
        self._pending: dict[str, dict[str, tuple[Any, Any]] | None] = {}
        self._pending_order: deque[str] = deque()
        # run_id -> the in-flight "take the before-snapshot" task. Awaited by
        # the matching completion before it reads `_pending` (see
        # `_handle_completed`), which is what makes a start/complete pair for
        # the SAME run_id race-free even though both are scheduled as
        # independent asyncio tasks (decision 4, module docstring).
        self._start_tasks: dict[str, asyncio.Task[None]] = {}
        # In-flight handler tasks, held so they are not garbage-collected
        # mid-snapshot; each removes itself on completion (held-task-set pattern).
        self._tasks: set[asyncio.Task[None]] = set()

    # -- the subscriber loop ----------------------------------------------

    async def run(self, broadcaster: Any) -> None:
        """Consume the broadcaster's stream forever; never raises past itself.

        Same resubscribe-on-desync contract as `ArtifactIngestor.run()`: a
        turn boundary lost in the gap is simply never
        diffed, not a crashed task.
        """
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
        """Look at one forwarded frame; maybe schedule a snapshot/diff.

        Synchronous and I/O-free (the consume-path contract shared with
        `ArtifactIngestor.observe`): the listing calls happen on a separate
        task.
        """
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
        """`(frame_type, run_id, project_id)` for a pairable turn boundary, else None.

        The pairing key is `run_id` -- see decision 4 in the module
        docstring. A `message.started`/`message.completed` frame with no
        `run_id` (unattributed, P2-2) cannot be safely paired and is
        skipped: guessing a pairing risks matching the wrong turn's other
        end, which is worse than not diffing this one.
        """
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

    # -- turn boundary handlers ---------------------------------------------

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
        if start_task is not None and not start_task.done():
            # Race guard (decision 4): make sure the before-snapshot this
            # same run_id's `message.started` is taking has actually landed
            # in `_pending` before this method reads it.
            with contextlib.suppress(Exception):
                await start_task
        if run_id not in self._pending:
            # No before-snapshot was ever paired for this run_id -- this
            # process started mid-turn, the pairing aged out under
            # `max_pending_runs` pressure, or the before-listing itself was
            # lost to a desync gap. Nothing to diff; not an error.
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
        # B-61: refuse Hermes's own operational churn BEFORE ingestion. The
        # walk already skips denylisted top-level directories; this catches
        # the root-level files (state.db-wal, channel_directory.json, ...).
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
            # A path straight out of a listing under `sandbox_root` should
            # never fail gateway-side confinement (HTTPException) -- caught
            # broadly regardless, because this subscriber must never crash.
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

    # -- snapshotting -------------------------------------------------------

    async def _snapshot(self, root: str) -> dict[str, tuple[Any, Any]] | None:
        """One full listing of the tree under `root`: `path -> (size, mtime)`.

        Walks breadth-first via repeated `files_list()` calls, bounded by
        `max_scan_dirs`/`max_scan_depth` (see the class docstring). Returns
        `None` -- not a partial result -- only when the ROOT listing itself
        fails: that is the one failure this treats as "the whole snapshot is
        unusable", since diffing against an empty/partial view would report
        every pre-existing file as newly created. A failure on a *deeper*
        subdirectory is best-effort: that subtree is skipped (logged) and the
        walk continues with whatever it already has.
        """
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
                continue  # best-effort: skip this subtree, keep the rest
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_path = entry.get("path")
                if not isinstance(entry_path, str) or not entry_path:
                    continue
                if entry.get("is_directory"):
                    if self._denylist.denies_dir(entry_path):
                        # B-61: Hermes's own trees (logs/, cron/, state/,
                        # cache/) are never entered -- nothing under them can
                        # be promoted, and skipping the descent spends the
                        # max_scan_dirs budget on real user directories.
                        continue
                    if depth < self._max_scan_depth:
                        queue.append((entry_path, depth + 1))
                    continue
                snapshot[entry_path] = (entry.get("size"), entry.get("mtime"))
        return snapshot

    async def _list_one(self, path: str) -> list[Any] | None:
        """One `files_list` call -> its `entries` list, or `None` on failure.

        Never raises: a transport error, a non-200, or a malformed body all
        degrade to "no usable entries from this directory" -- the graceful
        failure contract this whole class must hold (a crashed listing must
        never crash the subscriber).
        """
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


# ---------------------------------------------------------------------------
# MEDIA: tag detection -> ingestion (P3-10, B-59)
# ---------------------------------------------------------------------------

#: The second message-text frame the tag scanner watches. B-38 (canonical.py):
#: a turn that uses tools produces SEVERAL assistant segments, and
#: `message.complete` carries only the LAST one -- the commentary segments
#: arrive as `message.interim`. Scanning both covers the whole reply without
#: touching `message.delta` (deltas could split a tag across frame boundaries;
#: interim/completed each carry whole segments). Segments are disjoint, so no
#: tag is seen twice -- and even a double-detection would only re-run the
#: checksum-dedup'd `ingest()`.
_MESSAGE_INTERIM_TYPE = "message.interim"

#: A MEDIA tag is LINE-anchored: the tag must start its line (leading
#: whitespace, a list bullet, a wrapping inline-code / bold pair tolerated).
#: Case-INSENSITIVE since B-182 -- the model emits `Media:` as often as
#: `MEDIA:` -- but never matched mid-line, so prose like "see MEDIA:/x
#: above" or "the MEDIA: convention" cannot over-match. The path is
#: everything after the colon, trimmed -- paths may contain spaces. A tag
#: line whose remainder is not an absolute path is MALFORMED: ignored and
#: logged, never guessed at.
#:
#: These rules are MIRRORED by the app's `MediaTag.swift` (line for line,
#: with the same fixtures in `MediaTagTests.swift`), so what the app chips
#: is exactly what the gateway ingests. Change both or neither.
#:
#: `**MEDIA:** /x`, `Media: /x`, `MEDIA:/x` -- the prefix with optional
#: bold markers on either side of it and optional whitespace after the colon.
_MEDIA_TAG_RE = re.compile(r"^(?:\*\*|__)?\s*media:\s*(?:\*\*|__)?\s*(.*)$", re.IGNORECASE)

#: A markdown list bullet in front of the tag or path: `- MEDIA:/x`,
#: `* /opt/data/x.ogg`, `1. MEDIA:/x`.
_LIST_MARKER_RE = re.compile(r"^(?:[-*+•]|\d+[.)])\s+")

#: `[label](/abs/path)` -- the model sometimes delivers a file as a link.
_MARKDOWN_LINK_RE = re.compile(r"^\[[^\]]*\]\(\s*(.*?)\s*\)$")

#: Wrapping pairs the model puts around a tag line or a path; stripped only
#: as a PAIR (a lone backtick or `**` leaves the line untouched).
_WRAPPERS = ("`", "**", "__")

#: Punctuation the model hangs off the end of a path in prose
#: ("MEDIA:/x.ogg." / "...x.ogg,"). Never part of a real file name here.
_TRAILING_PUNCTUATION = ".,;:!?"

#: File extensions that mark a bare line as a deliverable rather than
#: prose (B-182: the model sometimes drops the absolute path on its own line
#: with no tag at all). MIRRORS `ViewerItem.knownFileExtensions` in the
#: app -- every extension the app can route to a viewer. Lower-case; the
#: check is case-insensitive.
MEDIA_PATH_EXTENSIONS: frozenset[str] = frozenset({
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus",  # audio
    "pdf", "md", "markdown", "html", "htm", "txt", "log", "csv", "json",  # documents
    "png", "jpg", "jpeg", "gif", "webp", "heic",  # images
    "mp4", "m4v", "mov", "webm",  # video
})  # fmt: skip

#: The first viewer-openable extension followed by end-of-line, whitespace
#: or sentence punctuation ends the path: "/x.ogg (the overview)" and
#: "/x.ogg, plus notes" both cut at `.ogg`; "/two words.ogg" stays whole
#: because nothing follows its only extension.
_EXTENSION_BOUNDARY_RE = re.compile(r"\.([A-Za-z0-9]+)(?=$|\s|[,;:!?)|])")

#: What may follow a BARE path for the line to still count as a reference:
#: a description the model appended, never a sentence. "/x.ogg (overview)"
#: and "/x.ogg - final" are deliveries; "/x.ogg is the file" is prose.
_DESCRIPTION_LEADERS = "(-—–:,;|[)"


@dataclass(frozen=True)
class MediaReference:
    """One file the model pointed the owner at inside a message text."""

    path: str
    #: True for an explicit `MEDIA:` tag; False for a bare absolute path
    #: on its own line whose extension the app can open (B-182). The
    #: distinction matters gateway-side only: an explicit tag is a claim
    #: strong enough to record an `unavailable` row when the file cannot be
    #: fetched, a bare path outside the sandbox root is just a mention.
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
    """The absolute path a tag remainder / bare line carries, or None.

    A tag's remainder may be a spaced file name with no known extension at
    all (the model tagged it; trust it). A bare line must be a path with a
    known extension followed by nothing or a description leader.
    """
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
    """`(references, malformed_lines)` from one message text.

    `references` are, in order of appearance and de-duplicated by path:
    every well-formed `MEDIA:` tag line (any case) and every bare line that
    is nothing but an absolute path with a viewer-openable extension.
    `malformed_lines` started as a tag but carried no absolute path -- the
    caller logs them; they are never ingested.
    """
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
            # A bare mention followed by an explicit tag of the same file
            # upgrades the reference; never the other way round.
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
    """Watches message text for `MEDIA:/abs/path` tags; ingests each file.

    The FOURTH internal broadcaster subscriber (module docstring, "MEDIA: tag
    detection"). It fetches nothing itself: every tagged path goes through
    the shared `ArtifactIngestor.ingest()` (validate -> fetch -> store ->
    row -> dedup), with `{"source": "media_tag"}` recorded in the row's
    metadata. Attribution rides the frame envelope (`run_id`/`project_id`,
    stamped by the RunRecorder before fan-out), honest NULLs when absent --
    unlike the diff ingestor, a tag needs no started/completed pairing, so a
    missing `run_id` does not block ingestion.

    B-61 denylist: deliberately NOT applied -- a `MEDIA:` tag is the model
    explicitly saying "this file is for the user", which outranks a
    heuristic churn filter. A tagged path that WOULD have matched is logged
    (it would be odd) and ingested anyway.
    """

    def __init__(
        self,
        ingestor: ArtifactIngestor,
        *,
        sandbox_root: str,
        denylist: SandboxDiffDenylist | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._sandbox_root = sandbox_root
        # Used ONLY to log the odd case of a tag pointing at a denylisted
        # path -- never to refuse it (see class docstring).
        self._denylist = denylist or SandboxDiffDenylist(sandbox_root)
        # In-flight ingest tasks, held so they are not garbage-collected
        # mid-fetch; each removes itself on completion (held-task-set pattern).
        self._tasks: set[asyncio.Task[None]] = set()

    # -- the subscriber loop ----------------------------------------------

    async def run(self, broadcaster: Any) -> None:
        """Consume the broadcaster's stream forever; never raises past itself.

        Same resubscribe-on-desync contract as the other three subscribers:
        a completed message lost in a desync gap is a lost tag detection --
        the P3-9 diff (or manual promotion) covers it, never a crashed task.
        """
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
        """Look at one forwarded frame; maybe schedule ingestions.

        Synchronous and I/O-free (the consume-path contract shared with the
        sibling subscribers): the fetches happen on separate tasks.
        """
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
        tags or bare media paths (B-182), else None.

        Triggers on the two whole-segment text frames (`message.completed` +
        `message.interim` -- see `_MESSAGE_INTERIM_TYPE`'s note). Malformed
        tag lines are logged here and dropped, never guessed at.
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
                # B-02 degraded frame: the text was dropped with the payload,
                # so any MEDIA tags in it are undetectable. Say so.
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

    # -- one tagged path through the shared pipeline -----------------------

    async def _ingest_one(
        self,
        path: str,
        run_id: str | None,
        project_id: str | None,
        *,
        explicit: bool = True,
    ) -> None:
        """One MEDIA-tagged (or bare, B-182) path through `ArtifactIngestor.ingest()`.

        Every failure is contained and recorded (the never-silent P3-1
        contract): a path that fails sandbox confinement is rejected before
        any fetch, logged, and -- for an explicit tag -- still gets its
        unavailable row. A BARE path outside the sandbox root is only a
        mention (`/etc/hosts.log` in a shell transcript), so it is logged
        and dropped without a row: the row would be noise in the library.
        Inside the root, bare and explicit are treated alike -- the model
        named a sandbox file, and a 404 on it is worth recording.
        """
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
            # Validation refused the path before any fetch (outside the
            # sandbox root / malformed).
            if not explicit:
                logger.info(
                    "bare media path %r is outside the sandbox (%s); treated as a "
                    "mention, not ingested",
                    path,
                    exc.detail,
                )
                return
            # Never silent for an explicit tag: record the unavailable row,
            # since ingest() had nothing to record.
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
