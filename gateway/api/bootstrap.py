"""Startup and shutdown of every `app.state` service, in the documented order."""

from __future__ import annotations

import asyncio
import contextlib
import posixpath
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters.hermes import HermesAdapter
from config.settings import Settings
from domain.artifact_ingest import (
    ArtifactIngestor,
    MediaTagIngestor,
    SandboxDiffDenylist,
    SandboxDiffIngestor,
)
from domain.artifact_store import ArtifactStore
from domain.attachment_orchestrator import AttachmentOrchestrator
from domain.audit_store import AuditRecorder, AuditStore
from domain.background_ledger import BackgroundLedger
from domain.chat_store import ChatStore
from domain.db import make_engine, make_sessionmaker
from domain.event_stream import EventBroadcaster
from domain.foreign_prompt_capture import ForeignPromptCapture
from domain.live_handles import LiveHandleCache
from domain.profile_connection import ProfileConnectionManager, SubprocessProfileLauncher
from domain.project_workspace import workspace_subdir
from domain.prompt_files import PromptFileStore, hermes_prompt_reader, prompt_dir
from domain.run_recorder import RunRecorder
from domain.sandbox_fs import SandboxFS
from domain.snapshot_sweeper import SnapshotSweeper
from domain.turn_backfill import run_turn_backfill


@dataclass(frozen=True)
class IngestTasks:
    """The three internal broadcaster subscribers' tasks, in start order."""

    ingest: asyncio.Task[None]
    diff_ingest: asyncio.Task[None]
    media_tag: asyncio.Task[None]


@dataclass(frozen=True)
class GatewayRuntime:
    """What `shutdown()` needs beyond `app.state`: the objects the lifespan
    held as locals and the tasks it cancels, in the order it closes them."""

    adapter: HermesAdapter
    broadcaster: EventBroadcaster
    ingest_tasks: IngestTasks
    attachment_orchestrator: AttachmentOrchestrator
    snapshot_sweeper: SnapshotSweeper


def _profile_adapter_factory(base_settings: Settings) -> Callable[[str, int], HermesAdapter]:
    """`(profile, port) -> HermesAdapter`, for `ProfileConnectionManager`."""
    profile_host = (
        base_settings.research_gateway_profile_dashboard_host or base_settings.hermes_host
    )

    def factory(profile: str, port: int) -> HermesAdapter:
        return HermesAdapter(
            base_settings.model_copy(update={"hermes_host": profile_host, "hermes_port": port})
        )

    return factory


def open_database(app_state: Any, settings: Settings) -> None:
    """`app.state.db_engine` / `db_sessions` -- the workspace's own store."""
    app_state.db_engine = make_engine(settings.research_gateway_db_path)
    app_state.db_sessions = make_sessionmaker(app_state.db_engine)


def open_chat_store(app_state: Any) -> None:
    """`app.state.chat_store` -- the durable chat-history store."""
    app_state.chat_store = ChatStore(app_state.db_sessions)


def build_profile_connection_manager(app_state: Any, settings: Settings) -> None:
    """`app.state.profile_connection_manager` -- one adapter per Hermes profile."""
    _docker_exec_target = settings.research_gateway_profile_docker_exec_target
    # No broadcaster= here: event_broadcaster is built later, so the manager resolves it lazily.
    app_state.profile_connection_manager = ProfileConnectionManager(
        app_state=app_state,
        launcher=SubprocessProfileLauncher(
            command_prefix=("docker", "exec", _docker_exec_target) if _docker_exec_target else (),
            dashboard_host="0.0.0.0" if _docker_exec_target else "127.0.0.1",
            # Reaping needs /proc, which is only there in the container reached by docker exec.
            reap_orphans=bool(_docker_exec_target),
        ),
        adapter_factory=_profile_adapter_factory(settings),
        chat_store=app_state.chat_store,
    )


def build_background_ledger(
    app_state: Any, adapter: HermesAdapter, live_handle_cache: LiveHandleCache
) -> BackgroundLedger:
    """`app.state.background_ledger` -- built, and its startup orphan sweep run."""
    background_ledger = BackgroundLedger(app_state.db_sessions, adapter, live_handle_cache)
    # Rows left `running` by a dead process have genuinely unknown outcomes; none are rescued.
    background_ledger.orphan_on_startup()
    app_state.background_ledger = background_ledger
    return background_ledger


def build_run_recorder(app_state: Any) -> RunRecorder:
    """`app.state.run_recorder` -- built, and its startup interrupt sweep run."""
    chat_store = getattr(app_state, "chat_store", None)
    capture = ForeignPromptCapture(app_state, chat_store) if chat_store is not None else None
    app_state.foreign_prompt_capture = capture
    run_recorder = RunRecorder(
        app_state.db_sessions,
        on_run_opened=capture.handle_run_opened if capture is not None else None,
    )
    run_recorder.interrupt_open_runs_on_startup()
    app_state.run_recorder = run_recorder
    return run_recorder


def backfill_chat_turn_ids(app_state: Any) -> None:
    """Link null-turn chat rows to their runs by timestamp, best-effort."""
    run_turn_backfill(app_state.db_sessions)


def wire_event_stream(
    app_state: Any,
    adapter: HermesAdapter,
    live_handle_cache: LiveHandleCache,
    background_ledger: BackgroundLedger,
    run_recorder: RunRecorder,
) -> EventBroadcaster:
    """`app.state.hermes_adapter` / `event_broadcaster` / `live_handle_cache` /
    `hermes_connect_lock`, and the pump started."""

    def _on_generation_change(generation: int | None, previous: int | None) -> None:
        # Cheap synchronous sweep first; the ledger additionally schedules an async re-resume.
        run_recorder.handle_generation_change(generation, previous)
        background_ledger.handle_generation_change(generation, previous)

    broadcaster = EventBroadcaster(
        adapter,
        live_handle_cache,
        on_background_complete=background_ledger.handle_completed,
        on_generation_change=_on_generation_change,
        on_canonical_event=run_recorder.handle_event,
        # A callable, not the dict: the per-profile caches are built lazily, after this point.
        profile_caches=lambda: getattr(app_state, "shared_profile_handle_caches", {}) or {},
    )
    app_state.hermes_adapter = adapter
    app_state.event_broadcaster = broadcaster
    app_state.live_handle_cache = live_handle_cache
    # Serializes connect(): a second concurrent connect tears down the first one's socket.
    app_state.hermes_connect_lock = asyncio.Lock()
    broadcaster.start()
    return broadcaster


def build_prompt_files(app_state: Any, settings: Settings) -> None:
    """`app.state.prompt_files` -- the file-backed rewrite prompts."""
    app_state.prompt_files = PromptFileStore(
        hermes_prompt_reader(app_state, sandbox_root=settings.hermes_sandbox_root),
        directory=prompt_dir(settings),
        ttl_s=settings.rewrite_prompt_file_ttl_s,
    )


def start_profile_reconciliation(app_state: Any, settings: Settings) -> None:
    """The default profile's chat capture, and the reconciliation timer if asked."""
    app_state.profile_connection_manager.start_default_capture()
    if settings.research_gateway_profile_reconcile_interval_s > 0:
        app_state.profile_connection_manager.start(
            settings.research_gateway_profile_reconcile_interval_s
        )


def start_artifact_ingestors(
    app_state: Any, settings: Settings, adapter: HermesAdapter, broadcaster: EventBroadcaster
) -> IngestTasks:
    """`app.state.artifact_store` / `artifact_ingestor` / `sandbox_diff_ingestor` /
    `media_tag_ingestor`, each ingestor subscribed to the broadcaster."""
    artifact_store = ArtifactStore(Path(settings.research_gateway_artifact_root))
    sandbox_denylist = SandboxDiffDenylist.from_settings(settings)
    artifact_ingestor = ArtifactIngestor(
        adapter,
        app_state.db_sessions,
        artifact_store,
        sandbox_root=settings.hermes_sandbox_root,
        denylist=sandbox_denylist,
        workspace_root=posixpath.join(
            settings.hermes_sandbox_root.rstrip("/"), workspace_subdir(settings)
        ),
    )
    app_state.artifact_store = artifact_store
    app_state.artifact_ingestor = artifact_ingestor
    ingest_task = asyncio.create_task(artifact_ingestor.run(broadcaster))
    sandbox_diff_ingestor = SandboxDiffIngestor(
        adapter,
        artifact_ingestor,
        sandbox_root=settings.hermes_sandbox_root,
        denylist=sandbox_denylist,
        sandbox_fs=getattr(app_state, "sandbox_fs", None),
    )
    app_state.sandbox_diff_ingestor = sandbox_diff_ingestor
    diff_ingest_task = asyncio.create_task(sandbox_diff_ingestor.run(broadcaster))
    media_tag_ingestor = MediaTagIngestor(
        artifact_ingestor,
        sandbox_root=settings.hermes_sandbox_root,
        denylist=sandbox_denylist,
    )
    app_state.media_tag_ingestor = media_tag_ingestor
    media_tag_task = asyncio.create_task(media_tag_ingestor.run(broadcaster))
    return IngestTasks(ingest=ingest_task, diff_ingest=diff_ingest_task, media_tag=media_tag_task)


def build_attachment_orchestrator(app_state: Any) -> AttachmentOrchestrator:
    """`app.state.attachment_orchestrator` -- built LAST of the request-serving
    services, and its startup orphan sweep run."""
    attachment_orchestrator = AttachmentOrchestrator(app_state)
    attachment_orchestrator.orphan_on_startup()
    app_state.attachment_orchestrator = attachment_orchestrator
    return attachment_orchestrator


def start_snapshot_sweeper(app_state: Any, settings: Settings) -> SnapshotSweeper:
    """`app.state.snapshot_sweeper` -- built after everything it reads, and started."""
    snapshot_sweeper = SnapshotSweeper.from_settings(app_state, settings)
    app_state.snapshot_sweeper = snapshot_sweeper
    snapshot_sweeper.start()
    return snapshot_sweeper


def build_audit_store(app_state: Any, settings: Settings) -> None:
    """`app.state.audit_store` / `audit_recorder` -- the read-only audit surface."""
    app_state.audit_store = AuditStore(
        endpoint=settings.audit_clickhouse_url,
        user=settings.audit_clickhouse_user,
        password=settings.audit_clickhouse_password.get_secret_value(),
        max_rows=settings.audit_query_max_rows,
        timeout_s=settings.audit_query_timeout_s,
    )
    app_state.audit_recorder = AuditRecorder(
        ingest_url=settings.audit_ingest_url,
        host=settings.audit_host_label or "gateway",
    )


def startup(
    app_state: Any,
    settings: Settings,
    adapter_factory: Callable[[Settings], HermesAdapter] = HermesAdapter,
    sandbox_fs: SandboxFS | None = None,
) -> GatewayRuntime:
    """Build every service, in the order `api.main.lifespan` always built them."""
    if sandbox_fs is not None:
        app_state.sandbox_fs = sandbox_fs
    adapter = adapter_factory(settings)
    open_database(app_state, settings)
    open_chat_store(app_state)
    build_profile_connection_manager(app_state, settings)
    live_handle_cache = LiveHandleCache(adapter)
    background_ledger = build_background_ledger(app_state, adapter, live_handle_cache)
    run_recorder = build_run_recorder(app_state)
    backfill_chat_turn_ids(app_state)
    broadcaster = wire_event_stream(
        app_state, adapter, live_handle_cache, background_ledger, run_recorder
    )
    build_prompt_files(app_state, settings)
    start_profile_reconciliation(app_state, settings)
    ingest_tasks = start_artifact_ingestors(app_state, settings, adapter, broadcaster)
    attachment_orchestrator = build_attachment_orchestrator(app_state)
    snapshot_sweeper = start_snapshot_sweeper(app_state, settings)
    build_audit_store(app_state, settings)
    return GatewayRuntime(
        adapter=adapter,
        broadcaster=broadcaster,
        ingest_tasks=ingest_tasks,
        attachment_orchestrator=attachment_orchestrator,
        snapshot_sweeper=snapshot_sweeper,
    )


async def shutdown(app_state: Any, runtime: GatewayRuntime) -> None:
    """Tear down in the order the lifespan's `finally` always did."""
    for name in ("audit_store", "audit_recorder"):
        client = getattr(app_state, name, None)
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()
    await app_state.profile_connection_manager.close()
    # Everything that can hold a resume closes before the adapter and broadcaster below.
    await runtime.snapshot_sweeper.close()
    await runtime.attachment_orchestrator.close()
    capture = getattr(app_state, "foreign_prompt_capture", None)
    if capture is not None:
        await capture.close()
    runtime.ingest_tasks.media_tag.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runtime.ingest_tasks.media_tag
    runtime.ingest_tasks.diff_ingest.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runtime.ingest_tasks.diff_ingest
    runtime.ingest_tasks.ingest.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runtime.ingest_tasks.ingest
    await runtime.broadcaster.close()
    await runtime.adapter.close()
    app_state.db_engine.dispose()
