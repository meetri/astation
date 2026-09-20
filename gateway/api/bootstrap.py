"""Startup and shutdown of every `app.state` service, in the documented order.

This was the body of `api.main.lifespan` (CLEANUP_PLAN step 3.8). One
function per subsystem, called by `startup()` in exactly the order the
lifespan built them -- that order is load-bearing (the comments on each step
say why: the manager is built before the broadcaster it reads lazily, the
ledger and recorder before the broadcaster their hooks are wired into, the
orchestrator after every `app.state` field it reads, the sweeper last) -- and
`shutdown()` tears them down in the order the lifespan's `finally` did.
Nothing starts earlier or later than it did in `api/main.py`, and every
`app.state` attribute keeps its name.

Deliberately does not call `login()`/`connect()` here: `HermesAdapter()`
itself makes no network calls at construction time (see its docstring),
and Phase 0 must not exercise real Hermes credentials just from the
gateway process starting up -- only an actual request that needs Hermes
(e.g. `GET /api/sessions`) triggers `login()`/`connect()`, lazily, once.
"""

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
    """`(profile, port) -> HermesAdapter`, for `ProfileConnectionManager`.

    Same scheme and credentials as the unified connection either way. The
    *host* is `research_gateway_profile_dashboard_host` if set, else
    `hermes_host` -- they differ under the Docker deploy (2026-09-05): a
    `docker exec`-launched dashboard's ephemeral port is reachable from the
    gateway container only via the Hermes container's name on their shared
    Docker network, not `hermes_host`'s LAN IP (see `config/settings.py`'s field
    docstring and `SubprocessProfileLauncher`'s `command_prefix`).
    """
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
    # The workspace's own store (Phase 1). Projects and the filing index live
    # here; Hermes owns the sessions themselves and this database only ever
    # *references* them by their stored id. Creating the engine opens no
    # connection and creates no schema -- `alembic upgrade head` owns the DDL,
    # see `domain/db.py`.
    app_state.db_engine = make_engine(settings.research_gateway_db_path)
    app_state.db_sessions = make_sessionmaker(app_state.db_engine)


def open_chat_store(app_state: Any) -> None:
    """`app.state.chat_store` -- the durable chat-history store."""
    # The durable chat-history store (P1, B-136): every profile's own live
    # capture and the turn-submit route both write through this one object.
    # See `domain/chat_store.py`.
    app_state.chat_store = ChatStore(app_state.db_sessions)


def build_profile_connection_manager(app_state: Any, settings: Settings) -> None:
    """`app.state.profile_connection_manager` -- one adapter per Hermes profile."""
    # `ProfileConnectionManager` (`docs/CHAT_HISTORY_DESIGN.md` §4,
    # `domain/profile_connection.py`): one `HermesAdapter` per Hermes profile,
    # auto-provisioned from `profiles.list`. Built with `app_state=app.state`
    # rather than handed this function's own local `adapter` -- every existing
    # test in this suite swaps `app.state.hermes_adapter` for a fake *after*
    # `lifespan` has already run (`test_main.py`'s `_make` fixture and its
    # siblings), so a manager built against a captured reference would go on
    # reconciling against the stale, pre-swap object for the rest of the
    # process's life. `app_state=` makes the manager re-resolve
    # `app.state.hermes_adapter` at the point of use instead, and additionally
    # rebinds its default-profile connection if that identity has already
    # moved since the last pass -- see `domain/profile_connection.py`'s module
    # docstring ("the base-adapter staleness hazard"), the same fix
    # `SnapshotSweeper` already uses for the same reason (`api/snapshot_sweep.py`).
    #
    # `SubprocessProfileLauncher` is the REAL launcher (shells out to
    # `hermes -p <profile> dashboard --isolated --port 0 --no-open
    # --skip-build`, parses `HERMES_DASHBOARD_READY port=<port>` from its
    # stdout) -- wired in here so the manager is capable of provisioning real
    # per-profile connections, not left permanently on `NullProfileLauncher`.
    # `RESEARCH_GATEWAY_PROFILE_DOCKER_EXEC_TARGET` (empty by default) routes
    # that shell-out through `docker exec <target>` -- the gateway and Hermes
    # run as separate containers, so `hermes` isn't in the gateway's own
    # image; see `config/settings.py`'s field docstring.
    # But the reconciliation timer that would ever actually call it is gated
    # behind `RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S` (default `0`,
    # disabled -- see `config/settings.py` and `snapshot_sweep_timer_disabled` in
    # `tests/conftest.py`, which pins it to `0` for the whole test suite
    # regardless). Constructing this object and storing it on `app.state`
    # spawns nothing; a positive interval turning the timer on is a separate,
    # owner-timed decision, not something this change turns on by shipping.
    #
    # B-137: no `broadcaster=` is passed here even though the default
    # connection's pump needs one. `app.state.event_broadcaster` is not set
    # until `wire_event_stream()` runs (`broadcaster = EventBroadcaster(...)`
    # there), so passing it now would pass `None` -- instead
    # `ProfileConnectionManager._resolve_broadcaster()` reads
    # `app_state.event_broadcaster` lazily, at the point of use, which by the
    # time the reconciliation timer's first pass can possibly run (gated on
    # `start_profile_reconciliation()`, itself after
    # `app.state.event_broadcaster` is set) is
    # always populated. This is what makes the default profile's capture
    # subscribe through the broadcaster (`EventBroadcaster.subscribe()`,
    # already used the same way by `ArtifactIngestor`/`SandboxDiffIngestor`/
    # `MediaTagIngestor` below) instead of calling `adapter.events()` a second
    # time on the same object `EventBroadcaster._run` already drains for
    # `/ws/events` -- see `domain/profile_connection.py`'s module docstring.
    _docker_exec_target = settings.research_gateway_profile_docker_exec_target
    app_state.profile_connection_manager = ProfileConnectionManager(
        app_state=app_state,
        launcher=SubprocessProfileLauncher(
            command_prefix=("docker", "exec", _docker_exec_target) if _docker_exec_target else (),
            dashboard_host="0.0.0.0" if _docker_exec_target else "127.0.0.1",
            # B-165: only meaningful where `/proc` exists -- i.e. inside the
            # Hermes container reached through `docker exec`.
            reap_orphans=bool(_docker_exec_target),
        ),
        adapter_factory=_profile_adapter_factory(settings),
        chat_store=app_state.chat_store,
    )


def build_background_ledger(
    app_state: Any, adapter: HermesAdapter, live_handle_cache: LiveHandleCache
) -> BackgroundLedger:
    """`app.state.background_ledger` -- built, and its startup orphan sweep run."""
    # The background-task ledger (P2-1, B-42). Startup first orphans any row
    # still `running` from a previous process: nothing was attached to any
    # session while the gateway was down, and the completion event is
    # delivered only to attached connections (P2-0b) -- so those outcomes are
    # genuinely unknown. Best-effort; an unmigrated DB logs and moves on.
    # No rescue is attempted here (the adapter is deliberately not connected
    # at startup); the first real connection bumps the generation, and the
    # generation-change hook runs the re-resume rescue then.
    background_ledger = BackgroundLedger(app_state.db_sessions, adapter, live_handle_cache)
    background_ledger.orphan_on_startup()
    app_state.background_ledger = background_ledger
    return background_ledger


def build_run_recorder(app_state: Any) -> RunRecorder:
    """`app.state.run_recorder` -- built, and its startup interrupt sweep run."""
    # The run recorder (P2-2): real attribution on every forwarded envelope +
    # selective event persistence into `runs`/`run_events`. Startup sweep
    # mirrors the background ledger's: a run left `running` by a previous
    # process lost its stream (nothing buffers or replays events), so it is
    # `interrupted`, honestly, before the pump starts. Best-effort; an
    # unmigrated DB logs and moves on (the /api/runs routes 503 with the fix).
    # B-188: the user row the submit route writes before a turn opens gets
    # its run id the moment the recorder opens one (`ChatStore.attach_turn`).
    # B-190: when no such row is waiting the turn was started outside this
    # gateway (the TUI, another client), and `ForeignPromptCapture` reads
    # the prompt back from Hermes instead -- attach first, capture only on a
    # miss; the composition lives in `ForeignPromptCapture.handle_run_opened`.
    # It reads the adapter / connect lock / profile manager off `app_state`
    # at capture time, so being built before `wire_event_stream` is fine.
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
    """Link null-turn chat rows to their runs by timestamp (B-192), best-effort."""
    # Ordered after the recorder's startup sweep (so a run left `running` by
    # the previous process has its `ended_at` and a bounded window) and
    # before `wire_event_stream` starts any pump (so nothing is capturing
    # into the rows this walks). `run_turn_backfill` logs its report and
    # swallows every failure: an unmigrated or broken DB costs the backfill,
    # never the startup. Idempotent, so running it on every boot is the
    # whole scheduling story -- there is no admin route for it.
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
        # Order matters: the recorder only sweeps its own tables (fast,
        # synchronous), the ledger additionally schedules the async re-resume
        # rescue -- run the cheap, purely-local one first.
        run_recorder.handle_generation_change(generation, previous)
        background_ledger.handle_generation_change(generation, previous)

    # The broadcaster reads that same cache to turn the live handle Hermes
    # stamps on an event back into the STORED id the app matches on.
    # One object, so a session resolved by a route is immediately attributable
    # on the event stream. The P2-1 hooks make a background task's single
    # completion event durable and injected; the P2-2 hook attributes and
    # persists -- see EventBroadcaster.__init__.
    broadcaster = EventBroadcaster(
        adapter,
        live_handle_cache,
        on_background_complete=background_ledger.handle_completed,
        on_generation_change=_on_generation_change,
        on_canonical_event=run_recorder.handle_event,
        # B-198: one connection serves every profile inside Hermes, so the
        # frame's profile has to come from whichever profile's cache knows the
        # session. Those caches are built lazily by
        # `resolve_live_handle_cache`, hence a callable rather than the dict.
        profile_caches=lambda: getattr(app_state, "shared_profile_handle_caches", {}) or {},
    )
    app_state.hermes_adapter = adapter
    app_state.event_broadcaster = broadcaster
    app_state.live_handle_cache = live_handle_cache
    # Serializes `_ensure_connected()`. Two requests arriving together on a
    # cold (or newly-dropped) adapter would otherwise both see
    # `is_connected == False` and both run `connect()`, and the second
    # connect tears down the socket the first just handed to its caller.
    # It belongs to the same `app.state` as the adapter it protects, and
    # `_ensure_connected()` is handed that state rather than reaching for a
    # module-global one -- see B-22.
    app_state.hermes_connect_lock = asyncio.Lock()
    # Drain the upstream stream for the whole process lifetime, so a client
    # that connects later is never handed a backlog of events that happened
    # before it existed. See `EventBroadcaster.start()`.
    broadcaster.start()
    return broadcaster


def build_prompt_files(app_state: Any, settings: Settings) -> None:
    """`app.state.prompt_files` -- the file-backed rewrite prompts.

    Owner, 2026-09-07: the speech prompts as files the app's own editor can
    open and save, rather than a redeploy or a Settings text field. The reader
    goes through Hermes because the files live in the SANDBOX -- the only
    filesystem `PUT /api/sandbox/text` will write to, and therefore the only
    one the editor can reach. Absent files change nothing: every prompt falls
    back to its configured setting. See `domain/prompt_files.py`.

    Ordered after the adapter is on `app_state` (the reader resolves it per
    call, so strictly this could sit anywhere, but a subsystem that reads
    through Hermes belongs after Hermes) and before any route can serve.
    """
    app_state.prompt_files = PromptFileStore(
        hermes_prompt_reader(app_state, sandbox_root=settings.hermes_sandbox_root),
        directory=prompt_dir(settings),
        ttl_s=settings.rewrite_prompt_file_ttl_s,
    )


def start_profile_reconciliation(app_state: Any, settings: Settings) -> None:
    """The default profile's chat capture, and the reconciliation timer if asked.

    **These are two jobs, and B-199 was them sharing one switch.** The timer
    launches a dashboard process per non-default profile and is disabled by
    default (`interval_s == 0`); capturing the default profile's assistant and
    tool messages into the chat store has nothing to do with that, but was
    only ever started from the same `run()` loop. Inside Hermes the timer is
    correctly off -- there is nothing to launch -- and chat capture went off
    with it, silently, so `chat_messages` grew only the user's own rows.
    """
    app_state.profile_connection_manager.start_default_capture()
    # The `ProfileConnectionManager` reconciliation timer -- disabled by
    # default (`interval_s == 0`); see its construction above for why. When
    # enabled, this is the ONLY thing that ever calls `SubprocessProfileLauncher`.
    if settings.research_gateway_profile_reconcile_interval_s > 0:
        app_state.profile_connection_manager.start(
            settings.research_gateway_profile_reconcile_interval_s
        )


def start_artifact_ingestors(
    app_state: Any, settings: Settings, adapter: HermesAdapter, broadcaster: EventBroadcaster
) -> IngestTasks:
    """`app.state.artifact_store` / `artifact_ingestor` / `sandbox_diff_ingestor` /
    `media_tag_ingestor`, each ingestor subscribed to the broadcaster."""
    # The Tier-2 artifact ingestor (P3-1): an *internal subscriber* to the
    # broadcaster -- it rides the same bounded per-subscriber queue as a
    # `/ws/events` client and costs `_forward_one` nothing beyond one
    # queue put. (The third-party notify relay that first used this pattern
    # was removed 2026-09-04 at the operator's direction; native notifications
    # are P6-5, `docs/NOTIFICATIONS_DESIGN.md`.) It watches write_file/patch completions and
    # pulls the produced files into the durable library; the promote route
    # reuses the same object (`app.state.artifact_ingestor`).
    artifact_store = ArtifactStore(Path(settings.research_gateway_artifact_root))
    # A5: ONE set of ignore rules, honoured by both producers. The diff walk
    # has checked them since B-61; the `tool.completed` path did not, which is
    # how `.npm/_logs`, `.curator_backups/blobs` and `profiles/<name>` rows
    # reached the operator's library.
    sandbox_denylist = SandboxDiffDenylist.from_settings(settings)
    artifact_ingestor = ArtifactIngestor(
        adapter,
        app_state.db_sessions,
        artifact_store,
        sandbox_root=settings.hermes_sandbox_root,
        denylist=sandbox_denylist,
        # C3: where a path is allowed to file itself. Only project workspaces
        # -- nothing else in the sandbox auto-tags.
        workspace_root=posixpath.join(
            settings.hermes_sandbox_root.rstrip("/"), workspace_subdir(settings)
        ),
    )
    app_state.artifact_store = artifact_store
    app_state.artifact_ingestor = artifact_ingestor
    ingest_task = asyncio.create_task(artifact_ingestor.run(broadcaster))
    # The before/after sandbox-listing diff auto-promoter (P3-9): a THIRD
    # internal subscriber, same pattern -- it snapshots the sandbox tree at
    # each turn's start/end and hands anything new-or-changed to the SAME
    # `artifact_ingestor` above (reused, not duplicated), which is what
    # catches `terminal`-produced files (audio, PDFs) that write_file/patch
    # auto-ingest never sees.
    # B-61: the diff must never promote Hermes's own operational churn
    # (logs/, cron/, state/, cache/, SQLite WALs, heartbeats, ...) -- built
    # from the module defaults in api/artifacts.py unless the
    # HERMES_SANDBOX_DENYLIST_* env overrides are set.
    sandbox_diff_ingestor = SandboxDiffIngestor(
        adapter,
        artifact_ingestor,
        sandbox_root=settings.hermes_sandbox_root,
        denylist=sandbox_denylist,
        sandbox_fs=getattr(app_state, "sandbox_fs", None),
    )
    app_state.sandbox_diff_ingestor = sandbox_diff_ingestor
    diff_ingest_task = asyncio.create_task(sandbox_diff_ingestor.run(broadcaster))
    # The MEDIA: tag detector (P3-10, B-59): a FOURTH internal subscriber,
    # same pattern -- it scans message.completed/message.interim text for
    # Hermes's native `MEDIA:/abs/path` file-delivery convention and hands
    # each tagged path to the SAME `artifact_ingestor` above, with
    # `source: media_tag` recorded in the row's metadata. The tag is the
    # model's explicit "this file is for the user" signal, so the B-61
    # denylist does not refuse it (only logs the odd overlap).
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
    # The composer-attachment orchestrator (P3-3): drives each uploaded
    # attachment through the measured two-turn chain (prime -> verify ->
    # image.attach / path reference) as a background task per row. Built
    # LAST, after every `app.state` field it reads is set (adapter, cache,
    # connect lock, sessionmaker, artifact store). Startup first orphans any
    # row a previous process left in flight -- its orchestration task died
    # with that process and, unlike background tasks, there is no completion
    # event to rescue the outcome with.
    attachment_orchestrator = AttachmentOrchestrator(app_state)
    attachment_orchestrator.orphan_on_startup()
    app_state.attachment_orchestrator = attachment_orchestrator
    return attachment_orchestrator


def start_snapshot_sweeper(app_state: Any, settings: Settings) -> SnapshotSweeper:
    """`app.state.snapshot_sweeper` -- built after everything it reads, and started."""
    # The snapshot sweep (P6-3, `api/snapshot_sweep.py`): the timer that
    # re-snapshots filed sessions whose transcript moved, so they are durable
    # without a tap. Reads adapter / run recorder / live-handle cache from
    # `app.state` at pass time, so it is built after all of them. Its first
    # pass WAITS for `adapter.is_connected` rather than connecting -- the
    # lifespan must not exercise Hermes credentials on its own (same rule as
    # the deliberately-unconnected adapter above). `INTERVAL_S=0` starts no
    # task; `POST /api/snapshot-sweeps` still runs a pass. `SnapshotSweeper`
    # lives in `domain/snapshot_sweeper.py`; its router is included by
    # `api/main.py`.
    snapshot_sweeper = SnapshotSweeper.from_settings(app_state, settings)
    app_state.snapshot_sweeper = snapshot_sweeper
    snapshot_sweeper.start()
    return snapshot_sweeper


def build_audit_store(app_state: Any, settings: Settings) -> None:
    """`app.state.audit_store` / `audit_recorder` -- the read-only audit surface.

    Always constructed, even with no endpoint configured. An unconfigured store
    answers every route with a 503 naming the reason, which is a better failure
    than a missing attribute raising somewhere deeper -- and it lets
    `GET /api/audit/health` explain what to set.
    """
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
    """Build every service, in the order `api.main.lifespan` always built them.

    Synchronous on purpose, like the lifespan body it replaces: nothing here
    awaits, and the `asyncio.create_task` calls inside need only the running
    loop the lifespan is called under.

    `adapter_factory` exists for the Hermes plugin, which runs inside the
    dashboard it talks to and must authenticate the way that dashboard was
    started (ticket when gated, session token when not). Every other caller
    gets the default and behaves exactly as before.

    `sandbox_fs` likewise: the plugin shares a filesystem with Hermes and
    passes a `DirectSandboxFS`, so sandbox reads, listings and writes become
    syscalls instead of loopback HTTP. It is installed on `app_state` FIRST,
    before any service is built, because the services below resolve it at
    construction and would otherwise capture the HTTP fallback for the life of
    the process. The sidecar passes nothing and keeps the HTTP path it must.
    """
    if sandbox_fs is not None:
        app_state.sandbox_fs = sandbox_fs
    adapter = adapter_factory(settings)
    open_database(app_state, settings)
    open_chat_store(app_state)
    build_profile_connection_manager(app_state, settings)
    # Live handles for the *current* Hermes connection only. Built
    # here, per process, and never persisted -- see `LiveHandleCache`.
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
    # The audit clients first: both are plain HTTP clients with no background
    # work, so closing them early cannot strand anything, and leaving them open
    # leaks a connection pool per restart in the plugin's long-lived process.
    for name in ("audit_store", "audit_recorder"):
        client = getattr(app_state, name, None)
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()
    # The profile connection manager first: it may hold its own separate
    # per-profile adapters/processes (when the reconciliation timer is
    # enabled), independent of everything below. A no-op when the timer
    # never ran (the default -- see `config/settings.py` / B-137).
    await app_state.profile_connection_manager.close()
    # The sweep next: a pass mid-flight holds a resume against the
    # adapter everything below is about to tear down.
    await runtime.snapshot_sweeper.close()
    await runtime.attachment_orchestrator.close()
    # B-190: a foreign-prompt capture mid-resume holds the adapter too.
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
