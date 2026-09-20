"""`ProfileConnectionManager` -- one `HermesAdapter` per Hermes profile."""

from __future__ import annotations

import abc
import asyncio
import asyncio.subprocess
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from adapters.hermes import HermesAdapter, HermesError
from domain.event_stream import (
    LIVE_SESSION_ID_FIELD,
    PROFILE_FIELD,
    STORED_SESSION_ID_FIELD,
    STREAM_DESYNCHRONIZED_EVENT_TYPE,
    session_identity_from_payload,
)
from events.canonical import EventContext
from events.normalizer import normalize_event, raw_event_type

logger = logging.getLogger(__name__)

_PUMP_RESTART_DELAY_S = 1.0

_MAX_LIVE_TO_STORED = 256

_STORED_SESSION_ID_FIELD = STORED_SESSION_ID_FIELD
_LIVE_SESSION_ID_FIELD = LIVE_SESSION_ID_FIELD
_PROFILE_FIELD = PROFILE_FIELD

_DESYNCHRONIZED_TYPE = STREAM_DESYNCHRONIZED_EVENT_TYPE


@dataclass(frozen=True)
class _EnvelopeAsCanonical:
    """Just enough of a `CanonicalEvent` for `ChatStore.capture_event`."""

    type: str | None


@dataclass(frozen=True)
class LaunchedProcess:
    """What a `ProfileLauncher` hands back for one non-default profile."""

    profile: str
    port: int
    handle: Any = None


class ProfileLauncher(abc.ABC):
    """Spawns/stops/health-checks the OS process behind one profile's isolated dashboard."""

    @abc.abstractmethod
    async def launch(self, profile: str) -> LaunchedProcess:
        """Start the profile's isolated dashboard process; return once it's ready."""

    @abc.abstractmethod
    async def stop(self, process: LaunchedProcess) -> None:
        """Stop a previously-launched process. Idempotent: stopping twice is fine."""

    @abc.abstractmethod
    def is_alive(self, process: LaunchedProcess) -> bool:
        """Whether the OS process behind `process` is still running."""


class NullProfileLauncher(ProfileLauncher):
    """The default concrete launcher: refuses every launch."""

    async def launch(self, profile: str) -> LaunchedProcess:
        raise HermesError(
            f"no ProfileLauncher configured; cannot launch a dashboard for profile {profile!r}"
        )

    async def stop(self, process: LaunchedProcess) -> None:  # pragma: no cover - unreachable
        return None

    def is_alive(self, process: LaunchedProcess) -> bool:  # pragma: no cover - unreachable
        return False


class ProfileDashboardDied(HermesError):
    """A launched dashboard process exited before ever reporting ready."""


class ProfileDashboardTimedOut(HermesError):
    """A launched dashboard process never reported ready within the deadline."""


SubprocessFactory = Callable[[list[str]], Awaitable[Any]]


async def _default_subprocess_factory(argv: list[str]) -> Any:  # pragma: no cover - real I/O
    return await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )


class SubprocessProfileLauncher(ProfileLauncher):
    """The real launcher: `hermes -p <profile> dashboard --port 0 --isolated --no-open --skip-build`."""

    _READY_PREFIX = "HERMES_DASHBOARD_READY port="

    def __init__(
        self,
        *,
        hermes_executable: str = "hermes",
        command_prefix: tuple[str, ...] = (),
        dashboard_host: str = "127.0.0.1",
        ready_timeout_s: float = 30.0,
        subprocess_factory: SubprocessFactory | None = None,
        reap_orphans: bool = False,
    ) -> None:
        """`command_prefix` and `dashboard_host` exist for the gateway-container /
        Hermes-container split: the gateway does not run on the
        same host as Hermes's CLI, so `hermes` cannot be exec'd directly.
        `command_prefix=("docker", "exec", "hermes")` runs the real CLI
        inside the Hermes container over the mounted Docker socket instead --
        see `api/main.py`'s wiring. `--host` defaults to Hermes's own CLI
        default (127.0.0.1) unless overridden; a `docker exec`-launched
        dashboard must bind `0.0.0.0` to be reachable from the gateway
        container at all (Hermes's own auth gate, already configured on the
        deployed host, is what makes that bind safe -- see
        `docs/CHAT_HISTORY_DESIGN.md`).
        """
        self._hermes_executable = hermes_executable
        self._command_prefix = command_prefix
        self._dashboard_host = dashboard_host
        self._ready_timeout_s = ready_timeout_s
        self._subprocess_factory = subprocess_factory or _default_subprocess_factory
        self._reap_orphans = reap_orphans

    def _argv(self, profile: str, *, extra: tuple[str, ...] = ()) -> list[str]:
        return [
            *self._command_prefix,
            self._hermes_executable,
            "-p",
            profile,
            "dashboard",
            *extra,
        ]

    def _launch_argv(self, profile: str) -> list[str]:
        return self._argv(
            profile,
            extra=(
                "--host",
                self._dashboard_host,
                "--port",
                "0",
                "--isolated",
                "--no-open",
                "--skip-build",
            ),
        )

    _REAPER_SCRIPT = (
        "import os, signal, sys\n"
        "profile, host = sys.argv[1], sys.argv[2]\n"
        "needle = ['-p', profile, 'dashboard', '--host', host, '--port', '0', '--isolated']\n"
        "me, n = os.getpid(), 0\n"
        "for pid in os.listdir('/proc'):\n"
        "    if not pid.isdigit() or int(pid) == me:\n"
        "        continue\n"
        "    try:\n"
        "        raw = open('/proc/%s/cmdline' % pid, 'rb').read()\n"
        "    except OSError:\n"
        "        continue\n"
        "    argv = [a.decode('utf-8', 'replace') for a in raw.split(b'\\0')]\n"
        "    if any(argv[i:i + len(needle)] == needle for i in range(len(argv))):\n"
        "        try:\n"
        "            os.kill(int(pid), signal.SIGTERM)\n"
        "            n += 1\n"
        "        except OSError:\n"
        "            pass\n"
        "print('REAPED', n, flush=True)\n"
    )

    def _reap_argv(self, profile: str) -> list[str]:
        return [
            *self._command_prefix,
            "python3",
            "-c",
            self._REAPER_SCRIPT,
            profile,
            self._dashboard_host,
        ]

    async def reap(self, profile: str) -> int | None:
        """B-165: terminate every isolated dashboard for `profile` in the target."""
        if not self._reap_orphans:
            return None
        try:
            process = await self._subprocess_factory(self._reap_argv(profile))
            line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=10.0)
        except Exception:
            logger.warning(
                "profile %r: orphan dashboard reaper could not run", profile, exc_info=True
            )
            return None
        text = line.decode("utf-8", errors="replace").strip()
        if not text.startswith("REAPED "):
            logger.warning("profile %r: reaper answered %r", profile, text)
            return None
        try:
            count = int(text[len("REAPED ") :])
        except ValueError:
            return None
        if count:
            logger.info(
                "profile %r: reaped %d orphan isolated dashboard(s) (B-165)", profile, count
            )
        return count

    async def launch(self, profile: str) -> LaunchedProcess:
        await self.reap(profile)
        argv = self._launch_argv(profile)
        process = await self._subprocess_factory(argv)
        try:
            port = await asyncio.wait_for(
                self._read_ready_port(profile, process), timeout=self._ready_timeout_s
            )
        except TimeoutError as exc:
            with contextlib.suppress(Exception):
                process.kill()
            raise ProfileDashboardTimedOut(
                f"profile {profile!r}: dashboard did not print "
                f"{self._READY_PREFIX}<port> within {self._ready_timeout_s:.0f}s"
            ) from exc
        return LaunchedProcess(profile=profile, port=port, handle=process)

    async def _read_ready_port(self, profile: str, process: Any) -> int:
        """Scan stdout for the ready line; raise if the process exits first."""
        stdout = process.stdout
        while True:
            line = await stdout.readline()
            if not line:
                returncode = getattr(process, "returncode", None)
                raise ProfileDashboardDied(
                    f"profile {profile!r}: dashboard process exited "
                    f"(code {returncode!r}) before printing "
                    f"{self._READY_PREFIX}<port>"
                )
            text = line.decode("utf-8", errors="replace").strip()
            if not text.startswith(self._READY_PREFIX):
                continue
            port_text = text[len(self._READY_PREFIX) :].strip()
            try:
                return int(port_text)
            except ValueError:
                logger.warning(
                    "profile %r: dashboard printed a malformed ready line: %r",
                    profile,
                    text,
                )
                continue

    async def stop(self, process: LaunchedProcess) -> None:
        """Always kills the local client process handle, `command_prefix` or not."""
        proc = process.handle
        if proc is not None and self.is_alive(process):
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                with contextlib.suppress(Exception):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        await self.reap(process.profile)

    def is_alive(self, process: LaunchedProcess) -> bool:
        proc = process.handle
        if proc is None:
            return False
        return proc.returncode is None


AdapterFactory = Callable[[str, int], HermesAdapter]


@dataclass
class ProfileConnection:
    """One profile's connection: its adapter, its process (if any), its pump."""

    profile: str
    adapter: HermesAdapter
    is_default: bool
    model: str | None = None
    process: LaunchedProcess | None = None
    pump_task: asyncio.Task[None] | None = None
    live_to_stored: dict[str, str] = field(default_factory=dict)
    live_handle_cache: Any = None

    @property
    def connected(self) -> bool:
        return self.adapter.is_connected


class ProfileConnectionManager:
    """Owns one `ProfileConnection` per Hermes profile, auto-provisioned."""

    def __init__(
        self,
        *,
        base_adapter: HermesAdapter | None = None,
        app_state: Any = None,
        broadcaster: Any = None,
        launcher: ProfileLauncher | None = None,
        adapter_factory: AdapterFactory | None = None,
        chat_store: Any = None,
        on_canonical_event: Any = None,
        default_profile_name: str = "default",
    ) -> None:
        """Exactly one of `base_adapter` or `app_state` must be given."""
        if (base_adapter is None) == (app_state is None):
            raise ValueError(
                "ProfileConnectionManager needs exactly one of base_adapter= "
                "or app_state= (see __init__'s docstring)"
            )
        self._base_adapter = base_adapter
        self._app_state = app_state
        self._broadcaster = broadcaster
        self._launcher = launcher or NullProfileLauncher()
        self._adapter_factory = adapter_factory
        self._chat_store = chat_store
        self._on_canonical_event = on_canonical_event
        self._default_profile_name = default_profile_name
        self._connections: dict[str, ProfileConnection] = {}
        self._run_task: asyncio.Task[None] | None = None
        self._capture_task: asyncio.Task[None] | None = None
        self._warned_no_broadcaster = False

    def _resolve_base_adapter(self) -> HermesAdapter:
        """The base/unified adapter, re-read at the point of use, never cached."""
        if self._app_state is not None:
            return self._app_state.hermes_adapter
        assert self._base_adapter is not None
        return self._base_adapter

    def _resolve_broadcaster(self) -> Any | None:
        """The `EventBroadcaster` the default connection must subscribe through."""
        if self._broadcaster is not None:
            return self._broadcaster
        if self._app_state is not None:
            return getattr(self._app_state, "event_broadcaster", None)
        return None

    def _resolve_canonical_event_hook(self) -> Any | None:
        """The per-frame hook for directly-drained (non-default) connections."""
        if self._on_canonical_event is not None:
            return self._on_canonical_event
        if self._app_state is not None:
            recorder = getattr(self._app_state, "run_recorder", None)
            if recorder is not None:
                return recorder.handle_event
        return None


    def get_connection(self, profile: str) -> ProfileConnection | None:
        return self._connections.get(profile)

    def list_profiles(self) -> list[dict[str, Any]]:
        """`[{name, model, connected}, ...]` for `GET /api/profiles`."""
        return [
            {"name": conn.profile, "model": conn.model, "connected": conn.connected}
            for conn in self._connections.values()
        ]


    async def _ensure_adapter_connected(self, adapter: HermesAdapter) -> None:
        """Mirrors `domain.hermes_runtime._ensure_connected` without importing
        it -- this module stays independent of `api/` (see its own module
        docstring on the `api.main` import cycle it already avoids).
        """
        if adapter.is_connected:
            return
        lock = getattr(self._app_state, "hermes_connect_lock", None) if self._app_state else None
        if lock is None:
            if not adapter.is_logged_in:
                await adapter.login()
            await adapter.connect()
            return
        async with lock:
            if adapter.is_connected:
                return
            if not adapter.is_logged_in:
                await adapter.login()
            await adapter.connect()

    async def reconcile(self) -> None:
        """One pass: launch missing connections, restart dead ones, tear down gone ones."""
        await self._rebind_default_if_stale()
        base_adapter = self._resolve_base_adapter()
        try:
            await self._ensure_adapter_connected(base_adapter)
            profiles = await base_adapter.profiles_list()
        except HermesError:
            logger.warning("profiles.list failed; skipping this reconciliation pass", exc_info=True)
            return

        rows = profiles.get("profiles") if isinstance(profiles, dict) else None
        if not isinstance(rows, list):
            logger.warning("profiles.list returned an unusable shape: %r", profiles)
            return

        wanted: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if isinstance(name, str) and name:
                wanted[name] = row
        if not wanted:
            wanted[self._default_profile_name] = {"is_default": True}

        for name, row in wanted.items():
            if name not in self._connections:
                await self._start(name, row)

        for name, conn in list(self._connections.items()):
            if conn.is_default or conn.process is None:
                continue
            if not self._launcher.is_alive(conn.process):
                logger.info("profile %r connection process died; restarting", name)
                await self._teardown(name)
                await self._start(name, wanted.get(name, {}))

        for name in list(self._connections):
            if name not in wanted:
                logger.info("profile %r no longer listed; tearing down its connection", name)
                await self._teardown(name)

    async def run(self, interval_s: float) -> None:
        """Production timer wrapper: `reconcile()` on an interval, forever."""
        while True:
            try:
                await self.reconcile()
            except Exception:  # pragma: no cover - defensive
                logger.exception("profile reconciliation pass failed; continuing")
            await asyncio.sleep(interval_s)

    def start(self, interval_s: float) -> None:
        """Create the `run()` timer task; a no-op if one is already running."""
        if self._run_task is not None:
            return
        self._run_task = asyncio.create_task(
            self.run(interval_s), name="profile-connection-reconcile"
        )

    async def ensure_default_capture(self) -> None:
        """Start the DEFAULT profile's chat capture, with no reconciliation."""
        name = self._default_profile_name
        conn = self._connections.get(name)
        if conn is not None and conn.pump_task is not None and not conn.pump_task.done():
            return
        await self._start(name, {"is_default": True})

    def start_default_capture(self) -> None:
        """`ensure_default_capture()` as a task, for a synchronous caller."""
        if self._capture_task is not None and not self._capture_task.done():
            return
        self._capture_task = asyncio.create_task(
            self.ensure_default_capture(), name="default-profile-chat-capture"
        )

    async def close(self) -> None:
        if self._capture_task is not None:
            self._capture_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await self._capture_task
            self._capture_task = None
        if self._run_task is not None:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None
        for name in list(self._connections):
            await self._teardown(name)

    async def _rebind_default_if_stale(self) -> None:
        """Re-point the default connection at the current base adapter, if it moved."""
        conn = self._connections.get(self._default_profile_name)
        if conn is None or not conn.is_default:
            return
        current = self._resolve_base_adapter()
        if conn.adapter is current:
            return
        logger.info(
            "default profile's base adapter changed identity since the last "
            "reconciliation pass; rebinding its connection instead of "
            "pumping the stale one"
        )
        if conn.pump_task is not None:
            conn.pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await conn.pump_task
        conn.adapter = current
        conn.pump_task = asyncio.create_task(self._launch_pump(conn))


    async def _start(self, name: str, row: dict[str, Any]) -> None:
        is_default = bool(row.get("is_default")) or name == self._default_profile_name
        model = row.get("model") if isinstance(row.get("model"), str) else None

        if is_default:
            adapter = self._resolve_base_adapter()
            process = None
        else:
            if self._adapter_factory is None:
                logger.warning(
                    "no adapter_factory configured; cannot open a connection for profile %r",
                    name,
                )
                return
            try:
                process = await self._launcher.launch(name)
            except HermesError:
                logger.warning(
                    "could not launch profile %r; will retry next pass", name, exc_info=True
                )
                return
            adapter = self._adapter_factory(name, process.port)
            try:
                await self._ensure_adapter_connected(adapter)
            except HermesError:
                logger.warning(
                    "profile %r: dashboard launched but could not connect; will retry next pass",
                    name,
                    exc_info=True,
                )
                await self._launcher.stop(process)
                return

        conn = ProfileConnection(
            profile=name, adapter=adapter, is_default=is_default, model=model, process=process
        )
        self._connections[name] = conn
        conn.pump_task = asyncio.create_task(self._launch_pump(conn))

    async def _teardown(self, name: str) -> None:
        conn = self._connections.pop(name, None)
        if conn is None:
            return
        if conn.pump_task is not None:
            conn.pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await conn.pump_task
        if not conn.is_default:
            with contextlib.suppress(Exception):
                await conn.adapter.close()
            if conn.process is not None:
                with contextlib.suppress(Exception):
                    await self._launcher.stop(conn.process)


    async def _launch_pump(self, conn: ProfileConnection) -> None:
        """Dispatch to the right pump coroutine for this connection."""
        if conn.is_default:
            broadcaster = self._resolve_broadcaster()
            if broadcaster is not None:
                await self._pump_default_via_broadcaster(conn, broadcaster)
                return
            if not self._warned_no_broadcaster:
                self._warned_no_broadcaster = True
                logger.warning(
                    "default profile connection has no EventBroadcaster to "
                    "subscribe through; falling back to draining its adapter "
                    "directly (B-137) -- safe only if nothing else drains the "
                    "same adapter's event queue"
                )
        await self._pump(conn)

    async def _pump_default_via_broadcaster(
        self, conn: ProfileConnection, broadcaster: Any
    ) -> None:
        """Capture the default profile's stream through the shared broadcaster."""
        while True:
            try:
                async with broadcaster.subscribe() as queue:
                    while True:
                        frame = await queue.get()
                        if not isinstance(frame, dict):
                            continue
                        if frame.get("type") == _DESYNCHRONIZED_TYPE:
                            logger.warning(
                                "profile %r: default connection's chat-store "
                                "capture fell behind the event stream and was "
                                "desynchronized; resubscribing (events in the "
                                "gap were not captured)",
                                conn.profile,
                            )
                            break
                        payload = frame.get("payload")
                        if isinstance(payload, dict):
                            origin = payload.get(_PROFILE_FIELD)
                            if isinstance(origin, str) and origin and origin != conn.profile:
                                continue
                        if self._chat_store is not None:
                            self._chat_store.capture_event(
                                conn.profile, _EnvelopeAsCanonical(frame.get("type")), frame
                            )
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "profile %r: broadcaster-subscriber capture pump failed; resubscribing",
                    conn.profile,
                )
            await asyncio.sleep(1.0)

    async def _pump(self, conn: ProfileConnection) -> None:
        """Drain `conn.adapter.events()` for the connection's whole lifetime."""
        while True:
            try:
                async for raw_event in conn.adapter.events():
                    try:
                        self._forward_one(conn, raw_event)
                    except Exception:  # pragma: no cover - defensive
                        logger.exception(
                            "profile %r: failed to capture an event; continuing",
                            conn.profile,
                        )
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "profile %r: event pump ended unexpectedly; restarting in %ss",
                    conn.profile,
                    _PUMP_RESTART_DELAY_S,
                )
            await asyncio.sleep(_PUMP_RESTART_DELAY_S)

    def _forward_one(self, conn: ProfileConnection, raw_event: dict[str, Any]) -> None:
        context = EventContext(project_id=None, session_id=None, run_id=None, seq=0)
        canonical = normalize_event(raw_event, context)
        if canonical is None:
            return
        envelope = canonical.to_dict()
        payload = envelope.get("payload")
        if isinstance(payload, dict):
            self._stamp_identity(conn, raw_event_type(raw_event), payload)
        hook = self._resolve_canonical_event_hook()
        if hook is not None:
            try:
                hook(canonical, envelope, conn.profile)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "profile %r: canonical-event hook failed; the frame is still captured",
                    conn.profile,
                )
        if self._chat_store is not None:
            self._chat_store.capture_event(conn.profile, canonical, envelope)
        broadcaster = self._resolve_broadcaster()
        inject = getattr(broadcaster, "inject", None)
        if inject is not None:
            try:
                inject(envelope, profile=conn.profile)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "profile %r: failed to fan a frame out to live subscribers; "
                    "the frame is still captured",
                    conn.profile,
                )

    def _stamp_identity(
        self, conn: ProfileConnection, raw_type: str | None, payload: dict[str, Any]
    ) -> None:
        """The B-29 stamping rule, scoped to one profile's own connection."""
        stored_id, live_id = session_identity_from_payload(raw_type, payload)

        cache = conn.live_handle_cache
        if stored_id is not None and live_id is not None:
            conn.live_to_stored.pop(live_id, None)
            conn.live_to_stored[live_id] = stored_id
            while len(conn.live_to_stored) > _MAX_LIVE_TO_STORED:
                del conn.live_to_stored[next(iter(conn.live_to_stored))]
            observe = getattr(cache, "observe_live_mapping", None)
            if observe is not None:
                observe(stored_id, live_id)
        elif live_id is not None:
            stored_id = conn.live_to_stored.get(live_id)
            if stored_id is None:
                lookup = getattr(cache, "stored_for_live", None)
                if lookup is not None:
                    stored_id = lookup(live_id)

        payload[_STORED_SESSION_ID_FIELD] = stored_id
        payload[_LIVE_SESSION_ID_FIELD] = live_id
