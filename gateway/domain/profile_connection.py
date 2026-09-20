"""`ProfileConnectionManager` -- one `HermesAdapter` per Hermes profile.

See `docs/CHAT_HISTORY_DESIGN.md` §4 for the design this implements. Replaces
the single global `HermesAdapter`/`EventBroadcaster` pairing the gateway has
today (`api/main.py`) with a small, automatically-managed set of them, one
per profile:

    ProfileConnectionManager
      - polls profiles.list on the "default"/unified connection every N minutes
      - for each profile with no live connection:
          launch  hermes -p <profile> dashboard --isolated --port 0 --no-open --skip-build
          (the default profile uses the existing unified dashboard -- no new process)
          parse HERMES_DASHBOARD_READY port=<port> from its stdout
          open a HermesAdapter against that port, same as today
      - for each profile whose process has died: relaunch
      - for a profile removed from profiles.list: close its connection, stop its process

**This module's own test suite never spawns a real process or calls the real
Hermes instance.** The actual `hermes -p <profile> dashboard --isolated
--port 0` launch (parsing `HERMES_DASHBOARD_READY port=<port>` from stdout)
is behind `ProfileLauncher`, an injectable interface. Two concrete launchers
ship here: `NullProfileLauncher`, which always refuses to launch (so a
deployment that has not wired a real one fails loudly rather than silently
pretending a profile connected -- this is what every test in this repo uses
by default), and `SubprocessProfileLauncher`, which does the real spawn --
see its own docstring. Every test that exercises `SubprocessProfileLauncher`
supplies a fake `subprocess_factory` instead of touching a real process, so
its stdout-parsing/timeout logic is covered without a live host; wiring it in
as `api.main.lifespan`'s *active* launcher is a separate, settings-gated
decision (`RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S`), left at `0`
(disabled) by default -- see `api/main.py`.

`reconcile()` is one pass and is what a test calls directly (`await
manager.reconcile()`) rather than driving a sleep loop; `run(interval_s)` is
the production wrapper that calls it on a timer, mirroring the polling
cadence the design's diagram describes ("polls `profiles.list` ... every N
minutes"). The design note "for each profile whose process has died:
relaunch, same backoff `EventBus` already has" describes a backoff mechanism
that does not actually exist anywhere in this codebase today (`EventBroadcaster`
reconnects on-demand per request, with no timer or backoff of its own) --
flagged rather than invented; `reconcile()` restarts a dead connection
immediately on the pass that notices it, and the polling interval itself is
what rate-limits how often that can happen.

## The base-adapter staleness hazard, and how this module avoids it

Every existing test in this suite (and, in production, a real Hermes
reconnect) can replace `app.state.hermes_adapter` with a different object
after this manager was constructed (`tests/test_main.py`'s `fake_client`
fixture is the pattern; see `api.main.lifespan`'s comment on the same
hazard). A manager built once against a captured adapter *reference* would
keep reconciling against the stale, replaced object for the rest of the
process's life. `ProfileConnectionManager` avoids this the same way
`api/snapshot_sweep.py`'s `SnapshotSweeper` does: it never stores the base
adapter directly. Construct it with `app_state=` (an object with a
`hermes_adapter` attribute, i.e. FastAPI's `app.state`) and
`_resolve_base_adapter()` re-reads `app_state.hermes_adapter` on every call,
never once at construction. `reconcile()` additionally checks, on every
pass, whether the *existing* default-profile connection's `.adapter` is
still identical to what `_resolve_base_adapter()` returns right now; if not
(the swap already happened before this manager noticed), it rebinds that
connection to the new adapter and restarts its capture pump, rather than
silently continuing to pump the old one. Tests that want a single fixed
adapter for the manager's whole life (no app.state indirection) still pass
`base_adapter=` directly, unchanged from before this fix.

## B-136, the last mile: non-default frames reach `/ws/events` too

`_forward_one` ends by handing the (normalized, identity-stamped, recorder-
attributed) envelope to `EventBroadcaster.inject(envelope, profile=...)`,
which tags it `_profile` and fans it out exactly like a default frame. Until
that call existed a non-default turn was captured and recorded but never
streamed -- the Run Inspector filled up while the chat sat still (owner,
2026-09-06). The default connection's own capture pump therefore has to skip
frames tagged for another profile (they were already captured under that
profile by the pump that injected them) -- see `_pump_default_via_broadcaster`.

## B-137: the default profile's capture must not drain the shared adapter directly

The `default` profile's connection reuses `app.state.hermes_adapter` --
literally the same object `EventBroadcaster._run` (`domain/event_stream.py`) already drains for
the whole app's `/ws/events` fan-out. `HermesAdapter.events()` is backed by
one shared `asyncio.Queue`, and `Queue.get()` hands each item to exactly one
waiter: two concurrent `async for ... in adapter.events()` loops split the
stream instead of each seeing all of it (`EventBroadcaster`'s own docstring;
this is the exact B-08 failure class). So unlike every other profile's
connection -- each with its own dedicated `HermesAdapter` from a
separately-launched dashboard process, with nobody else draining it, where a
direct `adapter.events()` loop is correct -- the `default` connection's pump
must instead subscribe through `EventBroadcaster.subscribe()`, the same
internal-subscriber seam `ArtifactIngestor`/`SandboxDiffIngestor`/
`MediaTagIngestor` (`api/artifacts.py`) already use to watch the live stream
without stealing frames from anyone else. That needs the broadcaster
available here: `broadcaster=` (a fixed object, mirroring `base_adapter=`) or,
in `app_state=` mode, resolved from `app_state.event_broadcaster` the same
way `_resolve_base_adapter()` resolves `app_state.hermes_adapter` -- see
`_resolve_broadcaster()`. Frames off that queue are already normalized and
identity-stamped by the broadcaster (a `CanonicalEvent.to_dict()`), so the
default connection's broadcaster-fed pump hands them to `ChatStore` as-is
rather than re-running `normalize_event`/`_stamp_identity`, which would not
only be redundant but would consult this connection's own (for `default`,
never-populated) `live_to_stored` cache instead of the global
`LiveHandleCache` the broadcaster already resolved against. If no broadcaster
can be resolved (only possible in a test built with `base_adapter=` and no
`broadcaster=`), the default connection falls back to draining its adapter
directly, logged once -- safe only because nothing else in that test is
consuming the same fixed fake adapter's queue; production always wires a
broadcaster and must never take this path.
"""

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

#: Seconds before a non-default profile's pump is restarted after it died
#: on an unexpected exception (CLEANUP_PLAN 3.9). Tests shrink this.
_PUMP_RESTART_DELAY_S = 1.0

#: Cap on `ProfileConnection.live_to_stored`, matching `LiveHandleCache`'s.
_MAX_LIVE_TO_STORED = 256

#: The B-29 attribution keys `ChatStore.capture_event` reads (see
#: `domain/chat_store.py`) and the B-136 profile tag. These used to be
#: by-literal copies (the broadcaster lived in `api.main`, which this module
#: cannot import); they are the broadcaster's own constants now
#: (`domain/event_stream.py`, CLEANUP_PLAN step 3.1), under the local names
#: the rest of this module and its tests already use.
_STORED_SESSION_ID_FIELD = STORED_SESSION_ID_FIELD
_LIVE_SESSION_ID_FIELD = LIVE_SESSION_ID_FIELD
_PROFILE_FIELD = PROFILE_FIELD

#: The control-frame type `EventBroadcaster` queues for a subscriber that fell
#: behind (`domain/event_stream.py`'s `desynchronized_frame`) -- pinned equal
#: to `api/artifacts.py`'s own `_DESYNCHRONIZED_TYPE` by a test. Only
#: meaningful on the default connection's `broadcaster.subscribe()` path
#:; the direct-drain path (`_pump`) never sees a synthetic frame like
#: this one.
_DESYNCHRONIZED_TYPE = STREAM_DESYNCHRONIZED_EVENT_TYPE


@dataclass(frozen=True)
class _EnvelopeAsCanonical:
    """Just enough of a `CanonicalEvent` for `ChatStore.capture_event`.

    `broadcaster.subscribe()` hands back already-`to_dict()`-ed envelopes, not
    `CanonicalEvent` objects -- but `ChatStore.capture_event(profile,
    canonical, envelope)` only ever reads `canonical.type` (see
    `domain/chat_store.py::_capture`; `payload` is read off `envelope`
    itself). This shim supplies exactly that one attribute rather than
    reconstructing a real `CanonicalEvent` from a dict that already lost its
    `timestamp`/`event_id` typing.
    """

    type: str | None


@dataclass(frozen=True)
class LaunchedProcess:
    """What a `ProfileLauncher` hands back for one non-default profile.

    Opaque to the manager beyond `port` -- `handle` is whatever the launcher
    needs to keep track of the process (a real launcher would keep a
    `subprocess.Popen`; a fake one for tests can put anything there, or
    nothing).
    """

    profile: str
    port: int
    handle: Any = None


class ProfileLauncher(abc.ABC):
    """Spawns/stops/health-checks the OS process behind one profile's isolated dashboard.

    The only thing in this module that would ever touch a real process or
    network port. Production wiring (not built by this task -- see the module
    docstring) implements `launch()` as
    `hermes -p <profile> dashboard --isolated --port 0 --no-open --skip-build`,
    parses `HERMES_DASHBOARD_READY port=<port>` from its stdout, and returns
    once that line is seen. Every test in this repo supplies a fake
    implementation instead.
    """

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
    """The default concrete launcher: refuses every launch.

    A deployment that has not wired a real launcher must fail loudly (a
    profile simply never gets a connection, logged once) rather than pretend
    to have spawned something. Used by every test in this repo unless a test
    is specifically about `SubprocessProfileLauncher` below.
    """

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


#: `(argv) -> an object with an async .stdout.readline(), .returncode,
#: .terminate()/.kill() and an async .wait()` -- i.e. the subset of
#: `asyncio.subprocess.Process` this launcher actually touches. The default
#: factory is `asyncio.create_subprocess_exec`; tests inject a fake instead so
#: this class is never exercised against a real process (see module
#: docstring).
SubprocessFactory = Callable[[list[str]], Awaitable[Any]]


async def _default_subprocess_factory(argv: list[str]) -> Any:  # pragma: no cover - real I/O
    return await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )


class SubprocessProfileLauncher(ProfileLauncher):
    """The real launcher: `hermes -p <profile> dashboard --port 0 --isolated --no-open --skip-build`.

    Never invoked for the `default` profile -- `ProfileConnectionManager._start`
    only calls `launch()` for a non-default profile, reusing the unified
    dashboard connection for `default`, so
    this class never needs to special-case an omitted `-p` flag itself.

    Verified live 2026-09-05: the process prints exactly one line of the form
    `HERMES_DASHBOARD_READY port=<port>` on stdout once its isolated dashboard
    is ready to accept connections. `launch()` reads stdout line by line until
    it sees that line (resolving with the parsed port), the process exits
    without ever printing it (`ProfileDashboardDied`), or `ready_timeout_s`
    elapses first (`ProfileDashboardTimedOut`) -- in the timeout case the
    half-started process is killed rather than left to leak.

    The actual subprocess creation is behind `subprocess_factory` precisely so
    a test can supply a fake process object with canned stdout lines and
    assert the parsing/timeout logic without spawning anything real -- this
    class must never be exercised against a live host by this repo's tests.
    """

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
        #: kill the container-side dashboard(s) this profile already has
        #: before launching a new one, and after stopping ours. Opt-in because
        #: it needs `/proc` (Linux -- the Hermes container) and because the
        #: repo's launcher tests assert the exact command list.
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

    #: Runs *inside* the target (`command_prefix`) as `python3 -c <script>
    #: <profile> <host>`. Walks `/proc/*/cmdline` and SIGTERMs every process
    #: whose argv contains, contiguously, exactly
    #: `-p <profile> dashboard --host <host> --port 0 --isolated` -- the
    #: fingerprint of an isolated per-profile dashboard THIS launcher started.
    #: The unified dashboard (`hermes dashboard --host 0.0.0.0 --port 9119`)
    #: has no `-p` and no `--isolated`, so it can never match; that is the
    #: whole reason this is an argv match and not a `dashboard --stop`
    #: (B-138 took the production dashboard down that way on 2026-09-05).
    #: Prints one line, `REAPED <n>`. `python3` rather than `pkill` because
    #: the Hermes image is not guaranteed to ship procps but always has Python.
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
        """B-165: terminate every isolated dashboard for `profile` in the target.

        Best-effort and never raises: the count on success, `None` when the
        reaper is disabled or could not run. Called before each `launch()`
        (a previous gateway container's dashboard for this profile is still
        there, unreachable and burning memory -- measured 2026-09-06: 31
        dashboards for 6 profiles, 7.4 GiB) and after each `stop()` (killing
        our `docker exec` client leaves the container-side process alive).
        """
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
        """Scan stdout for the ready line; raise if the process exits first.

        A line that starts with the prefix but fails to parse as an int is
        logged and skipped rather than treated as fatal -- the process is
        still running and may yet print a well-formed line; only EOF (the
        process actually exiting) or the outer timeout end this early.
        """
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
        """Always kills the local client process handle, `command_prefix` or not.

        this used to run `hermes -p <profile>
        dashboard --stop` under `command_prefix`, on the assumption `-p`
        scoped it to that profile's own dashboard. It does not -- the CLI's
        own `--help` text says `--stop` "Stop[s] all running Hermes web
        server processes", full stop, no profile scoping at all. Verified
        live 2026-09-05 the hard way: it killed the production unified
        dashboard on 9119 instead of the isolated one under test, a real
        outage. Killing our local `docker exec` client instead may leave the
        process it started inside the target container running (no
        init-level signal forwarding for `exec`, unlike `run`) -- a possible
        orphan isolated dashboard is a far smaller risk than a global stop
        that can take out every profile's connection including the default
        one, so that's the trade this makes on purpose. Since B-165 the
        orphan is no longer accepted either: with `reap_orphans` on, `reap()`
        then SIGTERMs the container-side process by its exact argv.
        """
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
        # the container-side process outlives our exec client. Reap it
        # by argv fingerprint -- see `reap()`; a no-op unless `reap_orphans`.
        await self.reap(process.profile)

    def is_alive(self, process: LaunchedProcess) -> bool:
        proc = process.handle
        if proc is None:
            return False
        return proc.returncode is None


#: `(profile_name, port_or_None) -> HermesAdapter`. `port` is `None` for the
#: default profile (it reuses the base/unified adapter and this factory is
#: never called for it) and the launched port for every other profile.
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
    #: live handle -> stored id, observed off this connection's own event
    #: stream (same idea as `LiveHandleCache`, kept local and minimal here --
    #: see `ProfileConnectionManager._stamp_identity`).
    live_to_stored: dict[str, str] = field(default_factory=dict)
    #: A `LiveHandleCache` (`domain/live_handles.py`), lazily constructed by
    #: `domain.hermes_runtime.resolve_live_handle_cache` the first time a
    #: session-scoped route needs one for this connection. Safe by construction
    #: rather than by an invalidation hook: a torn-down connection's
    #: `ProfileConnection` object is discarded, never mutated in place
    #: (`ProfileConnectionManager._start` always builds a fresh one), so a
    #: stale cache can never survive a reconnect that replaces this object.
    live_handle_cache: Any = None

    @property
    def connected(self) -> bool:
        return self.adapter.is_connected


class ProfileConnectionManager:
    """Owns one `ProfileConnection` per Hermes profile, auto-provisioned.

    `reconcile()` is the whole of the logic in one pass: read `profiles.list`
    off the base connection, launch a connection for every profile that
    doesn't have one, restart any whose process has died, and tear down any
    whose profile is no longer listed. Call it directly in tests; `run()` is
    the production timer wrapper.
    """

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
        """Exactly one of `base_adapter` or `app_state` must be given.

        `base_adapter` is a fixed `HermesAdapter` object, held for the
        manager's whole life -- what every test in this module passes,
        because within one test nothing swaps it out from under the manager.

        `app_state` is an object with a `hermes_adapter` attribute (FastAPI's
        `app.state`, in production) that this manager re-reads on every call
        instead of capturing once -- see the module docstring's "base-adapter
        staleness hazard" section. `api.main.lifespan` uses this form.

        `broadcaster` (B-137, see the module docstring's own section) is the
        `EventBroadcaster` the default profile's connection must subscribe
        through instead of draining `app.state.hermes_adapter` a second time.
        Optional and independent of the `base_adapter`/`app_state` choice: a
        fixed object works with either mode, and in `app_state=` mode it is
        also resolved from `app_state.event_broadcaster` if not given here
        directly (see `_resolve_broadcaster()`) -- production wires it via
        `app_state`, tests that don't care about the default profile's event
        stream may omit it entirely.

        `on_canonical_event` is called `(canonical, envelope,
        profile)` for every frame this manager drains DIRECTLY -- i.e. every
        non-default connection. In production it is `RunRecorder.handle_event`,
        which is what makes a non-default profile's turns appear in the Run
        Inspector at all (the operator reported that they did not: runs are
        recorded from the event stream, and only the default connection's
        stream reached the recorder).

        It is deliberately NOT called on the default connection's
        broadcaster-subscriber path (`_pump_default_via_broadcaster`): the
        `EventBroadcaster` already invokes the same recorder for those frames,
        and calling it here too would open and persist every default-profile
        turn twice.
        """
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
        #: the default profile's chat capture, started independently of
        #: the reconciliation timer because it is a different job.
        self._capture_task: asyncio.Task[None] | None = None
        self._warned_no_broadcaster = False

    def _resolve_base_adapter(self) -> HermesAdapter:
        """The base/unified adapter, re-read at the point of use, never cached."""
        if self._app_state is not None:
            return self._app_state.hermes_adapter
        assert self._base_adapter is not None  # guaranteed by __init__
        return self._base_adapter

    def _resolve_broadcaster(self) -> Any | None:
        """The `EventBroadcaster` the default connection must subscribe through.

        Re-read at the point of use, same discipline as
        `_resolve_base_adapter()` -- a fixed `broadcaster=` wins if given;
        otherwise, in `app_state=` mode, `app_state.event_broadcaster` is
        consulted (defaulting to `None` via `getattr` so an `app_state` stand-in
        that predates this attribute, e.g. an older test fixture, degrades to
        "no broadcaster" rather than raising). `None` means B-137's direct-drain
        fallback applies -- see `_launch_pump`.
        """
        if self._broadcaster is not None:
            return self._broadcaster
        if self._app_state is not None:
            return getattr(self._app_state, "event_broadcaster", None)
        return None

    def _resolve_canonical_event_hook(self) -> Any | None:
        """The per-frame hook for directly-drained (non-default) connections.

        Same lazy discipline, and for the same reason, as
        `_resolve_broadcaster()`: in production this is
        `RunRecorder.handle_event`, and the recorder is built AFTER this
        manager in `api.main.lifespan` -- so an explicit
        `on_canonical_event=` passed at construction time would have to be
        `None`. Reading `app_state.run_recorder` at the point of use gets the
        real object, and by the time any pump can run (gated on `.start()`,
        itself after the recorder is on `app.state`) it is always populated.

        `getattr` with a `None` default so an `app_state` stand-in predating
        the attribute degrades to "no hook" rather than raising.
        """
        if self._on_canonical_event is not None:
            return self._on_canonical_event
        if self._app_state is not None:
            recorder = getattr(self._app_state, "run_recorder", None)
            if recorder is not None:
                return recorder.handle_event
        return None

    # -- read access for routes -------------------------------------------

    def get_connection(self, profile: str) -> ProfileConnection | None:
        return self._connections.get(profile)

    def list_profiles(self) -> list[dict[str, Any]]:
        """`[{name, model, connected}, ...]` for `GET /api/profiles`."""
        return [
            {"name": conn.profile, "model": conn.model, "connected": conn.connected}
            for conn in self._connections.values()
        ]

    # -- the reconciliation pass -------------------------------------------

    async def _ensure_adapter_connected(self, adapter: HermesAdapter) -> None:
        """Mirrors `domain.hermes_runtime._ensure_connected` without importing
        it -- this module stays independent of `api/` (see its own module
        docstring on the `api.main` import cycle it already avoids).

        Used for two different adapters, for two different reasons:

        * The **base adapter**, in `reconcile()`. Found live 2026-09-05, the
          first time `RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S` was
          ever turned on in production: `reconcile()` called
          `base_adapter.profiles_list()` directly, with nothing
          establishing the connection first. Every other Hermes call in
          this codebase goes through `_ensure_connected`/`_with_reconnect`
          (`domain/hermes_runtime.py`) precisely because a fresh adapter is
          not connected until something asks it to be -- normally the
          first phone request or `/ws/events` subscriber, neither of which
          this manager's own timer waits for.
        * A **freshly-launched non-default profile's adapter**, in
          `_start()`, for the same reason one step earlier: `_launch_pump`
          would otherwise hand an unconnected adapter straight to `_pump`,
          which drains `events()` in a bare loop with no connect step of
          its own (unlike a route, which always goes through
          `_with_reconnect` first) -- the pump would sit forever on an
          empty queue nothing is filling, capturing nothing, with no error
          at all to say why.

        Invisible to this module's own test suite because
        `FakeBaseAdapter`/`FakeProfileAdapter` both start
        `is_connected = True` by construction, unlike a real `HermesAdapter`.
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
        """One pass: launch missing connections, restart dead ones, tear down gone ones.

        Best-effort against `profiles.list` itself: if the base connection
        can't answer (Hermes unreachable), this pass is a no-op rather than
        tearing down every connection that already exists -- a transient
        upstream blip must not be read as "every profile disappeared".
        """
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
            # Never seen a profile at all (e.g. a fresh/misreporting Hermes):
            # still guarantee the default connection exists so the gateway's
            # existing single-profile behavior is unaffected.
            wanted[self._default_profile_name] = {"is_default": True}

        # 1. Launch missing.
        for name, row in wanted.items():
            if name not in self._connections:
                await self._start(name, row)

        # 2. Restart dead (non-default only -- the default profile has no
        # process of its own to die; its adapter's own reconnect path is
        # `_ensure_connected`/`_with_reconnect`, unchanged by this module).
        for name, conn in list(self._connections.items()):
            if conn.is_default or conn.process is None:
                continue
            if not self._launcher.is_alive(conn.process):
                logger.info("profile %r connection process died; restarting", name)
                await self._teardown(name)
                await self._start(name, wanted.get(name, {}))

        # 3. Tear down removed.
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
        """Create the `run()` timer task; a no-op if one is already running.

        Mirrors `SnapshotSweeper.start()`'s convention (`api/snapshot_sweep.py`):
        the caller decides whether to call this at all -- `api.main.lifespan`
        only does when `RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S > 0` --
        and `close()` cancels whatever task this creates.
        """
        if self._run_task is not None:
            return
        self._run_task = asyncio.create_task(
            self.run(interval_s), name="profile-connection-reconcile"
        )

    async def ensure_default_capture(self) -> None:
        """Start the DEFAULT profile's chat capture, with no reconciliation.

        **B-199.** Capturing the default profile's assistant and tool messages
        into `chat_messages` and launching a dashboard process per non-default
        profile are two unrelated jobs that shared one switch: the capture
        pump was only ever started from `run()`, and `run()` only starts when
        `RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S > 0`.

        The sidecar set that to 300 because it had the Docker socket and did
        launch per-profile dashboards, so capture came along for the ride.
        Inside Hermes there is nothing to launch -- one connection reaches
        every profile -- so the interval stayed at its default of 0,
        the manager never reconciled, and **no assistant or tool message was
        written to the chat store at all.** Nothing failed; the timer simply
        never ran. What the operator saw was a chat that showed the history
        copied in at migration time and almost nothing after it.

        Idempotent: a second call while the pump is alive does nothing.
        """
        name = self._default_profile_name
        conn = self._connections.get(name)
        if conn is not None and conn.pump_task is not None and not conn.pump_task.done():
            return
        await self._start(name, {"is_default": True})

    def start_default_capture(self) -> None:
        """`ensure_default_capture()` as a task, for a synchronous caller.

        `api/bootstrap.startup()` is synchronous on purpose and already
        starts its other pumps this way (`start_artifact_ingestors`,
        `start_snapshot_sweeper`), so capture joins them rather than making
        the whole builder async.
        """
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
        """Re-point the default connection at the current base adapter, if it moved.

        `_resolve_base_adapter()` already re-reads `app_state.hermes_adapter`
        on every call it's asked to make (`profiles.list`, a fresh `_start`),
        but the *existing* default `ProfileConnection.adapter` set by a prior
        `_start()` call is a plain field -- nothing re-reads it on its own.
        If the resolved adapter's identity has changed since that connection
        was built (a test or a production reconnect replaced
        `app.state.hermes_adapter` since the last pass), rebind it here rather
        than silently continuing to pump the stale, replaced object. Restarts
        the pump task against the new adapter; leaves `live_to_stored` alone
        (that mapping is about live handles the pump observed, not about
        which adapter object served them).
        """
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

    # -- lifecycle of one connection ---------------------------------------

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
            # `RuntimeError` as well as `CancelledError`: a task created on a
            # loop that has already gone cannot be awaited from this one, and
            # there is nothing left to wait for -- it was cancelled and its
            # loop is closed. Shutdown must not fail over it.
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await conn.pump_task
        if not conn.is_default:
            with contextlib.suppress(Exception):
                await conn.adapter.close()
            if conn.process is not None:
                with contextlib.suppress(Exception):
                    await self._launcher.stop(conn.process)

    # -- capture: wiring ChatStore.capture_event to a connection's stream ---

    async def _launch_pump(self, conn: ProfileConnection) -> None:
        """Dispatch to the right pump coroutine for this connection.

        The `default` connection's `adapter` IS `app.state.hermes_adapter` --
        the same object `EventBroadcaster._run` already drains for
        `/ws/events` -- so it must subscribe through `EventBroadcaster.subscribe()`
        instead of calling `conn.adapter.events()` a second time (see the
        module docstring's "B-137" section). Every non-default connection has
        its own dedicated adapter that nobody else drains, so it keeps using
        the direct `_pump()` loop unchanged.

        If this is the default connection and no broadcaster can be resolved
        (only possible with a `base_adapter=`-built manager that never passed
        `broadcaster=`), fall back to `_pump()` -- logged once so a
        deployment that meant to wire one notices, rather than silently
        reintroducing B-137. Production always resolves a broadcaster (via
        `app_state.event_broadcaster`) and must never take this branch.
        """
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
        """Capture the default profile's stream through the shared broadcaster.

        Mirrors `ArtifactIngestor.run`'s subscriber loop (`api/artifacts.py`):
        subscribe, read frames until a `stream.desynchronized` control frame
        says this consumer fell behind, then resubscribe for a fresh queue.
        Falling behind or any internal failure resubscribes rather than
        ending the pump for good -- events lost in the gap are lost captures,
        recoverable the next time that session's history is fetched, never a
        crashed task (same contract `_pump()` keeps for the direct-drain
        case).

        Frames off `broadcaster.subscribe()` are already normalized and
        identity-stamped (`EventBroadcaster._forward_one` already ran
        `normalize_event`/`_stamp_session_identity` against the global
        `LiveHandleCache` before fan-out), so this method hands them to
        `ChatStore` as-is -- it must NOT re-normalize or re-stamp them via
        `_forward_one`/`_stamp_identity` below, which are for the
        direct-drain (raw-event) path only and would consult this
        connection's own `live_to_stored` cache instead of the global one the
        broadcaster already resolved against.
        """
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
                        # the broadcaster now also carries every OTHER
                        # profile's frames (injected by `_forward_one` below,
                        # which already captured them under their own
                        # profile). Capturing those here too would file a
                        # `kimi25` turn under `default` a second time.
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
        """Drain `conn.adapter.events()` for the connection's whole lifetime.

        Mirrors `EventBroadcaster._run`'s per-frame guard: one
        malformed event must not end the pump, or this profile's capture
        goes silent for good with nothing to restart it (the same failure
        class B-14 was originally about). Used for every non-default
        connection (each has its own dedicated adapter nobody else drains),
        and as the default connection's fallback when no broadcaster is
        configured -- see `_launch_pump`.
        """
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
                return  # the stream ended on its own (adapter closed): done
            except asyncio.CancelledError:
                raise
            except Exception:
                # CLEANUP_PLAN 3.9: an ended pump used to stay ended, and the
                # adapter's queue then grew for the life of the process with
                # nobody draining it. Restart instead, like the broadcaster
                # path does.
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
            # BEFORE the chat store, and never allowed to break it:
            # the recorder stamps `envelope["run_id"]`, which is the
            # `turn_id` the capture below writes on the row -- the same
            # order the broadcaster's own `_forward_one` already has for the
            # default profile. `RunRecorder.handle_event` swallows its own
            # exceptions, so this guard is for any other hook; a hook
            # failure costs the row its turn id, never the row.
            try:
                hook(canonical, envelope, conn.profile)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "profile %r: canonical-event hook failed; the frame is still captured",
                    conn.profile,
                )
        if self._chat_store is not None:
            self._chat_store.capture_event(conn.profile, canonical, envelope)
        # B-136, the last mile: hand the frame to `/ws/events` too, tagged
        # with this profile. Last, so the envelope already carries the
        # recorder's run attribution, and guarded like the hook -- a fan-out
        # failure must not cost the capture above. Only reached on the
        # direct-drain path, i.e. never for the default connection, whose
        # frames the broadcaster forwarded itself before this manager ever
        # saw them.
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
        """The B-29 stamping rule, scoped to one profile's own connection.

        Same trap as `EventBroadcaster._stamp_session_identity`
        (`docs/PROTOCOL_VERIFIED.md`: `session_id` means the live handle on
        almost every event, the stored id on `session.title`, and both on
        `session.info`) -- the classification itself is the shared
        `session_identity_from_payload` (`domain/event_stream.py`); what
        differs is the lookup. `ProfileConnection.live_to_stored` is this
        connection's own cache; it is never shared across profiles, since
        live handles from different profiles' processes have nothing to do
        with each other.
        """
        stored_id, live_id = session_identity_from_payload(raw_type, payload)

        # Two sources, consulted in order. `live_to_stored` learns only from
        # `session.info` frames; `conn.live_handle_cache` (built lazily by
        # `domain.hermes_runtime.resolve_live_handle_cache`) learns from every
        # `session.resume` a route ran on this profile -- which is how the
        # app's own open of a session teaches this connection whose frames
        # they are, even when Hermes never pushed a `session.info` for it.
        cache = conn.live_handle_cache
        if stored_id is not None and live_id is not None:
            # Bounded like `LiveHandleCache` (CLEANUP_PLAN 3.9): re-insert to
            # refresh recency, then evict the oldest past the cap.
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
