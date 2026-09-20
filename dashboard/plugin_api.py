"""Research Gateway routes, mounted inside the Hermes dashboard process.

Hermes's ``_mount_plugin_api_routes()`` imports this file and does
``app.include_router(router, prefix="/api/plugins/astation")``. So every
route the gateway has always served at ``/api/x`` is served here at
``/api/plugins/astation/x``, in Hermes's own process, with no second
container, no Docker socket and no Hermes-network membership.

Three things differ from ``api/main.py``, and only three:

1. **No ``/api`` prefix and no ``require_basic_auth``.** The mount supplies the
   prefix, and Hermes's own auth gate already refuses an unauthenticated
   request to ``/api/plugins/...`` before the route runs.
   ``RESEARCH_GATEWAY_USERNAME``/``_PASSWORD`` cease to exist.
2. **No FastAPI lifespan.** A plugin does not own the app, so the services
   ``api/bootstrap.startup()`` builds are created from a router startup
   handler instead, onto Hermes's own ``app.state`` -- verified collision-free
   against every name Hermes puts there.
3. **``/health`` is not public.** Hermes's own ``GET /api/status`` is the
   unauthenticated liveness probe now.

Nothing here may raise at import time: an exception means the router never
mounts and every route silently 404s, with only a line in the dashboard log.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket

log = logging.getLogger("astation.plugin")

router = APIRouter()

# --------------------------------------------------------------------------
# Locate the gateway package.
#
# Phase 1 runs the plugin directly out of this repo, so the source tree is a
# sibling of the plugin directory. TRG_GATEWAY_SRC overrides it. Phase 4
# packaging makes the plugin self-contained and this block goes away.
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    Path(os.environ["TRG_GATEWAY_SRC"]) if os.environ.get("TRG_GATEWAY_SRC") else None,
    _HERE.parent / "gateway",  # packaged (Phase 4)
    _HERE.parent.parent / "services" / "research-gateway",  # in-repo (Phase 1)
]


def _apply_loopback_defaults() -> dict[str, str]:
    """Point the gateway's Hermes client at the dashboard hosting this plugin.

    The sidecar dialled Hermes across the network and carried its own copy of
    the credential. In-process, the answer is always loopback, and the
    credential is the one Hermes itself was started with -- so nothing has to
    be configured twice and `HERMES_HOST`/`HERMES_PORT` stop being settings a
    user can get wrong.

    Applied at IMPORT time, before the gateway package is imported at all:
    `get_settings()` is cached on first call, and a router module may make that
    call while being imported. Anything already set in the environment wins, so
    an operator can still override.
    """
    applied: dict[str, str] = {}

    def default(key: str, value: str | None) -> None:
        if value and not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value if "PASSWORD" not in key else "<set>"

    default("HERMES_SCHEME", "http")
    default("HERMES_HOST", "127.0.0.1")
    default("HERMES_PORT", os.environ.get("HERMES_DASHBOARD_PORT") or "9119")
    # Hermes's own dashboard credential. `login()` presents this straight back
    # to the process we are running inside.
    default("HERMES_USERNAME", os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_USERNAME"))
    default("HERMES_PASSWORD", os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"))

    # The gateway's own data lives under HERMES_HOME, inside the bind mount,
    # beside everything else Hermes persists.
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    root = home / "astation"
    default("RESEARCH_GATEWAY_DB_PATH", str(root / "research.db"))
    default("RESEARCH_GATEWAY_ARTIFACT_ROOT", str(root / "artifacts"))
    # Speech. Both are plain directories under the same data root, so a voice
    # download or a Whisper model survives an image rebuild exactly like the
    # database does. Without these the defaults are RELATIVE ("./data/..."),
    # which resolve against whatever the dashboard's working directory happens
    # to be -- and the symptom is a 503 saying piper is not installed even
    # when it is.
    default("TTS_PIPER_VOICE_DIR", str(root / "piper-voices"))
    default("HF_HOME", str(root / "hf-cache"))
    # The gateway's own Basic auth is gone: Hermes's gate is the only one now.
    # `api/auth.py` fails closed when these are unset, and a few routes still
    # read them, so give them a value that can never be presented externally.
    default("RESEARCH_GATEWAY_USERNAME", "plugin")
    default("RESEARCH_GATEWAY_PASSWORD", os.urandom(24).hex())
    return applied


_loopback_defaults = _apply_loopback_defaults()

_gateway_src: Path | None = None
for candidate in _CANDIDATES:
    if candidate and (candidate / "api" / "bootstrap.py").exists():
        _gateway_src = candidate.resolve()
        break

_import_error: str | None = None
_routers: list[tuple[Any, str]] = []
def _audit_forwarder_stats() -> Any:
    """This PROCESS's audit-forwarder counters, for `GET /health`.

    Read from the forwarder module's singleton rather than rebuilt here: one
    forwarder per process is started by `register()`, which runs everywhere the
    plugin loads, while this route answers only from the dashboard. A profile
    gateway has its own forwarder and its own counters, not visible from here.
    """
    module = sys.modules.get("trg_audit_forwarder")
    forwarder = getattr(module, "INSTANCE", None) if module else None
    if forwarder is None:
        return "not started"
    return forwarder.stats if forwarder.configured else "not configured"


_routers: list[tuple[Any, str]] = []
_startup_error: str | None = None
_migration_status: str | None = None
_drain: Any = None
_started = False

if _gateway_src is None:
    _import_error = f"gateway source not found; looked in {[str(c) for c in _CANDIDATES if c]}"
    log.error("astation: %s", _import_error)
else:
    if str(_gateway_src) not in sys.path:
        sys.path.insert(0, str(_gateway_src))
    try:
        # Same order as api/main.py, so route precedence is unchanged. Each
        # module names its router <module>_router, not `router`.
        from api.artifacts import artifacts_router
        from api.attachments import attachment_serve_router, attachments_router
        from api.audit import audit_router
        from api.background import background_router
        from api.chat import chat_router
        from api.commands import commands_router
        from api.compress import compress_router
        from api.config import config_router
        from api.converse import converse_router
        from api.handoff import handoff_router
        from api.instance import instance_router
        from api.profile_admin import profile_admin_router
        from api.projects import projects_router
        from api.prompts import prompts_router
        from api.rewrite import rewrite_router
        from api.runs import runs_router
        from api.sandbox import sandbox_router
        from api.sandbox_text import sandbox_text_router
        from api.sessions import sessions_router
        from api.snapshot_sweep import snapshot_sweep_router
        from api.snapshots import snapshots_router
        from api.speak import speak_router
        from api.transcribe import transcribe_router

        _routers = [
            (sessions_router, "sessions"),
            (projects_router, "projects"),
            (prompts_router, "prompts"),
            (background_router, "background"),
            (runs_router, "runs"),
            (commands_router, "commands"),
            (sandbox_router, "sandbox"),
            (sandbox_text_router, "sandbox_text"),
            (artifacts_router, "artifacts"),
            (attachments_router, "attachments"),
            (transcribe_router, "transcribe"),
            (rewrite_router, "rewrite"),
            (handoff_router, "handoff"),
            (speak_router, "speak"),
            (converse_router, "converse"),
            (instance_router, "instance"),
            (profile_admin_router, "profile_admin"),
            (snapshots_router, "snapshots"),
            (compress_router, "compress"),
            (config_router, "config"),
            (snapshot_sweep_router, "snapshot_sweep"),
            (chat_router, "chat"),
            (audit_router, "audit"),
            # api/main.py mounts this one on `app` OUTSIDE the /api prefix: it
            # serves an attachment to the Hermes sandbox host by capability
            # URL. Under the plugin it lives beside everything else. §4.4
            # replaces the whole priming-turn flow with a direct sandbox
            # write, at which point this router goes away entirely.
            (attachment_serve_router, "attachment_serve"),
        ]
        # Mounted under an extra "/api" so the full path is
        #   /api/plugins/astation/api/<route>
        # matching the "/api/..." literals the iOS app has always used -- the
        # standalone app puts them behind `APIRouter(prefix="/api")` in
        # api/main.py. Keeping the segment means none of the ~58 route strings
        # in the app change, and the plugin's own /health stays at the mount
        # root where it does not collide with a gateway route.
        for sub, _name in _routers:
            router.include_router(sub, prefix="/api")
        log.info("astation: mounted %d gateway routers from %s", len(_routers), _gateway_src)
    except Exception as exc:
        _import_error = f"{type(exc).__name__}: {exc}"
        log.exception("astation: failed to import gateway routers")


def _hermes_app():
    """Hermes's FastAPI app, fetched lazily.

    This module is imported BY web_server while that module is still executing,
    so a module-level ``from hermes_cli.web_server import app`` would bind a
    half-initialised module. By startup time it is complete, and `app` is
    whichever of the two modules actually defines it.
    """
    for name in ("hermes_cli.web_server", "hermes_cli.web_server_chat"):
        mod = sys.modules.get(name)
        app = getattr(mod, "app", None) if mod else None
        if app is not None:
            return app
    return None


def _migrate() -> str:
    """Back up the workspace database, then bring it to the Alembic head.

    The sidecar deliberately never migrated on start: a crash-looping
    container must not fire a schema migration unattended. A plugin has no
    operator step between install and load, so the rule is kept in spirit
    instead of in letter -- take a copy first, migrate, and refuse to serve if
    it fails, so a half-migrated database is never served from.

    Returns a short status string for `GET /health`.
    """
    from alembic import command
    from alembic.config import Config

    from config.settings import get_settings

    settings = get_settings()
    db_path = Path(settings.research_gateway_db_path)
    if not db_path.is_absolute():
        db_path = (Path(str(_gateway_src)) / db_path).resolve()

    for candidate in (
        Path(str(_gateway_src)).parent.parent / "migrations",  # in-repo
        Path(str(_gateway_src)).parent / "migrations",  # packaged
        _HERE.parent / "migrations",
    ):
        if (candidate / "env.py").exists():
            script_location = candidate
            break
    else:
        raise RuntimeError("alembic migrations directory not found")

    cfg = Config()
    cfg.set_main_option("script_location", str(script_location))
    cfg.set_main_option("sqlalchemy.url", "")  # migrations/env.py derives it from settings

    if db_path.exists() and db_path.stat().st_size > 0:
        stamp = time.strftime("%Y-%m-%d-%H%M%S")
        backup = db_path.with_name(f"{db_path.name}.bak-{stamp}-pre-migrate")
        shutil.copy2(db_path, backup)
        note = f"backed up to {backup.name}"
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        note = "fresh database"

    command.upgrade(cfg, "head")
    return f"migrated to head ({note})"


# --------------------------------------------------------------------------
# The in-process Hermes client.
# --------------------------------------------------------------------------


def _session_token() -> str | None:
    """Hermes's loopback session token, if this build exposes one."""
    for name in ("hermes_cli.web_server", "hermes_cli.web_server_chat"):
        mod = sys.modules.get(name)
        token = getattr(mod, "_SESSION_TOKEN", None) if mod else None
        if token:
            return str(token)
    return os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or None


_fs_module = None


def _fs():
    """The direct-filesystem helpers, imported from the plugin directory.

    Loaded by path because the plugin is not an importable package from the
    route module's point of view -- the same reason `capture.py` is.
    """
    global _fs_module
    if _fs_module is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "trg_sandbox_fs", _HERE.parent / "sandbox_fs.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["trg_sandbox_fs"] = module
        spec.loader.exec_module(module)
        _fs_module = module
    return _fs_module


def _build_adapter_class():
    """An adapter that authenticates the way THIS dashboard was started.

    Hermes engages its auth gate only on a non-loopback bind. The two modes
    want different WebSocket credentials, and the wrong one is rejected:

      * gated (the deploy host, `--host 0.0.0.0`): cookie login, then a
        single-use `?ticket=`. Exactly what the sidecar always did.
      * ungated (a loopback dashboard, which is what a friend running Hermes
        locally gets): there is no `/auth/password-login` to call, and the
        socket wants `?token=<session token>`.

    Without this the plugin works on a gated instance and fails on a loopback
    one with `password-login failed with HTTP 404` -- measured, and the reason
    this class exists.
    """
    from adapters.hermes.client import HermesAdapter

    class InProcessHermesAdapter(HermesAdapter):
        def _gate_engaged(self) -> bool:
            app = _hermes_app()
            return bool(getattr(getattr(app, "state", None), "auth_required", False))

        async def login(self) -> None:
            if self._gate_engaged():
                await super().login()
                return
            # Ungated: nothing to log into. Mark the session usable so
            # `mint_ticket`'s guard and every caller's `_logged_in` check pass.
            self._logged_in = True

        async def _ws_auth_query(self) -> str:
            if self._gate_engaged():
                return await super()._ws_auth_query()
            token = _session_token()
            if not token:
                raise RuntimeError(
                    "this Hermes dashboard is ungated but exposes no session "
                    "token, so the plugin cannot open its own WebSocket"
                )
            return f"token={token}"

        # ---- direct filesystem (§4.4) -------------------------------------
        #
        # These routes were found by probing and are documented nowhere
        # upstream; two of them confine nothing. Inside Hermes the files are
        # just on disk, so four of the five become direct I/O returning the
        # same response shapes the callers already parse -- api/sandbox.py,
        # api/sandbox_text.py, domain/prompt_files.py, project_workspace.py
        # and artifact_ingest.py are untouched.
        #
        # `files_download` is deliberately NOT overridden: it is the one file
        # endpoint Hermes confines server-side, and it streams with native
        # Range support an in-memory replacement would lose.

        async def files_list(self, path: str):  # type: ignore[override]
            return _fs().list_directory(path, self._settings.hermes_sandbox_root)

        async def fs_read_text(self, path: str):  # type: ignore[override]
            return _fs().read_text(path, self._settings.hermes_sandbox_root)

        async def fs_write_text(self, path: str, content: str):  # type: ignore[override]
            return _fs().write_text(path, content, self._settings.hermes_sandbox_root)

        async def files_mkdir(self, path: str):  # type: ignore[override]
            return _fs().make_directory(path, self._settings.hermes_sandbox_root)

    return InProcessHermesAdapter


def _plugin_entry_module():
    """The SAME module object `register()` ran in.

    Hermes loads a plugin package as `hermes_plugins.<slug>`, where the slug is
    the name with `-` replaced by `_`. Finding it in `sys.modules` matters:
    importing `__init__.py` by path here would create a SECOND module with its
    own queue, so the hooks would enqueue into one and the drain would read the
    other, and capture would silently do nothing.
    """
    direct = sys.modules.get("hermes_plugins.astation")
    if direct is not None:
        return direct
    # The slug rule could change; fall back to any loaded plugin module that
    # looks like ours rather than silently capturing nothing.
    for name, mod in list(sys.modules.items()):
        if name.startswith("hermes_plugins.") and hasattr(mod, "CAPTURE_HOOKS"):
            return mod
    return None


@router.on_event("startup")
async def _start_gateway_services() -> None:
    """Build the gateway's services onto Hermes's ``app.state``.

    This is `api/main.lifespan`'s body. `startup()` is synchronous and makes no
    network call -- it opens the DB, builds the stores and starts the
    background tasks -- so it is safe to call here, where a loop is running.
    """
    global _started, _startup_error
    if _started or _import_error:
        return
    app = _hermes_app()
    if app is None:
        _startup_error = "could not reach Hermes's FastAPI app"
        log.error("astation: %s", _startup_error)
        return
    try:
        # Schema first: `startup()` opens the database and the routes assume a
        # schema, so a migration failure must stop us before either happens.
        global _migration_status
        _migration_status = _migrate()
        log.info("astation: %s", _migration_status)

        from api.bootstrap import startup
        from config.settings import get_settings
        from domain.sandbox_fs import DirectSandboxFS

        settings = get_settings()
        adapter_cls = _build_adapter_class()

        # §4.4: the sandbox is THIS process's filesystem. Listings, text reads
        # and writes become syscalls in a worker thread instead of loopback
        # HTTP -- which matters most for the artifact diff walker, hundreds of
        # listings per turn. `DirectSandboxFS` re-implements the two guards
        # Hermes's routes were providing (real-path confinement and the
        # credential-file denylist); see its module docstring. Downloads stay
        # on Hermes's streaming route either way.
        startup(
            app.state,
            settings,
            adapter_factory=adapter_cls,
            sandbox_fs=DirectSandboxFS(settings.hermes_sandbox_root),
        )
        _started = True

        # Hook enrichment (§4.3). Separate from `startup()` because it is the
        # plugin's own addition, not part of the gateway's service graph.
        global _drain
        entry = _plugin_entry_module()
        if entry is None:
            log.warning("astation: plugin entry module not found; hook enrichment is OFF")
        else:
            sys.path.insert(0, str(_HERE.parent)) if str(_HERE.parent) not in sys.path else None
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "trg_capture", _HERE.parent / "capture.py"
            )
            cap = importlib.util.module_from_spec(spec)
            sys.modules["trg_capture"] = cap
            spec.loader.exec_module(cap)
            _drain = cap.HookDrain(entry.events, app.state)
            _drain.start()

            # Foreign prompts without a network round trip. `pre_llm_call`
            # already carried the text of a turn started outside this gateway,
            # so `ForeignPromptCapture` no longer has to pay a `session.resume`
            # (up to 1.6 MB) per foreign turn to recover it. It falls back to
            # the resume whenever the source returns nothing, so this is
            # additive.
            # Attachments land in the sandbox by a direct write now, so the
            # priming turn, the capability URL and the 15-minute verify poll
            # all stop being used (§4.4).
            orchestrator = getattr(app.state, "attachment_orchestrator", None)
            store = getattr(app.state, "artifact_store", None)
            if (
                orchestrator is not None
                and store is not None
                and hasattr(orchestrator, "set_direct_delivery")
            ):
                from config.settings import get_settings as _settings

                root = _settings().hermes_sandbox_root

                def _deliver(sandbox_path: str, storage_key: str) -> tuple[bool, str]:
                    try:
                        source = store.path_for_key(storage_key)
                    except Exception as exc:
                        return False, f"could not locate the uploaded bytes: {exc}"
                    return _fs().deliver_attachment(sandbox_path, str(source), root)

                orchestrator.set_direct_delivery(_deliver)
                log.info("astation: attachments deliver directly into the sandbox")

            # Rewrite/handoff on a profile's OWN model, run by the host.
            # Resolving a profile to an OpenAI-compatible base URL plus a key
            # of our own cannot work for a local llama.cpp server, bedrock, moa
            # or openai-codex -- the honest answer there was a 503.
            # Hermes already knows how to talk to every provider it is
            # configured with, so hand it the messages instead.
            entry_mod = entry

            async def _profile_llm(
                *, profile: str, system_prompt: str, text: str, max_tokens: int
            ) -> str | None:
                ctx = getattr(entry_mod, "plugin_ctx", None)
                llm = getattr(ctx, "llm", None) if ctx is not None else None
                if llm is None:
                    return None
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ]
                try:
                    result = await llm.acomplete(
                        messages,
                        profile=profile,
                        max_tokens=max_tokens,
                        purpose="rewrite-for-listening",
                    )
                except Exception as exc:
                    # Returning None would silently fall back to the HTTP path
                    # and produce the same 503 the operator already saw, so say
                    # what went wrong instead.
                    raise RuntimeError(f"the host LLM refused the request: {exc}") from exc
                content = getattr(result, "content", None) or getattr(result, "text", None)
                if not content and isinstance(result, dict):
                    content = result.get("content") or result.get("text")
                return content.strip() if isinstance(content, str) and content.strip() else None

            app.state.profile_llm = _profile_llm
            log.info("astation: rewrite runs on the profile's own model via the host")

            fpc = getattr(app.state, "foreign_prompt_capture", None)
            if fpc is not None and hasattr(fpc, "set_prompt_source"):
                fpc.set_prompt_source(_drain.prompt_source)
                log.info("astation: foreign prompts now served from hooks")
            else:
                log.info(
                    "astation: no foreign-prompt capture to wire; "
                    "foreign prompts keep using session.resume"
                )

            log.info("astation: hook enrichment started")
        log.info("astation: gateway services started on Hermes app.state")
    except Exception as exc:
        _startup_error = f"{type(exc).__name__}: {exc}"
        log.exception("astation: gateway startup failed")


def _foreign_prompt_mode() -> str:
    """Whether foreign prompts cost a `session.resume` or come from a hook.

    Worth reporting rather than assuming: the difference is a 1.6 MB round trip
    per turn started outside this gateway, and the fallback is silent by design.
    """
    app = _hermes_app()
    fpc = getattr(getattr(app, "state", None), "foreign_prompt_capture", None)
    if fpc is None:
        return "no capture wired"
    return "hook" if getattr(fpc, "_prompt_source", None) is not None else "session.resume"


@router.get("/health")
async def health() -> dict[str, Any]:
    """Plugin liveness and wiring report.

    Unlike the sidecar's `/health` this sits behind Hermes's auth gate, and it
    deliberately reports *why* the plugin is unhealthy: a router import failure
    or a startup failure is otherwise only visible in the dashboard log.
    """
    hooks: Any
    try:
        home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        hooks = json.loads((home / "astation" / "plugin-status.json").read_text())
        hooks.pop("wall", None)
    except Exception as exc:
        hooks = {"error": f"register() status unavailable: {exc}"}

    return {
        "status": "ok" if (_started and not _import_error) else "degraded",
        "plugin": "astation",
        "gateway_src": str(_gateway_src) if _gateway_src else None,
        "routers_mounted": len(_routers),
        "services_started": _started,
        "import_error": _import_error,
        "startup_error": _startup_error,
        "migration": _migration_status,
        "pid": os.getpid(),
        "hooks": hooks,
        "loopback_defaults": sorted(_loopback_defaults),
        "ws_core_module": _resolve_ws_core()[1] or None,
        "ws_rejections": _ws_rejections[-5:],
        "enrichment": (_drain.stats if _drain is not None else "not started"),
        "foreign_prompts": _foreign_prompt_mode(),
        # Which filesystem backend the sandbox routes ended up on. Reported
        # for the same reason `foreign_prompts` is: the fallback to HTTP is
        # silent, and the two backends differ in confinement and in whether an
        # editor save can be clobbered.
        "sandbox_fs": _sandbox_fs_mode(),
        # A5: how many paths the ingest ignore rules kept out of the library
        # this process. Reported so "why did my file not show up" is
        # answerable from here rather than from a log dive.
        "artifacts": _artifact_ingest_stats(),
        # Session attribution for the audit store. `not started` means kernel
        # events can only be tied to a session by the time window they
        # happened in; a rising `dropped` means attribution is being lost and
        # the timeline will be quietly incomplete.
        # Reported from the PLUGIN module, which starts one forwarder per
        # process. This route answers from the dashboard, so these are the
        # dashboard's own counters; a profile gateway's forwarder has its own
        # and is not visible here.
        "audit_attribution": _audit_forwarder_stats(),
    }


def _artifact_ingest_stats() -> Any:
    app = _hermes_app()
    ingestor = getattr(getattr(app, "state", None), "artifact_ingestor", None)
    if ingestor is None:
        return "not started"
    return {"ignored": getattr(ingestor, "ignored_paths", 0)}


def _sandbox_fs_mode() -> Any:
    app = _hermes_app()
    try:
        from domain.sandbox_fs import describe_backend
    except Exception as exc:
        return f"unavailable: {exc}"
    return describe_backend(getattr(app, "state", None))


# --------------------------------------------------------------------------
# The event stream the app holds open.
# --------------------------------------------------------------------------

_WS_CORE_CANDIDATES = ("hermes_cli.web_server_chat", "hermes_cli.web_server")
_ws_rejections: list[dict] = []


def _resolve_ws_core():
    """Find Hermes's WebSocket auth helpers.

    These are PRIVATE and they MOVE: 0.20.5 had them in `hermes_cli.web_server`,
    0.21.3 has them in `hermes_cli.web_server_chat`. Resolution failure must be
    its own state -- during the probe round a swallowed `AttributeError` made a
    correctly-ticketed connection indistinguishable from a rejected credential,
    which reads as "plugin WebSockets do not work". `GET /health` reports this.
    """
    import importlib

    tried: list[str] = []
    for name in _WS_CORE_CANDIDATES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            tried.append(f"{name}: import failed ({exc})")
            continue
        if hasattr(mod, "_ws_auth_ok"):
            return mod, name, tried
        tried.append(f"{name}: imported but no _ws_auth_ok")
    return None, "", tried


def _ws_authorized(socket: WebSocket) -> tuple[bool, str]:
    """Authorize a WS upgrade using Hermes's own gate. FAILS CLOSED.

    Hermes's auth gates are all `@app.middleware("http")` and Starlette never
    runs HTTP middleware for a websocket scope, so without this a plugin socket
    is reachable with no credential, no Host/Origin rebinding guard and no
    peer-IP check -- on a `--host 0.0.0.0` dashboard, an open door.

    The bundled kanban plugin does the same thing but returns True when the
    import fails. That is fail-OPEN and is deliberately not copied.
    """
    mod, mod_name, tried = _resolve_ws_core()
    if mod is None:
        return False, f"core-unresolved: {tried}"
    try:
        if not mod._ws_auth_ok(socket):
            try:
                reason = mod._ws_auth_reason(socket)[0]
            except Exception:
                reason = "unknown"
            return False, f"credential-rejected[{mod_name}]: {reason}"
        checker = getattr(mod, "_ws_request_is_allowed", None)
        if checker is not None and not checker(socket):
            return False, f"host-origin-or-peer-rejected[{mod_name}]"
        return True, f"ok[{mod_name}]"
    except Exception as exc:
        return False, f"gate-error[{mod_name}]: {type(exc).__name__}: {exc}"


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """The same stream `api/main.py` serves, authenticated Hermes's way.

    The client mints a ticket with `POST /api/auth/ws-ticket` and appends it as
    `?ticket=` (30 s TTL, single use) exactly as it would for Hermes's own
    `/api/ws`. Everything after the auth decision is `stream_events()`, shared
    with the standalone app.
    """
    allowed, reason = _ws_authorized(websocket)
    if not allowed:
        # Closing before accept() makes Starlette answer the handshake with a
        # bare 403, so the reason never reaches the client -- record it.
        _ws_rejections.append({"wall": time.time(), "reason": reason})
        del _ws_rejections[:-10]
        await websocket.close(code=1008)
        return
    if not _started:
        await websocket.accept()
        await websocket.send_json(
            {"type": "error", "payload": {"message": "gateway services are not started"}}
        )
        await websocket.close(code=1011)
        return

    app = _hermes_app()
    from api.events_ws import stream_events

    await stream_events(websocket, app.state)
