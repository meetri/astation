"""Research Gateway routes, mounted inside the Hermes dashboard process."""

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

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    Path(os.environ["TRG_GATEWAY_SRC"]) if os.environ.get("TRG_GATEWAY_SRC") else None,
    _HERE.parent / "gateway",
    _HERE.parent.parent / "services" / "research-gateway",
]


def _apply_loopback_defaults() -> dict[str, str]:
    """Point the gateway's Hermes client at the dashboard hosting this plugin."""
    applied: dict[str, str] = {}

    def default(key: str, value: str | None) -> None:
        if value and not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value if "PASSWORD" not in key else "<set>"

    default("HERMES_SCHEME", "http")
    default("HERMES_HOST", "127.0.0.1")
    default("HERMES_PORT", os.environ.get("HERMES_DASHBOARD_PORT") or "9119")
    default("HERMES_USERNAME", os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_USERNAME"))
    default("HERMES_PASSWORD", os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"))

    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    root = home / "astation"
    default("RESEARCH_GATEWAY_DB_PATH", str(root / "research.db"))
    default("RESEARCH_GATEWAY_ARTIFACT_ROOT", str(root / "artifacts"))
    # Absolute: the defaults are relative and resolve against the dashboard's working directory.
    default("TTS_PIPER_VOICE_DIR", str(root / "piper-voices"))
    default("HF_HOME", str(root / "hf-cache"))
    # auth.py fails closed when these are unset; the random password can never be presented.
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
    """This PROCESS's audit-forwarder counters, for `GET /health`."""
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

        # Mount order sets route precedence; it matches api/main.py and must not be re-sorted.
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
            (attachment_serve_router, "attachment_serve"),
        ]
        for sub, _name in _routers:
            # The second /api segment is what keeps the app's existing route strings valid.
            router.include_router(sub, prefix="/api")
        log.info("astation: mounted %d gateway routers from %s", len(_routers), _gateway_src)
    except Exception as exc:
        _import_error = f"{type(exc).__name__}: {exc}"
        log.exception("astation: failed to import gateway routers")


def _hermes_app():
    """Hermes's FastAPI app, fetched lazily."""
    for name in ("hermes_cli.web_server", "hermes_cli.web_server_chat"):
        mod = sys.modules.get(name)
        app = getattr(mod, "app", None) if mod else None
        if app is not None:
            return app
    return None


def _migrate() -> str:
    """Back up the workspace database, then bring it to the Alembic head."""
    from alembic import command
    from alembic.config import Config

    from config.settings import get_settings

    settings = get_settings()
    db_path = Path(settings.research_gateway_db_path)
    if not db_path.is_absolute():
        db_path = (Path(str(_gateway_src)) / db_path).resolve()

    for candidate in (
        Path(str(_gateway_src)).parent.parent / "migrations",
        Path(str(_gateway_src)).parent / "migrations",
        _HERE.parent / "migrations",
    ):
        if (candidate / "env.py").exists():
            script_location = candidate
            break
    else:
        raise RuntimeError("alembic migrations directory not found")

    cfg = Config()
    cfg.set_main_option("script_location", str(script_location))
    # Empty on purpose: migrations/env.py derives the URL from settings.
    cfg.set_main_option("sqlalchemy.url", "")

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
    """The direct-filesystem helpers, imported from the plugin directory."""
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
    """An adapter that authenticates the way THIS dashboard was started."""
    from adapters.hermes.client import HermesAdapter

    class InProcessHermesAdapter(HermesAdapter):
        def _gate_engaged(self) -> bool:
            app = _hermes_app()
            return bool(getattr(getattr(app, "state", None), "auth_required", False))

        async def login(self) -> None:
            if self._gate_engaged():
                await super().login()
                return
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
    """The SAME module object `register()` ran in."""
    direct = sys.modules.get("hermes_plugins.astation")
    if direct is not None:
        return direct
    for name, mod in list(sys.modules.items()):
        if name.startswith("hermes_plugins.") and hasattr(mod, "CAPTURE_HOOKS"):
            return mod
    return None


@router.on_event("startup")
async def _start_gateway_services() -> None:
    """Build the gateway's services onto Hermes's ``app.state``."""
    global _started, _startup_error
    if _started or _import_error:
        return
    app = _hermes_app()
    if app is None:
        _startup_error = "could not reach Hermes's FastAPI app"
        log.error("astation: %s", _startup_error)
        return
    try:
        global _migration_status
        # Migrate before startup(): it opens the database and every route assumes the schema.
        _migration_status = _migrate()
        log.info("astation: %s", _migration_status)

        from api.bootstrap import startup
        from config.settings import get_settings
        from domain.sandbox_fs import DirectSandboxFS

        settings = get_settings()
        adapter_cls = _build_adapter_class()

        startup(
            app.state,
            settings,
            adapter_factory=adapter_cls,
            sandbox_fs=DirectSandboxFS(settings.hermes_sandbox_root),
        )
        _started = True

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
                # Returning None here would fall back to the HTTP path and repeat its 503.
                except Exception as exc:
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
    """Whether foreign prompts cost a `session.resume` or come from a hook."""
    app = _hermes_app()
    fpc = getattr(getattr(app, "state", None), "foreign_prompt_capture", None)
    if fpc is None:
        return "no capture wired"
    return "hook" if getattr(fpc, "_prompt_source", None) is not None else "session.resume"


@router.get("/health")
async def health() -> dict[str, Any]:
    """Plugin liveness and wiring report."""
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
        "sandbox_fs": _sandbox_fs_mode(),
        "artifacts": _artifact_ingest_stats(),
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


_WS_CORE_CANDIDATES = ("hermes_cli.web_server_chat", "hermes_cli.web_server")
_ws_rejections: list[dict] = []


def _resolve_ws_core():
    """Find Hermes's WebSocket auth helpers."""
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
    """Authorize a WS upgrade using Hermes's own gate. FAILS CLOSED."""
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
    """The same stream `api/main.py` serves, authenticated Hermes's way."""
    allowed, reason = _ws_authorized(websocket)
    if not allowed:
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
