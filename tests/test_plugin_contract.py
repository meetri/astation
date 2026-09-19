"""Contract tests for the astation Hermes plugin.

These run WITHOUT Hermes installed. They cover the failures that are silent in
production, which is the only reason they exist:

  * A malformed `plugin.yaml` or `dashboard/manifest.json` means Hermes never
    mounts the routes and every endpoint 404s, with one line in a log.
  * A `plugin_api.py` that raises at import does the same.
  * A hook callback that raises is SWALLOWED by Hermes, so a broken capture
    path looks exactly like an idle one.
  * A hook callback that blocks stalls the agent's turn it runs inside.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import yaml

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_entry(monkeypatch, home: Path):
    """Import plugin/__init__.py standalone, with HERMES_HOME redirected."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    spec = importlib.util.spec_from_file_location("trg_plugin_entry", PLUGIN_DIR / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trg_plugin_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeCtx:
    """Stands in for Hermes's PluginContext."""

    def __init__(self, reject: set[str] | None = None):
        self.hooks: dict[str, object] = {}
        self.cli: list[str] = []
        self.profile_name = "default"
        self._reject = reject or set()

    def register_hook(self, name, callback):
        if name in self._reject:
            raise ValueError(f"unknown hook: {name}")
        self.hooks[name] = callback

    def register_cli_command(self, name, help, setup_fn, handler_fn=None, description=""):
        self.cli.append(name)


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def test_plugin_yaml_is_valid_and_declares_no_capabilities():
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert data["name"] == "astation"
    assert data["api_version"] == 1
    assert "version" in data and "description" in data
    # Exactly ONE capability, and it must be the LLM profile override. That
    # one is load-bearing: without it the host refuses to run a completion on
    # a named profile's model, and "rewrite for listening" goes back to the
    # 503 that B-195 is. Anything MORE than this should be a deliberate
    # decision, because each capability is a consent prompt that fails closed
    # in a non-interactive install.
    assert data.get("capabilities") == ["llm.profile_override"], (
        "the plugin should declare exactly llm.profile_override; "
        f"got {data.get('capabilities')!r}"
    )


def test_dashboard_manifest_points_at_an_existing_api_file():
    manifest = json.loads((PLUGIN_DIR / "dashboard" / "manifest.json").read_text())
    assert manifest["name"] == "astation"
    api = manifest.get("api")
    assert api, "manifest must declare `api` or Hermes mounts no routes"
    assert (PLUGIN_DIR / "dashboard" / api).exists()


def test_plugin_api_exposes_a_module_level_router():
    """Hermes does getattr(module, 'router'); anything else mounts nothing.

    Parsed rather than imported: importing needs FastAPI and the gateway
    package, and this assertion is about the module's shape, not its behaviour.
    """
    tree = ast.parse((PLUGIN_DIR / "dashboard" / "plugin_api.py").read_text())
    names = {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "router" in names


def test_plugin_api_has_no_import_time_side_effects_that_can_raise():
    """Every top-level statement must be import-safe.

    A raise during import means `_mount_plugin_api_routes` logs a warning and
    moves on, leaving every route 404. Guard the shape: no bare calls at module
    level beyond the known-safe ones.
    """
    tree = ast.parse((PLUGIN_DIR / "dashboard" / "plugin_api.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            assert name in {"getLogger"}, f"unguarded module-level call: {name}"


# --------------------------------------------------------------------------
# register()
# --------------------------------------------------------------------------


def test_register_registers_every_capture_hook(tmp_path, monkeypatch):
    mod = _load_entry(monkeypatch, tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    assert set(ctx.hooks) == set(mod.CAPTURE_HOOKS)
    assert mod.failed_hooks == {}
    assert "trg" in ctx.cli


def test_register_survives_a_hook_name_this_build_rejects(tmp_path, monkeypatch):
    """An unknown hook name must degrade, not abort registration.

    Hook names have changed across Hermes versions. Losing one capture is
    recoverable; a raised exception in register() disables the whole plugin.
    """
    mod = _load_entry(monkeypatch, tmp_path)
    ctx = FakeCtx(reject={"post_api_request"})
    mod.register(ctx)
    assert "post_api_request" in mod.failed_hooks
    assert set(ctx.hooks) == set(mod.CAPTURE_HOOKS) - {"post_api_request"}


def test_register_writes_a_status_file_readable_by_the_routes(tmp_path, monkeypatch):
    mod = _load_entry(monkeypatch, tmp_path)
    mod.register(FakeCtx())
    status = json.loads(mod.status_path().read_text())
    assert status["registered_hooks"] == list(mod.CAPTURE_HOOKS)
    assert status["failed_hooks"] == {}


# --------------------------------------------------------------------------
# Hook discipline: never raise, never block, never grow without bound
# --------------------------------------------------------------------------


def test_hook_callbacks_never_raise(tmp_path, monkeypatch):
    """Hermes swallows hook exceptions, so a raising hook is invisible."""
    mod = _load_entry(monkeypatch, tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)

    class Hostile:
        def __repr__(self):
            raise RuntimeError("boom")

    for name, cb in ctx.hooks.items():
        assert cb(session_id="s", weird=Hostile()) is None, name


def test_hook_callbacks_return_none_so_they_never_rewrite_a_payload(tmp_path, monkeypatch):
    """pre_llm_call's return value is injected into the prompt; ours must not be."""
    mod = _load_entry(monkeypatch, tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    assert ctx.hooks["pre_llm_call"](session_id="s", user_message="hi") is None


def test_hook_callback_is_fast_enough_to_run_inside_a_turn(tmp_path, monkeypatch):
    """Hooks run inside the agent's turn under a 30 s budget shared by all
    plugins. Enqueue-and-return must stay far below that."""
    mod = _load_entry(monkeypatch, tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    cb = ctx.hooks["post_tool_call"]
    start = time.perf_counter()
    for _ in range(1000):
        cb(session_id="s", tool_name="t", result="x" * 1000, duration_ms=1)
    per_call_ms = (time.perf_counter() - start) * 1000 / 1000
    assert per_call_ms < 1.0, f"{per_call_ms:.3f} ms per hook call is too slow"


def test_queue_drops_rather_than_blocking_when_full(tmp_path, monkeypatch):
    """If the drain task dies, hooks must drop events, never block the turn."""
    mod = _load_entry(monkeypatch, tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    cb = (
        ctx.hooks["on_stream_delta"]
        if "on_stream_delta" in ctx.hooks
        else ctx.hooks["post_llm_call"]
    )

    while not mod.events.full():
        mod.events.put_nowait({"filler": True})

    start = time.perf_counter()
    assert cb(session_id="s") is None
    assert (time.perf_counter() - start) < 0.5, "a full queue must not block the hook"
    assert mod.stats["dropped"] >= 1


def test_hook_callbacks_are_thread_safe(tmp_path, monkeypatch):
    """Hooks fire on a different thread from the routes (measured on the
    deploy host), so concurrent callers must not corrupt the counters."""
    mod = _load_entry(monkeypatch, tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    cb = ctx.hooks["post_llm_call"]

    def hammer():
        for _ in range(200):
            cb(session_id="s")

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert mod.stats["errors"] == 0
    assert mod.stats["queued"] + mod.stats["dropped"] == 8 * 200
