"""astation — Hermes plugin entry point.

`register(ctx)` runs at plugin discovery, inside whatever process loaded the
plugin (the dashboard, for everything the app drives). It does two things:

  * registers the lifecycle hooks the gateway's capture path will consume, and
  * registers `hermes trg` for the operator commands that used to be
    `docker compose exec` one-liners.

The HTTP routes live in `dashboard/plugin_api.py`, which Hermes imports
separately. Nothing here imports the gateway package: `register()` may run
before the dependencies are importable, and a raised exception at discovery
disables the plugin.

Hook discipline, measured on the deploy host (plan R3):

  * Hooks are synchronous and Hermes bounds them with
    `plugins.hook_callback_timeout` (30 s). A slow hook stalls the turn it
    runs in, so every callback here appends to an in-memory queue and returns.
  * Hermes swallows hook exceptions, so a bug here is silent. Callbacks
    therefore catch their own errors and count them, and the count is
    reported by `GET /health`.
  * Hooks fire on a different thread from the routes, so the queue must be
    thread-safe.
  * Every hook carries the STORED session id, never a live handle -- which is
    why the capture path needs no live-handle mapping at all.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("astation.plugin")

# Hooks the capture path consumes, and what each one replaces. See
# docs/PLUGIN_V2_PLAN.md §4.3.
CAPTURE_HOOKS = (
    "on_session_start",  # opens a run
    "pre_llm_call",  # the user's message, including one typed in the TUI (B-190)
    "post_tool_call",  # carries the tool RESULT Hermes's transcript drops (B-156)
    "post_api_request",  # per-request token usage, not a cumulative total
    "post_llm_call",  # the assistant's reply
    "on_session_end",  # completed / interrupted, at the moment the turn ends (B-62)
    "on_session_reset",  # compaction and reset boundaries (B-175/176)
)

# Bounded: if the drain task dies, hooks must not grow the queue without limit
# and must never block the agent's turn.
MAX_QUEUED_EVENTS = 10_000

events: queue.Queue = queue.Queue(maxsize=MAX_QUEUED_EVENTS)

_stats_lock = threading.Lock()
stats: dict[str, int] = {"queued": 0, "dropped": 0, "errors": 0}


def _record(hook_name: str, kwargs: dict) -> None:
    """Enqueue one hook event. Must be fast, must never raise, never blocks."""
    try:
        item = {"hook": hook_name, "wall": time.time(), "kwargs": kwargs}
        try:
            events.put_nowait(item)
        except queue.Full:
            with _stats_lock:
                stats["dropped"] += 1
            return
        with _stats_lock:
            stats["queued"] += 1
    except Exception:
        with _stats_lock:
            stats["errors"] += 1


def _make(hook_name: str):
    def callback(**kwargs: Any):
        _record(hook_name, kwargs)
        return None  # never block a turn, never rewrite a payload

    callback.__name__ = f"trg_{hook_name}"
    return callback


registered_hooks: list[str] = []
failed_hooks: dict[str, str] = {}

#: The PluginContext `register()` was handed. The route module reaches
#: `ctx.llm` through this — it is the host-owned LLM facade, and the only way
#: to run a completion on a profile whose provider has no endpoint this
#: process could call itself.
plugin_ctx = None


def register(ctx) -> None:
    global plugin_ctx
    plugin_ctx = ctx
    registered_hooks.clear()
    failed_hooks.clear()
    for hook_name in CAPTURE_HOOKS:
        try:
            ctx.register_hook(hook_name, _make(hook_name))
            registered_hooks.append(hook_name)
        except Exception as exc:  # an unknown hook name on this Hermes build
            failed_hooks[hook_name] = str(exc)
            log.warning("astation: hook %s did not register: %s", hook_name, exc)

    try:
        ctx.register_cli_command(
            "trg",
            "Research Gateway plugin: migrate, status, backup",
            _setup_cli,
        )
    except Exception as exc:  # older builds may not expose it
        log.warning("astation: CLI command not registered: %s", exc)

    log.info(
        "astation: registered %d/%d hooks%s",
        len(registered_hooks),
        len(CAPTURE_HOOKS),
        f", failed: {sorted(failed_hooks)}" if failed_hooks else "",
    )
    _write_status(ctx)


def status_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return home / "astation" / "plugin-status.json"


def _write_status(ctx) -> None:
    """Record what register() saw, for `GET /health` and `hermes trg status`.

    register() runs in the plugin package's module namespace; the routes are a
    separately-imported module, so they cannot read these globals directly.
    A small file on disk is the seam, and it doubles as the only evidence that
    register() ran at all when a hook never fires.
    """
    try:
        path = status_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "wall": time.time(),
                    "pid": os.getpid(),
                    "profile": getattr(ctx, "profile_name", None),
                    "registered_hooks": registered_hooks,
                    "failed_hooks": failed_hooks,
                    "expected_hooks": list(CAPTURE_HOOKS),
                },
                default=str,
                indent=1,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("astation: could not write plugin status: %s", exc)


def _setup_cli(parser) -> None:
    """`hermes trg <sub>` — the operator surface that replaces docker exec."""
    sub = parser.add_subparsers(dest="trg_action")
    sub.add_parser("status", help="Show plugin wiring, hook counts and queue depth")
    sub.add_parser("migrate", help="Back up the database, then run Alembic to head")
    sub.add_parser("backup", help="Copy the research database and artifacts aside")
