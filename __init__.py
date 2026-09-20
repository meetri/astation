"""astation — Hermes plugin entry point."""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("astation.plugin")

CAPTURE_HOOKS = (
    "on_session_start",
    "pre_llm_call",
    "post_tool_call",
    "post_api_request",
    "post_llm_call",
    "on_session_end",
    "on_session_reset",
    # The only hook with session, turn and tool-call ids alongside the command about to run.
    "pre_tool_call",
)

# Bounded: if the drain dies, hooks must still return at once and must not grow memory.
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


# Built by register(), which runs in every plugin process; the route module starts in only one.
audit_forwarder: Any = None


def _start_audit_forwarder(ctx=None) -> None:
    """Build the forwarder for THIS process. Never raises."""
    global audit_forwarder
    if audit_forwarder is not None:
        return
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "trg_audit_forwarder", Path(__file__).resolve().parent / "audit_forwarder.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("trg_audit_forwarder", module)
        spec.loader.exec_module(module)
        audit_forwarder = module.AuditForwarder(
            ingest_url=os.environ.get("AUDIT_INGEST_URL", ""),
            host_label=os.environ.get("AUDIT_HOST_LABEL", ""),
            profile=str(getattr(ctx, "profile_name", "") or "default"),
        )
        audit_forwarder.start()
        module.INSTANCE = audit_forwarder
    except Exception as exc:
        audit_forwarder = None
        log.warning("astation: audit forwarder not started: %s", exc)


def _make(hook_name: str):
    def callback(**kwargs: Any):
        # Audit rows skip the queue below: only one process drains it, hooks fire in all of them.
        if audit_forwarder is not None and hook_name in ("pre_tool_call", "post_tool_call"):
            try:
                if hook_name == "pre_tool_call":
                    audit_forwarder.record_tool_call(kwargs)
                else:
                    audit_forwarder.record_tool_result(kwargs)
            except Exception:
                with _stats_lock:
                    stats["errors"] += 1
        _record(hook_name, kwargs)
        # A non-None return would rewrite the hook's payload; capture must stay transparent.
        return None

    callback.__name__ = f"trg_{hook_name}"
    return callback


registered_hooks: list[str] = []
failed_hooks: dict[str, str] = {}

plugin_ctx = None


def register(ctx) -> None:
    global plugin_ctx
    plugin_ctx = ctx
    _start_audit_forwarder(ctx)
    registered_hooks.clear()
    failed_hooks.clear()
    for hook_name in CAPTURE_HOOKS:
        try:
            ctx.register_hook(hook_name, _make(hook_name))
            registered_hooks.append(hook_name)
        except Exception as exc:
            failed_hooks[hook_name] = str(exc)
            log.warning("astation: hook %s did not register: %s", hook_name, exc)

    try:
        ctx.register_cli_command(
            "trg",
            "Research Gateway plugin: migrate, status, backup",
            _setup_cli,
        )
    except Exception as exc:
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
    """Record what register() saw, for `GET /health` and `hermes trg status`."""
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
