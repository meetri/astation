"""Ship session-attributed tool calls to the audit store."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any

log = logging.getLogger("astation.audit")

INSTANCE: AuditForwarder | None = None

MAX_QUEUED = 5_000

MAX_ARG_CHARS = 4_000

_SECRET_PATTERNS = (
    (r"(AKIA|ASIA)[0-9A-Z]{16}", "AWS_KEY_REDACTED"),
    (r"\b(sk|rk)-[A-Za-z0-9_\-]{16,}", "API_KEY_REDACTED"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}", "GITHUB_TOKEN_REDACTED"),
    (r"\bxox[baprs]-[A-Za-z0-9\-]{10,}", "SLACK_TOKEN_REDACTED"),
    (r"(?i)(password|passwd|secret|token|api[_-]?key)=\S+", "REDACTED=REDACTED"),
    (r"(?i)bearer [A-Za-z0-9._\-]{16,}", "Bearer REDACTED"),
)


def _redact(text: str) -> tuple[str, bool]:
    """Mask anything key-shaped. Returns the text and whether it changed."""
    import re

    out = text
    for pattern, replacement in _SECRET_PATTERNS:
        out = re.sub(pattern, replacement, out)
    return out, out != text


def summarize_args(args: Any) -> tuple[str, bool]:
    """One bounded, redacted string for a tool's arguments."""
    if args is None:
        return "", False
    try:
        raw = args if isinstance(args, str) else json.dumps(args, default=str, sort_keys=True)
    except Exception:
        raw = str(args)
    if len(raw) > MAX_ARG_CHARS:
        raw = raw[:MAX_ARG_CHARS] + f"…[+{len(raw) - MAX_ARG_CHARS} chars]"
    return _redact(raw)


def command_of(tool_name: str, args: Any) -> str:
    """The shell command a tool is about to run, when there is one."""
    if not isinstance(args, dict):
        return ""
    for key in ("command", "cmd", "script", "shell_command"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:MAX_ARG_CHARS]
    return ""


class AuditForwarder:
    """Queues `agent_*` rows and sends them to the shipper on its own thread."""

    def __init__(
        self,
        ingest_url: str,
        host_label: str = "",
        timeout_s: float = 5.0,
        profile: str = "",
    ):
        self._url = (ingest_url or "").strip().rstrip("/")
        self._host = host_label or "gateway"
        self._timeout_s = timeout_s
        self._profile = (profile or "").strip()
        self._queue: deque[dict[str, Any]] = deque(maxlen=MAX_QUEUED)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None
        self.stats: dict[str, int] = {
            "queued": 0,
            "sent": 0,
            "dropped": 0,
            "errors": 0,
            "no_session": 0,
        }

    @property
    def configured(self) -> bool:
        return bool(self._url)

    def start(self) -> None:
        if not self.configured:
            log.warning(
                "astation: AUDIT_INGEST_URL is unset, so tool calls are NOT being "
                "attributed to sessions in the audit store."
            )
            return
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="trg-audit-forwarder", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        self._thread = None


    def record_tool_call(self, kw: dict[str, Any]) -> None:
        """One `pre_tool_call`. Never raises."""
        try:
            session_id = str(kw.get("session_id") or "")
            if not session_id:
                self.stats["no_session"] += 1
                return
            args = kw.get("args")
            summary, redacted_args = summarize_args(args)
            command, redacted_cmd = _redact(command_of(str(kw.get("tool_name") or ""), args))
            self._enqueue(
                {
                    "audit_class": "agent_tool",
                    "host": self._host,
                    "profile": str(kw.get("profile") or self._profile),
                    "stored_session_id": session_id,
                    "turn_id": str(kw.get("turn_id") or ""),
                    "tool_call_id": str(kw.get("tool_call_id") or ""),
                    "tool_name": str(kw.get("tool_name") or ""),
                    "phase": "call",
                    "args": summary,
                    "command": command,
                    "blocked": "",
                    "redacted": 1 if (redacted_args or redacted_cmd) else 0,
                }
            )
        except Exception:
            self.stats["errors"] += 1

    def record_tool_result(self, kw: dict[str, Any]) -> None:
        """One `post_tool_call`: how long it took and whether it worked."""
        try:
            session_id = str(kw.get("session_id") or "")
            tool_call_id = str(kw.get("tool_call_id") or "")
            if not session_id:
                self.stats["no_session"] += 1
                return
            duration = kw.get("duration_ms")
            try:
                duration_ms = max(0, int(duration))
            except (TypeError, ValueError):
                duration_ms = 0
            self._enqueue(
                {
                    "audit_class": "agent_tool",
                    "host": self._host,
                    "profile": str(kw.get("profile") or self._profile),
                    "stored_session_id": session_id,
                    "turn_id": str(kw.get("turn_id") or ""),
                    "tool_call_id": tool_call_id,
                    "tool_name": str(kw.get("tool_name") or ""),
                    "phase": "result",
                    "duration_ms": duration_ms,
                    "status": str(kw.get("status") or "")[:40],
                    "error_type": str(kw.get("error_type") or "")[:60],
                    "args": "",
                    "command": "",
                    "blocked": "",
                    "redacted": 0,
                }
            )
        except Exception:
            self.stats["errors"] += 1

    def record_session_event(self, event: str, kw: dict[str, Any]) -> None:
        """A session lifecycle boundary, for the same store. Never raises."""
        try:
            session_id = str(kw.get("session_id") or "")
            if not session_id:
                self.stats["no_session"] += 1
                return
            self._enqueue(
                {
                    "audit_class": "agent_session",
                    "host": self._host,
                    "profile": str(kw.get("profile") or self._profile),
                    "stored_session_id": session_id,
                    "event": event,
                    "cwd": str(kw.get("cwd") or ""),
                    "detail": str(kw.get("status") or kw.get("reason") or "")[:500],
                }
            )
        except Exception:
            self.stats["errors"] += 1


    def _enqueue(self, row: dict[str, Any]) -> None:
        row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".000Z")
        with self._lock:
            was_full = len(self._queue) == MAX_QUEUED
            self._queue.append(row)
            if was_full:
                self.stats["dropped"] += 1
            else:
                self.stats["queued"] += 1
        self._wake.set()

    def _drain_batch(self) -> list[dict[str, Any]]:
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
        return batch

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=2.0)
            self._wake.clear()
            batch = self._drain_batch()
            if not batch:
                continue
            try:
                body = "\n".join(json.dumps(row, default=str) for row in batch).encode("utf-8")
                request = urllib.request.Request(
                    self._url, data=body, headers={"Content-Type": "application/x-ndjson"}
                )
                with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                    if response.status >= 400:
                        self.stats["errors"] += 1
                    else:
                        self.stats["sent"] += len(batch)
            except (urllib.error.URLError, OSError, ValueError):
                self.stats["errors"] += 1
                self.stats["dropped"] += len(batch)
            except Exception:
                self.stats["errors"] += 1
