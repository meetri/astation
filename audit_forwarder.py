"""Ship session-attributed tool calls to the audit store.

This is the half of the audit layer that only the agent's own process knows.
The kernel sensor records that `git` ran inside a container; it cannot know
which conversation asked for it. `pre_tool_call` does -- Hermes passes
`session_id`, `turn_id` and `tool_call_id` alongside the command about to run
(verified against the deployed `agent_runtime_helpers._pre_tool_block_message`,
2026-09-20). Writing those out is what turns "something ran on this machine"
into "this session ran this".

Three constraints shape the whole file:

**It must never slow or break a turn.** A hook runs inside the agent's turn
under Hermes's 30s callback budget, and Hermes swallows hook exceptions, so a
bug here would be both costly and silent. Nothing in this module is called from
the hook itself -- `plugin/__init__.py` only enqueues -- and every method here
runs on the drain task, catches its own errors, and counts them for `/health`.

**It must not become a second source of truth.** These rows are the agent's
CLAIM about what it asked for. The kernel's rows are what actually happened.
They are kept in separate tables and joined at read time, never merged: a
compromised agent can lie in this channel and cannot touch the other one.

**A dropped row is visible.** The queue is bounded, the drop is counted, and
`GET /health` reports it. An audit channel that silently stops is worse than
one that is visibly broken.

Delivery is over the shipper's HTTP ingest on the audit docker network -- the
same path every other audit event takes, so these rows get the same transform,
the same redaction and the same two sinks, including the archive the host
cannot alter.
"""

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

#: The forwarder this PROCESS is using, or None.
#:
#: A module-level singleton under a fixed `sys.modules` key, because the
#: plugin package and the route module are loaded from PATH by different
#: importers and would otherwise hold two different module objects. `register()`
#: sets it; `GET /health` reads it.
INSTANCE: AuditForwarder | None = None

#: Bounded. If the sender stalls, rows are dropped and COUNTED rather than
#: growing without limit inside a long-lived dashboard process.
MAX_QUEUED = 5_000

#: A tool's arguments can be a whole file's contents. Stored bounded: the
#: point of the row is the attribution, not a second copy of the payload.
MAX_ARG_CHARS = 4_000

#: Key-shaped strings, masked before the row leaves this process. Vector masks
#: again on the way in; doing it here too means a secret is never in flight.
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
    """One bounded, redacted string for a tool's arguments.

    Kept as text rather than structured JSON on purpose: every tool has a
    different argument shape, the store column is a string, and the value of
    this field is "what was asked for at a glance", not a re-parseable copy.
    """
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
    """The shell command a tool is about to run, when there is one.

    This is the field the kernel join keys on: a `host_exec` row's argv
    contains this text, in the same container, moments later. Tools that do not
    run a command return "" and are still recorded -- they simply join to
    nothing, which is the honest outcome rather than a forced match.
    """
    if not isinstance(args, dict):
        return ""
    for key in ("command", "cmd", "script", "shell_command"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:MAX_ARG_CHARS]
    return ""


class AuditForwarder:
    """Queues `agent_*` rows and sends them to the shipper on its own thread.

    A thread rather than an asyncio task because the drain that feeds it is
    itself driven from a queue populated by synchronous hooks, and because a
    blocking HTTP send must never touch the dashboard's event loop.
    """

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
        # The profile this PROCESS serves. Hermes does not pass `profile` to
        # `pre_tool_call` -- measured on the deploy host, where 44 of 46 stored
        # rows carry an empty profile -- but each `gateway run` process serves
        # exactly one, and `register()` is handed its name. Using it as the
        # fallback is what makes "which agent did this" answerable in the store
        # itself, rather than only by joining back through the session id.
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

    # -- recording ---------------------------------------------------------

    def record_tool_call(self, kw: dict[str, Any]) -> None:
        """One `pre_tool_call`. Never raises.

        A call with no session id is COUNTED rather than dropped silently: it
        would mean Hermes changed what it passes, and a quietly-empty
        attribution table is exactly the failure this whole feature exists to
        prevent.
        """
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
                    # Which half of the call this is. The call row is written
                    # BEFORE the tool runs, and is the one that must exist no
                    # matter what: a tool that hangs or is killed never reaches
                    # `post_tool_call`, and a call missing from the record
                    # because it never returned is the worst gap an audit trail
                    # can have.
                    "phase": "call",
                    "args": summary,
                    # The join key. Named separately from `args` so the
                    # attribution view does not have to parse a summary.
                    "command": command,
                    "blocked": "",
                    "redacted": 1 if (redacted_args or redacted_cmd) else 0,
                }
            )
        except Exception:
            self.stats["errors"] += 1

    def record_tool_result(self, kw: dict[str, Any]) -> None:
        """One `post_tool_call`: how long it took and whether it worked.

        **The tool's output is deliberately not sent**.
        Hermes hands the whole result to this hook, and a file read returns the
        file. The archive is under compliance-mode object lock, so anything
        written there cannot be edited or deleted for the retention window; a
        credential that reaches it is there for the year. What travels instead
        is the duration, the status, and the error CLASS -- three short, bounded
        values that answer "did it work and how long did it take" without
        carrying a payload that could contain anything at all.

        Joined to its call row on `tool_call_id` at read time rather than
        updating it, because the store is append-only by design.

        Never raises: this runs inside the agent's turn.
        """
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
                    # Short enums from Hermes, not free text. `error_message` is
                    # available on this hook and is NOT sent: it can quote the
                    # content that caused the failure.
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

    # -- plumbing ----------------------------------------------------------

    def _enqueue(self, row: dict[str, Any]) -> None:
        row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".000Z")
        with self._lock:
            # `deque(maxlen=)` discards silently, so the drop is detected by
            # length rather than trusted to be reported.
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
                # The rows are gone rather than retried: a retry queue that
                # grows during an outage is how a logging path takes down the
                # thing it is logging. The loss is counted, and the kernel's
                # own record of the same work is unaffected.
                self.stats["errors"] += 1
                self.stats["dropped"] += len(batch)
            except Exception:
                self.stats["errors"] += 1
