"""Hook enrichment: what the WebSocket cannot tell the gateway."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import OrderedDict
from datetime import UTC
from typing import Any

log = logging.getLogger("astation.capture")

MAX_TRACKED_TURNS = 2_000

TURN_TTL_S = 6 * 60 * 60


class TurnMap:
    """`turn_id` -> the context later hooks of that turn do not carry."""

    def __init__(self, max_entries: int = MAX_TRACKED_TURNS, ttl_s: float = TURN_TTL_S):
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_s

    def remember(self, turn_id: str | None, **fields: Any) -> None:
        if not turn_id:
            return
        entry = self._entries.pop(turn_id, None) or {}
        entry.update({k: v for k, v in fields.items() if v is not None})
        entry["_seen"] = time.time()
        self._entries[turn_id] = entry
        self._evict()

    def get(self, turn_id: str | None) -> dict[str, Any]:
        if not turn_id:
            return {}
        entry = self._entries.get(turn_id)
        if entry is None:
            return {}
        if time.time() - entry.get("_seen", 0) > self._ttl:
            self._entries.pop(turn_id, None)
            return {}
        return entry

    def forget(self, turn_id: str | None) -> None:
        if turn_id:
            self._entries.pop(turn_id, None)

    def _evict(self) -> None:
        now = time.time()
        while self._entries:
            oldest_key = next(iter(self._entries))
            oldest = self._entries[oldest_key]
            if len(self._entries) > self._max or now - oldest.get("_seen", 0) > self._ttl:
                self._entries.pop(oldest_key, None)
                continue
            break

    def __len__(self) -> int:
        return len(self._entries)


def usage_payload(hook_usage: dict[str, Any], model: str | None) -> dict[str, Any]:
    """Translate `post_api_request`'s usage into the shape the gateway reads."""
    usage: dict[str, Any] = {
        "prompt": hook_usage.get("prompt_tokens", hook_usage.get("input_tokens")),
        "input": hook_usage.get("input_tokens"),
        "completion": hook_usage.get("output_tokens"),
        "output": hook_usage.get("output_tokens"),
        "reasoning": hook_usage.get("reasoning_tokens"),
        "total": hook_usage.get("total_tokens"),
        "calls": hook_usage.get("request_count"),
        "cache_read": hook_usage.get("cache_read_tokens"),
        "cache_write": hook_usage.get("cache_write_tokens"),
    }
    if model:
        usage["model"] = model
    return {k: v for k, v in usage.items() if v is not None}


class HookDrain:
    """Drains the hook queue and applies enrichment. Never blocks a turn."""

    def __init__(self, events, app_state: Any, profile: str = "default", audit=None):
        self._events = events
        # Accepted and ignored: forwarding lives in register(), which runs in every process.
        self._audit = audit
        self._app_state = app_state
        self._profile = profile
        self.turns = TurnMap()
        # Keyed by session as well as turn: the foreign-prompt caller knows only the session.
        self._latest_prompt: OrderedDict[str, tuple[str, str, float]] = OrderedDict()
        self._closed_turns: OrderedDict[tuple[str, str], float] = OrderedDict()
        self.stats: dict[str, int] = {
            "drained": 0,
            "usage_recorded": 0,
            "runs_closed": 0,
            "dropped_no_session": 0,
            "duplicate_close": 0,
            "errors": 0,
        }
        self._task: asyncio.Task | None = None
        self._stop = False


    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="trg-hook-drain")

    async def close(self) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task
            self._task = None

    async def _run(self) -> None:
        # A thread queue fed by synchronous hooks: it is polled because it cannot be awaited.
        while not self._stop:
            try:
                item = self._events.get_nowait()
            except Exception:
                await asyncio.sleep(0.05)
                continue
            try:
                self.handle(item)
                self.stats["drained"] += 1
            except Exception:
                self.stats["errors"] += 1
                log.exception("astation: hook drain failed on %s", item.get("hook"))


    def handle(self, item: dict) -> None:
        hook = item.get("hook")
        kwargs = item.get("kwargs") or {}
        handler = getattr(self, f"_on_{hook}", None)
        if handler is not None:
            handler(kwargs)

    def _on_pre_llm_call(self, kw: dict) -> None:
        """The only hook that knows the turn's session AND model."""
        self.turns.remember(
            kw.get("turn_id"),
            session_id=kw.get("session_id"),
            model=kw.get("model"),
            platform=kw.get("platform"),
            user_message=kw.get("user_message"),
            is_first_turn=kw.get("is_first_turn"),
            parent_session_id=kw.get("parent_session_id"),
        )
        session_id = kw.get("session_id")
        text = kw.get("user_message")
        if session_id and isinstance(text, str) and text.strip():
            self._latest_prompt[str(session_id)] = (
                text,
                str(kw.get("turn_id") or ""),
                time.time(),
            )
            while len(self._latest_prompt) > MAX_TRACKED_TURNS:
                self._latest_prompt.popitem(last=False)

    def _on_post_api_request(self, kw: dict) -> None:
        """Per-request usage, attributed via the turn map."""
        turn_id = kw.get("turn_id")
        ctx = self.turns.get(turn_id)
        stored_id = kw.get("session_id") or ctx.get("session_id")
        if not stored_id:
            self.stats["dropped_no_session"] += 1
            return
        raw = kw.get("usage") or {}
        if not raw:
            return
        payload = {
            "usage": usage_payload(raw, ctx.get("model")),
            "_stored_session_id": stored_id,
            "_profile": self._profile,
            "_per_request": True,
            "_source": "hook",
            "turn_id": turn_id,
        }
        self._record("session.usage", payload)
        self.stats["usage_recorded"] += 1

    def _on_on_session_end(self, kw: dict) -> None:
        """Close a run whose closing frame never arrived."""
        stored_id = kw.get("session_id")
        turn_id = kw.get("turn_id")
        if not stored_id:
            return
        key = (str(stored_id), str(turn_id))
        if key in self._closed_turns:
            self.stats["duplicate_close"] += 1
            return
        recorder = getattr(self._app_state, "run_recorder", None)
        if recorder is None or not getattr(recorder, "has_open_run", None):
            return
        # The retry without `profile` is a signature shim for recorders that take one argument.
        try:
            if not recorder.has_open_run(stored_id, self._profile):
                return
        except TypeError:
            if not recorder.has_open_run(stored_id):
                return
        interrupted = bool(kw.get("interrupted"))
        self._closed_turns[key] = time.time()
        del_keys = len(self._closed_turns) - MAX_TRACKED_TURNS
        for _ in range(max(0, del_keys)):
            self._closed_turns.popitem(last=False)
        self._record(
            "message.completed",
            {
                "_stored_session_id": stored_id,
                "_profile": self._profile,
                "_source": "hook",
                "turn_id": turn_id,
                "status": "interrupted" if interrupted else None,
                "text": None,
                "_safety_net": True,
            },
        )
        self.stats["runs_closed"] += 1
        self.turns.forget(turn_id)


    def _record(self, event_type: str, payload: dict) -> None:
        """Hand one synthesized canonical event to the recorder."""
        recorder = getattr(self._app_state, "run_recorder", None)
        if recorder is None:
            return
        from datetime import datetime

        from events.canonical import CanonicalEvent

        canonical = CanonicalEvent(
            event_id="",
            project_id=None,
            session_id=payload.get("_stored_session_id"),
            run_id=None,
            seq=0,
            type=event_type,
            timestamp=datetime.now(UTC),
            payload=payload,
        )
        envelope = {
            "type": event_type,
            "payload": payload,
            "project_id": None,
            "session_id": payload.get("_stored_session_id"),
            "run_id": None,
        }
        recorder.handle_event(canonical, envelope, self._profile)


    def prompt_source(self, profile: str, stored_session_id: str) -> tuple[str, str] | None:
        """The newest prompt this session's `pre_llm_call` carried, ONCE."""
        del profile
        entry = self._latest_prompt.pop(str(stored_session_id), None)
        if entry is None:
            return None
        text, turn_id, wall = entry
        if time.time() - wall > TURN_TTL_S:
            return None
        return text, turn_id
