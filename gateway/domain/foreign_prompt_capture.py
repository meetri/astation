"""Capture the prompt of a turn that was started outside this gateway."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from domain.hermes_runtime import (
    _resume_for_live_id,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)

logger = logging.getLogger(__name__)

# Hermes serves its compaction summaries as role: user rows; they are never prompts.
SUMMARY_MARKERS: tuple[str, ...] = (
    "[Durable Summary",
    "[Session Arc Summary",
    "[Recent Summary",
    "[Current user objective preserved from compacted history]",
)

BACKFILL_SOURCE = "backfill"


def is_summary_text(text: str) -> bool:
    """Whether a user-row text is one of Hermes's compaction summaries."""
    stripped = text.lstrip()
    return any(stripped.startswith(marker) for marker in SUMMARY_MARKERS)


def newest_foreign_prompt(messages: Any) -> tuple[int, str] | None:
    """`(row_id, text)` of the newest real user prompt in a transcript, or None."""
    if not isinstance(messages, list | tuple):
        return None
    for row in reversed(messages):
        if not isinstance(row, dict) or row.get("role") != "user":
            continue
        if row.get("display_kind"):
            continue
        row_id = row.get("row_id")
        # bool is an int subclass: without this check True would pass as a row id.
        if isinstance(row_id, bool) or not isinstance(row_id, int):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if is_summary_text(text):
            continue
        return row_id, text
    return None


class ForeignPromptCapture:
    """The run-opened hook: attach the waiting user row, else fetch the prompt."""

    def __init__(
        self,
        app_state: Any,
        chat_store: Any,
        *,
        schedule: Callable[[Coroutine[Any, Any, Any]], Any] | None = None,
        prompt_source: Callable[[str, str], tuple[str, str] | None] | None = None,
    ) -> None:
        self._app_state = app_state
        self._chat_store = chat_store
        self._schedule = schedule
        self._prompt_source = prompt_source
        # Strong refs: asyncio holds only weak ones and would collect a live capture.
        self._tasks: set[Any] = set()

    def set_prompt_source(
        self, source: Callable[[str, str], tuple[str, str] | None] | None
    ) -> None:
        """Install (or clear) the in-process prompt source after construction."""
        self._prompt_source = source


    def handle_run_opened(self, profile: str, stored_id: str, run_id: str) -> bool:
        """`RunRecorder.on_run_opened`: attach first, capture only on a miss."""
        try:
            attached = self._chat_store.attach_turn(profile, stored_id, run_id)
        except Exception:  # pragma: no cover - attach_turn guards itself
            logger.exception("attach_turn failed for run %s; treating as no waiting row", run_id)
            attached = 0
        # Attaching nothing is the whole foreign-turn test: no prompt row was waiting.
        if attached:
            return False
        return self._spawn(profile, stored_id, run_id)

    # Scheduled, not awaited: the caller is the event pump and a resume is a round trip.
    def _spawn(self, profile: str, stored_id: str, run_id: str) -> bool:
        coro = self.capture(profile, stored_id, run_id)
        try:
            if self._schedule is not None:
                self._schedule(coro)
                return True
            task = asyncio.get_running_loop().create_task(
                coro, name=f"foreign-prompt-capture:{run_id}"
            )
        except RuntimeError:
            coro.close()
            logger.info(
                "no running loop to capture the foreign prompt for run %s on session %s; skipped",
                run_id,
                stored_id,
            )
            return False
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True


    async def capture(self, profile: str, stored_id: str, run_id: str) -> str | None:
        """Read the newest real user prompt for `stored_id` and store it."""
        try:
            return await self._capture(profile, stored_id, run_id)
        except Exception:
            logger.warning(
                "foreign-prompt capture failed for run %s (profile=%r session=%s); "
                "the turn's prompt is missing from chat history",
                run_id,
                profile,
                stored_id,
                exc_info=True,
            )
            return None

    async def _capture(self, profile: str, stored_id: str, run_id: str) -> str | None:
        if self._prompt_source is not None:
            try:
                found = self._prompt_source(profile, stored_id)
            except Exception:
                logger.exception(
                    "in-process prompt source failed for run %s; falling back to a resume",
                    run_id,
                )
                found = None
            if found is not None:
                text, _hermes_turn_id = found
                row = self._chat_store.capture_hook_user_row(
                    profile=profile,
                    stored_session_id=stored_id,
                    text=text,
                    turn_id=run_id,
                )
                logger.info(
                    "captured the foreign prompt for run %s from a hook (no resume); row=%s",
                    run_id,
                    row,
                )
                return row

        app_state = self._app_state
        adapter = resolve_profile_adapter(app_state, profile)
        cache = resolve_live_handle_cache(app_state, profile)
        _live_id, result = await _with_reconnect(
            app_state,
            adapter,
            lambda: _resume_for_live_id(adapter, stored_id, cache, profile=profile),
        )
        messages = result.get("messages") if isinstance(result, dict) else None
        found = newest_foreign_prompt(messages)
        if found is None:
            logger.info(
                "run %s on session %s (profile=%r) opened with no waiting user row and "
                "the transcript has no real user prompt; nothing captured",
                run_id,
                stored_id,
                profile,
            )
            return None
        row_id, text = found
        new_id = self._chat_store.capture_backfilled_user_row(
            profile=profile,
            stored_session_id=stored_id,
            text=text,
            hermes_row_id=row_id,
            turn_id=run_id,
        )
        if new_id is None:
            logger.info(
                "foreign prompt row %d for session %s already stored; run %s captured nothing",
                row_id,
                stored_id,
                run_id,
            )
        else:
            logger.info(
                "captured foreign prompt (hermes row %d) for run %s on session %s (profile=%r)",
                row_id,
                run_id,
                stored_id,
                profile,
            )
        return new_id


    async def close(self) -> None:
        """Cancel any capture still in flight (lifespan teardown)."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()


    @property
    def pending_count(self) -> int:
        return len(self._tasks)
