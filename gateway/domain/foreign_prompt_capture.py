"""Capture the prompt of a turn that was started outside this gateway.

## The gap

`chat_messages` gets its `user` rows from exactly one writer: the gateway's
own turn-submit route (`ChatStore.capture_submitted_user_row`). A prompt typed
in Hermes's own TUI, or sent by any other client, never passes through that
route -- so the store holds the whole of such a turn (segments, tool rows,
the final answer, all captured live off the event stream) and not the one
line that started it. `RUN_EVENT_UX_AUDIT.md` §6.2 lists this as the first
thing only Hermes's transcript has, and prerequisite 2 for reading the chat
from `GET /chat`. There is no `state.db` mount in `deploy/` (the backfill
reader in `domain/hermes_state_reader.py` needs the file), so the transcript
is read through the Hermes connection this gateway already holds.

## The rule

When `RunRecorder` opens a run it reports `(profile, stored_id, run_id)`
. `ChatStore.attach_turn` runs first and links the user row the
submit route wrote before the turn opened. **If it attached nothing, no
prompt was waiting, and the turn was started elsewhere** -- that is the
whole test, and it is decided from the store alone with no Hermes call. Only
then is a capture scheduled:

1. resolve the profile's adapter (`resolve_profile_adapter`) and go through
   `_with_reconnect`, like every other Hermes-touching path;
2. `session.resume(stored_id)` -- idempotent within a connection, returns
   the transcript rows with `role` / `row_id` / `text` / `timestamp`
. The freshly
   resolved live handle goes into the profile's `LiveHandleCache` as a side
   effect, exactly as a route's resume would;
3. take the NEWEST `role == "user"` row that is a real prompt: not a
   compaction summary (Hermes serves those as user rows -- B-176 -- with a
   text that begins with one of `SUMMARY_MARKERS`), not a display-only row
   (`display_kind`), with an integer `row_id` and non-blank text;
4. write it through `ChatStore.capture_backfilled_user_row` as `role=user`,
   `source=backfill`, `hermes_row_id=<row_id>`, `turn_id=<run id>`. The
   store's `hermes_row_id` dedup makes a repeat a no-op.

## Why it is scheduled, and what it costs

The run-open hook runs synchronously inside the event pump (both
`EventBroadcaster._forward_one` and `ProfileConnectionManager._forward_one`),
and a `session.resume` is a network round trip that can carry a 1.6 MB
transcript. The pump is never blocked on it: the capture is an
`asyncio` task on the pump's own loop, one per foreign-started run, and it
costs one resume per such turn. A gateway-started turn costs nothing here.

Best-effort throughout: every failure (no loop, profile not connected,
Hermes error, unreadable transcript, DB error) is logged at info/warning and
swallowed. A miss costs one prompt row; it never costs the run, the frame,
or the pump.
"""

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

#: Text prefixes of the `role: user` rows Hermes serves after a compaction
#:. They are Hermes's summary
#: of the conversation, not something the operator typed, and never a prompt.
SUMMARY_MARKERS: tuple[str, ...] = (
    "[Durable Summary",
    "[Session Arc Summary",
    "[Recent Summary",
    "[Current user objective preserved from compacted history]",
)

#: `source` on every row this module writes.
BACKFILL_SOURCE = "backfill"


def is_summary_text(text: str) -> bool:
    """Whether a user-row text is one of Hermes's compaction summaries."""
    stripped = text.lstrip()
    return any(stripped.startswith(marker) for marker in SUMMARY_MARKERS)


def newest_foreign_prompt(messages: Any) -> tuple[int, str] | None:
    """`(row_id, text)` of the newest real user prompt in a transcript, or None.

    Walks `session.resume`'s `messages` from the end. A row counts only if
    it is a dict with `role == "user"`, no `display_kind`, an integer
    `row_id`, and non-blank text that is not a compaction summary. Every
    access is a `.get()` on a key allowed to be missing -- B-34's rule: the
    only guaranteed key on a transcript row is `role`.
    """
    if not isinstance(messages, list | tuple):
        return None
    for row in reversed(messages):
        if not isinstance(row, dict) or row.get("role") != "user":
            continue
        if row.get("display_kind"):
            continue
        row_id = row.get("row_id")
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
    """The run-opened hook: attach the waiting user row, else fetch the prompt.

    Built once in `api/bootstrap.py::build_run_recorder` and handed to
    `RunRecorder` as `on_run_opened`. `app_state` is read at capture time,
    not at construction (the adapter, connect lock and profile manager are
    all set on it later in the same startup, and the profile manager's
    connections come and go).

    `schedule` is injectable for tests; the default puts the coroutine on
    the running loop. The pump coroutines that call the hook always have
    one; a caller with no loop (a synchronous test driving `RunRecorder`
    directly) gets the capture skipped with an info line, not an error.
    """

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
        # Optional in-process source of the same text this class otherwise
        # pays a `session.resume` to recover. The Hermes plugin supplies one
        # backed by `pre_llm_call`, which carries `user_message` for free.
        # Returns `(text, hermes_turn_id)` or None; on None the resume path
        # below runs exactly as before, so this is additive and reversible.
        self._prompt_source = prompt_source
        # Strong references so a scheduled capture is not garbage-collected
        # mid-flight (asyncio keeps only weak references to tasks).
        self._tasks: set[Any] = set()

    def set_prompt_source(
        self, source: Callable[[str, str], tuple[str, str] | None] | None
    ) -> None:
        """Install (or clear) the in-process prompt source after construction.

        The Hermes plugin builds its hook drain after `startup()` has already
        constructed this object, so the source arrives late. Public rather than
        a private poke, and clearable, so a caller can fall back to the resume
        path deliberately.
        """
        self._prompt_source = source

    # -- the hook -----------------------------------------------------------

    def handle_run_opened(self, profile: str, stored_id: str, run_id: str) -> bool:
        """`RunRecorder.on_run_opened`: attach first, capture only on a miss.

        Returns True when a capture was scheduled. Never raises.
        """
        try:
            attached = self._chat_store.attach_turn(profile, stored_id, run_id)
        except Exception:  # pragma: no cover - attach_turn guards itself
            logger.exception("attach_turn failed for run %s; treating as no waiting row", run_id)
            attached = 0
        if attached:
            return False
        return self._spawn(profile, stored_id, run_id)

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

    # -- the capture ---------------------------------------------------------

    async def capture(self, profile: str, stored_id: str, run_id: str) -> str | None:
        """Read the newest real user prompt for `stored_id` and store it.

        Returns the new `chat_messages` row id, or None when nothing was
        written (no usable prompt, already stored, or any failure -- all
        logged, none raised).
        """
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
        # In-process first. A `pre_llm_call` hook already carried this turn's
        # prompt into the gateway, so when it is available the whole
        # resume-per-foreign-turn round trip is skipped.
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
            app_state, adapter, lambda: _resume_for_live_id(adapter, stored_id, cache, profile=profile)
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

    # -- shutdown ------------------------------------------------------------

    async def close(self) -> None:
        """Cancel any capture still in flight (lifespan teardown)."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

    # -- introspection (tests) ---------------------------------------------

    @property
    def pending_count(self) -> int:
        return len(self._tasks)
