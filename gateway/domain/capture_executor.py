"""Where the per-event persistence stage runs: one worker thread, off the event loop.

Every streamed event is stamped, recorded (`RunRecorder`), captured (`ChatStore`) and journalled
(`EventInbox`) before it is fanned out. All of that is blocking SQLite work, and it used to run on the
asyncio loop that also serves every HTTP request and WebSocket, so a burst of events froze the whole
dashboard (event-loop stalls of 6-28 s were logged). It now runs here instead.

One thread, shared by every pipeline (the default profile's `EventBroadcaster` and each other profile's
pump): the stage was strictly serialized while it lived on the loop and the recorder and chat store keep
per-session in-memory state, so it stays strictly serialized. Event order is unchanged because each
pipeline awaits its stage before taking the next event.

What must stay on the loop (subscriber queues, `seq` assignment, tasks) goes through `call_on_loop`.

``ASTATION_CAPTURE_THREAD=0`` runs the stage inline on the loop again, the pre-change behaviour.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_ENV_SWITCH = "ASTATION_CAPTURE_THREAD"
_OFF_VALUES = frozenset({"0", "false", "no", "off"})

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: int | None = None
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get(_ENV_SWITCH, "1").strip().lower() not in _OFF_VALUES


def _bind(loop: asyncio.AbstractEventLoop) -> concurrent.futures.ThreadPoolExecutor:
    """Remember the loop (and its thread) the stage reports back to; create the worker once."""
    global _executor, _loop, _loop_thread
    with _lock:
        _loop = loop
        _loop_thread = threading.get_ident()
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="astation-capture"
            )
        return _executor


async def run_capture(fn: Callable[..., T], *args: Any) -> T:
    """Run `fn(*args)` on the capture thread and wait for it without blocking the loop."""
    if not enabled():
        return fn(*args)
    loop = asyncio.get_running_loop()
    executor = _bind(loop)
    # run_in_executor, unlike asyncio.to_thread, does not carry the context across.
    context = contextvars.copy_context()
    return await loop.run_in_executor(executor, context.run, fn, *args)


def on_loop_thread() -> bool:
    """True when it is safe to touch loop-owned state directly from here."""
    loop = _loop
    return loop is None or not loop.is_running() or threading.get_ident() == _loop_thread


def call_on_loop(fn: Callable[..., Any], *args: Any) -> None:
    """Run `fn(*args)` on the event loop: now when already there, else queued in order behind
    whatever the capture thread has already handed over."""
    if on_loop_thread():
        fn(*args)
        return
    try:
        _loop.call_soon_threadsafe(fn, *args)  # type: ignore[union-attr]
    except RuntimeError:
        # The loop closed between the check and the call: the process is going down.
        logger.debug("event loop closed; dropped a hand-off to it", exc_info=True)
