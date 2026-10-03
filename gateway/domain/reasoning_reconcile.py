"""Fill the chat store's reasoning from Hermes's own transcript, and archive every body it holds.

Hermes keeps reasoning on every assistant row it stores, including the tool-call steps that
`message.complete` never repeats. Two things happen per session, both idempotent:

- every Hermes reasoning body is copied into `reasoning_archive` (`source="hermes"`), one row
  per Hermes row id, so it survives whatever the chat rows end up holding;
- a chat row whose reasoning is missing, or a strict part of the Hermes body it came from, is
  given the full body. A row is only ever upgraded, never shortened or replaced by a body that
  does not contain what it already shows.

It runs on every transcript load (the messages are already in hand) and shortly after each turn
completes (`PostTurnReasoningSync`).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from .chat_store import ARCHIVE_SOURCE_HERMES
from .models import ChatMessage, ReasoningArchive, new_id, utcnow

logger = logging.getLogger(__name__)

# A reasoning-only row is matched by its text alone; shorter than this, the match means nothing.
_MIN_ANCHOR_CHARS = 24
# Opaque or encrypted parts of a structured reasoning item, never readable text.
_OPAQUE_KEYS = frozenset({"data", "encrypted_content", "signature", "id"})
_TEXT_KEYS = ("text", "summary", "content", "reasoning")
# Wait for Hermes to finish committing the turn before reading it back.
POST_TURN_DELAY_S = 2.0


def _structured_text(value: Any) -> Iterable[str]:
    """The readable text inside `reasoning_details` / `codex_reasoning_items`, in order."""
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from _structured_text(item)
    elif isinstance(value, dict):
        for key in _TEXT_KEYS:
            if key in value and key not in _OPAQUE_KEYS:
                yield from _structured_text(value[key])


def reasoning_body(message: dict[str, Any]) -> str | None:
    """A Hermes transcript row's reasoning, by the same key order every other reader uses."""
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("reasoning_details", "codex_reasoning_items"):
        parts = list(_structured_text(message.get(key)))
        if parts:
            return "\n\n".join(parts)
    return None


@dataclass(frozen=True)
class _HermesReasoning:
    row_id: int | None
    text: str
    body: str


@dataclass(frozen=True)
class ReconcileResult:
    archived: int
    upgraded: int


def _row_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _hermes_reasoning(messages: Any) -> list[_HermesReasoning]:
    found: list[_HermesReasoning] = []
    if not isinstance(messages, list):
        return found
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        body = reasoning_body(message)
        if body is None:
            continue
        text = message.get("text")
        found.append(
            _HermesReasoning(
                row_id=_row_id(message.get("row_id")),
                text=text.strip() if isinstance(text, str) else "",
                body=body,
            )
        )
    return found


def _archive(db: OrmSession, profile: str, stored_id: str, found: list[_HermesReasoning]) -> int:
    existing = {
        row.hermes_row_id: row
        for row in db.execute(
            select(ReasoningArchive).where(
                ReasoningArchive.profile == profile,
                ReasoningArchive.stored_session_id == stored_id,
                ReasoningArchive.source == ARCHIVE_SOURCE_HERMES,
            )
        ).scalars()
    }
    written = 0
    for item in found:
        if item.row_id is None:
            continue
        row = existing.get(item.row_id)
        if row is not None:
            if row.text != item.body:
                row.text = item.body
                row.updated_at = utcnow()
                written += 1
            continue
        db.add(
            ReasoningArchive(
                id=new_id("ra"),
                profile=profile,
                stored_session_id=stored_id,
                source=ARCHIVE_SOURCE_HERMES,
                hermes_row_id=item.row_id,
                text=item.body,
                sealed=True,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        written += 1
    return written


def _match(
    row: ChatMessage,
    found: list[_HermesReasoning],
    used: set[int],
    hermes_texts: Counter[str],
    chat_texts: Counter[str],
) -> int | None:
    """The index in `found` this chat row's reasoning came from, or None when it is not certain."""
    current = (row.reasoning or "").strip()
    text = (row.text or "").strip()
    if text:
        if not current and (hermes_texts[text] != 1 or chat_texts[text] != 1):
            return None
        for index, item in enumerate(found):
            if index in used or item.text != text:
                continue
            if current and current not in item.body:
                continue
            return index
        return None
    if len(current) < _MIN_ANCHOR_CHARS:
        return None
    candidates = [
        index
        for index, item in enumerate(found)
        if index not in used and not item.text and current in item.body
    ]
    return candidates[0] if len(candidates) == 1 else None


def _assistant_texts(messages: Any) -> Counter[str]:
    """Every Hermes assistant row's text, with or without reasoning: what "unique" is judged by."""
    texts: Counter[str] = Counter()
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            text = message.get("text")
            if isinstance(text, str) and text.strip():
                texts[text.strip()] += 1
    return texts


def _upgrade(
    db: OrmSession,
    profile: str,
    stored_id: str,
    found: list[_HermesReasoning],
    hermes_texts: Counter[str],
) -> int:
    rows = list(
        db.execute(
            select(ChatMessage)
            .where(
                ChatMessage.profile == profile,
                ChatMessage.stored_session_id == stored_id,
                ChatMessage.role == "assistant",
            )
            .order_by(ChatMessage.seq.asc())
        ).scalars()
    )
    chat_texts = Counter((row.text or "").strip() for row in rows if (row.text or "").strip())
    used: set[int] = set()
    upgraded = 0
    for row in rows:
        index = _match(row, found, used, hermes_texts, chat_texts)
        if index is None:
            continue
        used.add(index)
        body = found[index].body
        if len(body.strip()) > len((row.reasoning or "").strip()):
            row.reasoning = body
            upgraded += 1
    return upgraded


def reconcile_reasoning(
    session_factory: sessionmaker[OrmSession], profile: str, stored_id: str, messages: Any
) -> ReconcileResult:
    """Archive and back-fill one session's reasoning from its Hermes transcript `messages`."""
    found = _hermes_reasoning(messages)
    if not found:
        return ReconcileResult(archived=0, upgraded=0)
    with session_factory() as db:
        archived = _archive(db, profile, stored_id, found)
        upgraded = _upgrade(db, profile, stored_id, found, _assistant_texts(messages))
        if archived or upgraded:
            db.commit()
    return ReconcileResult(archived=archived, upgraded=upgraded)


class PostTurnReasoningSync:
    """Read a session's Hermes transcript back shortly after each turn and reconcile it."""

    def __init__(self, app_state: Any, *, delay_s: float = POST_TURN_DELAY_S) -> None:
        self._app_state = app_state
        self._delay_s = delay_s
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    def schedule(self, profile: str, stored_id: str) -> None:
        """Queue one reconcile for the session; a second request while one is pending is folded in."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        key = (profile, stored_id)
        pending = self._tasks.get(key)
        if pending is not None and not pending.done():
            return
        task = loop.create_task(self._run(profile, stored_id), name=f"reasoning-sync-{stored_id}")
        self._tasks[key] = task
        task.add_done_callback(lambda done: self._finished(key, done))

    def _finished(self, key: tuple[str, str], task: asyncio.Task[None]) -> None:
        if self._tasks.get(key) is task:
            del self._tasks[key]

    async def _run(self, profile: str, stored_id: str) -> None:
        # Imported here: hermes_runtime pulls in the adapter stack, which a store-only caller never needs.
        from .hermes_runtime import (
            _with_live_handle,
            _with_reconnect,
            resolve_live_handle_cache,
            resolve_profile_adapter,
        )
        from .transcript import _transcript

        await asyncio.sleep(self._delay_s)
        try:
            adapter = resolve_profile_adapter(self._app_state, profile)
            cache = resolve_live_handle_cache(self._app_state, profile)
            _live_id, history = await _with_reconnect(
                self._app_state,
                adapter,
                lambda: _with_live_handle(
                    adapter, cache, stored_id, adapter.session_history, profile=profile
                ),
            )
            _count, messages = _transcript(history, "count")
            result = await asyncio.to_thread(
                reconcile_reasoning, self._app_state.db_sessions, profile, stored_id, messages
            )
            if result.archived or result.upgraded:
                logger.info(
                    "reasoning sync for session %s (profile=%r): %d archived, %d row(s) upgraded",
                    stored_id, profile, result.archived, result.upgraded,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "post-turn reasoning sync failed for session %s (profile=%r); the next "
                "transcript load retries it",
                stored_id, profile, exc_info=True,
            )

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
