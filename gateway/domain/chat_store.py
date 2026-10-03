"""`ChatStore` -- the read/write interface over `chat_messages`."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from .models import ChatMessage, ReasoningArchive, new_id, utcnow

logger = logging.getLogger(__name__)

_STORED_SESSION_ID_FIELD = "_stored_session_id"

CAPTURED_EVENT_TYPES: frozenset[str] = frozenset(
    {"message.interim", "tool.completed", "message.completed", "status.update"}
)

# Not rows of their own: their text is held per run and written onto the next assistant row.
_REASONING_EVENT_TYPES: frozenset[str] = frozenset({"reasoning.delta"})
# The keys a `reasoning.delta` payload carries its text under, in the app's order.
_REASONING_TEXT_KEYS: tuple[str, ...] = ("text", "delta", "reasoning", "content", "chunk")
# Runs whose reasoning is held at once; an abandoned run's buffer is evicted oldest-first.
_MAX_REASONING_BUFFERS = 64
# A tool call ends a model step: the reasoning held for that step is written before it.
_STEP_END_EVENT_TYPES: frozenset[str] = frozenset({"tool.started"})
# A streaming step's archive row is rewritten at most this often, or once this much is unwritten.
_ARCHIVE_FLUSH_INTERVAL_S = 2.0
_ARCHIVE_FLUSH_CHARS = 8192

ARCHIVE_SOURCE_STREAM = "stream"
ARCHIVE_SOURCE_PAYLOAD = "payload"
ARCHIVE_SOURCE_HERMES = "hermes"

_MARKER_STATUS_KINDS: frozenset[str] = frozenset({"compacted", "process"})
_LIFECYCLE_STATUS_KIND = "lifecycle"

# Matched on the product name alone: the glyph and the dash vary across Hermes builds.
MEMORY_RECALL_MARKER = "Hindsight"

NOTICE_KIND_COMPACTED = "compacted"
NOTICE_KIND_MEMORY = "memory"
NOTICE_KIND_PROCESS = "process"

_MAX_SEQ_RETRIES = 5


def _turn_id(envelope: Any) -> str | None:
    """The envelope's `run_id` as a turn id, or None."""
    if not isinstance(envelope, dict):
        return None
    run_id = envelope.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def notice_kind(row: ChatMessage) -> str | None:
    """Which notice a `marker` row is, for the app to render without re-parsing."""
    if row.role != "marker":
        return None
    if row.compacted:
        return NOTICE_KIND_COMPACTED
    if isinstance(row.text, str) and MEMORY_RECALL_MARKER in row.text:
        return NOTICE_KIND_MEMORY
    return NOTICE_KIND_PROCESS


def _row_dict(row: ChatMessage) -> dict[str, Any]:
    """One `chat_messages` row as the JSON shape `GET .../chat` returns."""
    return {
        "id": row.id,
        "profile": row.profile,
        "stored_session_id": row.stored_session_id,
        "turn_id": row.turn_id,
        "seq": row.seq,
        "role": row.role,
        "text": row.text,
        "reasoning": row.reasoning,
        "tool_name": row.tool_name,
        "tool_call_id": row.tool_call_id,
        "tool_args": row.tool_args_json,
        "tool_result": row.tool_result_json,
        "hermes_row_id": row.hermes_row_id,
        "source": row.source,
        "compacted": bool(row.compacted),
        "notice_kind": notice_kind(row),
        "created_at": row.created_at.isoformat().replace("+00:00", "Z")
        if row.created_at.tzinfo is not None
        else row.created_at.isoformat() + "Z",
    }


def _iso(value: Any) -> str:
    return (
        value.isoformat().replace("+00:00", "Z")
        if value.tzinfo is not None
        else value.isoformat() + "Z"
    )


def _is_captured_notice(kind: Any, text: Any) -> bool:
    """Whether a `status.update` earns a marker row (B-191, module docstring)."""
    if kind in _MARKER_STATUS_KINDS:
        return True
    if kind == _LIFECYCLE_STATUS_KIND:
        return isinstance(text, str) and MEMORY_RECALL_MARKER in text
    return False


_BufferKey = tuple[str, str, str | None]


@dataclass
class _ArchiveCursor:
    """Where one streaming step's reasoning sits in `reasoning_archive`."""

    step: int
    row_id: str | None = None
    flushed_chars: int = 0
    flushed_at: float = 0.0


def _contains(outer: str, inner: str) -> bool:
    """Whether `inner` is part of `outer`, ignoring surrounding whitespace."""
    return inner.strip() in outer


def final_reasoning(streamed: str | None, payload: str | None, spent: list[str]) -> str | None:
    """The reasoning a turn's final row keeps, from what streamed and what `message.completed` says.

    Hermes's payload carries only the turn's most recent non-empty reasoning block, which can be
    a block already written onto an earlier step's row; repeating it would show it twice. When
    one text holds the other, the longer wins; otherwise the payload is authoritative.
    """
    streamed = streamed if streamed and streamed.strip() else None
    payload = payload if payload and payload.strip() else None
    if payload is None:
        return streamed
    if streamed is None:
        if any(spent_text.strip() == payload.strip() for spent_text in spent):
            return None
        return payload
    if _contains(streamed, payload):
        return streamed
    return payload


class ChatStore:
    """The durable chat-history read/write surface (`chat_messages`)."""

    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory
        # Streamed reasoning not yet written to a row, keyed by (profile, stored session, run).
        self._reasoning: dict[_BufferKey, str] = {}
        # Reasoning already written to a row this turn, so the final payload is not written twice.
        self._spent: dict[_BufferKey, list[str]] = {}
        # Each held buffer's archive row; the step counter outlives it until the turn completes.
        self._archive: dict[_BufferKey, _ArchiveCursor] = {}
        self._steps: dict[_BufferKey, int] = {}
        self._archive_failed_logged = False
        # Called with (profile, stored session id) once a turn's final row is written.
        self.on_turn_completed: Callable[[str, str], None] | None = None


    def capture_submitted_user_row(
        self,
        *,
        profile: str,
        stored_session_id: str,
        text: str,
        turn_id: str | None = None,
    ) -> str:
        """Write the user's own row, BEFORE Hermes is asked to act on it."""
        # Always inserts: a submit is a fresh action, and the route discards it if Hermes fails.
        with self._session_factory() as db:
            return self._insert(
                db,
                profile=profile,
                stored_session_id=stored_session_id,
                turn_id=turn_id,
                role="user",
                text=text,
                source="submit",
            )

    def attach_turn(self, profile: str, stored_session_id: str, turn_id: str) -> int:
        """Give the user row that started `turn_id` its run id."""
        try:
            with self._session_factory() as db:
                row = db.execute(
                    select(ChatMessage)
                    .where(
                        ChatMessage.profile == profile,
                        ChatMessage.stored_session_id == stored_session_id,
                        ChatMessage.role == "user",
                        ChatMessage.turn_id.is_(None),
                    )
                    .order_by(ChatMessage.seq.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if row is None:
                    return 0
                row.turn_id = turn_id
                db.commit()
                return 1
        except Exception:
            logger.exception(
                "could not attach run %s to its user row (profile=%r session=%s)",
                turn_id,
                profile,
                stored_session_id,
            )
            return 0

    def capture_backfilled_user_row(
        self,
        *,
        profile: str,
        stored_session_id: str,
        text: str,
        hermes_row_id: int,
        turn_id: str | None,
    ) -> str | None:
        """Write a user prompt read back from Hermes's own transcript."""
        return self._write(
            profile,
            stored_session_id,
            role="user",
            text=text,
            hermes_row_id=hermes_row_id,
            source="backfill",
            turn_id=turn_id,
        )

    def capture_hook_user_row(
        self,
        *,
        profile: str,
        stored_session_id: str,
        text: str,
        turn_id: str,
    ) -> str | None:
        """Write a foreign turn's user prompt as delivered by a Hermes plugin hook."""
        with self._session_factory() as db:
            duplicate = db.execute(
                select(ChatMessage.id).where(
                    ChatMessage.profile == profile,
                    ChatMessage.stored_session_id == stored_session_id,
                    ChatMessage.turn_id == turn_id,
                    ChatMessage.role == "user",
                )
            ).scalar()
            if duplicate is not None:
                return None
            return self._insert(
                db,
                profile=profile,
                stored_session_id=stored_session_id,
                turn_id=turn_id,
                role="user",
                text=text,
                source="hook",
            )

    def discard_row(self, row_id: str) -> None:
        """Roll back a row written by `capture_submitted_user_row`."""
        with self._session_factory() as db:
            db.execute(delete(ChatMessage).where(ChatMessage.id == row_id))
            db.commit()

    def capture_event(self, profile: str, canonical: Any, envelope: dict[str, Any]) -> str | None:
        """Map one forwarded canonical event to a `chat_messages` row, if any."""
        try:
            return self._capture(profile, canonical, envelope)
        except Exception:  # pragma: no cover - defensive; same contract as B-26
            logger.exception(
                "chat store failed to capture a %r event for profile %r; continuing",
                getattr(canonical, "type", None),
                profile,
            )
            return None

    def _capture(self, profile: str, canonical: Any, envelope: dict[str, Any]) -> str | None:
        event_type = getattr(canonical, "type", None)
        if (
            event_type not in CAPTURED_EVENT_TYPES
            and event_type not in _REASONING_EVENT_TYPES
            and event_type not in _STEP_END_EVENT_TYPES
        ):
            return None

        payload = envelope.get("payload") if isinstance(envelope, dict) else None
        if not isinstance(payload, dict):
            payload = getattr(canonical, "payload", None)
        if not isinstance(payload, dict):
            return None

        stored_id = payload.get(_STORED_SESSION_ID_FIELD)
        if not isinstance(stored_id, str) or not stored_id:
            return None


        # The run recorder stamps this id onto the envelope and must run before this capture.
        turn_id = _turn_id(envelope)
        buffer_key = (profile, stored_id, turn_id)

        if event_type in _REASONING_EVENT_TYPES:
            self._hold_reasoning(buffer_key, payload)
            return None
        if event_type in _STEP_END_EVENT_TYPES:
            # A step that called a tool without writing any text has no `message.interim` to
            # carry its reasoning, so it gets a reasoning-only row of its own, as the app draws
            # it live. Not returned: the id would stamp the tool frame with another row's id.
            held = self._reasoning.get(buffer_key)
            if held and held.strip():
                row_id = self._write(
                    profile, stored_id, role="assistant", text="", reasoning=held,
                    source="live", turn_id=turn_id,
                )
                self._spend(buffer_key, row_id)
            return None
        if event_type == "message.interim":
            # The reasoning streamed before this segment belongs to it, as the app shows it live.
            row_id = self._write(
                profile,
                stored_id,
                role="assistant",
                text=payload.get("text"),
                reasoning=self._reasoning.get(buffer_key),
                source="live",
                turn_id=turn_id,
            )
            self._spend(buffer_key, row_id)
            return row_id
        if event_type == "tool.completed":
            return self._write(
                profile,
                stored_id,
                role="tool",
                tool_name=payload.get("name"),
                tool_call_id=payload.get("tool_id"),
                tool_args_json=payload.get("args"),
                tool_result_json=payload.get("result"),
                source="live",
                turn_id=turn_id,
            )
        if event_type == "message.completed":
            # Some models carry the authoritative reasoning text under
            # `reasoning_content` rather than `reasoning` -- the same fallback
            # `transcript.py._measure_reasoning`, `SessionMessage.reasoningText`
            # (app), and `hermes_state_reader.to_backfill_message` already
            # apply. Without it here, this row's reasoning is silently dropped
            # from the chat store -- the app's primary transcript source --
            # even though it streamed live and Hermes's own transcript has it.
            # A model that only streams its reasoning as `reasoning.delta` leaves both keys
            # empty; the text held from those frames is the row's reasoning then.
            streamed = self._reasoning.get(buffer_key)
            claimed = payload.get("reasoning") or payload.get("reasoning_content")
            claimed = claimed if isinstance(claimed, str) else None
            reasoning = final_reasoning(streamed, claimed, self._spent.get(buffer_key, []))
            row_id = self._write(
                profile,
                stored_id,
                role="assistant",
                text=payload.get("text"),
                reasoning=reasoning,
                source="live",
                turn_id=turn_id,
            )
            spent = self._spent.get(buffer_key, [])
            self._spend(buffer_key, row_id)
            if (
                claimed
                and claimed.strip()
                and not _contains(streamed or "", claimed)
                and not any(text.strip() == claimed.strip() for text in spent)
            ):
                # The payload's own text, kept even where the stream's text was chosen over it.
                self._archive_text(
                    profile, stored_id, turn_id, claimed, source=ARCHIVE_SOURCE_PAYLOAD,
                    chat_message_id=row_id,
                )
            self._end_turn(buffer_key)
            if self.on_turn_completed is not None:
                try:
                    self.on_turn_completed(profile, stored_id)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("turn-completed hook failed for session %r", stored_id)
            return row_id
        if event_type == "status.update":
            kind = payload.get("kind")
            text = payload.get("text")
            if not _is_captured_notice(kind, text):
                return None
            return self._write(
                profile,
                stored_id,
                role="marker",
                text=text,
                source="live",
                compacted=kind == "compacted",
                turn_id=turn_id,
            )
        return None  # pragma: no cover - CAPTURED_EVENT_TYPES is exhaustive above

    def _hold_reasoning(self, key: _BufferKey, payload: dict[str, Any]) -> None:
        """Append one `reasoning.delta`'s text to its run's held reasoning."""
        text = next(
            (v for k in _REASONING_TEXT_KEYS if isinstance(v := payload.get(k), str) and v),
            None,
        )
        if text is None:
            return
        if key not in self._reasoning:
            while len(self._reasoning) >= _MAX_REASONING_BUFFERS:
                evicted = next(iter(self._reasoning))
                # Abandoned, not lost: its archive row keeps everything that streamed.
                self._flush_archive(evicted, force=True)
                self._forget(evicted)
            self._reasoning[key] = ""
        self._reasoning[key] += text
        self._flush_archive(key, force=False)

    def _spend(self, key: _BufferKey, row_id: str | None) -> None:
        """The held reasoning is on a row now: seal its archive row and start the next step."""
        held = self._reasoning.pop(key, None)
        if held is None:
            return
        self._flush_archive(key, force=True, held=held, sealed=True, chat_message_id=row_id)
        self._archive.pop(key, None)
        self._steps[key] = self._steps.get(key, 0) + 1
        if held.strip():
            spent = self._spent.setdefault(key, [])
            spent.append(held)
            while len(self._spent) > _MAX_REASONING_BUFFERS:
                del self._spent[next(iter(self._spent))]

    def _end_turn(self, key: _BufferKey) -> None:
        self._spent.pop(key, None)
        self._steps.pop(key, None)

    def _forget(self, key: _BufferKey) -> None:
        self._reasoning.pop(key, None)
        self._archive.pop(key, None)
        self._end_turn(key)

    def _flush_archive(
        self,
        key: _BufferKey,
        *,
        force: bool,
        held: str | None = None,
        sealed: bool = False,
        chat_message_id: str | None = None,
    ) -> None:
        """Write a step's streamed reasoning to `reasoning_archive`, at most every few seconds."""
        text = held if held is not None else self._reasoning.get(key)
        if not text:
            return
        cursor = self._archive.get(key)
        if cursor is None:
            cursor = self._archive[key] = _ArchiveCursor(step=self._steps.get(key, 0))
        now = time.monotonic()
        unwritten = len(text) - cursor.flushed_chars
        if not force and (
            unwritten < _ARCHIVE_FLUSH_CHARS
            and cursor.row_id is not None
            and now - cursor.flushed_at < _ARCHIVE_FLUSH_INTERVAL_S
        ):
            return
        if unwritten == 0 and not sealed:
            return
        profile, stored_id, turn_id = key
        try:
            with self._session_factory() as db:
                row = db.get(ReasoningArchive, cursor.row_id) if cursor.row_id else None
                if row is None:
                    row = ReasoningArchive(
                        id=new_id("ra"),
                        profile=profile,
                        stored_session_id=stored_id,
                        turn_id=turn_id,
                        step=cursor.step,
                        source=ARCHIVE_SOURCE_STREAM,
                        text=text,
                        created_at=utcnow(),
                    )
                    db.add(row)
                row.text = text
                row.sealed = sealed
                if chat_message_id is not None:
                    row.chat_message_id = chat_message_id
                row.updated_at = utcnow()
                db.commit()
                cursor.row_id = row.id
        except Exception:
            # The archive is a second copy: failing it must never cost the chat row itself.
            if not self._archive_failed_logged:
                self._archive_failed_logged = True
                logger.exception("reasoning archive write failed; chat capture continues")
            return
        cursor.flushed_chars = len(text)
        cursor.flushed_at = now

    def _archive_text(
        self,
        profile: str,
        stored_id: str,
        turn_id: str | None,
        text: str,
        *,
        source: str,
        chat_message_id: str | None = None,
    ) -> None:
        """One finished reasoning body into `reasoning_archive`, best-effort."""
        try:
            with self._session_factory() as db:
                db.add(
                    ReasoningArchive(
                        id=new_id("ra"),
                        profile=profile,
                        stored_session_id=stored_id,
                        turn_id=turn_id,
                        source=source,
                        chat_message_id=chat_message_id,
                        text=text,
                        sealed=True,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )
                db.commit()
        except Exception:
            if not self._archive_failed_logged:
                self._archive_failed_logged = True
                logger.exception("reasoning archive write failed; chat capture continues")

    def reasoning_archive(self, *, profile: str, stored_session_id: str) -> list[dict[str, Any]]:
        """Every archived reasoning body for one session, oldest first."""
        with self._session_factory() as db:
            rows = list(
                db.execute(
                    select(ReasoningArchive)
                    .where(
                        ReasoningArchive.profile == profile,
                        ReasoningArchive.stored_session_id == stored_session_id,
                    )
                    .order_by(ReasoningArchive.created_at.asc(), ReasoningArchive.step.asc())
                ).scalars().all()
            )
        return [
            {
                "id": row.id,
                "turn_id": row.turn_id,
                "step": row.step,
                "source": row.source,
                "hermes_row_id": row.hermes_row_id,
                "chat_message_id": row.chat_message_id,
                "sealed": bool(row.sealed),
                "chars": len(row.text),
                "text": row.text,
                "created_at": _iso(row.created_at),
                "updated_at": _iso(row.updated_at),
            }
            for row in rows
        ]

    def _write(
        self,
        profile: str,
        stored_session_id: str,
        *,
        role: str,
        text: str | None = None,
        reasoning: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_args_json: Any = None,
        tool_result_json: Any = None,
        hermes_row_id: int | None = None,
        source: str,
        turn_id: str | None = None,
        compacted: bool = False,
    ) -> str | None:
        """Idempotent insert for one live-captured event. See module docstring."""
        with self._session_factory() as db:
            if hermes_row_id is not None:
                duplicate = db.execute(
                    select(ChatMessage.id).where(
                        ChatMessage.profile == profile,
                        ChatMessage.stored_session_id == stored_session_id,
                        ChatMessage.hermes_row_id == hermes_row_id,
                    )
                ).scalar()
                if duplicate is not None:
                    return None
            elif tool_call_id is not None:
                duplicate = db.execute(
                    select(ChatMessage.id).where(
                        ChatMessage.profile == profile,
                        ChatMessage.stored_session_id == stored_session_id,
                        ChatMessage.tool_call_id == tool_call_id,
                    )
                ).scalar()
                if duplicate is not None:
                    return None
            else:
                # No live event carries a row id: a redelivery is caught by the newest row's text.
                last = db.execute(
                    select(ChatMessage)
                    .where(
                        ChatMessage.profile == profile,
                        ChatMessage.stored_session_id == stored_session_id,
                    )
                    .order_by(ChatMessage.seq.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if (
                    last is not None
                    and last.role == role
                    and last.text == text
                    and last.reasoning == reasoning
                    and last.source == source
                ):
                    return None

            return self._insert(
                db,
                profile=profile,
                stored_session_id=stored_session_id,
                turn_id=turn_id,
                role=role,
                text=text,
                reasoning=reasoning,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_args_json=tool_args_json,
                tool_result_json=tool_result_json,
                hermes_row_id=hermes_row_id,
                source=source,
                compacted=compacted,
            )

    def _next_seq(self, db: OrmSession, profile: str, stored_session_id: str) -> int:
        current = db.execute(
            select(func.max(ChatMessage.seq)).where(
                ChatMessage.profile == profile,
                ChatMessage.stored_session_id == stored_session_id,
            )
        ).scalar()
        return (current or 0) + 1

    def _insert(
        self,
        db: OrmSession,
        *,
        profile: str,
        stored_session_id: str,
        turn_id: str | None,
        role: str,
        text: str | None = None,
        reasoning: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_args_json: Any = None,
        tool_result_json: Any = None,
        hermes_row_id: int | None = None,
        source: str,
        compacted: bool = False,
    ) -> str:
        """Allocate the next `seq` and insert, retrying on a real collision."""
        row = ChatMessage(
            id=new_id("cm"),
            profile=profile,
            stored_session_id=stored_session_id,
            turn_id=turn_id,
            seq=self._next_seq(db, profile, stored_session_id),
            role=role,
            text=text,
            reasoning=reasoning,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args_json=tool_args_json,
            tool_result_json=tool_result_json,
            hermes_row_id=hermes_row_id,
            source=source,
            compacted=compacted,
            created_at=utcnow(),
        )
        # A seq collision is two writers racing, not duplicate content: recompute, never drop.
        for _attempt in range(_MAX_SEQ_RETRIES):
            db.add(row)
            try:
                db.commit()
                return row.id
            except IntegrityError:
                db.rollback()
                row.seq = self._next_seq(db, profile, stored_session_id)
        raise RuntimeError(
            f"could not allocate a unique chat_messages.seq for "
            f"(profile={profile!r}, stored_session_id={stored_session_id!r}) "
            f"after {_MAX_SEQ_RETRIES} attempts"
        )


    def tool_rows(self, *, profile: str, stored_session_id: str) -> list[dict[str, Any]]:
        """Every captured tool row for one session, ascending by `seq`."""
        with self._session_factory() as db:
            query = (
                select(ChatMessage)
                .where(
                    ChatMessage.profile == profile,
                    ChatMessage.stored_session_id == stored_session_id,
                    ChatMessage.role == "tool",
                )
                .order_by(ChatMessage.seq.asc())
            )
            rows = list(db.execute(query).scalars().all())
        return [_row_dict(row) for row in rows]

    def tool_result(
        self, *, stored_session_id: str, tool_call_id: str, max_chars: int
    ) -> dict[str, Any] | None:
        """One captured tool result, by the call id the audit trail carries."""
        # Not keyed on profile: the call id is unique and a ledger profile can be wrong.
        with self._session_factory() as db:
            row = db.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.stored_session_id == stored_session_id,
                    ChatMessage.tool_call_id == tool_call_id,
                    ChatMessage.role == "tool",
                )
                .order_by(ChatMessage.seq.asc())
                .limit(1)
            ).scalar_one_or_none()
        if row is None:
            return None

        # A JSON column returns a decoded dict, not text: len() would count keys, not characters.
        raw = row.tool_result_json
        if raw is None:
            text = ""
        elif isinstance(raw, str):
            text = raw
        else:
            try:
                text = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(raw)
        total = len(text)
        truncated = total > max_chars
        return {
            "tool_call_id": row.tool_call_id,
            "tool_name": row.tool_name,
            "stored_session_id": row.stored_session_id,
            "profile": row.profile,
            "text": text[:max_chars] if truncated else text,
            "total_chars": total,
            "truncated": truncated,
        }

    def page(
        self,
        *,
        profile: str,
        stored_session_id: str,
        before_seq: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Up to `limit` rows immediately before `before_seq`, ascending by `seq`."""
        with self._session_factory() as db:
            query = select(ChatMessage).where(
                ChatMessage.profile == profile,
                ChatMessage.stored_session_id == stored_session_id,
            )
            if before_seq is not None:
                query = query.where(ChatMessage.seq < before_seq)
            query = query.order_by(ChatMessage.seq.desc()).limit(limit)
            rows = list(db.execute(query).scalars().all())
        rows.reverse()
        return [_row_dict(row) for row in rows]
