"""`ChatStore` -- the read/write interface over `chat_messages` (B-136).

See `docs/CHAT_HISTORY_DESIGN.md` §3/§4 for the design this implements.
Follows the same shape as the P2-1/P2-2 ledgers (`api/background.py`'s
`BackgroundLedger`, `api/runs.py`'s `RunRecorder`): owns its own
`sessionmaker`, opens a session per call, and never raises past its own
per-event entry point -- a chat-history write failing must never take down
the event pump or the turn-submit route it rides inside.

## What gets captured, and the idempotency rule

These write rows here (`docs/CHAT_HISTORY_DESIGN.md` §3's table, extended):

* `message.interim` (a segment finished) -> one `assistant` row.
* `tool.completed` (raw wire name `tool.complete`) -> one `tool` row.
* `message.completed` (turn finished) -> one `assistant` row, carrying the
  turn's full reasoning (D-3, never truncated). The design note "plus record
  the turn's final usage" is NOT implemented here -- `chat_messages` (§4) has
  no usage/token column, and `session.usage` is already durable in
  `run_events` (`events/persistence.py::PERSISTED_RUN_EVENT_TYPES`) via
  `RunRecorder`. Recording it a second time in this table would need a schema
  change this migration does not make; flagged rather than silently
  invented.
* `status.update` -> one `marker` row for three notices, text verbatim
  (B-191; `RUN_EVENT_UX_AUDIT.md` §2.3):
  - `kind == "compacted"`: a bookmark that a compaction happened
    (`compacted=True`). Compaction never deletes Hermes's own rows (§1.4),
    so nothing needs backfilling around it.
  - `kind == "lifecycle"` whose text names Hindsight (measured: "👁️
    Hindsight — recalled 43 memories"): the turn's memory-recall fact,
    which the app renders as a durable note row at the top of the turn.
  - `kind == "process"`: a background process ended.
  Every other notice is forwarded live and recorded nowhere here -- in
  particular the compression warning ("Session compressed N times …", kind
  `lifecycle`), which repeats on every step of a turn and whose count is
  already durable in `session.usage.compressions`.
* the gateway's own turn-submit route -> one `user` row, written via
  `capture_submitted_user_row` BEFORE Hermes is called (so a durable copy of
  what the user typed exists even if the Hermes call then fails; the route
  calls `discard_row` in that case).
* `domain/foreign_prompt_capture.py` (B-190) -> one `user` row with
  `source="backfill"` and its `hermes_row_id`, for a turn started from
  Hermes's own TUI or another client -- the one prompt the submit route
  never sees. Written via `capture_backfilled_user_row`.

**Idempotency ("use `hermes_row_id` where you have it, else the event's own
natural key"):** none of the three live event payloads Hermes actually pushes
(`message.interim`, `tool.complete`, `message.complete`) carry a Hermes-side
row id on the wire -- only the transcript rows `session.resume` /
`session.history` return do (`docs/PROTOCOL_VERIFIED.md`). `hermes_row_id`
is therefore populated by backfill, not by live capture, and this store's
live-capture idempotency has to substitute something else:

* `tool.completed` carries Hermes's own `tool_id` (verified live,
  `docs/PROTOCOL_VERIFIED.md`'s `tool.complete` capture) -- mapped onto the
  `tool_call_id` column, which *is* a stable natural key, so a redelivered
  `tool.completed` for the same `tool_id` is detected directly.
* `message.interim` / `message.completed` / the compaction marker carry no
  id of any kind. This store's chosen natural key for those is "the same
  (profile, stored_session_id) pair's most recently captured row is
  byte-identical in every column this event would write" -- i.e. a redelivery
  right behind the original is caught; two genuinely distinct occurrences of
  identical text are extremely rare for prose and are not what "redelivered"
  means here (the codebase's own measurements elsewhere record that nothing
  buffers or replays the live stream, so an exact redelivery is itself a rare
  edge, not routine traffic). This is a judgment call the design doc leaves
  open ("the event's own natural key") and is recorded here rather than
  silently assumed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from .models import ChatMessage, new_id, utcnow

logger = logging.getLogger(__name__)

#: B-29 identity key, by literal (same import-cycle avoidance as `api/runs.py`
#: and `api/background.py`; pinned equal to `api.main.STORED_SESSION_ID_FIELD`
#: by `tests/test_chat_store.py`).
_STORED_SESSION_ID_FIELD = "_stored_session_id"

#: Canonical event types this store captures live, per
#: `docs/CHAT_HISTORY_DESIGN.md` §3's table. Everything else is a no-op.
CAPTURED_EVENT_TYPES: frozenset[str] = frozenset(
    {"message.interim", "tool.completed", "message.completed", "status.update"}
)

#: `status.update` kinds captured as marker rows outright (B-191). A
#: `lifecycle` notice is captured only when its text names Hindsight.
_MARKER_STATUS_KINDS: frozenset[str] = frozenset({"compacted", "process"})
_LIFECYCLE_STATUS_KIND = "lifecycle"
#: Measured 2026-08-29 (`tests/test_event_normalizer.py`'s live capture):
#: "👁️ Hindsight — recalled 32 memories". Matched on the product name only,
#: since the glyph and the dash have both varied across Hermes builds.
MEMORY_RECALL_MARKER = "Hindsight"

#: `notice_kind` values on a `marker` row (`_row_dict`), derived -- no column.
NOTICE_KIND_COMPACTED = "compacted"
NOTICE_KIND_MEMORY = "memory"
NOTICE_KIND_PROCESS = "process"

#: How many times `_write` retries after a `seq` collision (a genuine race
#: between two writers for the same (profile, stored_session_id), not a
#: duplicate-content skip -- see `_write`'s docstring) before giving up.
_MAX_SEQ_RETRIES = 5


def _turn_id(envelope: Any) -> str | None:
    """The envelope's `run_id` as a turn id, or None (B-188)."""
    if not isinstance(envelope, dict):
        return None
    run_id = envelope.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def notice_kind(row: ChatMessage) -> str | None:
    """Which notice a `marker` row is, for the app to render without re-parsing.

    Derived from the row, not stored (B-191): `compacted` from the flag,
    `memory` when the text names Hindsight, and `process` for the only other
    marker this store writes. Null for every non-marker row. If a fourth
    marker source is ever added, it needs its own rule here -- the fallback
    is `process` because that is the one remaining writer today, not
    because "unknown" means process.
    """
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


def _is_captured_notice(kind: Any, text: Any) -> bool:
    """Whether a `status.update` earns a marker row (B-191, module docstring).

    `compacted` and `process` by kind; `lifecycle` only when the text names
    Hindsight -- the other lifecycle notice measured live is the per-step
    compression warning, which is deliberately not a row.
    """
    if kind in _MARKER_STATUS_KINDS:
        return True
    if kind == _LIFECYCLE_STATUS_KIND:
        return isinstance(text, str) and MEMORY_RECALL_MARKER in text
    return False


class ChatStore:
    """The durable chat-history read/write surface (`chat_messages`).

    One instance per gateway process (`app.state.chat_store`, built in
    `api.main.lifespan` alongside the other P1/P2 ledgers), shared by every
    `ProfileConnection`'s capture hook and by the turn-submit route.
    """

    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    # -- writes -----------------------------------------------------------

    def capture_submitted_user_row(
        self,
        *,
        profile: str,
        stored_session_id: str,
        text: str,
        turn_id: str | None = None,
    ) -> str:
        """Write the user's own row, BEFORE Hermes is asked to act on it.

        Always inserts (never treated as a possible duplicate): a submit is a
        fresh user action every time, and the caller (the turn-submit route)
        calls `discard_row(row_id)` itself if the subsequent Hermes call
        fails. Returns the new row's id.
        """
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
        """Give the user row that started `turn_id` its run id (B-188).

        The submit route writes the user's row BEFORE Hermes is called, so
        no run exists yet to stamp it with; `RunRecorder` reports the open
        (`on_run_opened`) and this attaches the run to the newest user row of
        that session that has no turn yet. One row at most: a message
        Hermes queued behind a running turn keeps waiting for its own run,
        and one it folded into the running reply (a steer) is left with no
        turn of its own, which is what happened to it.

        Positional arguments, deliberately: this is the exact
        `(profile, stored_id, run_id)` shape `RunRecorder.on_run_opened`
        calls with. Returns how many rows were attached (0 or 1). Never
        raises past this method -- a miss here costs one link, not the run.
        """
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
        """Write a user prompt read back from Hermes's own transcript (B-190).

        The foreign-prompt path (`domain/foreign_prompt_capture.py`): a turn
        started from the TUI or another client has no submit-route row, so
        its prompt is read from `session.resume` and stored here with
        `source="backfill"` and the Hermes `row_id` it came with. Idempotent
        on `hermes_row_id` (the store's strongest natural key, module
        docstring): a row already stored for this `(profile, session,
        hermes_row_id)` is a no-op and returns None, so a repeated run open
        for the same turn cannot duplicate it. Raises on a DB failure --
        the caller owns the best-effort guard.
        """
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
        """Write a foreign turn's user prompt as delivered by a Hermes plugin hook.

        Same job as `capture_backfilled_user_row`, different source. That one
        reads the prompt back out of Hermes's transcript with a whole
        `session.resume` (up to 1.6 MB) and dedups on the `hermes_row_id` the
        transcript carries. A `pre_llm_call` hook hands the same text over
        in-process for nothing -- but carries **no** Hermes row id, so it needs
        a different natural key.

        The key is `(profile, stored_session_id, turn_id)` with `role="user"`:
        a run has exactly one user prompt, so a second user row for a run this
        store has already recorded one for is by definition a duplicate. That
        makes a repeated run-open, or a hook and a transcript read racing for
        the same turn, a no-op returning None.

        `turn_id` is required here, unlike on the backfill path -- without it
        there is no key at all and the row could duplicate silently on every
        retry. Raises on a DB failure; the caller owns the best-effort guard.
        """
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
        """Roll back a row written by `capture_submitted_user_row`.

        Used only when the Hermes call that was supposed to follow it never
        happened -- mirrors the existing `submit_turn` route's own
        error-handling style (structured, never a bare crash). A row that
        does not exist (already discarded, or never written) is a silent
        no-op: discarding is idempotent by construction.
        """
        with self._session_factory() as db:
            db.execute(delete(ChatMessage).where(ChatMessage.id == row_id))
            db.commit()

    def capture_event(self, profile: str, canonical: Any, envelope: dict[str, Any]) -> str | None:
        """Map one forwarded canonical event to a `chat_messages` row, if any.

        `canonical` is a `events.canonical.CanonicalEvent` (typed `Any` here
        to avoid an import cycle with `events`, which this module does not
        otherwise need); `envelope` is that event's `to_dict()` -- the same
        two arguments `RunRecorder.handle_event` receives, so a
        `ProfileConnection` can wire this in exactly the same way.

        Never raises: a capture failure costs this one row, never the event
        stream it rides inside (same contract as `RunRecorder.handle_event`
        and `BackgroundLedger.handle_completed`). Returns the new row's id,
        or `None` when the event was not one of `CAPTURED_EVENT_TYPES`, was
        unattributed (no `_stored_session_id` on the payload), or was
        recognized as a duplicate/redelivery.
        """
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
        if event_type not in CAPTURED_EVENT_TYPES:
            return None

        payload = envelope.get("payload") if isinstance(envelope, dict) else None
        if not isinstance(payload, dict):
            payload = getattr(canonical, "payload", None)
        if not isinstance(payload, dict):
            return None

        stored_id = payload.get(_STORED_SESSION_ID_FIELD)
        if not isinstance(stored_id, str) or not stored_id:
            # Unattributed frame (no live->stored mapping resolved yet on
            # this connection). Honest nothing, same rule as every other
            # attribution path in this codebase (P2-2).
            return None

        # B-188: the run this frame belongs to, stamped on the envelope by
        # `RunRecorder.handle_event` (which runs BEFORE this capture on both
        # pump paths -- `ProfileConnectionManager._forward_one` and the
        # broadcaster's `_forward_one`). `run_id` == `turn_id`: a run is one
        # turn (`domain/runs.py`). None when the recorder could not
        # attribute the frame, which is the honest column value.
        turn_id = _turn_id(envelope)

        if event_type == "message.interim":
            return self._write(
                profile,
                stored_id,
                role="assistant",
                text=payload.get("text"),
                source="live",
                turn_id=turn_id,
            )
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
            return self._write(
                profile,
                stored_id,
                role="assistant",
                text=payload.get("text"),
                reasoning=payload.get("reasoning"),
                source="live",
                turn_id=turn_id,
            )
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
        """Allocate the next `seq` and insert, retrying on a real collision.

        A `seq` collision (`IntegrityError` on the unique constraint) means
        two writers raced for the same `(profile, stored_session_id)` --
        e.g. the submit route and a live capture landing at the same instant
        -- and is NOT a duplicate-content signal (that is decided by the
        caller before this is reached). Retrying with a freshly recomputed
        `seq` is what keeps a race from silently dropping a real message.
        """
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
        for _attempt in range(_MAX_SEQ_RETRIES):
            db.add(row)
            try:
                db.commit()
                return row.id
            except IntegrityError:
                # A failed flush already detaches `row` back to transient
                # (SQLAlchemy's own rollback behavior for pending objects),
                # so it is safe to mutate and re-`add()` on the next
                # iteration without an explicit `expunge()`.
                db.rollback()
                row.seq = self._next_seq(db, profile, stored_session_id)
        raise RuntimeError(
            f"could not allocate a unique chat_messages.seq for "
            f"(profile={profile!r}, stored_session_id={stored_session_id!r}) "
            f"after {_MAX_SEQ_RETRIES} attempts"
        )

    # -- reads --------------------------------------------------------------

    def tool_rows(self, *, profile: str, stored_session_id: str) -> list[dict[str, Any]]:
        """Every captured tool row for one session, ascending by `seq` (B-156).

        The input to `domain.tool_result_backfill.attach_captured_results`:
        Hermes's own transcript keeps a call's arguments but never its result,
        and these rows are the only durable copy of what each call returned.
        """
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
        """One captured tool result, by the call id the audit trail carries.

        The audit store deliberately does NOT hold tool output: a file read
        returns the file, and the audit archive cannot be edited or deleted for
        the retention window (owner, 2026-09-20). This store does hold it, as
        part of the conversation, and it is what the audit detail screen offers
        behind an explicit tap.

        The two are NOT the same kind of evidence and the caller is expected to
        say so. This row lives in a database the gateway writes freely, so it
        can be rewritten; the audit trail cannot. Returned unredacted, because
        it was never passed through the audit path's masking.

        `profile` is not a parameter: the audit trail's tool call id is
        unique on its own, and requiring a profile here would make the lookup
        fail for exactly the sessions whose profile the run ledger records
        wrongly (B-206).

        Bounded by `max_chars`, and says whether it truncated. Results run past
        200,000 characters on the owner's host; a screen must not be handed the
        whole thing by default.
        """
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
        # `tool_result_json` is a JSON column, so SQLAlchemy hands back the
        # DECODED value -- usually a dict, sometimes a list, occasionally a
        # bare string. Treating it as text made `len()` count keys: a 1,763
        # character result reported itself as 4 and the dict went out under a
        # field the client decodes as a string, so the screen rendered empty
        # with no error anywhere. Caught live, 2026-09-20.
        #
        # Re-serialised with indentation rather than compactly: this is going
        # into a source viewer, and pretty-printed JSON is both what a person
        # wants to read and what the highlighter can colour.
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
        """Up to `limit` rows immediately before `before_seq`, ascending by `seq`.

        `before_seq=None` (the default) returns the newest `limit` rows --
        the initial page a chat screen opens to. Passing the oldest `seq`
        already shown pages further into the past ("reveal earlier",
        `docs/CHAT_HISTORY_DESIGN.md` §6's `revealEarlier()`).
        """
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
