"""`BackgroundLedger`: the background-task completion pipeline (P2-1, fixes B-42).

Moved out of `api/background.py` (CLEANUP_PLAN step 3.5); the submit/list
routes stay there and re-export these names.

Why a ledger at all: `prompt.background` is fire-and-forget and **Hermes
keeps no trace of it** (PV "Phase 2 probe", measured live 2026-08-30). The
submit returns `{"task_id": "bg_..."}` and then the wire is silent -- no
events, no pollable status, no transcript row, and a background-only session
is never persisted. Completion is exactly one `background.complete`
`{task_id, text}` event, delivered only to connections attached to the
session at that instant. Until P2-1 the gateway dropped that event unmapped
(B-42), so the app could never learn a task finished. The `background_tasks`
table (written at submit, updated on completion) plus that one event are the
background pill's entire data source.

2. **`BackgroundLedger`** -- the event-side orchestration, wired to
   `EventBroadcaster` by `api.main.lifespan`:

   * `handle_completed` runs when the pump forwards a
     `background.completed` frame: marks the row finished, stores the
     result text, backfills the frame's `_stored_session_id` from the
     ledger (the completion frame carries only a LIVE handle, and the
     ledger's submit-time record is the authoritative mapping), and returns
     a synthesized `message.interim` frame for the broadcaster to inject
     into the live stream -- so the open transcript shows the task ran.
   * `handle_generation_change` runs when the Hermes connection is
     replaced: every running row is marked **orphaned** (P2-0b measured the
     completion event as lost-by-default across a reconnect: delivered only
     to attached connections, never buffered, never replayed -- 3/3 lost on
     a bare reconnect), then the **rescue** re-resumes every session that
     still has an unfinished task, because the same probe measured that a
     completion firing *after* a re-resume on the new connection is
     delivered normally (2/2). A rescued row goes `orphaned -> finished`
     when its completion arrives; one that completed inside the disconnect
     window stays orphaned, honestly, forever.
   * `orphan_on_startup` covers the gateway-restart half of the same rule:
     nothing was attached while the process was down, so every running row's
     outcome is unknown. The first real connection after startup bumps the
     adapter's generation (0 -> 1), so the reconnect rescue also runs then
     and can still upgrade a survivor.

3. **Transcript injection** (`append_finished_background_results`): Hermes
   writes no transcript row for a background turn, so on every transcript
   read (`POST /resume`, `GET /messages`) the finished results for that
   session are appended as synthesized rows. Best-effort and never fatal,
   exactly like `_annotate_filing_status` -- a local storage problem must
   not take down the transcript.

The synthesized live frame is a `message.interim` with
`already_streamed: false`, deliberately: B-38 pinned that type's contract as
"the client has never seen this text and must insert it as its own
segment", which is precisely what an injected result is. A synthesized
`message.completed` would *replace* the app's streaming buffer (B-38's
bug), and a `message.delta` would append into whatever bubble is open. It
carries `seq` 0 like every gateway-synthesized frame: `seq` is the client's
replay cursor over the *upstream* stream and a frame Hermes never sent must
not spend one.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import sessionmaker

from domain import background_tasks as ledger_ops

# The canonical type `events/canonical.py` maps `background.complete` to.
# Defined in `domain/event_stream.py` (the broadcaster keys its hook on it);
# re-exported here, and from `api/background.py`, for the ledger's callers
# and tests.
from domain.event_stream import BACKGROUND_COMPLETED_EVENT_TYPE  # noqa: F401
from domain.hermes_runtime import _resume_for_live_id
from domain.models import BackgroundTask
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

#: The type of the synthesized transcript frame -- see module docstring for
#: why it is `message.interim` and not `message.completed`/`message.delta`.
SYNTHESIZED_MESSAGE_EVENT_TYPE = "message.interim"

#: Marker stamped on every synthesized message (frame payload and transcript
#: row alike) naming the raw event it was synthesized from. `_`-prefixed:
#: the gateway's namespace, same convention as `_stored_session_id` /
#: `_truncated` / `_raw_type`, so upstream cannot forge it.
SYNTHESIZED_FROM_FIELD = "_synthesized_from"
SYNTHESIZED_FROM_VALUE = "background.complete"


#: Honest copy for an orphaned row, served with every listing so the client
#: does not have to invent its own explanation (P2-1: "surfaced honestly").
ORPHANED_EXPLANATION = (
    "submitted before a gateway restart or reconnect; the work most likely "
    "completed, but its one completion announcement fired while nothing was "
    "listening and Hermes keeps no copy, so the outcome is unknown"
)


def task_row(task: BackgroundTask) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task.task_id,
        "stored_session_id": task.stored_session_id,
        "prompt_text": task.prompt_text,
        "state": task.state,
        "submitted_at": iso_z(task.submitted_at),
        "finished_at": iso_z(task.finished_at),
        "result_text": task.result_text,
        "connection_generation": task.connection_generation,
    }
    if task.state == ledger_ops.STATE_ORPHANED:
        row["state_explanation"] = ORPHANED_EXPLANATION
    return row


# ---------------------------------------------------------------------------
# Event-side orchestration
# ---------------------------------------------------------------------------


class BackgroundLedger:
    """Wires the ledger into the event pump and the reconnect lifecycle.

    Constructed once per app in `api.main.lifespan` and handed to
    `EventBroadcaster` as its `on_background_complete` /
    `on_generation_change` hooks. DB access is the same deliberate
    synchronous SQLite the rest of the gateway uses (`domain/db.py`): one
    single-row write per task completion and one bounded UPDATE per
    reconnect, both microseconds against a local file -- not worth a second
    storage story. Revisit alongside P2-2's sync-in-async-pump decision if
    event persistence ever makes the pump write-heavy.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        adapter: Any,
        live_handle_cache: Any,
        *,
        schedule: Callable[[Any], Any] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._adapter = adapter
        self._cache = live_handle_cache
        # Injectable for tests; defaults to real task scheduling. The rescue
        # is async (it performs session.resume round trips) while the
        # generation-change hook is called synchronously from the pump.
        self._schedule = schedule if schedule is not None else asyncio.create_task
        self._rescue_task: Any = None

    # -- startup ----------------------------------------------------------

    def orphan_on_startup(self) -> int:
        """`running -> orphaned` for rows left over from a previous process.

        A gateway restart resets everything that could deliver a completion
        (the pump, the adapter socket, every subscription), so any row still
        `running` was submitted by a dead process and its announcement was
        either missed or is un-deliverable until a rescue re-attach. Best
        effort: an unmigrated or unreadable DB logs and moves on -- startup
        must not crash over a ledger sweep (the routes will 503 with the
        real explanation).
        """
        try:
            with self._session_factory() as db:
                rows = ledger_ops.orphan_all_running(db)
                db.commit()
        except Exception:
            logger.warning(
                "could not sweep the background-task ledger on startup; "
                "continuing (is the database migrated?)",
                exc_info=True,
            )
            return 0
        if rows:
            logger.info(
                "marked %d running background task(s) orphaned on startup: %s",
                len(rows),
                [row.task_id for row in rows],
            )
        return len(rows)

    # -- reconnect --------------------------------------------------------

    def handle_generation_change(self, generation: int | None, previous: int | None) -> None:
        """Orphan running rows, then schedule the re-resume rescue (P2-0b).

        Synchronous by contract (called from the pump / generation watcher),
        so the async rescue is scheduled rather than awaited. Ordering is
        deliberate: orphan first (fast, guaranteed, honest -- from this
        instant the outcome genuinely is unknown), rescue second (best
        effort); a rescued completion upgrades its row back to finished.
        """
        try:
            with self._session_factory() as db:
                rows = ledger_ops.orphan_all_running(db)
                db.commit()
        except Exception:
            logger.warning(
                "could not orphan running background tasks on generation change %s -> %s",
                previous,
                generation,
                exc_info=True,
            )
            rows = []
        if rows:
            logger.info(
                "Hermes connection generation moved %s -> %s with %d running "
                "background task(s); marked orphaned, attempting re-resume rescue",
                previous,
                generation,
                len(rows),
            )
        try:
            self._rescue_task = self._schedule(self.rescue_unfinished_sessions())
        except RuntimeError:  # pragma: no cover - no running loop (tests)
            logger.warning("no event loop to schedule the background-task rescue on")

    async def rescue_unfinished_sessions(self) -> int:
        """Re-resume every session holding an unfinished task; return sessions tried.

        This is the measured rescue (PV "Phase 2a probes"): the completion
        event is delivered to connections attached to the session at the
        instant it fires, so re-attaching via `session.resume` on the new
        connection *before* the task completes gets the announcement
        delivered normally (2/2, vs 0/3 without). `_resume_for_live_id` also
        seeds `LiveHandleCache` with the fresh live handle, which is what
        lets the broadcaster attribute the arriving frame to the stored id.

        Per-session best effort: one un-resumable session (deleted, `[4007]`,
        Hermes mid-restart) must not abandon the rescue of the others.
        """
        try:
            with self._session_factory() as db:
                stored_ids = ledger_ops.stored_session_ids_with_unfinished_tasks(db)
        except Exception:
            logger.warning("could not read the ledger for the rescue", exc_info=True)
            return 0
        rescued = 0
        for stored_id in stored_ids:
            try:
                await _resume_for_live_id(self._adapter, stored_id, self._cache)
                rescued += 1
            except Exception:
                logger.warning(
                    "re-resume rescue failed for session %s; a background task "
                    "completion there may be lost",
                    stored_id,
                    exc_info=True,
                )
        return rescued

    # -- completion -------------------------------------------------------

    def handle_completed(self, envelope: dict[str, Any]) -> dict[str, Any] | None:
        """Apply one forwarded `background.completed` frame to the ledger.

        Called by `EventBroadcaster._forward_one` after session-identity
        stamping and before fan-out. Three jobs:

        1. `running|orphaned -> finished`, storing the result text -- the
           only copy of it anywhere.
        2. Backfill the frame's `_stored_session_id` from the ledger when
           the live-handle cache could not attribute it (the completion is
           stamped with a live handle the gateway may never have resolved
           on this connection; the ledger's submit-time record is the
           authoritative mapping). This is what lets the app's B-29 filter
           match the frame at all.
        3. Return the synthesized `message.interim` frame to inject into the
           live stream (or None when there is nothing renderable), so the
           open transcript shows the task ran.

        Never raises: a ledger failure must not cost the frame its fan-out.
        """
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            return None
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            logger.warning(
                "background.completed frame carries no usable task_id; "
                "forwarding it unledgered (keys: %s)",
                sorted(payload),
            )
            return None
        raw_text = payload.get("text")
        result_text = raw_text if isinstance(raw_text, str) else None
        # The broadcaster's own attribution, present when the live handle was
        # already known to the cache (e.g. after a rescue re-resume).
        stamped_stored = payload.get("_stored_session_id")
        hint = stamped_stored if isinstance(stamped_stored, str) and stamped_stored else None

        try:
            with self._session_factory() as db:
                row = ledger_ops.record_completed(
                    db,
                    task_id=task_id,
                    result_text=result_text,
                    stored_session_id_hint=hint,
                )
                db.commit()
                stored_id = row.stored_session_id
        except Exception:
            logger.exception(
                "failed to record background.complete for task %r in the ledger",
                task_id,
            )
            return None

        if hint is None and stored_id:
            # Job 2: the ledger knew what the cache did not.
            payload["_stored_session_id"] = stored_id

        if not stored_id or result_text is None:
            # Unattributable or resultless: the ledger row is still the
            # record; there is no session timeline to inject into (or nothing
            # to say). The background.completed frame itself is forwarded
            # regardless by the broadcaster.
            return None
        return self._synthesized_message_frame(envelope, payload, task_id, result_text, stored_id)

    @staticmethod
    def _synthesized_message_frame(
        envelope: dict[str, Any],
        payload: dict[str, Any],
        task_id: str,
        result_text: str,
        stored_id: str,
    ) -> dict[str, Any]:
        """The injected transcript frame -- shaped as an ordinary canonical envelope.

        `seq` 0 (gateway-synthetic, spends no upstream cursor -- same rule as
        `stream.ready`/`stream.resync`); identity fields copied from the
        completion frame it was synthesized from, with `_stored_session_id`
        guaranteed non-null (that is the field the app matches on, B-29).
        `already_streamed: false` is the B-38 contract for "insert this text
        as its own segment -- you have never seen it".
        """
        return {
            "event_id": f"evt_{uuid.uuid4().hex}",
            "project_id": envelope.get("project_id"),
            "session_id": envelope.get("session_id"),
            "run_id": envelope.get("run_id"),
            "seq": 0,
            "type": SYNTHESIZED_MESSAGE_EVENT_TYPE,
            "timestamp": envelope.get("timestamp")
            or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "payload": {
                "role": "assistant",
                "text": result_text,
                "already_streamed": False,
                "task_id": task_id,
                SYNTHESIZED_FROM_FIELD: SYNTHESIZED_FROM_VALUE,
                "_stored_session_id": stored_id,
                "_live_session_id": payload.get("_live_session_id"),
                "_connection_generation": payload.get("_connection_generation"),
            },
        }


# ---------------------------------------------------------------------------
# Transcript injection on the read path
# ---------------------------------------------------------------------------


def synthesized_transcript_row(task: BackgroundTask) -> dict[str, Any]:
    """One finished task as a Hermes-shaped transcript row.

    Follows the measured row contract (B-34): `role` is the one guaranteed
    key and the app treats everything else as optional, so extra keys are
    safe. `role: "assistant"` + `text` is the shape the Phase 1 renderer
    already displays (the result is already markdown). The marker keys let a
    future build render it distinctly and let anyone reading a raw
    transcript see it was the gateway, not Hermes, that put it there.
    """
    return {
        "role": "assistant",
        "text": task.result_text,
        SYNTHESIZED_FROM_FIELD: SYNTHESIZED_FROM_VALUE,
        "task_id": task.task_id,
        "finished_at": iso_z(task.finished_at),
    }


def append_finished_background_results(
    app_state: Any, stored_session_id: str, messages: Any
) -> int:
    """Append this session's finished background results to a transcript, in place.

    Returns the number of rows appended so the caller can keep its `count`
    consistent with `len(messages)`. Appended at the end, oldest result
    first: a background turn has no position in Hermes's row order (it never
    had a row), and end-of-transcript in completion order is the only
    placement that cannot mis-interleave.

    Best-effort and never fatal, exactly like `_annotate_filing_status`: if
    the DB is missing, unmigrated or unreadable, the transcript is served
    without the injected rows rather than not at all. Non-list `messages`
    (forwarded verbatim from a Hermes this code has never seen) are left
    untouched.
    """
    if not isinstance(messages, list):
        return 0
    factory = getattr(app_state, "db_sessions", None)
    if factory is None:
        return 0
    try:
        with factory() as db:
            rows = ledger_ops.finished_results_for_session(db, stored_session_id)
    except Exception:
        logger.warning(
            "could not read the background-task ledger; serving the transcript "
            "without injected background results",
            exc_info=True,
        )
        return 0
    for task in rows:
        messages.append(synthesized_transcript_row(task))
    return len(rows)
