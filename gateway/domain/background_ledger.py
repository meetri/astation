"""`BackgroundLedger`: the background-task completion pipeline (P2-1, fixes B-42)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import sessionmaker

from domain import background_tasks as ledger_ops

from domain.event_stream import BACKGROUND_COMPLETED_EVENT_TYPE  # noqa: F401
from domain.hermes_runtime import _resume_for_live_id
from domain.models import BackgroundTask
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

SYNTHESIZED_MESSAGE_EVENT_TYPE = "message.interim"

SYNTHESIZED_FROM_FIELD = "_synthesized_from"
SYNTHESIZED_FROM_VALUE = "background.complete"


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


class BackgroundLedger:
    """Wires the ledger into the event pump and the reconnect lifecycle."""

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
        self._schedule = schedule if schedule is not None else asyncio.create_task
        self._rescue_task: Any = None


    def orphan_on_startup(self) -> int:
        """`running -> orphaned` for rows left over from a previous process."""
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


    def handle_generation_change(self, generation: int | None, previous: int | None) -> None:
        """Orphan running rows, then schedule the re-resume rescue (P2-0b)."""
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
        """Re-resume every session holding an unfinished task; return sessions tried."""
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


    def handle_completed(self, envelope: dict[str, Any]) -> dict[str, Any] | None:
        """Apply one forwarded `background.completed` frame to the ledger."""
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
            payload["_stored_session_id"] = stored_id

        if not stored_id or result_text is None:
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
        """The injected transcript frame -- shaped as an ordinary canonical envelope."""
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


def synthesized_transcript_row(task: BackgroundTask) -> dict[str, Any]:
    """One finished task as a Hermes-shaped transcript row."""
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
    """Append this session's finished background results to a transcript, in place."""
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
