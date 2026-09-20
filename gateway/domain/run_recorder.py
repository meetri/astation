"""`RunRecorder`: run attribution + selective event persistence (P2-2)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from domain import runs as runs_ops
from domain.event_stream import STORED_SESSION_ID_FIELD
from domain.models import HERMES_RUNTIME, Run, RunEvent, Session, utcnow
from events import is_persisted_event_type, persist_run_event
from events.canonical import CanonicalEvent

logger = logging.getLogger(__name__)

_STORED_SESSION_ID_FIELD = STORED_SESSION_ID_FIELD

RUN_CLOSING_TYPE = "message.completed"

COMPLETION_STATUS_ERROR = "error"
COMPLETION_STATUS_INTERRUPTED = "interrupted"

HERMES_INTERRUPTED_EXPLANATION = (
    "Hermes reported this turn interrupted before it finished (stopped by a "
    "user or by the agent); the transcript holds whatever it had produced"
)
FAILED_WITHOUT_REASON_EXPLANATION = "Hermes reported this turn failed but did not say why"


def completion_close(payload: Any) -> tuple[str, str | None]:
    """(run status, note) for a `message.completed` payload."""
    if not isinstance(payload, dict):
        return runs_ops.STATUS_COMPLETED, None
    status = payload.get("status")
    if status == COMPLETION_STATUS_ERROR:
        error = payload.get("error")
        text = error.strip() if isinstance(error, str) else ""
        return runs_ops.STATUS_FAILED, text or FAILED_WITHOUT_REASON_EXPLANATION
    if status == COMPLETION_STATUS_INTERRUPTED:
        return runs_ops.STATUS_INTERRUPTED, HERMES_INTERRUPTED_EXPLANATION
    return runs_ops.STATUS_COMPLETED, None


# A compress emits these with no message.completed, so a run opened by one never closes.
RUN_ATTACH_ONLY_TYPES: frozenset[str] = frozenset({"status.update"})

# message.completed opens one too, so a turn this process joined late still gets a run.
RUN_OPENING_TYPES: frozenset[str] = frozenset(
    {
        "message.started",
        "message.delta",
        "message.interim",
        "message.completed",
        "reasoning.delta",
        "thinking.status",
        "status.update",
        "tool.generating",
        "tool.started",
        "tool.progress",
        "tool.completed",
        "approval.requested",
        "clarify.requested",
        "sudo.requested",
        "secret.requested",
    }
)

NON_RUN_TYPES: frozenset[str] = frozenset({"session.updated", "background.completed"})


DEFAULT_RUN_PROFILE = "default"

SYNTHETIC_RESOLVED_TYPES: frozenset[str] = frozenset(
    {"approval.resolved", "clarify.resolved", "sudo.resolved", "secret.resolved"}
)

_REQUESTED_SUFFIX = ".requested"
_RESOLVED_SUFFIX = ".resolved"

_REQUEST_LOOKUP_TYPES: frozenset[str] = frozenset(
    {kind.removesuffix(_RESOLVED_SUFFIX) + _REQUESTED_SUFFIX for kind in SYNTHETIC_RESOLVED_TYPES}
    | SYNTHETIC_RESOLVED_TYPES
)

# Bounded because the request_id match runs in Python over JSON payloads, not in SQL.
REQUEST_LOOKUP_LIMIT = 200


@dataclass
class _OpenRun:
    """In-memory record of one open run: attribution + the seq counter."""

    run_id: str
    runtime_session_id: str
    workspace_session_id: str | None
    project_id: str | None
    last_seq: int = 0
    profile: str = DEFAULT_RUN_PROFILE
    pending_requests: dict[str, str] = field(default_factory=dict)


class RunRecorder:
    """Attributes every forwarded frame and persists the run-relevant ones."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        on_run_opened: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._open: dict[str, _OpenRun] = {}
        self._on_run_opened = on_run_opened
        self._db_failure_logged: set[str] = set()


    def interrupt_open_runs_on_startup(self) -> int:
        """`running -> interrupted` for rows left over from a previous process."""
        return self._sweep("startup")

    def handle_generation_change(self, generation: int | None, previous: int | None) -> None:
        """The Hermes connection was replaced: every open run's stream is dead."""
        self._open.clear()
        swept = self._sweep(f"generation change {previous} -> {generation}")
        if swept:
            logger.info(
                "interrupted %d open run(s) on Hermes generation change %s -> %s",
                swept,
                previous,
                generation,
            )

    def _sweep(self, reason: str) -> int:
        try:
            with self._session_factory() as db:
                rows = runs_ops.interrupt_all_running(db)
                db.commit()
        except Exception:
            logger.warning(
                "could not sweep running runs (%s); continuing (is the database migrated?)",
                reason,
                exc_info=True,
            )
            return 0
        if rows:
            logger.info(
                "marked %d running run(s) interrupted (%s): %s",
                len(rows),
                reason,
                [row.id for row in rows],
            )
        return len(rows)


    def handle_event(
        self,
        canonical: CanonicalEvent,
        envelope: dict[str, Any],
        profile: str = DEFAULT_RUN_PROFILE,
    ) -> None:
        """Attribute one forwarded frame; persist it if run-relevant."""
        try:
            self._handle(canonical, envelope, profile)
        except Exception:  # pragma: no cover - defensive; same contract as B-26
            logger.exception(
                "run recorder failed on a %r frame; the frame is still forwarded",
                canonical.type,
            )

    def _handle(self, canonical: CanonicalEvent, envelope: dict[str, Any], profile: str) -> None:
        event_type = canonical.type
        payload = envelope.get("payload")
        stored_raw = payload.get(_STORED_SESSION_ID_FIELD) if isinstance(payload, dict) else None
        stored_id = stored_raw if isinstance(stored_raw, str) and stored_raw else None

        if stored_id is None or event_type in NON_RUN_TYPES:
            return

        record = self._open.get(stored_id)
        opening = (
            record is None
            and event_type in RUN_OPENING_TYPES
            and event_type not in RUN_ATTACH_ONLY_TYPES
        )
        if record is None and not opening:
            return

        persists = is_persisted_event_type(event_type)
        closes = event_type == RUN_CLOSING_TYPE

        if record is not None and not persists and not closes:
            self._stamp(envelope, record)
            return

        preexisting = record is not None
        seq = record.last_seq + 1 if (record is not None and persists) else 1
        try:
            with self._session_factory() as db:
                if record is None:
                    record = self._open_run_in(db, stored_id, canonical.timestamp, profile)
                if persists:
                    persist_run_event(
                        db,
                        run_id=record.run_id,
                        seq=seq,
                        event_type=event_type,
                        payload=payload if isinstance(payload, dict) else None,
                        timestamp=canonical.timestamp,
                    )
                if closes:
                    run = db.get(Run, record.run_id)
                    if run is not None:
                        status, note = completion_close(payload)
                        if note is None:
                            runs_ops.close_run(run, now=canonical.timestamp)
                        else:
                            runs_ops.close_run_abnormal(
                                run, status=status, note=note, now=canonical.timestamp
                            )
                db.commit()
        except Exception:
            if stored_id not in self._db_failure_logged:
                self._db_failure_logged.add(stored_id)
                logger.exception(
                    "could not persist run state for session %s; the stream "
                    "continues, this run's durable timeline has a gap "
                    "(logged once per session until a write succeeds)",
                    stored_id,
                )
            # A run that failed to open was rolled back; its id would name no row.
            if preexisting and record is not None:
                self._stamp(envelope, record)
            return

        self._db_failure_logged.discard(stored_id)
        if persists:
            record.last_seq = seq
            self._remember_request(record, event_type, payload)
        self._stamp(envelope, record)
        if closes:
            self._open.pop(stored_id, None)
        else:
            self._open[stored_id] = record
        if not preexisting and self._on_run_opened is not None:
            try:
                self._on_run_opened(profile, stored_id, record.run_id)
            except Exception:  # pragma: no cover - defensive; same contract as the DB guard
                logger.exception(
                    "on_run_opened hook failed for run %s; the run and the frame stand",
                    record.run_id,
                )

    def _open_run_in(
        self, db: OrmSession, stored_id: str, now: datetime, profile: str = DEFAULT_RUN_PROFILE
    ) -> _OpenRun:
        """Create the Run row + open-run record; the filing lookup happens here."""
        filing = db.execute(
            select(Session.id, Session.project_id).where(
                Session.runtime == HERMES_RUNTIME,
                Session.runtime_session_id == stored_id,
            )
        ).first()
        workspace_session_id = filing[0] if filing is not None else None
        project_id = filing[1] if filing is not None else None
        run = runs_ops.open_run(
            db,
            runtime_session_id=stored_id,
            workspace_session_id=workspace_session_id,
            project_id=project_id,
            profile=profile,
            now=now,
        )
        # Flush first: run.id is a python-side default and is unset until the flush.
        db.flush()
        return _OpenRun(
            run_id=run.id,
            runtime_session_id=stored_id,
            workspace_session_id=workspace_session_id,
            project_id=project_id,
            profile=profile,
        )

    @staticmethod
    def _remember_request(record: _OpenRun, event_type: str, payload: Any) -> None:
        """Note a `*.requested` frame's `request_id` on its open run."""
        if not event_type.endswith(_REQUESTED_SUFFIX) or not isinstance(payload, dict):
            return
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and request_id:
            record.pending_requests[request_id] = event_type

    @staticmethod
    def _stamp(envelope: dict[str, Any], record: _OpenRun) -> None:
        envelope["project_id"] = record.project_id
        envelope["session_id"] = record.workspace_session_id
        envelope["run_id"] = record.run_id


    def stored_id_for_request(
        self, request_id: str, profile: str = DEFAULT_RUN_PROFILE
    ) -> str | None:
        """The stored session id whose run saw `<kind>.requested` with this id."""
        for stored_id, record in self._open.items():
            if record.profile == profile and request_id in record.pending_requests:
                return stored_id
        try:
            with self._session_factory() as db:
                found = self._find_requested_row(db, request_id, profile)
        except Exception:
            logger.warning(
                "could not look up request %s in the run ledger; treating it as unknown",
                request_id,
                exc_info=True,
            )
            return None
        if found is None:
            return None
        run, _event = found
        return run.runtime_session_id

    @staticmethod
    def _find_requested_row(
        db: OrmSession, request_id: str, profile: str, stored_id: str | None = None
    ) -> tuple[Run, RunEvent] | None:
        """The newest persisted `*.requested` row carrying `request_id`."""
        stmt = (
            select(Run, RunEvent)
            .join(Run, Run.id == RunEvent.run_id)
            .where(RunEvent.event_type.in_(_REQUEST_LOOKUP_TYPES), Run.profile == profile)
        )
        if stored_id is not None:
            stmt = stmt.where(Run.runtime_session_id == stored_id)
        stmt = stmt.order_by(RunEvent.timestamp.desc(), RunEvent.seq.desc()).limit(
            REQUEST_LOOKUP_LIMIT
        )
        for run, event in db.execute(stmt).all():
            payload = runs_ops.payload_dict(event)
            if payload.get("request_id") != request_id:
                continue
            # Already answered: a second resolved row would say it was answered twice.
            if event.event_type.endswith(_RESOLVED_SUFFIX):
                return None
            return run, event
        return None

    def record_synthetic(
        self,
        stored_id: str,
        event_type: str,
        payload: dict[str, Any],
        profile: str = DEFAULT_RUN_PROFILE,
    ) -> bool:
        """Append one gateway-authored row to the run that asked."""
        try:
            return self._record_synthetic(stored_id, event_type, payload, profile)
        except Exception:  # pragma: no cover - defensive; same contract as handle_event
            logger.exception(
                "could not record synthetic %r on session %s; the ledger has a gap",
                event_type,
                stored_id,
            )
            return False

    def _record_synthetic(
        self, stored_id: str, event_type: str, payload: dict[str, Any], profile: str
    ) -> bool:
        if event_type not in SYNTHETIC_RESOLVED_TYPES:
            logger.warning(
                "refusing to record synthetic %r: not a gateway-authored run event type",
                event_type,
            )
            return False
        record = self._open.get(stored_id)
        if record is None or record.profile != profile:
            return self._record_synthetic_durable(stored_id, event_type, payload, profile)
        row_payload = dict(payload)
        row_payload[_STORED_SESSION_ID_FIELD] = stored_id
        seq = record.last_seq + 1
        try:
            with self._session_factory() as db:
                persist_run_event(
                    db,
                    run_id=record.run_id,
                    seq=seq,
                    event_type=event_type,
                    payload=row_payload,
                    timestamp=utcnow(),
                )
                db.commit()
        except Exception:
            logger.exception(
                "could not persist synthetic %r for run %s; the run's timeline has a gap",
                event_type,
                record.run_id,
            )
            return False
        record.last_seq = seq
        request_id = payload.get("request_id")
        if isinstance(request_id, str):
            record.pending_requests.pop(request_id, None)
        return True

    def _record_synthetic_durable(
        self, stored_id: str, event_type: str, payload: dict[str, Any], profile: str
    ) -> bool:
        """The B-193 fallback: find the run through the ledger, append after its last row."""
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return False
        try:
            with self._session_factory() as db:
                found = self._find_requested_row(db, request_id, profile, stored_id)
                if found is None:
                    return False
                run, requested = found
                requested_stored = runs_ops.payload_dict(requested).get(_STORED_SESSION_ID_FIELD)
                row_payload = dict(payload)
                row_payload[_STORED_SESSION_ID_FIELD] = (
                    requested_stored
                    if isinstance(requested_stored, str) and requested_stored
                    else run.runtime_session_id or stored_id
                )
                seq = runs_ops.last_seq_by_run(db, [run.id]).get(run.id, 0) + 1
                persist_run_event(
                    db,
                    run_id=run.id,
                    seq=seq,
                    event_type=event_type,
                    payload=row_payload,
                    timestamp=utcnow(),
                )
                db.commit()
        except Exception:
            logger.exception(
                "could not persist synthetic %r for request %s via the ledger; "
                "the run's timeline has a gap",
                event_type,
                request_id,
            )
            return False
        return True


    def has_open_run(self, stored_id: str) -> bool:
        """Whether a turn this gateway can see is open on `stored_id` right now."""
        return stored_id in self._open


    @property
    def open_run_count(self) -> int:
        return len(self._open)
