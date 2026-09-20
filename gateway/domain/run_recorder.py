"""`RunRecorder`: run attribution + selective event persistence (P2-2).

Moved out of `api/runs.py` (CLEANUP_PLAN step 3.5); the run read routes stay
there and re-export these names. This is the phase's biggest gateway build,
scoped honestly: before P2-2, **nothing persisted events** --
`events/persistence.py` had zero live call sites and the broadcaster stamped
Phase-0 placeholder attribution ids on every envelope.

1. **`RunRecorder`** -- the event-side orchestration, wired into
   `EventBroadcaster` by `api.main.lifespan` as its `on_canonical_event`
   hook. For every forwarded frame it (a) attributes the canonical envelope
   for real, and (b) persists the run-relevant ones. Its per-frame decisions:

   * **Attribution (P2-2a).** The frame's `_stored_session_id` (stamped by
     `_stamp_session_identity`, B-29) names the Hermes session; the filing
     lookup (sessions table, done once per run open, cached on the open-run
     record) yields the workspace `session_id`/`project_id` -- or None for
     an unfiled session (P2-2d). One `Run` row per turn, **flat**: the wire
     produces no child runs (PV "Phase 2 probe"), so the §9 tree is
     explicitly cut and `parent_run_id` is never written. A frame that
     belongs to no run (unattributable, or a non-run type like
     `session.updated`) carries explicit nulls -- the honest answer, and
     the app already degrades a null envelope field ("").

   * **Turn boundaries.** There is no `turn.started` on the wire. A run
     opens on the first turn-activity frame for a stored session with no
     open run (usually `reasoning.delta` -- the first ~30s of a real turn
     are reasoning only, B-14) and closes on `message.completed`, the
     turn's single end-of-turn signal (B-38: one per turn, carrying
     `final_response`).

   * **Persistence (P2-2b) -- the sync-SQLite-in-async-pump decision.** The
     DB layer is deliberately synchronous (`domain/db.py`); the pump is
     async. That is fine *because of the selective policy*: after
     `PERSISTED_RUN_EVENT_TYPES` filters out the token-scale streams
     (`message.delta`/`reasoning.delta`/`thinking.status` -- see
     `events/persistence.py::NOT_PERSISTED_EVENT_TYPES` for every reason),
     what remains is ~a dozen coarse rows per multi-tool turn, and a
     single-row SQLite insert against a local file is microseconds --
     measured well under the per-frame budget the pump already spends on
     `json.dumps` size-checking. **Batching boundary: one forwarded
     frame = one session = one commit** (run-open + event row + run-close
     share the transaction when they coincide). Rejected alternatives:
     per-turn batching (a crash loses the whole turn's timeline, and an
     `approval.requested` row must be durable when the P2-3 notify relay
     wants it, not at end-of-turn) and a background flusher task (a second
     silently-dying pump is the exact B-26 failure class). **A DB error is
     logged and never fatal to the stream**: every DB touch is wrapped, a
     failed write costs that row only, and fan-out proceeds regardless --
     same contract as the P2-1 ledger hook.

   * **Per-run cursor (P2-2c).** `run_events.seq` starts at 1 per run and
     is allocated at persist time from the open-run record's in-memory
     counter. It survives restarts *structurally*, not by recovery: a
     restart (or Hermes reconnect) interrupts every open run and later
     activity opens a NEW run with a fresh counter -- so no counter ever
     spans a restart, written rows keep their numbers forever, and
     `after_seq` is durable. The broadcaster's global `seq` (which resets
     with the process) is never used for this.

   * **Synthesized rows.** An answered prompt produces no
     `*.resolved` frame on the wire -- Hermes emits one only for a sudo/
     secret EXPIRY -- so the ledger could see a turn block on
     `approval.requested` and never see it unblock, and the blocked span
     was unmeasurable (RUN_EVENT_UX_AUDIT §2.5/§4). `record_synthetic`
     lets the gateway's own respond routes (`api/prompts.py`) append a
     gateway-authored `<kind>.resolved` row to the open run: attach-only
     (never opens a run), `seq` allocated exactly like a persisted frame,
     one commit, never raises, and NOT fanned out to `/ws/events` -- it is
     a ledger fact, not a live frame. Three of the four respond routes are
     keyed on `request_id` alone (no session on the route), so the open-run
     record also remembers every `request_id` its `*.requested` frames
     carried (`stored_id_for_request`). That mapping is the fast path; when
     it misses (a gateway restart between the prompt and the answer, or a
     `*.requested` frame this process never saw) the persisted ledger is
     the fallback: the newest `*.requested` row carrying that
     `request_id` names the run, and the resolved row is appended to THAT
     run even when it is closed, `seq` continuing from the run's last
     persisted row. A request whose newest ledger row is already a
     `*.resolved` is treated as answered and not recorded twice.
"""

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

#: B-29 identity key -- the broadcaster's own constant (this used to be a
#: by-literal copy to dodge an import cycle with `api.main`; a test still pins
#: the two equal).
_STORED_SESSION_ID_FIELD = STORED_SESSION_ID_FIELD

#: The turn's one end-of-turn signal (B-38: exactly one per turn).
RUN_CLOSING_TYPE = "message.completed"

#: What the closing frame's own `status` says about how the turn ended
#:. Measured 2026-09-17 on the ledger, 198 closes: `complete` 149,
#: `interrupted` 27, `error` 22 -- and every one had been closed as
#: `completed` because nothing read the field. `error` rows also carry
#: `error` (the text), `error_surface` and `recoverable`.
COMPLETION_STATUS_ERROR = "error"
COMPLETION_STATUS_INTERRUPTED = "interrupted"

#: Honest copy, stored on the row by `runs_ops.close_run_abnormal` and served
#: verbatim as `status_explanation` (same convention as `api/runs.py`'s
#: `INTERRUPTED_EXPLANATION`: the client never invents its own wording).
HERMES_INTERRUPTED_EXPLANATION = (
    "Hermes reported this turn interrupted before it finished (stopped by a "
    "user or by the agent); the transcript holds whatever it had produced"
)
FAILED_WITHOUT_REASON_EXPLANATION = "Hermes reported this turn failed but did not say why"


def completion_close(payload: Any) -> tuple[str, str | None]:
    """(run status, note) for a `message.completed` payload.

    `error` -> `failed` with the frame's `error` text; `interrupted` ->
    `interrupted` with the Hermes-side note; anything else (`complete`, a
    missing or unknown status) -> `completed` with no note -- the status
    field is a bonus, and a frame without one still closes the turn as it
    always has.
    """
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


#: **`status.update` attaches to an open run and never opens one.**
#: Measured 2026-09-04 (PV "Compress + the LCM context engine"):
#: `session.compress` emits `status.update` kinds `compressing` ->
#: `compacting` -> `compacted` -> `status` ("ready") and **no
#: `message.completed`**, so a run opened by any of them could never close --
#: the spike's run sat `running` forever after one compress and every later
#: compress was refused as "mid-turn" by this recorder's own guard. Every
#: status notice ever measured inside a turn (B-14: "Hindsight -- recalled 32
#: memories") arrived after `message.started` had already opened the run, so
#: nothing is lost by refusing to open on one: a notice with a turn attaches
#: and persists as before; a notice without one (a compress, an idle "ready")
#: is forwarded live and recorded nowhere, which is the truth.
RUN_ATTACH_ONLY_TYPES: frozenset[str] = frozenset({"status.update"})

#: Canonical types that prove a turn is producing frames and may OPEN a run.
#: Everything here is turn activity observed live (B-14's measured turn).
#: `message.completed` is included so a turn whose start this process never
#: saw (restart mid-turn) still gets a (single-event, immediately-closed)
#: run rather than vanishing.
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

#: Types that are never run activity. `session.updated` is session lifecycle
#: (session.info fires on every resume, title changes on renames);
#: `background.completed` belongs to the background-task LEDGER (P2-1) --
#: the wire is silent while a background task runs, so there is no turn to
#: attach it to. Neither may open a run nor be attributed to one.
NON_RUN_TYPES: frozenset[str] = frozenset({"session.updated", "background.completed"})

#: Types that join an already-open run but never open one: they can arrive
#: outside any turn (a sudo/secret expiry long after the prompt; a usage
#: report on resume) and a run containing only one of these would be noise.
#: Derived membership: anything not opening, not non-run.


#: The connection every run was recorded on before `Run.profile` existed, and
#: the one `RunRecorder`'s `EventBroadcaster` hook still carries. Kept local
#: rather than imported from `api.chat` so `api/runs.py` does not grow a
#: dependency on the chat surface for one string.
DEFAULT_RUN_PROFILE = "default"

#: The gateway-authored rows `record_synthetic` may write: one
#: `*.resolved` per human-in-the-loop prompt kind. Anything else is refused
#: (returns False) so the ledger's persisted-type policy stays the single
#: decision surface -- every entry here is in `PERSISTED_RUN_EVENT_TYPES`.
SYNTHETIC_RESOLVED_TYPES: frozenset[str] = frozenset(
    {"approval.resolved", "clarify.resolved", "sudo.resolved", "secret.resolved"}
)

#: The request-id bookkeeping on the open run keys off `<kind>.requested`.
_REQUESTED_SUFFIX = ".requested"
_RESOLVED_SUFFIX = ".resolved"

#: The persisted prompt rows the B-193 fallback scans: the four `*.requested`
#: kinds and their `*.resolved` counterparts (so an already-answered request
#: is recognized as such).
_REQUEST_LOOKUP_TYPES: frozenset[str] = frozenset(
    {kind.removesuffix(_RESOLVED_SUFFIX) + _REQUESTED_SUFFIX for kind in SYNTHETIC_RESOLVED_TYPES}
    | SYNTHETIC_RESOLVED_TYPES
)

#: How many of the newest prompt rows the fallback reads before giving up
#:. Prompts are a few per turn at most, so 200 spans many turns; the
#: payload is JSON and the match is Python-side, which is why it is bounded.
REQUEST_LOOKUP_LIMIT = 200


# ---------------------------------------------------------------------------
# Event-side orchestration
# ---------------------------------------------------------------------------


@dataclass
class _OpenRun:
    """In-memory record of one open run: attribution + the seq counter.

    Attribution is resolved once, at run open, and reused for every frame of
    the turn -- so the token-scale streams (`reasoning.delta` at ~945/turn)
    cost zero DB work per frame. A session filed *mid-turn* keeps its
    at-open attribution for that run; the next run picks the change up.
    """

    run_id: str
    runtime_session_id: str
    workspace_session_id: str | None
    project_id: str | None
    last_seq: int = 0
    #: Which Hermes connection the run was observed on; the
    #: request-id lookup below refuses a match from another profile, since
    #: `request_id` is per-Hermes-process, not global (`api/prompts.py`).
    profile: str = DEFAULT_RUN_PROFILE
    #: `request_id -> canonical requested type` for every `*.requested`
    #: frame persisted on this run. Lets the session-less respond
    #: routes find the run their answer belongs to. Cleared with the run.
    pending_requests: dict[str, str] = field(default_factory=dict)


class RunRecorder:
    """Attributes every forwarded frame and persists the run-relevant ones.

    See the module docstring for the P2-2 decisions this implements. Wired
    by `api.main.lifespan`:

    * `handle_event` -- the broadcaster's `on_canonical_event` hook, called
      synchronously in the pump for every forwarded frame, after
      `_stamp_session_identity` and before the size check (so real
      attribution is part of what gets measured, and survives B-02
      degradation via `_kept_session_identity`... the envelope keys are not
      payload keys, so they survive regardless). NEVER raises past the
      guard; a DB failure costs durability for one row, never the stream.
    * `interrupt_open_runs_on_startup` -- lifespan sweep: a run left
      `running` by a previous process is `interrupted` (frames emitted while
      the gateway was down are unrecoverable -- nothing buffers or replays
      them, P2-0b).
    * `handle_generation_change` -- the same sweep on a Hermes reconnect,
      composed with the background ledger's hook by lifespan.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        on_run_opened: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._open: dict[str, _OpenRun] = {}
        # `(profile, stored_id, run_id)` after a run row is committed.
        # `ChatStore.attach_turn` hangs off this so the user row the submit
        # route wrote BEFORE the turn opened gets the run's id -- the one
        # row of a turn the capture hook can never stamp itself. Never
        # allowed to fail the frame.
        self._on_run_opened = on_run_opened
        # DB failures logged once per stored session until a success, so a
        # permanently-broken DB does not log 945 tracebacks per turn
        # (reasoning.delta rate) -- the same tune-out risk as B-14's
        # per-frame warnings.
        self._db_failure_logged: set[str] = set()

    # -- lifecycle sweeps -------------------------------------------------

    def interrupt_open_runs_on_startup(self) -> int:
        """`running -> interrupted` for rows left over from a previous process."""
        return self._sweep("startup")

    def handle_generation_change(self, generation: int | None, previous: int | None) -> None:
        """The Hermes connection was replaced: every open run's stream is dead.

        In-memory open runs are forgotten (a post-reconnect frame belongs to
        a NEW turn from this gateway's point of view -- the old turn's
        remaining frames were never delivered and never will be), and the DB
        sweep records the honest `interrupted` state.
        """
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

    # -- the per-frame hook -----------------------------------------------

    def handle_event(
        self,
        canonical: CanonicalEvent,
        envelope: dict[str, Any],
        profile: str = DEFAULT_RUN_PROFILE,
    ) -> None:
        """Attribute one forwarded frame; persist it if run-relevant.

        Mutates `envelope`'s `project_id`/`session_id`/`run_id` in place
        (they arrive as None from the normalizer -- P2-2a). Never raises.

        `profile` is which Hermes connection the frame came off, and
        is recorded on any run this frame OPENS. It defaults so the
        `EventBroadcaster` hook -- which only ever carries the default
        connection's stream -- keeps calling this with two arguments;
        `ProfileConnectionManager` passes its own connection's name.
        """
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
            # Explicit nulls are already on the envelope (the normalizer's
            # context). Unattributable frames stay that way honestly.
            return

        record = self._open.get(stored_id)
        opening = (
            record is None
            and event_type in RUN_OPENING_TYPES
            and event_type not in RUN_ATTACH_ONLY_TYPES
        )
        if record is None and not opening:
            # e.g. a stray session.usage / sudo.resolved with no turn open.
            return

        persists = is_persisted_event_type(event_type)
        closes = event_type == RUN_CLOSING_TYPE

        if record is not None and not persists and not closes:
            # The hot path: token-scale streams attach to the open run with
            # zero DB work.
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
            # A run we failed to OPEN was rolled back -- its id names no row,
            # so the frame stays unattributed (nulls) rather than pointing at
            # a phantom run. One that already existed still attributes from
            # memory; only the durable row for this frame is lost.
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
        """Create the Run row + open-run record; the filing lookup happens here.

        One query against the sessions table per run open (not per frame):
        a filed session yields its workspace `sess_...` id and project, an
        unfiled one yields None/None (P2-2d).
        """
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
        db.flush()  # assigns run.id (python-side default) before we record it
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

    # -- gateway-authored rows --------------------------------------

    def stored_id_for_request(
        self, request_id: str, profile: str = DEFAULT_RUN_PROFILE
    ) -> str | None:
        """The stored session id whose run saw `<kind>.requested` with this id.

        For the three respond routes that carry no session (`clarify`,
        `sudo`, `secret` -- Hermes keys them on `request_id` alone). Scoped
        to `profile`, because a request id is only unique per Hermes
        process. The open runs this process saw are checked first (no DB
        work); on a miss the persisted ledger is searched, so the
        answer survives a gateway restart and a closed turn. None when no
        run anywhere carries the id, or its newest ledger row already
        resolves it. Never raises.
        """
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
        """The newest persisted `*.requested` row carrying `request_id`.

        Reads the newest `REQUEST_LOOKUP_LIMIT` prompt rows (`*.requested`
        and `*.resolved`) for `profile` -- and for `stored_id`'s runs only,
        when given -- newest first, and returns the first whose payload
        `request_id` matches, with its run. If that newest matching row is
        a `*.resolved`, the request is already answered and None comes
        back: a second resolved row would say the prompt was answered
        twice.
        """
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
        """Append one gateway-authored row to the run that asked.

        Attach-only: it never opens a run. The open run on `stored_id` is
        the fast path (`seq` from its in-memory counter). With no open run
        -- a restart between the prompt and the answer, or a turn that has
        already closed -- the persisted `*.requested` row carrying
        `payload["request_id"]` on one of `stored_id`'s runs names the run
, and the row lands on it, closed or not, with `seq` after
        the run's last persisted row. Nothing to hang it on means nothing
        is recorded and False comes back. `event_type` must be one of
        `SYNTHETIC_RESOLVED_TYPES`. The stored row's payload is `payload`
        plus the B-29 `_stored_session_id` key every persisted frame carries
        (from the requested row on the fallback path), so a reader cannot
        tell a synthesized row from a wire one by shape -- only by
        `by == "app"`.

        Same contract as `handle_event`: one commit, a DB failure logged and
        swallowed (False). Never raises. The caller must never put a
        credential in `payload` (ARCHITECTURE §14): sudo/secret resolutions
        record the fact, not the value.
        """
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

    # -- guards for other components ----------------------------------------

    def has_open_run(self, stored_id: str) -> bool:
        """Whether a turn this gateway can see is open on `stored_id` right now.

        `stored_id` is the Hermes STORED id (the same key `_open` uses; a live
        handle here would never match). The snapshot sweep's open-run guard
        (`api/snapshot_sweep.py`, §6.5): a session mid-turn *through this
        gateway* is skipped for the pass. It only knows about turns whose
        frames flowed through this process -- a turn driven from the TUI or
        any other Hermes client is invisible here and is the `session.active_list` guard's job.
        """
        return stored_id in self._open

    # -- introspection (tests) --------------------------------------------

    @property
    def open_run_count(self) -> int:
        return len(self._open)
