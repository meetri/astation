"""Write run-relevant canonical events into the `run_events` table (P2-2).

Uses the real SQLAlchemy models from `domain/models.py` (`RunEvent`) -- see
that module's docstring for the schema's provenance (`ARCHITECTURE.md` §13).

## History: this module predates its first call site

The Phase 0 version of this file shipped a `persist_event()` with **zero live
call sites** -- nothing in the service ever invoked it, `run_events` stayed
empty, and the broadcaster stamped placeholder attribution ids. P2-2 is where
persistence becomes real: `api/runs.py::RunRecorder` (wired into the
broadcast path by `api.main.lifespan`) is the one caller, and this module is
deliberately reduced to what that caller actually needs -- the persistence
*policy* (which canonical types earn a durable row, with a written reason for
every type that does not) and the single row-writing function.

The Phase 0 version also grew `messages` rows from `message.delta` /
`message.completed`. That machinery was **cut, on purpose**, not wired up:

* **Hermes owns the transcript.** The gateway serves it verbatim from
  `session.resume` / `session.history` (the B-34 rule: no shape assumptions,
  element-for-element passthrough), and nothing reads the gateway's
  `messages` table. A second, gateway-grown copy of every transcript would
  be a divergence bug factory with no reader.
* **It could not have worked as written.** `Message.session_id` is a
  NOT-NULL FK to `sessions.id` (a workspace `sess_...` row), and a workspace
  Session row only exists for *filed* sessions -- most real traffic (spike
  sessions, everything unfiled) has no such row to point at.
* **It was token-scale writing in the event pump.** One measured turn is 328
  `message.delta` frames, each of which would have been a SELECT + UPDATE in
  the synchronous pump path for data nobody reads back.

## What gets persisted, and what deliberately does not

`RunEvent` is a flat, append-only per-run event log (`run_id`, `seq`,
`event_type`, `payload_json`, `timestamp` -- one row in, one row out, never
mutated). `seq` here is the **per-run, restart-surviving cursor** (P2-2c):
allocated by the caller at persist time, monotonic within one run, stored on
the row -- NOT the broadcaster's global stream counter, which resets on every
gateway restart and must never be used as a durable cursor.

`PERSISTED_RUN_EVENT_TYPES` is the allow-list, following `ARCHITECTURE.md`
§7.1's intent ("stores important state transitions"): the run's timeline of
significant, human-scale transitions -- tool lifecycle, human-in-the-loop
prompts and their resolutions, message/segment boundaries, run-relevant
status. These are dozens-per-turn at most, so there is no volume problem in
logging every one, and each is independently meaningful on a run timeline
(`tool.completed` in particular carries `result.diff` -- the colored-diff
data source, ROADMAP Phase 2).

`NOT_PERSISTED_EVENT_TYPES` records every canonical type that is forwarded
live but earns no durable row, with the reason. The distinction from "not in
either table" matters exactly as it does in `events/canonical.py`: an
unlisted type is a gap in our knowledge (`is_persisted_event_type` treats it
as not-persisted, and a new canonical type should be added to one table or
the other deliberately), while these are decisions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from domain.models import RunEvent

#: Canonical types that get one `run_events` row each, verbatim (P2-2).
#: Coarse-grained, human-in-the-loop-scale transitions only -- see module
#: docstring. Every entry here must be a type `RAW_TO_CANONICAL_TYPE` can
#: produce.
PERSISTED_RUN_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # Message/segment boundaries: "the model started / sealed a segment /
        # finished the turn" are exactly §7.1's significant transitions, and
        # `message.completed` carries the final text + reasoning.
        # `message.interim` carries each interim segment's full text,
        # so the persisted timeline keeps every segment, not just the last.
        "message.started",
        "message.interim",
        "message.completed",
        # Tool lifecycle -- the Run Inspector's core content, and
        # `tool.completed.result.diff` is the file-edit diff source.
        "tool.generating",
        "tool.started",
        "tool.progress",
        "tool.completed",
        # Human-in-the-loop prompts and their resolutions. The wire only ever
        # produces two of the resolutions (sudo.expire/secret.expire ->
        # *.resolved with payload.resolution == "expired"); an ANSWERED
        # prompt never comes back as an event -- only the gateway's own
        # *.respond route knows, and it synthesizes the matching *.resolved
        # row (payload.resolution == "answered", by == "app") through
        # `RunRecorder.record_synthetic`. `approval.resolved` and
        # `clarify.resolved` therefore have no raw producer in
        # `RAW_TO_CANONICAL_TYPE`; they are gateway-authored rows.
        "approval.requested",
        "approval.resolved",
        "clarify.requested",
        "clarify.resolved",
        "sudo.requested",
        "sudo.resolved",
        "secret.requested",
        "secret.resolved",
        # Run-relevant status: one human-readable notice per occurrence
        # ("Hindsight -- recalled 32 memories"), append semantics (§7.1).
        "status.update",
        # Token/context usage -- one frame per turn, cheap, and the only
        # durable record of what a turn cost.
        "session.usage",
    }
)

#: Canonical types that are forwarded live but deliberately NOT persisted,
#: with the reason recorded (the B-14 convention, applied to storage).
NOT_PERSISTED_EVENT_TYPES: dict[str, str] = {
    "message.delta": (
        "token-scale (328 frames in one measured turn) and fully redundant: "
        "the finished text lands on the persisted message.interim/"
        "message.completed rows, and the authoritative transcript is "
        "Hermes's own, served verbatim (B-34). Persisting every token would "
        "bury the run timeline and triple the pump's write volume for data "
        "with no reader."
    ),
    "reasoning.delta": (
        "the highest-volume stream on the wire (945 frames vs 328 "
        "message.delta in one measured turn), display-only; the "
        "authoritative reasoning text for a turn arrives once on "
        "message.completed's payload (B-14/B-35) and is persisted there."
    ),
    "thinking.status": (
        "a transient TUI status label with replace-not-append semantics and "
        "no durable meaning ('( ͡° ͜ʖ ͡°) brainstorming...', then '' to "
        "clear it)."
    ),
    "session.updated": (
        "session lifecycle, not run activity: session.info fires on every "
        "resume and session.title on renames, neither belongs to a turn, and "
        "the durable home for session metadata is the sessions table / "
        "Hermes itself -- not a run's event log."
    ),
    "background.completed": (
        "the background-task LEDGER owns this event (P2-1/B-42: "
        "`background_tasks.result_text` is the durable copy, written by "
        "BackgroundLedger.handle_completed before fan-out). A background "
        "task is not a turn -- the wire is silent while it runs, so there is "
        "no run to attach it to, and a second copy in run_events would just "
        "be a divergence risk."
    ),
}


def is_persisted_event_type(event_type: str) -> bool:
    """Whether this canonical type earns a `run_events` row.

    An unknown type (in neither table) is NOT persisted: the allow-list is
    the decision surface, and a new canonical type should be added to one of
    the two tables above deliberately rather than persisted by accident.
    """
    return event_type in PERSISTED_RUN_EVENT_TYPES


def persist_run_event(
    db: Session,
    *,
    run_id: str,
    seq: int,
    event_type: str,
    payload: dict[str, Any] | None,
    timestamp: datetime,
) -> RunEvent:
    """Write one `run_events` row.

    Does not call `db.commit()` / `db.flush()` -- the caller owns transaction
    boundaries (`api/runs.py::RunRecorder` batches all rows for one forwarded
    frame into one commit; see its docstring for the P2-2b decision).

    `seq` is the caller-allocated **per-run** cursor value (P2-2c), not the
    broadcaster's global counter. `payload` is stored as forwarded --
    including the gateway's `_stored_session_id` / `_live_session_id` /
    `_connection_generation` attribution keys (B-29), which are part of the
    honest record of what was sent.
    """
    run_event = RunEvent(
        run_id=run_id,
        seq=seq,
        event_type=event_type,
        payload_json=payload,
        timestamp=timestamp,
    )
    db.add(run_event)
    return run_event
