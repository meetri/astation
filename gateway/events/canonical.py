"""Canonical event envelope + the raw-Hermes -> canonical type mapping table.

Source of truth for the envelope shape: `docs/ARCHITECTURE.md` §7 (the
`event_id` / `project_id` / `session_id` / `run_id` / `seq` / `type` /
`timestamp` / `payload` JSON block) and §7.1 (the canonical event names
table).

Source of truth for the raw wire event names: `docs/PROTOCOL_VERIFIED.md`'s
event catalog (`message.delta`, `message.complete`, `tool.start/progress/
complete`, `approval.request`, `clarify.request`, `sudo.request`,
`secret.request`, `sudo.expire`, `secret.expire`). That document explicitly
flags the *payload* shapes of these events as unverified against the live
instance -- only the event *names* and the request/response resolution
pattern (four `*.request` events resolved by a matching `*.respond` RPC,
carrying a `request_id`) are confirmed. Treat every raw-payload field read
in this package as best-effort until P0-5's fixtures (captured from the real
catalog) pin the exact shape down.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# --- raw Hermes event name -> canonical workspace event type -------------
#
# `docs/ARCHITECTURE.md` §7.1 only spells out one pair for the whole
# human-in-the-loop family: `approval.requested` / `approval.resolved`.
# `docs/PROTOCOL_VERIFIED.md` then corrects §4 to say `clarify.request`,
# `sudo.request`, and `secret.request` are siblings of `approval.request` --
# same server->client-request, client-resolves-via-RPC pattern, just a
# different kind of prompt. There is no separate table entry for those
# siblings, so this module generalizes the one documented pair across the
# whole family (`clarify.requested/resolved`, `sudo.requested/resolved`,
# `secret.requested/resolved`) rather than inventing an unrelated name.
#
# `sudo.expire` / `secret.expire` are folded into the corresponding
# `*.resolved` type (with `payload["resolution"] = "expired"`, added in
# `normalize_event`) rather than given their own canonical type: from the
# workspace's point of view an expiry *is* how that pending prompt got
# resolved (PROTOCOL_VERIFIED.md: "the adapter should clear only the
# matching pending prompt... on an expiry"), and a client only ever needs to
# know "is this request still open, and if not, how did it end" -- it does
# not need a fourth event name to learn that.
# --- B-14: the types a real turn actually pushes --------------------------
#
# The table below was originally written from PROTOCOL_VERIFIED.md's event
# catalog, which lists the message/tool/human-in-the-loop families and then
# says "plus session lifecycle and error events". That elision hid most of
# the real traffic. One complete live turn (2026-08-29) pushed:
#
#     reasoning.delta   x945    message.delta      x328
#     sessions.changed   x12    thinking.delta      x11
#     tool.generating     x1    tool.start           x1    tool.complete  x1
#     session.usage       x1    reasoning.available  x1    message.complete x1
#     session.info        x1    session.title       x2    message.start  x1
#     gateway.ready       x1    session.reclaimed    x1
#
# ...plus `status.update`, which that measurement missed entirely and which
# the once-per-type logger below surfaced on the very first live turn after
# this table was widened. That is the whole argument for the logger: the
# unknown-unknown is the expensive case.
#
# Only four of those names were in this table, and none of the reasoning
# ones -- so a turn that reasons for thirty seconds before it starts writing
# forwarded ZERO frames and the phone looked hung.
#
# Every raw name observed above is now either in `RAW_TO_CANONICAL_TYPE` or
# in `DELIBERATELY_DROPPED_RAW_EVENTS` with a reason. Nothing is handled by
# omission; anything in neither is logged once per type by the broadcaster
# so the next unknown name is discoverable instead of invisible.
#
# Naming follows `ARCHITECTURE.md` §7.1: `<noun>.<past-tense>` for a
# transition (`message.started`, `session.updated`), `<noun>.<stream>` for a
# repeating sample (`message.delta`, `tool.progress`, `metric.sample`).
#
# `reasoning.delta` is the headline addition and is deliberately a type of
# its own, NOT folded into `message.delta`: reasoning is not the answer, and
# a client that appended it to the answer buffer would render the model's
# scratchpad as the reply.
#
# `thinking.delta` is NOT a second reasoning stream, despite the name. Live
# payloads: `{'text': '( ͡° ͜ʖ ͡°) brainstorming...'}` then `{'text': ''}`
# twice -- a transient TUI status label with REPLACE semantics (empty string
# = clear it), 11 frames per turn against 945 real reasoning tokens. Folding
# it into `reasoning.delta` would splice "brainstorming..." into the middle
# of the reasoning transcript, so it gets its own `thinking.status` type
# whose payload replaces rather than appends.
#
# `status.update` is a third, different thing again, and is the one type here
# that carries a `kind` discriminator of its own:
# `{'kind': 'lifecycle', 'text': 'Hindsight -- recalled 32 memories'}`. It is a
# human-readable notice about what the agent just did, one per occurrence,
# append semantics -- exactly the sort of thing a research UI should show in
# the activity rail, so it is forwarded verbatim under its own name.
#
# --- B-38: `message.interim` is the SEGMENT BOUNDARY, and it is not optional -
#
# A Hermes turn that uses tools does not produce one assistant message. It
# produces several -- commentary, tool call, commentary, tool call, answer --
# and each one is persisted as its own transcript row. Measured on one
# 3-step turn: four assistant rows (53, 54, 39 and 150 characters) with tool
# rows between them.
#
# `message.complete` carries only the LAST of those (`result["final_response"]`
# in Hermes's `tui_gateway/server.py`), while `message.delta` streamed all of
# them into one buffer. Hermes announces every boundary with `message.interim`
# and says why in its own source: "Surface interim assistant text (commentary
# emitted alongside tool calls...) so the desktop can seal it as its own
# segment instead of losing it when message.complete replaces the streaming
# buffer."
#
# This gateway dropped that frame -- it was in neither table, so
# `normalize_event` returned None and one warning per process was all that was
# ever said about it (`unmapped Hermes event type 'message.interim'`, present
# in the owner's own gateway log for the failing turn). Without the boundary a
# client pours every segment into one bubble and `message.complete` then
# replaces the whole accumulation with the final segment, deleting text the
# user already watched arrive. That is B-38.
#
# Forwarded verbatim under its own name, with `already_streamed` intact:
# `True` means the text also went out as `message.delta`s (the client already
# has it and should seal what it has), `False` means it did not (the client
# has never seen this text and must insert it).
#
# --- B-42: `background.complete` is a task's ONLY announcement, ever ---------
#
# A `prompt.background` task is fire-and-forget on the wire (PV "Phase 2
# probe", measured live 2026-08-30 across five tasks): the submit returns
# `{"task_id": "bg_..."}` immediately, then **nothing** -- no deltas, no
# tool frames, no status; `delegation.status` / `agents.list` /
# `process.list` are all empty mid-run and `session.resume` reports `idle`.
# Completion is exactly one event, `background.complete`
# `{task_id, text}`, stamped with the LIVE session handle -- and Hermes
# persists no trace of the task at all (no transcript row; a background-only
# session is never saved, `[4007]` on resume). So this frame is
# unrecoverable: dropping it means the result text is gone forever, and
# there is nothing to poll later.
#
# This name being in neither table below is B-42 -- every completed
# background task's one announcement was silently discarded. Mapped as a
# transition per §7.1 naming (`<noun>.<past-tense>`), like
# `message.complete` -> `message.completed`. The broadcaster additionally
# hands this one event to the background-task ledger (`api/background.py`),
# which is what makes the result durable; forwarding alone would only fix
# the case where a client happens to be subscribed at the exact instant it
# fires.
RAW_TO_CANONICAL_TYPE: dict[str, str] = {
    "background.complete": "background.completed",
    "message.start": "message.started",
    "message.delta": "message.delta",
    "message.interim": "message.interim",
    "message.complete": "message.completed",
    "reasoning.delta": "reasoning.delta",
    "thinking.delta": "thinking.status",
    "status.update": "status.update",
    "tool.generating": "tool.generating",
    "tool.start": "tool.started",
    "tool.progress": "tool.progress",
    "tool.complete": "tool.completed",
    "session.info": "session.updated",
    "session.title": "session.updated",
    "session.usage": "session.usage",
    "approval.request": "approval.requested",
    "clarify.request": "clarify.requested",
    "sudo.request": "sudo.requested",
    "secret.request": "secret.requested",
    "sudo.expire": "sudo.resolved",
    "secret.expire": "secret.resolved",
}

# --- deliberate drops (B-14) ---------------------------------------------
#
# A raw event name in here is dropped ON PURPOSE, with the reason recorded.
# The distinction from "not in either table" matters: an unlisted name is a
# gap in our knowledge and gets logged, while these are decisions.
DELIBERATELY_DROPPED_RAW_EVENTS: dict[str, str] = {
    "gateway.ready": (
        "transport handshake. The adapter itself waits for this before sending "
        "requests; by the time a client is subscribed the socket is already up, "
        "so forwarding it would tell the UI nothing it can act on."
    ),
    "sessions.changed": (
        "contentless poke, ~12 per turn. Every observed payload is "
        "{'session_id': ''} -- it names nothing, so a client can only respond by "
        "refetching the whole session list, 12 times a turn. The information a "
        "client actually wanted from it (a session's title changed) arrives with "
        "content as session.title -> session.updated. Revisit if a later Hermes "
        "build gives it a real payload."
    ),
    "platforms.changed": (
        "contentless poke, same family as sessions.changed and from the same "
        "Hermes machinery (`_CHANGE_WATCHES`, payload `lambda: {}`). It names "
        "nothing and this workspace has no platform surface to refresh, so a "
        "client could only respond by refetching something it cannot identify. "
        "Observed in the owner's gateway log 2026-08-29 (B-38)."
    ),
    "cron.changed": (
        "contentless poke from the same `_CHANGE_WATCHES` table (payload "
        "`lambda: {}`), fired when Hermes's cron store changes on disk. This "
        "workspace does not surface Hermes's schedules, and the frame carries "
        "nothing to act on. Observed in the owner's gateway log 2026-08-29 "
        "(B-38). Revisit when scheduled runs become a product surface here."
    ),
    "session.reclaimed": (
        "Hermes-internal orphan reaping of a LIVE handle "
        "({'session_id': '87162bde', 'reason': 'ws_orphan_reap'}). It concerns "
        "process-local handles this workspace never stores (PROTOCOL_VERIFIED.md, "
        "two id spaces) and is self-healed by LiveHandleCache's [4001] path (B-01), "
        "so a client has nothing to do with it."
    ),
    "reasoning.available": (
        "its payload contradicts its name and would corrupt the reasoning pane. "
        "Observed live, the same turn's frames were: "
        "reasoning.available {'text': \"Hello! I'm Hermes Agent, ready to help...\"} "
        "-- i.e. the ANSWER -- while message.complete carried "
        "reasoning='The user is asking for a greeting in a single sentence...'. "
        "Re-confirmed on a second live turn: a prompt answered 'RAWCAP OK' "
        "produced reasoning.available {'text': 'RAWCAP OK'}. "
        "Forwarding it as end-of-reasoning would print the answer a second time "
        "in the reasoning pane. The authoritative reasoning text is already on "
        "message.completed's payload, and end-of-reasoning is observable from the "
        "first message.delta. Revisit if a later Hermes build fixes the payload."
    ),
}

# Raw names that share a canonical type with another raw name. Derived, not
# hand-written, so adding a merge to the table above cannot forget to keep
# this in sync. `normalize_event` stamps `payload["_raw_type"]` on these so
# the merge is lossless -- `session.info` and `session.title` both arrive as
# `session.updated` but carry very different payloads, and a client should
# not have to guess which by sniffing keys.
_CANONICAL_FAN_IN: Counter[str] = Counter(RAW_TO_CANONICAL_TYPE.values())
MERGED_RAW_EVENTS: frozenset[str] = frozenset(
    raw for raw, canonical in RAW_TO_CANONICAL_TYPE.items() if _CANONICAL_FAN_IN[canonical] > 1
)

# Raw event names whose canonical counterpart is a `*.resolved` type purely
# because the underlying prompt expired (as opposed to being answered via
# the matching `*.respond` RPC, which this normalizer never sees directly --
# only the gateway's own record of having called `*.respond` would produce
# that side; see events/README note in normalizer.py).
EXPIRY_RAW_EVENTS: frozenset[str] = frozenset({"sudo.expire", "secret.expire"})


@dataclass(frozen=True)
class EventContext:
    """Everything the raw Hermes event itself does not carry.

    Hermes only knows about its own runtime session; it has no notion of our
    `project_id`, our internal `session_id`/`turn_id`, or which of our `Run`
    rows this activity belongs to. The caller (the code pumping Hermes's
    event stream) is responsible for tracking "which workspace run is this
    socket's activity currently attributed to" and supplying it here, along
    with a per-run monotonic `seq` it owns. Keeping that bookkeeping outside
    `normalize_event` is what keeps the normalizer itself a pure function of
    (raw_event, context) -> CanonicalEvent.

    All three attribution ids are optional (P2-2). `None` means "this frame
    is not attributed", which is a *true* statement the moment the normalizer
    runs -- attribution needs the live-handle -> stored-id resolution and the
    sessions-table lookup, both of which happen downstream in the broadcast
    path (`api/runs.py::RunRecorder`). Phase 0 instead stamped placeholder
    strings (`proj_phase0`/`sess_phase0`/`run_phase0`) here, which made every
    envelope *look* attributed while attributing nothing; an explicit null is
    the honest version of the same envelope, and the app already degrades a
    null envelope field to "" (`GatewayEvent.init(from:)`, B-34's rule
    applied to the socket).
    """

    project_id: str | None
    session_id: str | None
    run_id: str | None
    seq: int
    turn_id: str | None = None


@dataclass(frozen=True)
class CanonicalEvent:
    """The workspace's canonical event envelope (`ARCHITECTURE.md` §7).

    `project_id` / `session_id` / `run_id` are `None` until (unless) the
    broadcast path attributes the frame -- see `EventContext`. The envelope
    keys are always present in `to_dict()` (the app's decoder expects the
    shape); a null value is the explicit "not attributed" answer.
    """

    event_id: str
    project_id: str | None
    session_id: str | None
    run_id: str | None
    seq: int
    type: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render exactly the JSON shape shown in `ARCHITECTURE.md` §7."""
        return {
            "event_id": self.event_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "seq": self.seq,
            "type": self.type,
            "timestamp": self.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "payload": self.payload,
        }
