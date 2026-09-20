"""Canonical event envelope + the raw-Hermes -> canonical type mapping table."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

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
    "clarify.cancel": "clarify.resolved",
    "approval.cancel": "approval.resolved",
    "sudo.cancel": "sudo.resolved",
    "secret.cancel": "secret.resolved",
}

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

_CANONICAL_FAN_IN: Counter[str] = Counter(RAW_TO_CANONICAL_TYPE.values())
MERGED_RAW_EVENTS: frozenset[str] = frozenset(
    raw for raw, canonical in RAW_TO_CANONICAL_TYPE.items() if _CANONICAL_FAN_IN[canonical] > 1
)

EXPIRY_RAW_EVENTS: frozenset[str] = frozenset({"sudo.expire", "secret.expire"})

CANCEL_RAW_EVENTS: frozenset[str] = frozenset(
    {"clarify.cancel", "approval.cancel", "sudo.cancel", "secret.cancel"}
)


@dataclass(frozen=True)
class EventContext:
    """Everything the raw Hermes event itself does not carry."""

    project_id: str | None
    session_id: str | None
    run_id: str | None
    seq: int
    turn_id: str | None = None


@dataclass(frozen=True)
class CanonicalEvent:
    """The workspace's canonical event envelope (`ARCHITECTURE.md` §7)."""

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
