"""Event normalization + persistence for the Research Gateway (P0-3, P2-2).

Maps raw Hermes JSON-RPC events onto the canonical workspace event shape
from `docs/ARCHITECTURE.md` §7/§7.1, and persists the run-relevant ones into
`run_events` (`domain/models.py`) -- see `events/persistence.py` for the
persistence policy and `api/runs.py` for the caller that wires it into the
broadcast path.
"""

from __future__ import annotations

from .canonical import (
    DELIBERATELY_DROPPED_RAW_EVENTS,
    MERGED_RAW_EVENTS,
    RAW_TO_CANONICAL_TYPE,
    CanonicalEvent,
    EventContext,
)
from .normalizer import normalize_event, raw_event_type
from .persistence import (
    NOT_PERSISTED_EVENT_TYPES,
    PERSISTED_RUN_EVENT_TYPES,
    is_persisted_event_type,
    persist_run_event,
)

__all__ = [
    "DELIBERATELY_DROPPED_RAW_EVENTS",
    "MERGED_RAW_EVENTS",
    "NOT_PERSISTED_EVENT_TYPES",
    "PERSISTED_RUN_EVENT_TYPES",
    "RAW_TO_CANONICAL_TYPE",
    "CanonicalEvent",
    "EventContext",
    "is_persisted_event_type",
    "normalize_event",
    "persist_run_event",
    "raw_event_type",
]
