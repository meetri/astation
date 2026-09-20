"""Event normalization + persistence for the Research Gateway (P0-3, P2-2)."""

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
