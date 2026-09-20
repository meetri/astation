"""Measured latency and spend per profile, from the gateway's own history (P6)."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from domain.models import Run, RunEvent
from domain.runs import RUN_KIND_TURN

logger = logging.getLogger(__name__)

FIRST_RESPONSE_EVENT_TYPES: frozenset[str] = frozenset(
    {"tool.generating", "tool.started", "message.interim", "message.completed"}
)

USAGE_EVENT_TYPE = "session.usage"

DEFAULT_WINDOW_DAYS = 7

SPEND_NOTE = (
    "≈ prompt × input price + completion × output price, per turn, using each turn's own model"
)

_IN_CHUNK = 400

PriceLookup = Callable[[str], "tuple[float, float] | None"]


def _aware(value: datetime | None) -> datetime | None:
    """SQLite hands datetimes back naive; the recorder only ever writes UTC
    (`api/runs.py::_run_age_seconds` relies on the same convention)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _percentiles(values: list[float]) -> dict[str, float] | None:
    """`{median, p90}` rounded to 2 dp, or None for no data."""
    if not values:
        return None
    if len(values) == 1:
        one = round(values[0], 2)
        return {"median": one, "p90": one}
    median = statistics.median(values)
    p90 = statistics.quantiles(values, n=10, method="inclusive")[-1]
    return {"median": round(median, 2), "p90": round(p90, 2)}


def _usage_from_payload(payload: Any) -> dict[str, Any] | None:
    """The `usage` dict inside a persisted `session.usage` payload, or None."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _count(usage: dict[str, Any], primary: str, fallback: str) -> int:
    """`usage[primary]` as a non-negative int, else `usage[fallback]`, else 0."""
    for key in (primary, fallback):
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, float) and value.is_integer():
            return max(int(value), 0)
    return 0


def _chunks(items: list[str], size: int = _IN_CHUNK) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _load_events(
    db: OrmSession, run_ids: list[str], event_types: frozenset[str]
) -> dict[str, list[RunEvent]]:
    """`{run_id: [events in seq order]}` for the given types, chunked."""
    by_run: dict[str, list[RunEvent]] = defaultdict(list)
    for chunk in _chunks(run_ids):
        rows = db.execute(
            select(RunEvent)
            .where(RunEvent.run_id.in_(chunk), RunEvent.event_type.in_(event_types))
            .order_by(RunEvent.run_id, RunEvent.seq)
        ).scalars()
        for event in rows:
            by_run[event.run_id].append(event)
    return by_run


def _session_key(run: Run) -> str:
    """Runs are grouped for the cumulative-usage subtraction by the Hermes
    stored id; a row without one (pre-P2-2 in principle) falls back to its
    workspace session, then to itself."""
    return run.runtime_session_id or run.session_id or run.id


def _last_usage(events: list[RunEvent]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.event_type == USAGE_EVENT_TYPE:
            usage = _usage_from_payload(event.payload_json)
            if usage is not None:
                return usage
    return None


def compute_profile_stats(
    db: OrmSession,
    profile: str,
    days: int = DEFAULT_WINDOW_DAYS,
    *,
    price_lookup: PriceLookup | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The §8 `stats` block for `profile` over the last `days` days."""
    now_utc = _aware(now) or datetime.now(UTC)
    window_start = now_utc - timedelta(days=days)

    runs = list(
        db.execute(
            select(Run)
            .where(Run.kind == RUN_KIND_TURN, Run.profile == profile, Run.started_at.is_not(None))
            .order_by(Run.started_at, Run.id)
        ).scalars()
    )

    in_window: list[Run] = []
    predecessor_by_session: dict[str, Run] = {}
    latest_before_window: dict[str, Run] = {}
    for run in runs:
        started = _aware(run.started_at)
        if started is None:
            continue
        key = _session_key(run)
        if started < window_start:
            latest_before_window[key] = run
        else:
            in_window.append(run)
    for run in in_window:
        key = _session_key(run)
        if key in latest_before_window and key not in predecessor_by_session:
            predecessor_by_session[key] = latest_before_window[key]

    in_window_ids = [run.id for run in in_window]
    events_by_run = _load_events(db, in_window_ids, FIRST_RESPONSE_EVENT_TYPES | {USAGE_EVENT_TYPE})
    baseline_events = _load_events(
        db, [run.id for run in predecessor_by_session.values()], frozenset({USAGE_EVENT_TYPE})
    )

    baseline: dict[str, tuple[int, int]] = {}
    for key, run in predecessor_by_session.items():
        usage = _last_usage(baseline_events.get(run.id, []))
        if usage is not None:
            baseline[key] = (
                _count(usage, "prompt", "input"),
                _count(usage, "completion", "output"),
            )

    first_response: list[float] = []
    turn_seconds: list[float] = []
    prompt_total = 0
    completion_total = 0
    spend = 0.0
    priced_any = False

    for run in in_window:
        started = _aware(run.started_at)
        ended = _aware(run.ended_at)
        assert started is not None
        if ended is not None:
            turn_seconds.append(max((ended - started).total_seconds(), 0.0))

        events = events_by_run.get(run.id, [])
        first = min(
            (
                _aware(e.timestamp)
                for e in events
                if e.event_type in FIRST_RESPONSE_EVENT_TYPES and e.timestamp is not None
            ),
            default=None,
        )
        if first is not None:
            first_response.append(max((first - started).total_seconds(), 0.0))

        usage = _last_usage(events)
        if usage is None:
            continue
        key = _session_key(run)
        prev_prompt, prev_completion = baseline.get(key, (0, 0))
        last_prompt = _count(usage, "prompt", "input")
        last_completion = _count(usage, "completion", "output")
        prompt_delta = last_prompt - prev_prompt
        completion_delta = last_completion - prev_completion
        if prompt_delta < 0 or completion_delta < 0:
            prompt_delta, completion_delta = last_prompt, last_completion
        baseline[key] = (last_prompt, last_completion)
        prompt_total += prompt_delta
        completion_total += completion_delta

        model = usage.get("model")
        price = (
            price_lookup(model)
            if (price_lookup is not None and isinstance(model, str) and model)
            else None
        )
        if price is not None:
            input_price, output_price = price
            spend += (
                prompt_delta * input_price / 1_000_000 + completion_delta * output_price / 1_000_000
            )
            priced_any = True

    return {
        "window_days": days,
        "turns": len(in_window),
        "first_response_s": _percentiles(first_response),
        "turn_s": _percentiles(turn_seconds),
        "tokens": {"prompt": prompt_total, "completion": completion_total},
        "spend_usd": round(spend, 4) if priced_any else None,
        "spend_note": SPEND_NOTE,
    }


def empty_stats(days: int = DEFAULT_WINDOW_DAYS) -> dict[str, Any]:
    """The `stats` block for "no turns in the window" -- the same shape, all nulls."""
    return {
        "window_days": days,
        "turns": 0,
        "first_response_s": None,
        "turn_s": None,
        "tokens": {"prompt": 0, "completion": 0},
        "spend_usd": None,
        "spend_note": SPEND_NOTE,
    }


__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "FIRST_RESPONSE_EVENT_TYPES",
    "SPEND_NOTE",
    "USAGE_EVENT_TYPE",
    "compute_profile_stats",
    "empty_stats",
]
