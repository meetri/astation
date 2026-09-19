"""Measured latency and spend per profile, from the gateway's own history (P6).

`docs/AGENT_MODEL_DESIGN.md` §1: *"Latency and spend are not in any catalog.
Nothing upstream gives them a priori."* What the gateway does have is every
turn it observed (`runs`, one row per turn with `profile`, `started_at`,
`ended_at`) and every `session.usage` frame it persisted (`run_events`). Owner
decision 5 (§6): latency is measured from this history, never a paid live
test. This module turns those rows into the §8 `stats` block:

    {"window_days": 7, "turns": 41,
     "first_response_s": {"median": 1.8, "p90": 6.2} | null,
     "turn_s":           {"median": 52.0, "p90": 410.0} | null,
     "tokens": {"prompt": 2598126, "completion": 16693},
     "spend_usd": 3.40 | null,
     "spend_note": "..."}

**Three measured facts shape the arithmetic (§7, deployed DB):**

1. **First response** is the earliest persisted `tool.generating` /
   `tool.started` / `message.interim` / `message.completed` after the run's
   `started_at`. **Not** `message.started` and **not** `status.update`:
   measured on the deployed DB (2026-09-06, kimi25), a run's `started_at` IS
   the timestamp of its own `message.started` (seq 1) and a `status.update`
   follows within ~15 ms, so counting either yields a flat `0.0` for every
   turn. The first frame that means "the model produced something" is the
   first tool call or the first sealed segment -- measured gaps of 12.3 s,
   13.9 s and 46.8 s on the same profile. It is "first thing the owner could
   see", not a vendor time-to-first-token: `reasoning.delta` is not
   persisted, so a reasoning-first model's first response is when its
   reasoning *ends*. A run with none of those four frames contributes
   nothing to the distribution.
2. **`session.usage` is CUMULATIVE per session**, ~15 frames per turn, one per
   model call: `{usage: {model, prompt, completion, input, output, reasoning,
   total, calls, context_used, context_max}}`. A turn's own tokens are
   therefore a **delta**: the last usage frame in this run minus the last
   usage frame of the previous run *in the same session* (zero baseline for a
   session's first turn). Runs are grouped by `runtime_session_id` (the
   stored id) and ordered by `started_at` for that subtraction. A negative
   delta means the counter reset underneath us (Hermes restarted; the session
   was re-resumed into a fresh process) and the run's own last value is taken
   as-is rather than reported as negative tokens.
3. **Spend uses each turn's own `usage.model`**, not the profile's current
   one: the owner may have switched the profile's model mid-week, and the
   turns before the switch cost what *that* model cost. The price comes from
   the injected `price_lookup(model_id) -> (input, output) | None` (per 1M
   tokens; `domain/model_catalog.py::price_lookup_from_catalog`). `spend_usd`
   is `null` when no run in the window had a known price -- never a
   reassuring `0.0` for "we could not price it".

The window boundary is handled so the first in-window turn of a session is
not charged the whole session's cumulative count: for each session the last
usage frame of the most recent run *before* the window is loaded as that
session's baseline.

Medians and p90 are plain `statistics`; `null` when there is no data.
"""

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

#: The persisted event types that count as "the agent produced something the
#: owner can see". `message.started` is deliberately absent -- it is the
#: frame that opens the run, so its gap to `started_at` is zero by
#: construction (measured, module docstring) -- and so is `status.update`
#: (a "recalled 32 memories" notice ~15 ms in). `reasoning.delta` and
#: `message.delta` are not persisted (`events/persistence.py`), so they
#: cannot be in this set.
FIRST_RESPONSE_EVENT_TYPES: frozenset[str] = frozenset(
    {"tool.generating", "tool.started", "message.interim", "message.completed"}
)

#: The one persisted frame that carries token counts.
USAGE_EVENT_TYPE = "session.usage"

#: The §8 window default.
DEFAULT_WINDOW_DAYS = 7

#: Verbatim in the response, so the app can show where the number came from.
SPEND_NOTE = (
    "≈ prompt × input price + completion × output price, per turn, using each turn's own model"
)

#: SQLite's default `SQLITE_MAX_VARIABLE_NUMBER` is 999 on older builds;
#: `IN (...)` lists are chunked well under it.
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
    """`{median, p90}` rounded to 2 dp, or None for no data.

    `statistics.quantiles` needs at least two points; a single observation is
    its own median and p90 rather than an error.
    """
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
    """The §8 `stats` block for `profile` over the last `days` days.

    `price_lookup(model_id)` answers `(input_per_million, output_per_million)`
    or None; `None` for the whole callable means nothing can be priced and
    `spend_usd` is `null`. `now` is injectable for the tests.
    """
    now_utc = _aware(now) or datetime.now(UTC)
    window_start = now_utc - timedelta(days=days)

    # Every turn for this profile, oldest first. The whole history is read
    # (rows are small) because the previous run of each in-window session --
    # possibly outside the window -- is the baseline for its cumulative usage.
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

    # Baseline per session: the last usage of the most recent pre-window run.
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

    for run in in_window:  # already oldest-first, so per-session deltas chain correctly
        started = _aware(run.started_at)
        ended = _aware(run.ended_at)
        assert started is not None  # filtered above
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
            # Counter reset under us (fresh Hermes process): the run's own
            # cumulative value is the best available reading of its cost.
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
