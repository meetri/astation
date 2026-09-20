"""Transcript projection: what a served Hermes transcript row may leave out."""

from __future__ import annotations

import json
from typing import Any


def _transcript(result: Any, count_key: str) -> tuple[int, Any]:
    """`(count, messages)` for a resume/history result -- B-34's shape rule."""
    body = result if isinstance(result, dict) else {}
    messages = body.get("messages")
    if messages is None:
        messages = []
    # A get() default never fires on an explicit null; the type is checked instead.
    count = body.get(count_key)
    if isinstance(count, bool) or not isinstance(count, int):
        count = len(messages) if isinstance(messages, (list, tuple)) else 0
    return count, messages


TRANSCRIPT_DETAIL_FULL = "full"
TRANSCRIPT_DETAIL_LIGHT = "light"

_REASONING_KEY = "reasoning"
_REASONING_CONTENT_KEY = "reasoning_content"
# args never leaves: a tool row has no row_id and could not fetch it back.
# A blacklist, not a whitelist: a whitelist would delete row keys Hermes adds later.
_LIGHT_OMITTED_KEYS = frozenset({_REASONING_KEY, _REASONING_CONTENT_KEY})


def _measure_body(value: Any) -> tuple[bool, int]:
    """`(is there anything to fetch, characters omitted)` for a reasoning body."""
    if value is None:
        return False, 0
    if isinstance(value, str):
        return (True, len(value)) if value.strip() else (False, 0)
    if isinstance(value, (list, tuple, dict)) and not value:
        return False, 0
    try:
        chars = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):  # pragma: no cover - it came from JSON
        chars = len(str(value))
    return chars > 0, chars


def _measure_reasoning(row: dict) -> tuple[bool, int]:
    """`(has_reasoning, reasoning_chars)` for a row, by the app's own rule."""
    present, chars = _measure_body(row.get(_REASONING_KEY))
    if present:
        return present, chars
    return _measure_body(row.get(_REASONING_CONTENT_KEY))


def _reasoning_content_is_duplicate(row: dict) -> bool:
    """Whether `reasoning_content` is an exact copy of `reasoning` on this row."""
    # Exact equality only: a reasoning_content differing by one character is content.
    if _REASONING_KEY not in row or _REASONING_CONTENT_KEY not in row:
        return False
    primary = row[_REASONING_KEY]
    secondary = row[_REASONING_CONTENT_KEY]
    return isinstance(primary, str) and isinstance(secondary, str) and primary == secondary


def _project_message(row: Any, *, light: bool) -> Any:
    """One transcript row as it should be served. Never mutates its input."""
    if not isinstance(row, dict):
        return row
    if not light:
        if not _reasoning_content_is_duplicate(row):
            return row
        return {key: value for key, value in row.items() if key != _REASONING_CONTENT_KEY}

    projected = {key: value for key, value in row.items() if key not in _LIGHT_OMITTED_KEYS}
    has_reasoning, reasoning_chars = _measure_reasoning(row)
    projected["has_reasoning"] = has_reasoning
    projected["reasoning_chars"] = reasoning_chars
    return projected


def _project_transcript(messages: Any, *, light: bool) -> Any:
    """`messages` with `_project_message` applied to every element."""
    if not isinstance(messages, list):
        return messages
    return [_project_message(row, light=light) for row in messages]


def _find_transcript_row(messages: Any, row_id: str) -> Any:
    """The first row whose `row_id` renders as `row_id`, or `None`."""
    if not isinstance(messages, (list, tuple)):
        return None
    for row in messages:
        if not isinstance(row, dict):
            continue
        # Compared as strings: row_id is an int here but a path parameter is not.
        candidate = row.get("row_id")
        if isinstance(candidate, bool) or not isinstance(candidate, (int, str)):
            continue
        if str(candidate).strip() == row_id:
            return row
    return None
