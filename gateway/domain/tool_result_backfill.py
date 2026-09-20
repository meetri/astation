"""Put captured tool results back onto a reloaded transcript."""

from __future__ import annotations

from typing import Any

RESULT_SOURCE_FIELD = "_result_source"
RESULT_SOURCE_CAPTURE = "gateway_capture"

_LONG_TEXT_FIELDS = ("output", "content", "stdout", "stderr")
_LONG_TEXT_LIMIT = 4_000


def _args_consistent(captured: Any, transcript: Any) -> bool:
    """Every argument the capture recorded is on the transcript row, equal."""
    if not isinstance(captured, dict) or not captured:
        return False
    if not isinstance(transcript, dict):
        return False
    for key, value in captured.items():
        if key not in transcript:
            return False
        if transcript[key] != value:
            return False
    return True


def _bounded_result(result: Any) -> Any:
    """A copy safe to put on every row of a 300-row transcript."""
    if not isinstance(result, dict):
        if isinstance(result, str) and len(result) > _LONG_TEXT_LIMIT:
            return (
                result[:_LONG_TEXT_LIMIT]
                + f"\n… ({len(result) - _LONG_TEXT_LIMIT} more characters)"
            )
        return result
    bounded: dict[str, Any] = {}
    for key, value in result.items():
        if key in _LONG_TEXT_FIELDS and isinstance(value, str) and len(value) > _LONG_TEXT_LIMIT:
            bounded[key] = (
                value[:_LONG_TEXT_LIMIT] + f"\n… ({len(value) - _LONG_TEXT_LIMIT} more characters)"
            )
            bounded["_truncated"] = True
        else:
            bounded[key] = value
    return bounded


def attach_captured_results(messages: Any, captured: list[dict[str, Any]]) -> int:
    """Attach captured results to `messages`'s tool rows in place. Returns how many."""
    if not isinstance(messages, list) or not captured:
        return 0
    candidates = [
        row
        for row in captured
        if isinstance(row, dict) and row.get("tool_result") is not None and row.get("tool_name")
    ]
    if not candidates:
        return 0
    cursor = 0
    attached = 0
    for row in messages:
        if not isinstance(row, dict) or row.get("role") != "tool" or "result" in row:
            continue
        name = row.get("name")
        if not name:
            continue
        for index in range(cursor, len(candidates)):
            candidate = candidates[index]
            if candidate.get("tool_name") != name:
                continue
            if not _args_consistent(candidate.get("tool_args"), row.get("args")):
                continue
            row["result"] = _bounded_result(candidate["tool_result"])
            row[RESULT_SOURCE_FIELD] = RESULT_SOURCE_CAPTURE
            if candidate.get("tool_call_id") and "tool_id" not in row:
                row["tool_id"] = candidate["tool_call_id"]
            cursor = index + 1
            attached += 1
            break
    return attached
