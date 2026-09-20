"""Put captured tool results back onto a reloaded transcript.

Hermes's stored transcript (`session.resume` / `session.history`) keeps every
tool call's ARGUMENTS and never its RESULT -- measured 2026-09-06 on the
owner's `gpt` session: 27 tool rows, 0 with a `result` key. The result is
where the diff, the exit code, the error and the written-bytes count live, so
after a reload every file-change row lost its diff stat and coloured lines
and every failed command looked like it had succeeded. Live, the same rows had
all of it, because `tool.completed` carries `result` -- and `ChatStore`
captured that frame, per profile, for exactly this session (91 rows for the
one above).

`attach_captured_results` marries the two: for each transcript tool row that
has no `result`, the first not-yet-used captured row with the same tool name
whose arguments are consistent with the transcript's is taken as that call's
completion, and its result is attached under `result` (bounded -- see
`_bounded_result`) with `_result_source: "gateway_capture"` so a reader can
tell it from a result Hermes itself sent.

**Matching is by content, not by position.** Both sequences are
chronological, but the capture can have gaps (a desynchronized subscriber, a
gateway restart mid-turn), and matching the Nth transcript call to the Nth
captured row would then attach the wrong output to a row -- a corrupted
transcript, silently. So a captured row is only used when every argument it
recorded is present with the same value on the transcript row (Hermes persists
a superset: the call's arguments plus the tool's defaults), scanning forward
from the last match so an unmatched capture is skipped rather than forced.
No match means no result, which is exactly what the row had before.

Pure: no I/O. The routes fetch the captured rows and hand them in.
"""

from __future__ import annotations

from typing import Any

RESULT_SOURCE_FIELD = "_result_source"
RESULT_SOURCE_CAPTURE = "gateway_capture"

#: Result string fields worth carrying whole are short by nature; these two
#: are the ones that can be a whole file or a whole build log.
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
