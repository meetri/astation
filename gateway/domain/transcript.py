"""Transcript projection: what a served Hermes transcript row may leave out.

Pure functions over the `messages` list `session.resume` / `session.history`
return, moved here from `api/main.py` (CLEANUP_PLAN step 3.2) so the session
routes (`api/sessions.py`) and the snapshot reader (`api/snapshots.py`) share
THE one place a transcript row can lose anything (B-86) without one of them
importing a private from the app module.

Two rules, both measured, both load-bearing:

* **B-34** -- a transcript row has exactly one guaranteed key, `role`; nothing
  here assumes a shape, and a non-dict row is forwarded untouched.
* **B-86** -- exactly two things may leave a row: a `reasoning_content`
  byte-identical to `reasoning` (always), and the reasoning bodies under
  `detail=light` (replaced by `has_reasoning` / `reasoning_chars`). `args` is
  never omitted.
"""

from __future__ import annotations

import json
from typing import Any


def _transcript(result: Any, count_key: str) -> tuple[int, Any]:
    """`(count, messages)` for a resume/history result -- B-34's shape rule.

    **A transcript row has exactly one guaranteed key, `role`.** Measured
    exhaustively over all 47 live sessions / 16,572 messages (see
    `docs/PROTOCOL_VERIFIED.md`, "Transcript message shape"): **58%** of real
    messages are `role: "tool"` calls carrying `args`/`context`/`name` and
    **no `text`, no `row_id` and no `timestamp` at all**. Assuming otherwise
    is B-34, which made every tool-using session -- i.e. almost every real
    one -- undecodable on the phone.

    So this function, and the two routes that use it, deliberately do **not**
    look inside a message. The list is forwarded element for element, exactly
    as Hermes stored it: the gateway has no shape to be wrong about, and a
    row type nobody has seen yet reaches the client intact instead of being
    dropped or mangled by a normalizer that predates it.

    What *is* normalized is the scalar beside it. `count_key` is Hermes's own
    total (`message_count` on resume, `count` on history) and the old code
    read it as `result.get(count_key, len(messages))` -- which returns `None`
    when Hermes sends an explicit null, because a present key never takes the
    default. A null (or a string) there is the same class of failure as B-34
    one field over: a client decoding it as an `Int` fails on the whole
    response, transcript included. It is always an int here, falling back to
    the measured length.
    """
    body = result if isinstance(result, dict) else {}
    messages = body.get("messages")
    if messages is None:
        messages = []
    count = body.get(count_key)
    if isinstance(count, bool) or not isinstance(count, int):
        # `len()` only where it is defined: `messages` came off the wire and
        # a non-list there must not turn a transcript into a 500.
        count = len(messages) if isinstance(messages, (list, tuple)) else 0
    return count, messages


# --- B-86: what a transcript row is allowed to leave out -------------------
#
# Measured live 2026-09-01 against the operator's own session
# (`20260829_223119_a8da0069`, 1,379 messages): the transcript these two
# routes served was **3,958,334 B**, of which `reasoning` was 1,638,139 B and
# `reasoning_content` was *the same* **1,638,139 B** -- byte-identical on all
# 494 rows that carried both, zero differences (the same result as the app's
# earlier 1,998-row / 16-session count). `SessionMessage.reasoningText` reads
# `reasoning` and only falls back to `reasoning_content` when `reasoning` is
# empty; that fallback has never once fired in any measurement. So ~41% of
# every transcript download was a copy no client has ever read -- fetched in
# full to render the 50 bubbles `TranscriptWindow.initialCount` shows.
#
# Exactly two things may leave a row here, and nothing else:
#
#   1. **A `reasoning_content` byte-identical to `reasoning`** -- always, on
#      both routes, no opt-in. When the two DIFFER, or when only
#      `reasoning_content` is present, it is kept verbatim: the app's
#      documented fallback is a real contract and this must not be the thing
#      that finally breaks it.
#   2. **The reasoning bodies** (`reasoning`, `reasoning_content`) -- under
#      `?detail=light` only, replaced by `has_reasoning` / `reasoning_chars`
#      so the app can still draw the "Reasoning" affordance and know whether
#      there is anything behind it. The body itself is then one
#      `GET /sessions/{id}/messages/{row_id}` away.
#
# **`args` is NOT omitted, at any detail level, and this is deliberate.**
# Measured on the same session 2026-09-01: all **799** tool rows carry **no
# `row_id`** -- only assistant (525) and user (55) rows have one, exactly as
# B-34 recorded. A tool row's `args` *is* the command text the UI has to show,
# and a row with no `row_id` is not addressable by the per-row detail route, so
# omitting `args` would strip a body with no path back and force the client to
# re-download the whole transcript to render one command. The size case agrees:
# `reasoning` + `reasoning_content` is 3.28 MB of the 3.96 MB payload (82%)
# while `args` is 346 KB (9%) -- dropping the duplicate and omitting reasoning
# already takes the transcript to ~680 KB, an ~83% cut, and the last 9% is not
# worth making the command text unfetchable.
#
# This is the first code in the gateway that looks *inside* a transcript row,
# which B-34's rule ("the gateway has no shape to be wrong about") deliberately
# avoided. The rule is honoured the only way it can be while still doing this:
# every access below is a `.get()` on a key that is allowed to be missing, no
# key is required, no key is invented, no value is rewritten, and a row that is
# not a dict at all is forwarded untouched. A row shape nobody has seen yet
# passes through with only the two reasoning keys at risk, and in `full` mode
# only the exact-duplicate one.
TRANSCRIPT_DETAIL_FULL = "full"
TRANSCRIPT_DETAIL_LIGHT = "light"

_REASONING_KEY = "reasoning"
_REASONING_CONTENT_KEY = "reasoning_content"
# Omitted by `detail=light`. A blacklist, not an envelope whitelist, on
# purpose: `display_kind` (PV "Transcript message shape") and the gateway's own
# synthesized background-result markers (`api/background.py`) are both real row
# keys that a whitelist would silently delete, and the next row key Hermes adds
# would go the same way. Only a body that is both heavy *and* re-fetchable
# leaves -- see the note on `args` above.
_LIGHT_OMITTED_KEYS = frozenset({_REASONING_KEY, _REASONING_CONTENT_KEY})


def _measure_body(value: Any) -> tuple[bool, int]:
    """`(is there anything to fetch, characters omitted)` for a reasoning body.

    "Anything to fetch" mirrors what the app would actually render:
    `SessionMessage.reasoningText` trims, so a whitespace-only reasoning shows
    nothing and is reported as nothing here. The two values always agree --
    `has_reasoning == (reasoning_chars > 0)` -- so a client can branch on
    either one and a "Reasoning" affordance can never open onto an empty sheet.

    Reasoning is a string on every row ever measured. A non-string one is
    measured by its compact JSON encoding rather than refused, on the same
    principle as everything else here: an unfamiliar shape must degrade to a
    number, not to a 500.
    """
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
    """`(has_reasoning, reasoning_chars)` for a row, by the app's own rule.

    `reasoning` first, `reasoning_content` only when `reasoning` is empty --
    exactly `SessionMessage.reasoningText`. The count therefore describes the
    text the client would have displayed, not the sum of both copies.
    """
    present, chars = _measure_body(row.get(_REASONING_KEY))
    if present:
        return present, chars
    return _measure_body(row.get(_REASONING_CONTENT_KEY))


def _reasoning_content_is_duplicate(row: dict) -> bool:
    """Whether `reasoning_content` is an exact copy of `reasoning` on this row.

    Both keys present, both strings, equal. Anything else -- one key missing,
    a non-string on either side, any difference at all -- is not a duplicate
    and both keys survive. The comparison is never fuzzy: no trimming, no
    normalization, no prefix match. A `reasoning_content` that differs from
    `reasoning` by one character is content, and it stays.
    """
    if _REASONING_KEY not in row or _REASONING_CONTENT_KEY not in row:
        return False
    primary = row[_REASONING_KEY]
    secondary = row[_REASONING_CONTENT_KEY]
    return isinstance(primary, str) and isinstance(secondary, str) and primary == secondary


def _project_message(row: Any, *, light: bool) -> Any:
    """One transcript row as it should be served. Never mutates its input.

    `light=False` is today's response minus a duplicated `reasoning_content`;
    a row with nothing to drop is passed through as the very same object.
    `light=True` additionally omits the reasoning bodies and adds
    `has_reasoning` / `reasoning_chars`, both always present so a client can
    decode them unconditionally. `args` survives both settings -- a tool row
    has no `row_id` and could not fetch it back (see the note above).
    """
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
    """`messages` with `_project_message` applied to every element.

    A non-list `messages` -- forwarded verbatim from a Hermes this code has
    never seen -- is returned untouched, same rule as
    `append_finished_background_results`.
    """
    if not isinstance(messages, list):
        return messages
    return [_project_message(row, light=light) for row in messages]


def _find_transcript_row(messages: Any, row_id: str) -> Any:
    """The first row whose `row_id` renders as `row_id`, or `None`.

    Compared as strings because Hermes's `row_id` is an int on this instance
    but the app already tolerates a quoted one (`SessionMessage`'s decoder
    coerces `"30808"`), and a path parameter is a string either way. `bool` is
    excluded explicitly -- it is an `int` subclass and `str(True)` is not a row
    id anybody meant.
    """
    if not isinstance(messages, (list, tuple)):
        return None
    for row in messages:
        if not isinstance(row, dict):
            continue
        candidate = row.get("row_id")
        if isinstance(candidate, bool) or not isinstance(candidate, (int, str)):
            continue
        if str(candidate).strip() == row_id:
            return row
    return None
