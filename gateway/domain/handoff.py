"""Continue in a new session -- the pure half.

The operator: *"when I want to start a new session continuing from where I left
off ... analyze the last several messages and come up with a good prompt to
start a new session."* This module turns a transcript into the text a model
is asked to distil, and nothing else: it makes no request, reads no setting
and touches no session. The route (`api/handoff.py`) owns the I/O.

## What counts as "the last several messages"

Only the CONVERSATION: `role: "user"` and `role: "assistant"` rows carrying
non-blank `text`. Everything else is dropped before counting, so `last_n`
means "the last N things the two of them said" and not "the last N rows",
which -- measured over the operator's own transcripts (`docs/PROTOCOL_VERIFIED.md`,
"Transcript message shape") -- would be 58% tool calls:

* `role: "tool"` rows (a call's `name`/`args`/`context`) -- the *outcome* of a
  tool call is in the assistant's next text row; the call itself is noise
  to a fresh session.
* display-only rows (`display_kind`, e.g. `compacted 4 messages`) -- the
  app's timeline furniture, not something anyone said.
* rows with no `text`, or only whitespace.

## The size cap

The rewrite path has an input cap (`REWRITE_MAX_INPUT_CHARS`, 24 000 by
default) and this text goes down that same path, so it must fit. Older rows
are dropped first -- the point of a handoff is where the conversation IS,
not where it started -- and if the newest row alone is over the cap its
head is cut and marked, never its tail: the end of the last reply is the
part that says what happens next.

`truncated` in the result is true whenever anything was cut, so the app can
say so instead of presenting a partial excerpt as the whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The roles a handoff distils. Everything else is dropped (module docstring).
HANDOFF_ROLES = frozenset({"user", "assistant"})

#: How a role is labelled in the excerpt the model reads. Plain words, not the
#: wire role, because a small model follows "User:" / "Assistant:" more
#: reliably than a JSON-ish transcript, and because the excerpt is what an
#: editable prompt file (`handoff.md`) is written against.
ROLE_LABELS = {"user": "User", "assistant": "Assistant"}

#: Marks the head of a row that had to be cut to fit the cap.
OMISSION_MARK = "[... earlier part of this message omitted ...]"

#: The suffix a continued session's suggested title carries.
CONTINUED_SUFFIX = "(continued)"

#: The title a session with no title continues under.
UNTITLED_CONTINUATION = "Continued session"


@dataclass(frozen=True)
class HandoffExcerpt:
    """The transcript excerpt handed to the model, plus what it cost to make."""

    text: str
    #: Rows that made it into `text` (after the cap).
    messages_used: int
    #: Conversational rows the transcript had before `last_n` and the cap.
    messages_available: int
    #: Whether anything -- whole rows or the head of one -- was cut for size.
    truncated: bool


def _row_text(row: Any) -> str | None:
    """The spoken text of a conversational row, or `None` if it is not one."""
    if not isinstance(row, dict):
        return None
    if row.get("role") not in HANDOFF_ROLES:
        return None
    if row.get("display_kind"):
        return None
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def conversational_rows(messages: Any) -> list[tuple[str, str]]:
    """`(role, text)` for every row a handoff may read, in transcript order."""
    if not isinstance(messages, (list, tuple)):
        return []
    rows: list[tuple[str, str]] = []
    for row in messages:
        text = _row_text(row)
        if text is not None:
            rows.append((row["role"], text))
    return rows


def _format(rows: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{ROLE_LABELS[role]}:\n{text}" for role, text in rows)


def build_excerpt(messages: Any, *, last_n: int, max_chars: int) -> HandoffExcerpt:
    """The last `last_n` conversational rows, formatted, within `max_chars`.

    `last_n` and `max_chars` must be positive; the route's schema and the
    settings guarantee that, and a non-positive value here is a programming
    error rather than a request to return nothing.
    """
    if last_n < 1:
        raise ValueError("last_n must be at least 1")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    available = conversational_rows(messages)
    chosen = available[-last_n:] if available else []
    truncated = False
    # Drop from the oldest end until it fits, keeping at least the newest row.
    while len(chosen) > 1 and len(_format(chosen)) > max_chars:
        chosen = chosen[1:]
        truncated = True
    text = _format(chosen)
    if chosen and len(text) > max_chars:
        # One row is itself over the cap: keep its TAIL (module docstring).
        role, body = chosen[0]
        label = f"{ROLE_LABELS[role]}:\n{OMISSION_MARK}\n"
        keep = max(0, max_chars - len(label))
        text = label + body[-keep:] if keep else label.rstrip("\n")
        truncated = True
    return HandoffExcerpt(
        text=text,
        messages_used=len(chosen),
        messages_available=len(available),
        truncated=truncated,
    )


def suggested_title(source_title: str | None) -> str:
    """The title the new session is offered under: `<source> (continued)`.

    A title that is already a continuation counts up rather than stacking
    the suffix: `X (continued)` -> `X (continued 2)` -> `X (continued 3)`.
    An empty or missing title becomes `Continued session`. The operator edits
    the field before starting, so this only has to be a sensible default.
    """
    title = (source_title or "").strip()
    if not title:
        return UNTITLED_CONTINUATION
    if title.endswith(CONTINUED_SUFFIX):
        base = title[: -len(CONTINUED_SUFFIX)].rstrip()
        return f"{base} (continued 2)" if base else "(continued 2)"
    head, sep, tail = title.rpartition("(continued ")
    if sep and tail.endswith(")") and tail[:-1].isdigit():
        return f"{head.rstrip()} (continued {int(tail[:-1]) + 1})"
    return f"{title} {CONTINUED_SUFFIX}"


__all__ = [
    "CONTINUED_SUFFIX",
    "HANDOFF_ROLES",
    "OMISSION_MARK",
    "UNTITLED_CONTINUATION",
    "HandoffExcerpt",
    "build_excerpt",
    "conversational_rows",
    "suggested_title",
]
