"""Continue in a new session -- the pure half."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HANDOFF_ROLES = frozenset({"user", "assistant"})

ROLE_LABELS = {"user": "User", "assistant": "Assistant"}

OMISSION_MARK = "[... earlier part of this message omitted ...]"

CONTINUED_SUFFIX = "(continued)"

UNTITLED_CONTINUATION = "Continued session"


@dataclass(frozen=True)
class HandoffExcerpt:
    """The transcript excerpt handed to the model, plus what it cost to make."""

    text: str
    messages_used: int
    messages_available: int
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
    """The last `last_n` conversational rows, formatted, within `max_chars`."""
    if last_n < 1:
        raise ValueError("last_n must be at least 1")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    available = conversational_rows(messages)
    chosen = available[-last_n:] if available else []
    truncated = False
    while len(chosen) > 1 and len(_format(chosen)) > max_chars:
        chosen = chosen[1:]
        truncated = True
    text = _format(chosen)
    if chosen and len(text) > max_chars:
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
    """The title the new session is offered under: `<source> (continued)`."""
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
