"""Request-body validation rules shared by more than one router."""

from __future__ import annotations

from typing import Any

from adapters.hermes import REWIND_FIELDS

BLANK_TEXT_MESSAGE = "text must contain non-whitespace characters"


def reject_blank_text(value: str, *, message: str = BLANK_TEXT_MESSAGE) -> str:
    """Reject whitespace-only text with a 422, before anything upstream is touched."""
    if not value.strip():
        raise ValueError(message)
    return value


def reject_rewind_fields(data: Any, *, submission: str) -> Any:
    """Refuse any `prompt.submit` rewind/truncate key smuggled into a body."""
    if isinstance(data, dict):
        smuggled = REWIND_FIELDS & data.keys()
        if smuggled:
            raise ValueError(
                f"rewind-only field(s) {sorted(smuggled)} are not accepted on a "
                f"{submission} submission: they destructively rewrite session history "
                "and are never forwarded from an HTTP body"
            )
    return data
