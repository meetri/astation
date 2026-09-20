"""Small, shape-tolerant coercions for values that came off the wire or the DB."""

from __future__ import annotations

from typing import Any


def int_or_none(value: Any) -> int | None:
    """An int, excluding bool (an int subclass no caller means), else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
