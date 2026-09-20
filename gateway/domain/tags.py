"""Tag names: one normalization, used by every write path."""

from __future__ import annotations

import re

TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9 _-]*$")

MAX_TAG_CHARS = 32

_WHITESPACE = re.compile(r"\s+")


class TagNameError(ValueError):
    """A tag name that cannot be stored, with a message naming the rule."""


def normalize_tag_name(raw: str) -> str:
    """The canonical form of `raw`, or raise `TagNameError`."""
    if not isinstance(raw, str):
        raise TagNameError("a tag must be text")
    collapsed = _WHITESPACE.sub(" ", raw).strip().lower()
    if not collapsed:
        raise TagNameError("a tag cannot be empty")
    if len(collapsed) > MAX_TAG_CHARS:
        raise TagNameError(
            f"a tag can be at most {MAX_TAG_CHARS} characters; "
            f"{collapsed[:MAX_TAG_CHARS]!r}… is {len(collapsed)}"
        )
    if not TAG_PATTERN.match(collapsed):
        raise TagNameError(
            "a tag can contain letters, digits, spaces, hyphens and underscores, "
            "and must start with a letter or a digit"
        )
    return collapsed


def normalize_tag_names(raw: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize many, de-duplicated, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for name in raw or ():
        normalized = normalize_tag_name(name)
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out
