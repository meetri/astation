"""Tag names: one normalization, used by every write path.

`docs/ARTIFACT_ORGANIZATION_PLAN.md` §4.2. Tags are a shared vocabulary for
projects and artifacts — one `tags` table, two join tables — because the
owner's corpus is one project holding 95% of everything, and a tag that meant
different things on a project and on a file would be two vocabularies wearing
one name.

**Why normalization lives in exactly one function.** A tag vocabulary is only
useful if `Results`, `results` and `  results ` are the same tag. If each
write path folded case its own way, the library would grow near-duplicate
tags that look identical in a list and filter to different sets — and nothing
would ever surface the divergence, because both spellings render the same.

The rules, and the reason for each:

* **Lowercased.** The display is the storage; there is no separate "display
  name" column to drift from it.
* **Inner whitespace collapsed, outer stripped.** `"  L328  results "` is the
  same intent as `"l328 results"`, and a trailing space is invisible in a
  chip.
* **A small alphabet**: letters, digits, space, hyphen, underscore, starting
  with a letter or digit. Enough for `l328`, `kink-rung`, `figure 2`; not
  enough for a path, a URL or an emoji, which are not tags.
* **32 characters.** A chip has to fit on a phone. Longer is a note, and
  notes have their own home.

Every rejection names the rule it broke, because "invalid tag" tells the operator
nothing about what to type instead.
"""

from __future__ import annotations

import re

#: What a tag may contain after normalization.
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9 _-]*$")

MAX_TAG_CHARS = 32

_WHITESPACE = re.compile(r"\s+")


class TagNameError(ValueError):
    """A tag name that cannot be stored, with a message naming the rule."""


def normalize_tag_name(raw: str) -> str:
    """The canonical form of `raw`, or raise `TagNameError`.

    Idempotent: normalizing an already-normalized name returns it unchanged,
    which is what lets the create route be safely called with either.
    """
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
    """Normalize many, de-duplicated, order preserved.

    Order is the caller's, not sorted: a client that sent
    `["results", "l328"]` gets them back in that order, and the one place
    sorting matters (the row's `tags` field) does it explicitly.
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in raw or ():
        normalized = normalize_tag_name(name)
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out
