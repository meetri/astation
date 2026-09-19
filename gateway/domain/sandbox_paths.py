"""Gateway-side sandbox path confinement (defense in depth).

`validate_sandbox_path` was `api.sandbox._validate_sandbox_path`; it lives here
(CLEANUP_PLAN step 3.4) so `domain/artifact_ingest.py` can apply the same rule
as the browse / text / attachment routes without importing a router module.
The rule is lexical, `posixpath` semantics, no filesystem access.

**Two rules, not one.** `validate_sandbox_path` is the write/ingest rule: one
root, `Settings.hermes_sandbox_root`. `validate_view_path` is the READ-ONLY
viewer rule: that same root plus any extra root in
`Settings.research_gateway_view_roots`. They are separate because the sandbox
root is not only a permission -- `domain/artifact_ingest.py` WALKS it, so
widening it to browse a file would point the auto-ingest sweep at whatever it
was widened to. Only `GET /api/sandbox/text` uses the viewer rule; the write
route, the browse routes and ingest all keep the strict one.
"""

from __future__ import annotations

import posixpath

from fastapi import HTTPException


def _normalized_or_422(raw_path: str) -> str:
    """The shared 422 gate: absolute, non-empty, no NUL. Returns `normpath`."""
    if not raw_path or "\x00" in raw_path:
        raise HTTPException(
            status_code=422, detail="path must be a non-empty absolute sandbox path"
        )
    if not raw_path.startswith("/"):
        raise HTTPException(
            status_code=422,
            detail=f"path must be absolute (inside the sandbox root); got {raw_path!r}",
        )
    return posixpath.normpath(raw_path)


def is_under_root(normalized_path: str, root: str) -> bool:
    """Whether an already-normalized path sits at or under `root`.

    `/` is handled explicitly: `normpath("/") == "/"`, so the usual
    `startswith(root + "/")` would test for `//` and reject everything.
    """
    root = posixpath.normpath(root)
    if root == "/":
        return normalized_path.startswith("/")
    return normalized_path == root or normalized_path.startswith(root + "/")


def parse_roots(raw: str) -> tuple[str, ...]:
    """`"/opt,/srv"` -> `("/opt", "/srv")`, order kept, duplicates dropped.

    Anything not absolute is ignored rather than raising: this parses a
    server-side setting at request time, and one bad entry must not take the
    route down. `/` is a legitimate value meaning "the whole filesystem".
    """
    roots: list[str] = []
    for chunk in (raw or "").split(","):
        candidate = chunk.strip()
        if not candidate or not candidate.startswith("/") or "\x00" in candidate:
            continue
        normalized = posixpath.normpath(candidate)
        if normalized not in roots:
            roots.append(normalized)
    return tuple(roots)


def validate_view_path(raw_path: str, sandbox_root: str, view_roots: tuple[str, ...] = ()) -> str:
    """The READ-ONLY viewer rule: inside the sandbox root, or any `view_roots` entry.

    Same 422s and the same 403 status as `validate_sandbox_path`; the detail
    names every root that would have been accepted, so a 403 tells the owner
    what to add to `RESEARCH_GATEWAY_VIEW_ROOTS` rather than only what failed.

    Read-only is the whole point: a path this admits but
    `validate_sandbox_path` would not is reported `editable: false` by
    `api/sandbox_text.py`, and `PUT /api/sandbox/text` still runs the strict
    rule, so widening the viewer never widens what can be written.
    """
    normalized = _normalized_or_422(raw_path)
    if is_under_root(normalized, sandbox_root):
        return normalized
    for root in view_roots:
        if is_under_root(normalized, root):
            return normalized
    allowed = ", ".join(repr(r) for r in (posixpath.normpath(sandbox_root), *view_roots))
    raise HTTPException(
        status_code=403,
        detail=(
            f"path {raw_path!r} is outside every root this gateway may read "
            f"({allowed}); add a root to RESEARCH_GATEWAY_VIEW_ROOTS to widen "
            "the read-only viewer"
        ),
    )


def validate_sandbox_path(raw_path: str, sandbox_root: str) -> str:
    """Reject anything outside the configured sandbox root, lexically.

    422 for a malformed path (empty, relative, NUL); 403 for a well-formed
    path that normalizes outside the root -- mirroring Hermes's own 403 for
    the same offense, so a client sees one consistent status for "you may
    not touch that" regardless of which layer caught it.

    Returns the normalized path, which is what gets sent upstream: `..`
    segments are resolved *before* the request leaves the gateway, so the
    upstream never sees a traversal attempt this side already understood.
    """
    normalized = _normalized_or_422(raw_path)
    root = posixpath.normpath(sandbox_root)
    if not is_under_root(normalized, root):
        raise HTTPException(
            status_code=403,
            detail=(
                f"path {raw_path!r} is outside the sandbox root {root!r} "
                "(gateway-side confinement; Hermes enforces the same rule upstream)"
            ),
        )
    return normalized
