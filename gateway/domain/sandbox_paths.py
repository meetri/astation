"""Gateway-side sandbox path confinement (defense in depth)."""

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
    """Whether an already-normalized path sits at or under `root`."""
    root = posixpath.normpath(root)
    if root == "/":
        return normalized_path.startswith("/")
    return normalized_path == root or normalized_path.startswith(root + "/")


def parse_roots(raw: str) -> tuple[str, ...]:
    """`"/opt,/srv"` -> `("/opt", "/srv")`, order kept, duplicates dropped."""
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
    """The READ-ONLY viewer rule: inside the sandbox root, or any `view_roots` entry."""
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
    """Reject anything outside the configured sandbox root, lexically."""
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
