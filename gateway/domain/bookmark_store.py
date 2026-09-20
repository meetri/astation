"""Stars, per project.

Owner ask 2026-09-19: *"I just created a new project and I see bookmarked
artifacts from other projects."* A star used to be a column on the artifact,
so there was one shelf and every scope showed it. It is now a decision taken
**inside a scope**, so the same file can be starred in the project that
produced it and not in the one that is merely reading it.

One rule runs through everything here: **the scope in the request is the
scope the stars are read in.** A project listing shows that project's stars;
the unfiled listing shows the unfiled scope's; a request that names no scope
is asking "starred anywhere", and gets the union. There is no second rule for
writes -- starring while standing somewhere records the star there.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from domain.models import ArtifactBookmark, utcnow

#: The scope an unfiled artifact's star lives in. Empty string rather than
#: NULL so the composite primary key actually enforces one star per scope --
#: SQLite treats NULLs as distinct, so a nullable scope would let the same row
#: be starred twice and the shelf would show it twice.
UNFILED_SCOPE = ""


def scope_of(project_id: str | None) -> str:
    """A project id as a star scope. `None` is the unfiled scope."""
    return project_id or UNFILED_SCOPE


def set_bookmark(db: OrmSession, artifact_id: str, scope: str, *, starred: bool) -> datetime | None:
    """Star or un-star one artifact in one scope. Returns the new timestamp.

    Starring is idempotent but DOES refresh the timestamp: it records when the
    owner last said this matters, which is the order they expect to find it
    in. Un-starring what was never starred is not an error -- the caller's
    intent is already true.
    """
    existing = db.get(ArtifactBookmark, (artifact_id, scope))
    if not starred:
        if existing is not None:
            db.delete(existing)
        return None
    now = utcnow()
    if existing is None:
        db.add(ArtifactBookmark(artifact_id=artifact_id, scope=scope, bookmarked_at=now))
    else:
        existing.bookmarked_at = now
    return now


def is_bookmarked(db: OrmSession, artifact_id: str, scope: str) -> bool:
    return db.get(ArtifactBookmark, (artifact_id, scope)) is not None


def starred_at(db: OrmSession, artifact_ids: list[str], scope: str | None) -> dict[str, datetime]:
    """When each of these was starred in `scope`, for the rows that were.

    One query for a whole page rather than one per row: a 200-row listing that
    asked per row would be 200 queries for a glyph. `scope=None` means "any
    scope", and then the newest star wins -- the honest answer to "is this
    starred anywhere".
    """
    if not artifact_ids:
        return {}
    query = select(ArtifactBookmark.artifact_id, ArtifactBookmark.bookmarked_at).where(
        ArtifactBookmark.artifact_id.in_(artifact_ids)
    )
    if scope is not None:
        query = query.where(ArtifactBookmark.scope == scope)
    out: dict[str, datetime] = {}
    for artifact_id, at in db.execute(query):
        if artifact_id not in out or at > out[artifact_id]:
            out[artifact_id] = at
    return out


def scoped_ids(db: OrmSession, scope: str | None) -> list[str]:
    """Every artifact starred in `scope`, newest star first.

    `scope=None` is "starred anywhere", deduplicated to one entry per
    artifact -- a file starred in two projects is still one file.
    """
    query = select(ArtifactBookmark.artifact_id, ArtifactBookmark.bookmarked_at)
    if scope is not None:
        query = query.where(ArtifactBookmark.scope == scope)
    newest: dict[str, datetime] = {}
    for artifact_id, at in db.execute(query):
        if artifact_id not in newest or at > newest[artifact_id]:
            newest[artifact_id] = at
    return [
        artifact_id
        for artifact_id, _ in sorted(newest.items(), key=lambda kv: (-kv[1].timestamp(), kv[0]))
    ]


def scopes_of(db: OrmSession, artifact_id: str) -> list[dict[str, Any]]:
    """Which scopes hold a star on this artifact, for its detail screen.

    `project_id` is `null` for the unfiled scope, because that is what the
    artifact's own `project_id` says and two spellings of "no project" on one
    screen is one too many.
    """
    from domain.timeutil import iso_z

    rows = db.execute(
        select(ArtifactBookmark.scope, ArtifactBookmark.bookmarked_at)
        .where(ArtifactBookmark.artifact_id == artifact_id)
        .order_by(ArtifactBookmark.bookmarked_at.desc())
    )
    return [{"project_id": scope or None, "bookmarked_at": iso_z(at)} for scope, at in rows]
