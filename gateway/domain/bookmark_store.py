"""Stars, per project."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from domain.models import ArtifactBookmark, utcnow

UNFILED_SCOPE = ""


def scope_of(project_id: str | None) -> str:
    """A project id as a star scope. `None` is the unfiled scope."""
    return project_id or UNFILED_SCOPE


def set_bookmark(db: OrmSession, artifact_id: str, scope: str, *, starred: bool) -> datetime | None:
    """Star or un-star one artifact in one scope. Returns the new timestamp."""
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
    """When each of these was starred in `scope`, for the rows that were."""
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
    """Every artifact starred in `scope`, newest star first."""
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
    """Which scopes hold a star on this artifact, for its detail screen."""
    from domain.timeutil import iso_z

    rows = db.execute(
        select(ArtifactBookmark.scope, ArtifactBookmark.bookmarked_at)
        .where(ArtifactBookmark.artifact_id == artifact_id)
        .order_by(ArtifactBookmark.bookmarked_at.desc())
    )
    return [{"project_id": scope or None, "bookmarked_at": iso_z(at)} for scope, at in rows]
