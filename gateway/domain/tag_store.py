"""Reading and writing tags, for both owners."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy import false as sa_false
from sqlalchemy.orm import Session as OrmSession

from domain.models import Artifact, ArtifactTag, ProjectTag, Tag, new_id, utcnow
from domain.tags import normalize_tag_name

_JOINS: dict[str, tuple[Any, Any]] = {
    "artifact": (ArtifactTag, ArtifactTag.artifact_id),
    "project": (ProjectTag, ProjectTag.project_id),
}


def _join_for(owner: str) -> tuple[Any, Any]:
    try:
        return _JOINS[owner]
    except KeyError as exc:  # pragma: no cover - callers are internal
        raise ValueError(f"unknown tag owner {owner!r}") from exc


def get_or_create(db: OrmSession, name: str) -> Tag:
    """The tag called `name`, created if new. Idempotent by the unique index."""
    normalized = normalize_tag_name(name)
    existing = db.execute(select(Tag).where(Tag.name == normalized)).scalar_one_or_none()
    if existing is not None:
        return existing
    tag = Tag(id=new_id("tag"), name=normalized, created_at=utcnow())
    db.add(tag)
    db.flush()
    return tag


def names_for(db: OrmSession, owner: str, owner_ids: list[str]) -> dict[str, list[str]]:
    """`{owner_id: [tag names]}`, sorted, for many owners in one query."""
    if not owner_ids:
        return {}
    join, owner_column = _join_for(owner)
    rows = db.execute(
        select(owner_column, Tag.name)
        .join(Tag, Tag.id == join.tag_id)
        .where(owner_column.in_(owner_ids))
    )
    out: dict[str, list[str]] = {}
    for owner_id, name in rows:
        out.setdefault(owner_id, []).append(name)
    for names in out.values():
        names.sort()
    return out


def names_of(db: OrmSession, owner: str, owner_id: str) -> list[str]:
    """One owner's tag names, sorted."""
    return names_for(db, owner, [owner_id]).get(owner_id, [])


def attach(db: OrmSession, owner: str, owner_id: str, name: str) -> bool:
    """Give `owner_id` the tag `name`. `True` when it was not already there."""
    join, owner_column = _join_for(owner)
    tag = get_or_create(db, name)
    existing = db.execute(
        select(join).where(owner_column == owner_id, join.tag_id == tag.id)
    ).scalar_one_or_none()
    if existing is not None:
        return False
    row = join(tag_id=tag.id, added_at=utcnow())
    setattr(row, owner_column.key, owner_id)
    db.add(row)
    return True


def detach(db: OrmSession, owner: str, owner_id: str, name: str) -> bool:
    """Take the tag off. `True` when it was there."""
    normalized = normalize_tag_name(name)
    join, owner_column = _join_for(owner)
    tag = db.execute(select(Tag).where(Tag.name == normalized)).scalar_one_or_none()
    if tag is None:
        return False
    result = db.execute(delete(join).where(owner_column == owner_id, join.tag_id == tag.id))
    return bool(result.rowcount)


def apply_bulk(
    db: OrmSession, owner: str, owner_ids: list[str], *, add: list[str], remove: list[str]
) -> int:
    """Add and remove tags across many owners. Returns how many owners changed."""
    changed: set[str] = set()
    for owner_id in owner_ids:
        for name in add:
            if attach(db, owner, owner_id, name):
                changed.add(owner_id)
        for name in remove:
            if detach(db, owner, owner_id, name):
                changed.add(owner_id)
    return len(changed)


def owner_ids_with_all(db: OrmSession, owner: str, names: list[str]) -> list[str]:
    """Owners carrying EVERY one of `names` (AND, not OR)."""
    if not names:
        return []
    join, owner_column = _join_for(owner)
    wanted = [normalize_tag_name(name) for name in names]
    rows = db.execute(
        select(owner_column)
        .join(Tag, Tag.id == join.tag_id)
        .where(Tag.name.in_(wanted))
        .group_by(owner_column)
        .having(func.count(func.distinct(Tag.name)) == len(set(wanted)))
    )
    return [owner_id for (owner_id,) in rows]


def catalog(
    db: OrmSession, *, project_id: str | None = None, unfiled: bool = False
) -> list[dict[str, Any]]:
    """Every tag with its two counts, most used first, ties alphabetical."""
    scoped = project_id is not None or unfiled
    artifact_query = select(ArtifactTag.tag_id, func.count()).group_by(ArtifactTag.tag_id)
    if scoped:
        artifact_query = artifact_query.join(
            Artifact, Artifact.id == ArtifactTag.artifact_id
        ).where(Artifact.project_id.is_(None) if unfiled else Artifact.project_id == project_id)
    artifact_counts = dict(db.execute(artifact_query).all())

    project_query = select(ProjectTag.tag_id, func.count()).group_by(ProjectTag.tag_id)
    if project_id is not None:
        project_query = project_query.where(ProjectTag.project_id == project_id)
    elif unfiled:
        project_query = project_query.where(sa_false())
    project_counts = dict(db.execute(project_query).all())

    tags = db.execute(select(Tag).order_by(Tag.name)).scalars()
    rows = [
        {
            "id": tag.id,
            "name": tag.name,
            "artifact_count": artifact_counts.get(tag.id, 0),
            "project_count": project_counts.get(tag.id, 0),
        }
        for tag in tags
    ]
    if scoped:
        rows = [row for row in rows if row["artifact_count"] or row["project_count"]]
    rows.sort(key=lambda row: (-(row["artifact_count"] + row["project_count"]), row["name"]))
    return rows


def delete_tag(db: OrmSession, tag_id: str) -> tuple[bool, int]:
    """Remove a tag from the vocabulary. `(deleted, how many owners lost it)`."""
    tag = db.get(Tag, tag_id)
    if tag is None:
        return False, 0
    detached = (
        db.execute(
            select(func.count()).select_from(ArtifactTag).where(ArtifactTag.tag_id == tag_id)
        ).scalar()
        or 0
    ) + (
        db.execute(
            select(func.count()).select_from(ProjectTag).where(ProjectTag.tag_id == tag_id)
        ).scalar()
        or 0
    )
    db.execute(delete(ArtifactTag).where(ArtifactTag.tag_id == tag_id))
    db.execute(delete(ProjectTag).where(ProjectTag.tag_id == tag_id))
    db.delete(tag)
    return True, detached


def rename(db: OrmSession, tag_id: str, name: str) -> Tag | None:
    """Rename a tag, merging into an existing one if the new name is taken."""
    tag = db.get(Tag, tag_id)
    if tag is None:
        return None
    normalized = normalize_tag_name(name)
    if normalized == tag.name:
        return tag
    target = db.execute(select(Tag).where(Tag.name == normalized)).scalar_one_or_none()
    if target is None:
        tag.name = normalized
        db.flush()
        return tag

    for join, owner_column in (
        (ArtifactTag, ArtifactTag.artifact_id),
        (ProjectTag, ProjectTag.project_id),
    ):
        owners = [
            owner_id
            for (owner_id,) in db.execute(select(owner_column).where(join.tag_id == tag.id))
        ]
        already = {
            owner_id
            for (owner_id,) in db.execute(select(owner_column).where(join.tag_id == target.id))
        }
        for owner_id in owners:
            if owner_id in already:
                continue
            row = join(tag_id=target.id, added_at=utcnow())
            setattr(row, owner_column.key, owner_id)
            db.add(row)
        db.execute(delete(join).where(join.tag_id == tag.id))
    db.delete(tag)
    db.flush()
    return target
