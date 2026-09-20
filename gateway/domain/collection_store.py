"""Reading and writing collections."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session as OrmSession

from domain.models import Artifact, Collection, CollectionArtifact, new_id, utcnow

MAX_COLLECTION_NAME_CHARS = 64
MAX_COLLECTION_DESCRIPTION_CHARS = 280


class CollectionNameError(ValueError):
    """A collection name that cannot be stored, with the rule it broke."""


def normalize_collection_name(raw: str) -> str:
    """Trim and collapse whitespace; reject empty or over-long."""
    if not isinstance(raw, str):
        raise CollectionNameError("a collection name must be text")
    collapsed = " ".join(raw.split())
    if not collapsed:
        raise CollectionNameError("a collection needs a name")
    if len(collapsed) > MAX_COLLECTION_NAME_CHARS:
        raise CollectionNameError(
            f"a collection name can be at most {MAX_COLLECTION_NAME_CHARS} characters; "
            f"that one is {len(collapsed)}"
        )
    return collapsed


def _counts(db: OrmSession, collection_ids: list[str]) -> dict[str, int]:
    if not collection_ids:
        return {}
    rows = db.execute(
        select(CollectionArtifact.collection_id, func.count())
        .where(CollectionArtifact.collection_id.in_(collection_ids))
        .group_by(CollectionArtifact.collection_id)
    )
    return dict(rows.all())


def _first_artifact_ids(db: OrmSession, collection_ids: list[str]) -> dict[str, str]:
    """The lowest-positioned member of each collection, for the cover."""
    if not collection_ids:
        return {}
    rows = db.execute(
        select(CollectionArtifact.collection_id, CollectionArtifact.artifact_id)
        .where(CollectionArtifact.collection_id.in_(collection_ids))
        .order_by(CollectionArtifact.collection_id, CollectionArtifact.position)
    )
    out: dict[str, str] = {}
    for collection_id, artifact_id in rows:
        out.setdefault(collection_id, artifact_id)
    return out


def collection_json(
    collection: Collection, *, count: int = 0, cover: dict[str, Any] | None = None
) -> dict[str, Any]:
    from domain.timeutil import iso_z

    return {
        "id": collection.id,
        "name": collection.name,
        "description": collection.description,
        "count": count,
        "created_at": iso_z(collection.created_at),
        "updated_at": iso_z(collection.updated_at),
        "cover": cover,
    }


def listing(db: OrmSession, render: Any) -> list[dict[str, Any]]:
    """Every collection, most recently touched first."""
    collections = list(
        db.execute(
            select(Collection).order_by(Collection.updated_at.desc(), Collection.name)
        ).scalars()
    )
    ids = [collection.id for collection in collections]
    counts = _counts(db, ids)
    covers = _first_artifact_ids(db, ids)
    artifacts = {
        artifact.id: artifact
        for artifact in db.execute(
            select(Artifact).where(Artifact.id.in_(list(covers.values())))
        ).scalars()
    }
    return [
        collection_json(
            collection,
            count=counts.get(collection.id, 0),
            cover=(
                render(artifacts[covers[collection.id]])
                if covers.get(collection.id) in artifacts
                else None
            ),
        )
        for collection in collections
    ]


def create(db: OrmSession, name: str, description: str | None) -> Collection:
    """Make a collection. Idempotent on the name, like tag creation."""
    normalized = normalize_collection_name(name)
    existing = db.execute(
        select(Collection).where(Collection.name == normalized)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    collection = Collection(
        id=new_id("col"),
        name=normalized,
        description=(description or "").strip() or None,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db.add(collection)
    db.flush()
    return collection


def _next_position(db: OrmSession, collection_id: str) -> int:
    highest = db.execute(
        select(func.max(CollectionArtifact.position)).where(
            CollectionArtifact.collection_id == collection_id
        )
    ).scalar()
    return (highest or 0) + 1


def add(db: OrmSession, collection: Collection, artifact_ids: list[str]) -> int:
    """Append artifacts to the end. Returns how many were actually added."""
    present = {
        artifact_id
        for (artifact_id,) in db.execute(
            select(CollectionArtifact.artifact_id).where(
                CollectionArtifact.collection_id == collection.id
            )
        )
    }
    known = [
        artifact_id
        for (artifact_id,) in db.execute(select(Artifact.id).where(Artifact.id.in_(artifact_ids)))
    ]
    ordered = [artifact_id for artifact_id in artifact_ids if artifact_id in set(known)]
    position = _next_position(db, collection.id)
    added = 0
    for artifact_id in ordered:
        if artifact_id in present:
            continue
        db.add(
            CollectionArtifact(
                collection_id=collection.id,
                artifact_id=artifact_id,
                position=position,
                added_at=utcnow(),
            )
        )
        position += 1
        added += 1
    if added:
        collection.updated_at = utcnow()
    return added


def remove(db: OrmSession, collection: Collection, artifact_id: str) -> bool:
    """Drop one member. **Never deletes the artifact.**"""
    result = db.execute(
        delete(CollectionArtifact).where(
            CollectionArtifact.collection_id == collection.id,
            CollectionArtifact.artifact_id == artifact_id,
        )
    )
    if result.rowcount:
        collection.updated_at = utcnow()
        return True
    return False


def member_ids(db: OrmSession, collection_id: str) -> list[str]:
    """Members in curated order."""
    return [
        artifact_id
        for (artifact_id,) in db.execute(
            select(CollectionArtifact.artifact_id)
            .where(CollectionArtifact.collection_id == collection_id)
            .order_by(CollectionArtifact.position, CollectionArtifact.added_at)
        )
    ]


def reorder(db: OrmSession, collection: Collection, artifact_ids: list[str]) -> bool:
    """Rewrite the whole order. `False` when `artifact_ids` is not the
    membership.
    """
    current = set(member_ids(db, collection.id))
    if set(artifact_ids) != current or len(artifact_ids) != len(current):
        return False
    rows = {
        row.artifact_id: row
        for row in db.execute(
            select(CollectionArtifact).where(CollectionArtifact.collection_id == collection.id)
        ).scalars()
    }
    for position, artifact_id in enumerate(artifact_ids, start=1):
        rows[artifact_id].position = position
    collection.updated_at = utcnow()
    return True


def containing(db: OrmSession, artifact_id: str) -> list[dict[str, str]]:
    """Which collections hold this artifact, for its detail screen."""
    rows = db.execute(
        select(Collection.id, Collection.name)
        .join(CollectionArtifact, CollectionArtifact.collection_id == Collection.id)
        .where(CollectionArtifact.artifact_id == artifact_id)
        .order_by(Collection.name)
    )
    return [{"id": collection_id, "name": name} for collection_id, name in rows]


def delete_collection(db: OrmSession, collection_id: str) -> tuple[bool, int]:
    """Remove a collection. `(deleted, how many memberships went with it)`."""
    collection = db.get(Collection, collection_id)
    if collection is None:
        return False, 0
    members = (
        db.execute(
            select(func.count())
            .select_from(CollectionArtifact)
            .where(CollectionArtifact.collection_id == collection_id)
        ).scalar()
        or 0
    )
    db.execute(delete(CollectionArtifact).where(CollectionArtifact.collection_id == collection_id))
    db.delete(collection)
    return True, members
