"""Reading and writing tags, for both owners.

`docs/ARTIFACT_ORGANIZATION_PLAN.md` §4.2/§5.2. One vocabulary, two join
tables, and one module so the two owners cannot drift: an artifact and a
project attach a tag by the same rules, and the day a third owner appears it
gets the same ones.

**Every write normalizes.** Nothing here takes a raw name on trust, because
the whole value of a tag vocabulary is that `Results` and `results` are one
tag (`domain/tags.py`).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy import false as sa_false
from sqlalchemy.orm import Session as OrmSession

from domain.models import Artifact, ArtifactTag, ProjectTag, Tag, new_id, utcnow
from domain.tags import normalize_tag_name

#: The join table for each owner kind, and the column naming the operator. One
#: table rather than a branch at every call site: adding an owner is a row
#: here, not an `if` in six functions.
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
    """The tag called `name`, created if new. Idempotent by the unique index.

    Normalizing first is what makes it idempotent: without it "Results" would
    create a second row that renders identically to "results" in every chip
    and filters to a different set.
    """
    normalized = normalize_tag_name(name)
    existing = db.execute(select(Tag).where(Tag.name == normalized)).scalar_one_or_none()
    if existing is not None:
        return existing
    tag = Tag(id=new_id("tag"), name=normalized, created_at=utcnow())
    db.add(tag)
    db.flush()
    return tag


def names_for(db: OrmSession, owner: str, owner_ids: list[str]) -> dict[str, list[str]]:
    """`{owner_id: [tag names]}`, sorted, for many owners in one query.

    Batched because the alternative is a query per row in a 200-row listing.
    An owner with no tags is simply absent; the caller renders `[]`, which is
    always present on the wire so a client never has to treat absence as a
    special case.
    """
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
    """Give `owner_id` the tag `name`. `True` when it was not already there.

    Idempotent: re-tagging is not an error, and unlike a bookmark it does not
    refresh a timestamp -- a tag has no ordering that a re-tag could mean
    anything about.
    """
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
    """Take the tag off. `True` when it was there.

    The TAG itself survives an empty detach: it is a vocabulary entry the
    owner typed, and deleting it because its last artifact lost it would make
    the tag list flicker with their own work.
    """
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
    """Add and remove tags across many owners. Returns how many owners changed.

    One transaction, and **adds run before removes** so a request carrying the
    same name in both lists ends with it removed. Arbitrary either way, but it
    has to be decided somewhere rather than left to dictionary order.
    """
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
    """Owners carrying EVERY one of `names` (AND, not OR).

    AND because tags narrow: an owner filtering by `l328` and `results` is
    asking for the intersection, and OR would hand back more rows the more
    precisely they asked.
    """
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
    """Every tag with its two counts, most used first, ties alphabetical.

    Counts come from the joins rather than being stored, so they cannot drift
    from the rows they describe.

    **Scoped on request.** `project_id` (or `unfiled=True`)
    narrows to the tags that scope's artifacts actually carry, with that
    scope's counts, and drops the rest of the vocabulary entirely. The operator's
    report was about stars, but the same complaint applies here: a brand-new
    project offering forty tags that match nothing in it is a menu of dead
    ends. The vocabulary itself stays shared -- this only changes which part
    of it a scope is shown.
    """
    scoped = project_id is not None or unfiled
    artifact_query = select(ArtifactTag.tag_id, func.count()).group_by(ArtifactTag.tag_id)
    if scoped:
        artifact_query = artifact_query.join(
            Artifact, Artifact.id == ArtifactTag.artifact_id
        ).where(Artifact.project_id.is_(None) if unfiled else Artifact.project_id == project_id)
    artifact_counts = dict(db.execute(artifact_query).all())

    project_query = select(ProjectTag.tag_id, func.count()).group_by(ProjectTag.tag_id)
    if project_id is not None:
        # In a project, "1 project carries this" can only mean this one.
        project_query = project_query.where(ProjectTag.project_id == project_id)
    elif unfiled:
        # The unfiled scope has no project, so no project can carry a tag in
        # it -- reporting the workspace's project counts there would be a
        # count of something the operator is not looking at.
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
        # A tag nothing here carries is not part of this scope's vocabulary.
        rows = [row for row in rows if row["artifact_count"] or row["project_count"]]
    rows.sort(key=lambda row: (-(row["artifact_count"] + row["project_count"]), row["name"]))
    return rows


def delete_tag(db: OrmSession, tag_id: str) -> tuple[bool, int]:
    """Remove a tag from the vocabulary. `(deleted, how many owners lost it)`.

    Never deletes an artifact or a project; the cascade only clears the joins.
    """
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
    """Rename a tag, merging into an existing one if the new name is taken.

    Merging rather than refusing: the operator asking to rename `fig` to
    `figure` when `figure` exists means "these are the same thing", and a 409
    would leave them to do it by hand across every artifact.
    """
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
