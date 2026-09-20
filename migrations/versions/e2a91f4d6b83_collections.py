"""collections, the pinned deliverable, and filing at promotion

A collection is a named, ORDERED, cross-project set of artifacts the owner
curates: "figures for the L328 paper". Ordered because a curated set has an
order -- figure 1 before figure 2 -- which is the whole difference between a
collection and a tag. Cross-project because the owner's corpus is one project
holding 95% of everything, so "which project" answers nothing.

`projects.pinned_artifact_id` is the one deliverable a project is currently
about. Deliberately NOT a foreign key: `artifacts.project_id` already points
at `projects`, and a key pointing back makes the two tables circular, which
SQLAlchemy cannot order for `create_all`/`drop_all`. The constraint would buy
little -- there is no delete for an artifact anywhere in this system, and the
read path renders `null` for a row it cannot find.

**Starred is deliberately NOT migrated into this.** `bookmarked_at` shipped
this week and works; the app presents it as a built-in collection. Doing two
data models' worth of churn to remove one column is not worth it, and the
column can fold in later if collections prove out
(`docs/ARTIFACT_ORGANIZATION_PLAN.md` §10).

Revision ID: e2a91f4d6b83
Revises: d4b8e6c1729f
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e2a91f4d6b83"
down_revision = "d4b8e6c1729f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", name="uq_collections_name"),
    )
    op.create_index("ix_collections_updated_at", "collections", ["updated_at"])

    op.create_table(
        "collection_artifacts",
        sa.Column(
            "collection_id",
            sa.String(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "artifact_id",
            sa.String(),
            sa.ForeignKey("artifacts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        # The curated order. Not a timestamp: "figure 1 before figure 2" is a
        # decision, not a chronology, and appending would otherwise be the
        # only order a collection could have.
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_collection_artifacts_order", "collection_artifacts", ["collection_id", "position"]
    )

    # No foreign key: `artifacts.project_id` already points at this table, and
    # a key pointing back makes the two circular -- which SQLAlchemy cannot
    # order for `create_all`/`drop_all`. There is no delete for an artifact in
    # this system, and the read path renders `null` for a row it cannot find.
    op.add_column("projects", sa.Column("pinned_artifact_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "pinned_artifact_id")
    op.drop_index("ix_collection_artifacts_order", table_name="collection_artifacts")
    op.drop_table("collection_artifacts")
    op.drop_index("ix_collections_updated_at", table_name="collections")
    op.drop_table("collections")
