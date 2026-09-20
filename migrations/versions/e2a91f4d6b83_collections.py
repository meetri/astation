"""collections, the pinned deliverable, and filing at promotion"""

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
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_collection_artifacts_order", "collection_artifacts", ["collection_id", "position"]
    )

    op.add_column("projects", sa.Column("pinned_artifact_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "pinned_artifact_id")
    op.drop_index("ix_collection_artifacts_order", table_name="collection_artifacts")
    op.drop_table("collection_artifacts")
    op.drop_index("ix_collections_updated_at", table_name="collections")
    op.drop_table("collections")
