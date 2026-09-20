"""a star belongs to a project, not to the whole workspace"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f1c3d7a49e52"
down_revision = "e2a91f4d6b83"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifact_bookmarks",
        sa.Column(
            "artifact_id",
            sa.String(),
            sa.ForeignKey("artifacts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("scope", sa.String(), primary_key=True),
        sa.Column("bookmarked_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_artifact_bookmarks_scope", "artifact_bookmarks", ["scope", "bookmarked_at"])

    op.execute(
        """
        INSERT INTO artifact_bookmarks (artifact_id, scope, bookmarked_at)
        SELECT id, COALESCE(project_id, ''), bookmarked_at
        FROM artifacts
        WHERE bookmarked_at IS NOT NULL
        """
    )

    op.drop_index("ix_artifacts_bookmarked_at", table_name="artifacts")
    with op.batch_alter_table("artifacts") as batch:
        batch.drop_column("bookmarked_at")


def downgrade() -> None:
    with op.batch_alter_table("artifacts") as batch:
        batch.add_column(sa.Column("bookmarked_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_artifacts_bookmarked_at", "artifacts", ["bookmarked_at"])
    op.execute(
        """
        UPDATE artifacts
        SET bookmarked_at = (
            SELECT MAX(bookmarked_at) FROM artifact_bookmarks
            WHERE artifact_bookmarks.artifact_id = artifacts.id
        )
        """
    )
    op.drop_index("ix_artifact_bookmarks_scope", table_name="artifact_bookmarks")
    op.drop_table("artifact_bookmarks")
