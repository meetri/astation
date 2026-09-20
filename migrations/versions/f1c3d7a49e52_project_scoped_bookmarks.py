"""a star belongs to a project, not to the whole workspace

Owner report 2026-09-19: *"I just created a new project and I see bookmarked
artifacts from other projects."* A star lived on the artifact row
(`artifacts.bookmarked_at`), so there was exactly one shelf and every scope
showed it -- which was the earlier ask ("reachable from every project") and is
now the wrong answer: a new project opened onto forty files it had nothing to
do with.

So a star becomes a row in `artifact_bookmarks`, keyed by
`(artifact_id, scope)` where `scope` is a project id or `""` for the unfiled
scope. The same file can be starred in the project that produced it and not in
one that merely reads it.

**Nothing starred is lost.** Every existing `bookmarked_at` becomes a row in
the project that owns the artifact -- `COALESCE(project_id, '')`, so an
unfiled artifact's star lands in the unfiled scope, which is exactly where the
owner was standing when they made it.

The column is then dropped rather than left behind. A column nothing
maintains is a drift trap: the next reader cannot tell that it stopped being
the truth. `downgrade()` puts it back from the newest star per artifact, which
is the most faithful single value a one-column shape can hold.

Revision ID: f1c3d7a49e52
Revises: e2a91f4d6b83
"""

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
        # A project id, or "" for the unfiled scope. NOT nullable: SQLite
        # treats NULLs as distinct in a unique index, so a nullable scope
        # would let the same artifact be starred twice in "no project" and
        # the shelf would show it twice.
        sa.Column("scope", sa.String(), primary_key=True),
        sa.Column("bookmarked_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_artifact_bookmarks_scope", "artifact_bookmarks", ["scope", "bookmarked_at"])

    # Carry every existing star into the scope that owns its artifact.
    op.execute(
        """
        INSERT INTO artifact_bookmarks (artifact_id, scope, bookmarked_at)
        SELECT id, COALESCE(project_id, ''), bookmarked_at
        FROM artifacts
        WHERE bookmarked_at IS NOT NULL
        """
    )

    # SQLite cannot drop an indexed column while the index exists.
    op.drop_index("ix_artifacts_bookmarked_at", table_name="artifacts")
    with op.batch_alter_table("artifacts") as batch:
        batch.drop_column("bookmarked_at")


def downgrade() -> None:
    with op.batch_alter_table("artifacts") as batch:
        batch.add_column(sa.Column("bookmarked_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_artifacts_bookmarked_at", "artifacts", ["bookmarked_at"])
    # The newest star per artifact: a single column cannot hold "starred in
    # two projects", and the most recent is the one the old shelf would have
    # ordered by.
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
