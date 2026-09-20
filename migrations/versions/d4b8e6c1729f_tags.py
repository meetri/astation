"""tags, shared by artifacts and projects

One `tags` vocabulary with two join tables, rather than a tag table per
owner kind: the owner's corpus is one project holding 95% of everything, and
a tag meaning different things on a project and on a file would be two
vocabularies wearing one name.

`tags.name` is unique and stored already-normalized (`domain/tags.py`):
lowercased, whitespace collapsed, a small alphabet, 32 characters. The
uniqueness is what makes "create a tag" idempotent -- the route looks the
name up and returns the existing row rather than growing near-duplicates that
render identically and filter to different sets.

Both joins cascade on delete, so removing a tag detaches it everywhere and
never leaves a row pointing at nothing. Deleting a tag never deletes an
artifact or a project.

Revision ID: d4b8e6c1729f
Revises: c7f3a2e18b40
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "d4b8e6c1729f"
down_revision = "c7f3a2e18b40"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tags",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", name="uq_tags_name"),
    )
    op.create_index("ix_tags_name", "tags", ["name"])

    op.create_table(
        "artifact_tags",
        sa.Column(
            "artifact_id",
            sa.String(),
            sa.ForeignKey("artifacts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "tag_id", sa.String(), sa.ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
    )
    # The read path is "which artifacts carry this tag"; the primary key
    # already covers the other direction.
    op.create_index("ix_artifact_tags_tag_id", "artifact_tags", ["tag_id"])

    op.create_table(
        "project_tags",
        sa.Column(
            "project_id",
            sa.String(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "tag_id", sa.String(), sa.ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_project_tags_tag_id", "project_tags", ["tag_id"])


def downgrade() -> None:
    op.drop_index("ix_project_tags_tag_id", table_name="project_tags")
    op.drop_table("project_tags")
    op.drop_index("ix_artifact_tags_tag_id", table_name="artifact_tags")
    op.drop_table("artifact_tags")
    op.drop_index("ix_tags_name", table_name="tags")
    op.drop_table("tags")
