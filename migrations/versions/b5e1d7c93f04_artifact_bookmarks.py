"""artifact bookmarks

Adds `artifacts.bookmarked_at` plus its index.

A nullable timestamp rather than a boolean: it orders the bookmark shelf by
when the user starred something, which is not the same as when the artifact
was produced, and re-bookmarking reads as a fresh entry instead of quietly
keeping an old position.

Revision ID: b5e1d7c93f04
Revises: a1c4e7f2b9d0
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b5e1d7c93f04"
down_revision = "a1c4e7f2b9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("artifacts", sa.Column("bookmarked_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_artifacts_bookmarked_at", "artifacts", ["bookmarked_at"])


def downgrade() -> None:
    op.drop_index("ix_artifacts_bookmarked_at", table_name="artifacts")
    op.drop_column("artifacts", "bookmarked_at")
