"""artifact bookmarks"""

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
