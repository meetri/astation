"""artifact archive"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c7f3a2e18b40"
down_revision = "b5e1d7c93f04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("artifacts", sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_artifacts_archived_at", "artifacts", ["archived_at"])


def downgrade() -> None:
    op.drop_index("ix_artifacts_archived_at", table_name="artifacts")
    op.drop_column("artifacts", "archived_at")
