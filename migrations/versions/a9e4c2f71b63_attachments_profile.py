"""attachments: persist the Hermes profile used for background routing"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a9e4c2f71b63"
down_revision = "f1c3d7a49e52"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "attachments",
        sa.Column("profile", sa.String(), nullable=False, server_default="default"),
    )


def downgrade() -> None:
    with op.batch_alter_table("attachments") as batch:
        batch.drop_column("profile")
