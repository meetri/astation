"""attachments: persist the Hermes profile used for background routing

attachment orchestration outlives the upload request, but the ledger
recorded only a stored session id. Stored ids are profile-scoped; resuming a
non-default session against the default profile returns Hermes [4007] session
not found after the bytes have already reached the sandbox.

Existing rows predate profile-aware attachment uploads and therefore belong to
the only route they could use: default. The server default makes that backfill
explicit and keeps rolling deploys readable.

Revision ID: a9e4c2f71b63
Revises: f1c3d7a49e52
"""

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
