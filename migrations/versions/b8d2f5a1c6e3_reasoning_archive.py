"""reasoning_archive: every streamed and Hermes-held reasoning body, per session"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b8d2f5a1c6e3"
down_revision = "b7d1e4a92c63"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reasoning_archive",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("profile", sa.String(), nullable=False),
        sa.Column("stored_session_id", sa.String(), nullable=False),
        sa.Column("turn_id", sa.String(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("hermes_row_id", sa.Integer(), nullable=True),
        sa.Column("chat_message_id", sa.String(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("sealed", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("reasoning_archive") as batch:
        batch.create_index(
            "ix_reasoning_archive_session", ["profile", "stored_session_id"], unique=False
        )
        batch.create_index(
            "ix_reasoning_archive_hermes_row",
            ["profile", "stored_session_id", "hermes_row_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("reasoning_archive") as batch:
        batch.drop_index("ix_reasoning_archive_hermes_row")
        batch.drop_index("ix_reasoning_archive_session")
    op.drop_table("reasoning_archive")
