"""Durable exact-origin remote job completions and dispatch state."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "6f4c2a9d8e10"
down_revision = "a9e4c2f71b63"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "remote_job_origins",
        sa.Column("job_id", sa.String(128), primary_key=True),
        sa.Column("request_digest", sa.String(256), nullable=False),
        sa.Column("profile", sa.String(128), nullable=False),
        sa.Column("origin_session_id", sa.String(256), nullable=False),
        sa.Column("stored_session_id", sa.String(256), nullable=False),
        sa.Column("turn_id", sa.String(256), nullable=True),
        sa.Column("host", sa.String(256), nullable=False),
        sa.Column("task_label", sa.String(160), nullable=False),
        sa.Column("task_summary", sa.Text(), nullable=False),
        sa.Column("ownership_generation", sa.Integer(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "remote_job_completions",
        sa.Column("event_id", sa.String(256), primary_key=True),
        sa.Column("job_id", sa.String(128), sa.ForeignKey("remote_job_origins.job_id", ondelete="RESTRICT"), nullable=False, unique=True),
        sa.Column("request_digest", sa.String(256), nullable=False),
        sa.Column("profile", sa.String(128), nullable=False),
        sa.Column("stored_session_id", sa.String(256), nullable=False),
        sa.Column("turn_id", sa.String(256), nullable=True),
        sa.Column("owner_generation", sa.Integer(), nullable=False),
        sa.Column("host", sa.String(256), nullable=False),
        sa.Column("delivery_identity", sa.String(256), nullable=False),
        sa.Column("delivery_id", sa.String(256), nullable=False),
        sa.Column("event_sequence", sa.Integer(), nullable=False),
        sa.Column("terminal_state", sa.String(32), nullable=False),
        sa.Column("verification_state", sa.String(32), nullable=False),
        sa.Column("observation_state", sa.String(32), nullable=False),
        sa.Column("verification_error", sa.Text(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dispatch_state", sa.String(32), nullable=False),
        sa.Column("dispatch_attempt_count", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_owner", sa.String(64), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dispatch_error", sa.Text(), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("takeover_reconcile_required", sa.Boolean(), nullable=False),
        sa.Column("reconciliation_session_id", sa.String(256), nullable=True),
    )
    op.create_index("ix_remote_job_completions_profile", "remote_job_completions", ["profile"])
    op.create_index("ix_remote_job_completions_stored_session_id", "remote_job_completions", ["stored_session_id"])
    op.create_index("ix_remote_job_completions_received_at", "remote_job_completions", ["received_at"])
    op.create_index("ix_remote_job_completions_dispatch_state", "remote_job_completions", ["dispatch_state"])


def downgrade() -> None:
    op.drop_table("remote_job_completions")
    op.drop_table("remote_job_origins")