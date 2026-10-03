"""Profile-qualify background-task ledger rows and identity."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b7d1e4a92c63"
down_revision = "6f4c2a9d8e10"
branch_labels = None
depends_on = None

_TABLE = "background_tasks"
_REBUILT = "_background_tasks_profile_identity"


def _create_profile_qualified_table(name: str) -> None:
    op.create_table(
        name,
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column(
            "profile",
            sa.String(),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("stored_session_id", sa.String(), nullable=True),
        sa.Column("prompt_text", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("result_text", sa.Text(), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("connection_generation", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("task_id", "profile"),
    )


def _create_indexes() -> None:
    op.create_index(
        "ix_background_tasks_stored_session_id",
        _TABLE,
        ["stored_session_id"],
        unique=False,
    )
    op.create_index(
        "ix_background_tasks_profile_stored_session_id",
        _TABLE,
        ["profile", "stored_session_id"],
        unique=False,
    )
    op.create_index(
        "ix_background_tasks_state",
        _TABLE,
        ["state"],
        unique=False,
    )


def upgrade() -> None:
    _create_profile_qualified_table(_REBUILT)
    op.execute(
        sa.text(
            f"INSERT INTO {_REBUILT} "
            "(task_id, profile, stored_session_id, prompt_text, submitted_at, state, "
            "result_text, finished_at, connection_generation) "
            f"SELECT task_id, 'default', stored_session_id, prompt_text, submitted_at, state, "
            f"result_text, finished_at, connection_generation FROM {_TABLE}"
        )
    )
    op.drop_table(_TABLE)
    op.rename_table(_REBUILT, _TABLE)
    _create_indexes()


def downgrade() -> None:
    bind = op.get_bind()
    duplicate = bind.execute(
        sa.text("SELECT task_id FROM background_tasks GROUP BY task_id HAVING COUNT(*) > 1 LIMIT 1")
    ).first()
    if duplicate is not None:
        raise RuntimeError(
            "cannot downgrade background_tasks while task ids collide across profiles"
        )

    old = "_background_tasks_task_identity"
    op.create_table(
        old,
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("stored_session_id", sa.String(), nullable=True),
        sa.Column("prompt_text", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("result_text", sa.Text(), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("connection_generation", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("task_id"),
    )
    op.execute(
        sa.text(
            f"INSERT INTO {old} "
            "(task_id, stored_session_id, prompt_text, submitted_at, state, result_text, "
            "finished_at, connection_generation) "
            "SELECT task_id, stored_session_id, prompt_text, submitted_at, state, result_text, "
            f"finished_at, connection_generation FROM {_TABLE}"
        )
    )
    op.drop_table(_TABLE)
    op.rename_table(old, _TABLE)
    op.create_index(
        "ix_background_tasks_stored_session_id",
        _TABLE,
        ["stored_session_id"],
        unique=False,
    )
    op.create_index(
        "ix_background_tasks_state",
        _TABLE,
        ["state"],
        unique=False,
    )
