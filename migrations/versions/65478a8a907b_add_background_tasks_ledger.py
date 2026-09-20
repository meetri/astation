"""add background_tasks ledger"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '65478a8a907b'
down_revision: Union[str, None] = '3cddea4c8cf6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('background_tasks',
    sa.Column('task_id', sa.String(), nullable=False),
    sa.Column('stored_session_id', sa.String(), nullable=True),
    sa.Column('prompt_text', sa.Text(), nullable=True),
    sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('state', sa.String(), nullable=False),
    sa.Column('result_text', sa.Text(), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('connection_generation', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('task_id')
    )
    with op.batch_alter_table('background_tasks', schema=None) as batch_op:
        batch_op.create_index('ix_background_tasks_state', ['state'], unique=False)
        batch_op.create_index('ix_background_tasks_stored_session_id', ['stored_session_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('background_tasks', schema=None) as batch_op:
        batch_op.drop_index('ix_background_tasks_stored_session_id')
        batch_op.drop_index('ix_background_tasks_state')

    op.drop_table('background_tasks')
