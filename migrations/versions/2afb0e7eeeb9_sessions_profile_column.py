"""sessions: add profile column, widen the stored-id uniqueness to include it"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '2afb0e7eeeb9'
down_revision: Union[str, None] = 'fefd36f63ff6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('sessions', schema=None, recreate='always') as batch_op:
        batch_op.add_column(
            sa.Column('profile', sa.String(), nullable=False, server_default='default')
        )
        batch_op.drop_constraint('uq_sessions_runtime_runtime_session_id', type_='unique')
        batch_op.create_unique_constraint(
            'uq_sessions_runtime_profile_runtime_session_id',
            ['runtime', 'profile', 'runtime_session_id'],
        )


def downgrade() -> None:
    with op.batch_alter_table('sessions', schema=None, recreate='always') as batch_op:
        batch_op.drop_constraint(
            'uq_sessions_runtime_profile_runtime_session_id', type_='unique'
        )
        batch_op.create_unique_constraint(
            'uq_sessions_runtime_runtime_session_id', ['runtime', 'runtime_session_id']
        )
        batch_op.drop_column('profile')
