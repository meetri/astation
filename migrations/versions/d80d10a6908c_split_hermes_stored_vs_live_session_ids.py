"""split hermes stored vs live session ids"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd80d10a6908c'
down_revision: Union[str, None] = '261bd8308291'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('runtime_live_session_id', sa.String(), nullable=True))
        batch_op.create_unique_constraint('uq_sessions_runtime_runtime_session_id', ['runtime', 'runtime_session_id'])


def downgrade() -> None:
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.drop_constraint('uq_sessions_runtime_runtime_session_id', type_='unique')
        batch_op.drop_column('runtime_live_session_id')

