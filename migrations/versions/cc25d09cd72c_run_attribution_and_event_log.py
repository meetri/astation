"""run attribution and event log"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'cc25d09cd72c'
down_revision: Union[str, None] = '65478a8a907b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('run_events', schema=None) as batch_op:
        batch_op.create_index('ix_run_events_run_id_seq', ['run_id', 'seq'], unique=False)

    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('runtime_session_id', sa.String(), nullable=True))
        batch_op.alter_column('project_id',
               existing_type=sa.VARCHAR(),
               nullable=True)
        batch_op.create_index('ix_runs_project_id', ['project_id'], unique=False)
        batch_op.create_index('ix_runs_runtime_session_id', ['runtime_session_id'], unique=False)


def downgrade() -> None:
    op.execute(
        "DELETE FROM run_events WHERE run_id IN "
        "(SELECT id FROM runs WHERE project_id IS NULL)"
    )
    op.execute("DELETE FROM runs WHERE project_id IS NULL")
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_index('ix_runs_runtime_session_id')
        batch_op.drop_index('ix_runs_project_id')
        batch_op.alter_column('project_id',
               existing_type=sa.VARCHAR(),
               nullable=False)
        batch_op.drop_column('runtime_session_id')

    with op.batch_alter_table('run_events', schema=None) as batch_op:
        batch_op.drop_index('ix_run_events_run_id_seq')

