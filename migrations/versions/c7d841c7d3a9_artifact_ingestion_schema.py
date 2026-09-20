"""artifact ingestion schema"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c7d841c7d3a9'
down_revision: Union[str, None] = 'cc25d09cd72c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('artifacts', schema=None) as batch_op:
        batch_op.add_column(sa.Column('status', sa.String(), server_default='available', nullable=False))
        batch_op.add_column(sa.Column('source_path', sa.String(), nullable=True))
        batch_op.alter_column('project_id',
               existing_type=sa.VARCHAR(),
               nullable=True)
        batch_op.alter_column('storage_key',
               existing_type=sa.VARCHAR(),
               nullable=True)
        batch_op.create_index('ix_artifacts_source_path', ['source_path'], unique=False)


def downgrade() -> None:
    op.execute(
        "DELETE FROM artifact_links WHERE artifact_id IN "
        "(SELECT id FROM artifacts WHERE project_id IS NULL OR storage_key IS NULL)"
    )
    op.execute(
        "DELETE FROM artifacts WHERE project_id IS NULL OR storage_key IS NULL"
    )
    with op.batch_alter_table('artifacts', schema=None) as batch_op:
        batch_op.drop_index('ix_artifacts_source_path')
        batch_op.alter_column('storage_key',
               existing_type=sa.VARCHAR(),
               nullable=False)
        batch_op.alter_column('project_id',
               existing_type=sa.VARCHAR(),
               nullable=False)
        batch_op.drop_column('source_path')
        batch_op.drop_column('status')

