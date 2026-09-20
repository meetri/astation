"""index sessions.project_id for the project session list"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '3cddea4c8cf6'
down_revision: Union[str, None] = 'd80d10a6908c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.create_index('ix_sessions_project_id', ['project_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.drop_index('ix_sessions_project_id')

