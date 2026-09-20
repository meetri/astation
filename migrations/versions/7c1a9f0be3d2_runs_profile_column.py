"""runs: add profile column"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '7c1a9f0be3d2'
down_revision: Union[str, None] = '2afb0e7eeeb9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'runs',
        sa.Column('profile', sa.String(), nullable=False, server_default='default'),
    )


def downgrade() -> None:
    op.drop_column('runs', 'profile')
