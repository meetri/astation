"""projects: add folder_path (the file-browser bookmark)"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1c4e7f2b9d0"
down_revision: Union[str, None] = "7c1a9f0be3d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("folder_path", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "folder_path")
