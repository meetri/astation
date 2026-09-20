"""composer attachments ledger"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '6473154d2901'
down_revision: Union[str, None] = 'c7d841c7d3a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'attachments',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('stored_session_id', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('mime_type', sa.String(), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('checksum', sa.String(), nullable=False),
        sa.Column('storage_key', sa.String(), nullable=False),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('sandbox_path', sa.String(), nullable=False),
        sa.Column('serve_token', sa.String(), nullable=False),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.Column('attach_result_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('attachments', schema=None) as batch_op:
        batch_op.create_index(
            'ix_attachments_stored_session_id', ['stored_session_id'], unique=False
        )
        batch_op.create_index(
            'ix_attachments_serve_token', ['serve_token'], unique=True
        )


def downgrade() -> None:
    with op.batch_alter_table('attachments', schema=None) as batch_op:
        batch_op.drop_index('ix_attachments_serve_token')
        batch_op.drop_index('ix_attachments_stored_session_id')
    op.drop_table('attachments')
