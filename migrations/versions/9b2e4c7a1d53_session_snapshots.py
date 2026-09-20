"""session snapshots"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9b2e4c7a1d53'
down_revision: Union[str, None] = '6473154d2901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'session_snapshots',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('project_id', sa.String(), nullable=True),
        sa.Column('workspace_session_id', sa.String(), nullable=True),
        sa.Column('stored_session_id', sa.String(), nullable=False),
        sa.Column('taken_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('reason', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('message_rows', sa.Integer(), nullable=False),
        sa.Column('list_message_count', sa.Integer(), nullable=True),
        sa.Column('messages_omitted', sa.Boolean(), nullable=True),
        sa.Column('raw_bytes', sa.Integer(), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('checksum', sa.String(), nullable=False),
        sa.Column('content_checksum', sa.String(), nullable=False),
        sa.Column('storage_key', sa.String(), nullable=False),
        sa.Column('hermes_meta_json', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(
            ['workspace_session_id'], ['sessions.id'], ondelete='SET NULL'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('session_snapshots', schema=None) as batch_op:
        batch_op.create_index(
            'ix_session_snapshots_stored_taken',
            ['stored_session_id', 'taken_at'],
            unique=False,
        )
        batch_op.create_index(
            'ix_session_snapshots_project_id', ['project_id'], unique=False
        )

    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.drop_column('archived_at')

    with op.batch_alter_table('session_snapshots', schema=None) as batch_op:
        batch_op.drop_index('ix_session_snapshots_project_id')
        batch_op.drop_index('ix_session_snapshots_stored_taken')
    op.drop_table('session_snapshots')
