"""composer attachments ledger

P3-3. New `attachments` table -- the ledger for the asynchronous two-turn
composer attach (`domain/models.py::Attachment` carries the full rationale).
Hermes has no upload endpoint, so an
attachment's bytes travel gateway -> capability URL -> priming-turn `curl` ->
sandbox -> byte-exact verify -> `image.attach`/path-reference (measured
chain, PV "Phase 3 build probes", P3-0c). The flow takes minutes, so its
state must be pollable and must survive honest reporting across a gateway
restart (`orphaned`), which is exactly the `background_tasks` ledger's
reasoning applied to attachments.

Purely additive: no existing table or row is touched, so the downgrade is a
clean drop (in-flight attachment ledgers are transient orchestration state;
the underlying bytes remain in the content-addressed artifact store either
way).

Hand-written against head `c7d841c7d3a9`; `alembic check` reports "No new
upgrade operations detected" with this file applied.

Revision ID: 6473154d2901
Revises: c7d841c7d3a9
Create Date: 2026-08-31 12:00:00.000000+00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
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
