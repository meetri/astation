"""session snapshots

P6-3. Two changes, one revision, because
`tests/test_projects.py::test_alembic_chain_matches_the_models` requires the
models and the chain to land together:

* New `session_snapshots` table -- the index over the gateway's own complete,
  point-in-time copies of Hermes sessions (`domain/models.py::SessionSnapshot`
  carries the full rationale). The bytes live in the content-addressed
  `ArtifactStore`; this row names them (`storage_key`) and describes them
  (`checksum`, `content_checksum`, `message_rows`, `hermes_meta_json`).
  **Both foreign keys are `ON DELETE SET NULL`** -- in this DDL, not only in the
  SQLAlchemy model -- so `DELETE /api/projects/{id}`, unfiling and
  `DELETE /api/sessions/{id}` need no snapshot cleanup code and can never
  delete a snapshot by accident: SQLite (`PRAGMA foreign_keys=ON`,
  `domain/db.py`) nulls the reference and the copy survives.
* `sessions.archived_at` (nullable, tz-aware) -- the user-intent flag behind
  a project's Archived section. It is set only after a snapshot row exists, so
  it never stands in for a copy that is not there.

Purely additive: no existing row is touched. The downgrade drops the column
and the table; the gzipped documents stay in the artifact store either way
(content-addressed, unreferenced, harmless -- nothing here deletes bytes).

Hand-written against head `6473154d2901`; `alembic check` reports "No new
upgrade operations detected" with this file applied.

Revision ID: 9b2e4c7a1d53
Revises: 6473154d2901
Create Date: 2026-09-03 12:00:00.000000+00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
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
        # ON DELETE SET NULL on both: the snapshot outlives the project and the
        # filing row it was taken under. See the module docstring.
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
