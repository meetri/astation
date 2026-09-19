"""chat history: chat_messages, retiring messages/turns

P1 chat-history redesign (B-136, `docs/CHAT_HISTORY_DESIGN.md` §3/§4). Two
changes, one revision, because `tests/test_projects.py::test_alembic_chain_matches_the_models`
requires the models and the chain to land together:

* **Drops `messages` and `turns`.** Both have had 0 rows since P2-2 --
  `events/persistence.py`'s own module docstring records why that Phase-0
  "grow a second transcript copy in the gateway" machinery was cut rather
  than wired up (Hermes owns the transcript; nothing ever read this copy).
  Nothing else in the schema is FK-dependent on their *data* (both are
  empty), but `runs.turn_id` pointed at `turns.id` at the DDL level, so that
  column is rebuilt first, as a plain unenforced String (mirroring the
  existing, deliberately-unenforced `context_snapshots.turn_id` back
  -reference) -- it was never written by any code path either.
* **Adds `chat_messages`** -- the new durable chat-history table, captured
  from each Hermes profile's own live event stream
  (`domain/profile_connection.py`) plus the app's own turn-submit route
  (`domain/chat_store.py`). Schema matches `docs/CHAT_HISTORY_DESIGN.md` §4
  exactly: uniqueness is `(profile, stored_session_id, seq)`, never
  `stored_session_id` alone, because profiles are separate Hermes runtimes
  with separate id spaces.

Hand-written against head `9b2e4c7a1d53`.

Revision ID: fefd36f63ff6
Revises: 9b2e4c7a1d53
Create Date: 2026-09-05 00:00:00.000000+00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fefd36f63ff6'
down_revision: Union[str, None] = '9b2e4c7a1d53'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # `messages` first (child of `turns` at the DDL level; both are empty).
    op.drop_table('messages')

    # `runs.turn_id` pointed at `turns.id`. Drop that FK before dropping
    # `turns` -- SQLite batch mode's recreate otherwise preserves every
    # constraint it reflects off the existing table, FK included, even when
    # only unrelated columns are touched. The original migration declared
    # the FK unnamed, so `naming_convention` assigns it a deterministic name
    # (from the reflected column/target) purely so `drop_constraint` has
    # something to call -- verified against a real copy of this exact table:
    # the other three FKs on `runs` (project_id/session_id/parent_run_id)
    # survive the recreate untouched.
    _runs_fk_naming = {
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    }
    with op.batch_alter_table(
        'runs', schema=None, recreate='always', naming_convention=_runs_fk_naming
    ) as batch_op:
        batch_op.drop_constraint('fk_runs_turn_id_turns', type_='foreignkey')

    op.drop_table('turns')

    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('profile', sa.String(), nullable=False),
        sa.Column('stored_session_id', sa.String(), nullable=False),
        sa.Column('turn_id', sa.String(), nullable=True),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('tool_name', sa.String(), nullable=True),
        sa.Column('tool_call_id', sa.String(), nullable=True),
        sa.Column('tool_args_json', sa.JSON(), nullable=True),
        sa.Column('tool_result_json', sa.JSON(), nullable=True),
        sa.Column('hermes_row_id', sa.Integer(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column(
            'compacted', sa.Boolean(), nullable=False, server_default='0'
        ),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'profile', 'stored_session_id', 'seq',
            name='uq_chat_messages_profile_session_seq',
        ),
    )
    with op.batch_alter_table('chat_messages', schema=None) as batch_op:
        batch_op.create_index(
            'ix_chat_messages_profile', ['profile'], unique=False
        )
        batch_op.create_index(
            'ix_chat_messages_stored_session_id', ['stored_session_id'], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table('chat_messages', schema=None) as batch_op:
        batch_op.drop_index('ix_chat_messages_stored_session_id')
        batch_op.drop_index('ix_chat_messages_profile')
    op.drop_table('chat_messages')

    op.create_table(
        'turns',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('context_snapshot_id', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['context_snapshot_id'], ['context_snapshots.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    with op.batch_alter_table('runs', schema=None, recreate='always') as batch_op:
        batch_op.drop_column('turn_id')
        batch_op.add_column(sa.Column('turn_id', sa.String(), nullable=True))
        batch_op.create_foreign_key(
            'fk_runs_turn_id_turns', 'turns', ['turn_id'], ['id']
        )

    op.create_table(
        'messages',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('turn_id', sa.String(), nullable=True),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content_json', sa.JSON(), nullable=True),
        sa.Column('runtime_message_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.ForeignKeyConstraint(['turn_id'], ['turns.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
