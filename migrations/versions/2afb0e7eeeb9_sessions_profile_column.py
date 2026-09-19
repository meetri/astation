"""sessions: add profile column, widen the stored-id uniqueness to include it

B-136 multi-profile routing. `sessions` had no way to record which Hermes
profile a workspace session belongs to -- every session-scoped route
(resume, messages, turns, interrupt, the four prompt-respond routes)
assumed the single default connection. A stored session id is only unique
*within* a profile (`docs/CHAT_HISTORY_DESIGN.md` §4's `chat_messages`
uniqueness), so profile has to travel with the session, not be inferred.

NOT NULL DEFAULT 'default': every existing row was created against the one
connection that existed before this column, i.e. the default profile --
this backfills them correctly, not just satisfies the constraint.

Also widens `uq_sessions_runtime_runtime_session_id` to
`(runtime, profile, runtime_session_id)`. Without this, two different
profiles that happened to generate the same stored id (astronomically
unlikely -- Hermes's ids are timestamp + random hex -- but no longer
impossible now that more than one profile is real) would collide on the
old two-column constraint and one session would fail to file. Recreated in
one batch (SQLite cannot ALTER a constraint in place); the column add is
folded into the same batch rather than a separate `add_column` pass so the
table is only rebuilt once.

Revision ID: 2afb0e7eeeb9
Revises: fefd36f63ff6
Create Date: 2026-09-05 16:43:46.250549+00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2afb0e7eeeb9'
down_revision: Union[str, None] = 'fefd36f63ff6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('sessions', schema=None, recreate='always') as batch_op:
        batch_op.add_column(
            sa.Column('profile', sa.String(), nullable=False, server_default='default')
        )
        batch_op.drop_constraint('uq_sessions_runtime_runtime_session_id', type_='unique')
        batch_op.create_unique_constraint(
            'uq_sessions_runtime_profile_runtime_session_id',
            ['runtime', 'profile', 'runtime_session_id'],
        )


def downgrade() -> None:
    with op.batch_alter_table('sessions', schema=None, recreate='always') as batch_op:
        batch_op.drop_constraint(
            'uq_sessions_runtime_profile_runtime_session_id', type_='unique'
        )
        batch_op.create_unique_constraint(
            'uq_sessions_runtime_runtime_session_id', ['runtime', 'runtime_session_id']
        )
        batch_op.drop_column('profile')
