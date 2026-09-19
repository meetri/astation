"""runs: add profile column

B-136's remaining multi-profile gap, owner-reported as "runs / turns aren't
listed for sessions that belong to non-default profiles". Runs are recorded
from the event stream, and only the DEFAULT connection's stream reached
`RunRecorder` -- so a `kimi25`/`qwen38-flash` turn produced no run row at
all and the Run Inspector had nothing to show.

Recording them needs this column for one specific reason on the READ path:
`api/runs.py::_reconcile_stale_running` asks Hermes "is this session still
working" to close runs whose `message.completed` went missing (B-62), and
it asks the default connection. A non-default profile's session is simply
absent from that answer, which the reconciler reads as "over" and closes --
so without a per-run profile, turning recording on would have started
silently marking live turns as completed. `Run.session_id` cannot stand in:
it is NULL for every unfiled session, which is most of them.

NOT NULL DEFAULT 'default' backfills correctly rather than merely
satisfying the constraint: every run recorded before this column existed
was, by construction, observed on the only connection there was.

Plain `add_column` -- no constraint is being changed, so SQLite does not
need the table rebuilt.

Revision ID: 7c1a9f0be3d2
Revises: 2afb0e7eeeb9
Create Date: 2026-09-05 18:05:00.000000+00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
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
