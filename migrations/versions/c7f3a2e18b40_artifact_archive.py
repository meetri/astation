"""artifact archive

Adds `artifacts.archived_at` plus its index.

A nullable timestamp rather than a boolean, for the same reason
`bookmarked_at` is one (`b5e1d7c93f04`): it orders "recently archived",
un-archiving is a null write rather than a state machine, and the column
answers *when* as well as *whether*.

**Archive is not delete.** An archived artifact keeps its bytes, its
provenance and every link into it from a transcript; it is only hidden from
the library's default listings. There is deliberately no delete for an
artifact anywhere in this system.

Revision ID: c7f3a2e18b40
Revises: b5e1d7c93f04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c7f3a2e18b40"
down_revision = "b5e1d7c93f04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("artifacts", sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_artifacts_archived_at", "artifacts", ["archived_at"])
    # No index on `source_path` here: `c7d841c7d3a9` already built
    # `ix_artifacts_source_path`, which is what the folders view and the
    # `?prefix=` filter scan.


def downgrade() -> None:
    op.drop_index("ix_artifacts_archived_at", table_name="artifacts")
    op.drop_column("artifacts", "archived_at")
