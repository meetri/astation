"""projects: add folder_path (the file-browser bookmark)

Owner ask 2026-09-07: *"I just wanna be able to create a place where I can go
and say, this is my default folder. So when I click on file browser, this is
where I go. Has nothing to do with anything else."* One nullable column
holding an absolute sandbox path. The file browser opens there; nothing else
reads it. NULL means "no bookmark set" -- the browser opens at the sandbox
root, exactly as it did before this column existed.

Deliberately NOT the project's *instructions*: those are a `HERMES.md` file in
a gateway-owned per-project folder (`domain/project_workspace.py`), edited with
the app's own file editor and baked into each new session's system prompt via
its `cwd` at `session.create` (measured 2026-09-07,
`docs/PROJECT_INSTRUCTIONS_DESIGN.md`). The bookmark and the instructions are
two separate layers by the operator's explicit instruction, so the bookmark is
the only new column.

Plain `add_column` -- no constraint changes, so SQLite needs no table rebuild.

Revision ID: a1c4e7f2b9d0
Revises: 7c1a9f0be3d2
Create Date: 2026-09-07 09:20:00.000000+00:00

"""
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
