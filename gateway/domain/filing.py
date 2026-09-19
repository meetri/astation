"""The one workspace row for a stored runtime session.

`find_filing` was `api.projects._find_filing`; it lives here (CLEANUP_PLAN
step 3.5) because `domain/snapshot_builder.py` resolves a session's filing
while building a snapshot, and domain/ must not import a router module.
`api.projects` still exposes it as `_find_filing` for its own routes and for
`api/instance.py` / `api/snapshots.py`.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from domain.models import Session


def find_filing(
    db: OrmSession, runtime: str, stored_session_id: str, *, profile: str | None = None
) -> Session | None:
    """The one workspace row for a stored runtime session, if it is filed.

    "The one" is guaranteed by `UNIQUE (runtime, profile, runtime_session_id)`
    in the schema (B-136), which is also what makes filing idempotent and
    makes "move" a single-row update rather than a delete-and-recreate.

    `profile=None` (the default) matches on `(runtime, stored_session_id)`
    alone, same as before that column existed -- used by `move_session()`/
    `unfile_session()`, which look a row up by stored id without knowing its
    profile ahead of time, and additionally re-check the caller's
    `project_id` against the result, which is by far the stronger
    disambiguator in practice (a real cross-profile stored-id collision
    landing in the same project too is not a risk worth threading `profile`
    through routes that have no other use for it). `file_stored_session()`
    passes a real value, since it is the one place actually deciding what a
    *new* row's profile is.
    """
    clauses = [Session.runtime == runtime, Session.runtime_session_id == stored_session_id]
    if profile is not None:
        clauses.append(Session.profile == profile)
    return db.execute(select(Session).where(*clauses)).scalar_one_or_none()
