"""The one workspace row for a stored runtime session."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from domain.models import Session


def find_filing(
    db: OrmSession, runtime: str, stored_session_id: str, *, profile: str | None = None
) -> Session | None:
    """The one workspace row for a stored runtime session, if it is filed."""
    clauses = [Session.runtime == runtime, Session.runtime_session_id == stored_session_id]
    if profile is not None:
        clauses.append(Session.profile == profile)
    return db.execute(select(Session).where(*clauses)).scalar_one_or_none()
