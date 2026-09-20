"""The workspace database: one SQLite engine, one session factory, one dependency.

Phase 0 shipped `domain/models.py` and the Alembic chain but never opened the
database from the service -- every route keyed off Hermes's own stored ids and
nothing was persisted. Phase 1 is where that changes, so this module is the one
place that decides how the gateway talks to its own storage.

Three rules it exists to hold:

* **The gateway never creates its own schema.** `alembic upgrade head` owns the
  DDL (see `README.md`). A process that quietly `create_all()`-ed on boot would
  make a missing migration invisible until the first column drifted, and would
  write a schema the migration chain does not describe. If the tables are not
  there, a route says so (503) instead of inventing them.
* **Every request gets its own `Session` and it is always closed.**
  `db_dependency` is a generator dependency, so FastAPI closes the session on
  the way out of the request whether it succeeded or raised.
* **Foreign keys are enforced.** SQLite disables them per *connection* by
  default, which would let a `sessions.project_id` point at a project that no
  longer exists -- exactly the dangling row this index must not grow.

Deliberately synchronous. The store is a local SQLite file holding, at v1
scale, a few dozen rows; the queries here are single-row lookups and one
`COUNT(*)`. Introducing an async driver to avoid microseconds of event-loop
blocking would buy nothing and would make the schema/migration story two
stories. If this ever fronts Postgres with real concurrency, that is the moment
to revisit it -- not before.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import sqlalchemy
from fastapi import HTTPException, Request
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

#: What a route says when the workspace tables are not there. A 503 rather than
#: a 500 because the service is fine and the fix is one documented command --
#: this is "not ready", not "broken".
SCHEMA_MISSING_DETAIL = (
    "the workspace database has no schema yet; run `uv run alembic upgrade head` "
    "in services/research-gateway"
)


def database_url(db_path: str) -> str:
    """`sqlite:///<abs path>` for a configured `RESEARCH_GATEWAY_DB_PATH`.

    Relative paths resolve against the process cwd, matching how
    `migrations/env.py` builds the very same URL -- the app and the migration
    chain must never disagree about which file they mean.
    """
    resolved = Path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved}"


def make_engine(db_path: str) -> Engine:
    """Build the engine for `db_path`, with SQLite's footguns disarmed.

    `check_same_thread=False` because FastAPI may hand a sync dependency to a
    worker thread; the session itself is still never shared between requests.
    `PRAGMA foreign_keys=ON` is per-connection in SQLite and off by default, so
    it is set on every connection rather than once at startup.
    """
    engine = create_engine(
        database_url(db_path),
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    return engine


def make_sessionmaker(engine: Engine) -> sessionmaker[OrmSession]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def db_dependency(request: Request) -> Iterator[OrmSession]:
    """FastAPI dependency yielding one workspace DB session per request.

    Reads the factory from `request.app.state`, never a module global, for the
    same reason `_ensure_connected()` takes the caller's state: a router
    mounted on a second `FastAPI()` must use *that* app's engine.
    """
    factory: sessionmaker[OrmSession] = request.app.state.db_sessions
    session = factory()
    try:
        yield session
    finally:
        session.close()


def schema_is_present(engine: Engine) -> bool:
    """Whether the workspace tables exist. Used to turn a crash into a 503."""
    try:
        inspector = sqlalchemy.inspect(engine)
        return inspector.has_table("projects") and inspector.has_table("sessions")
    except OperationalError:  # pragma: no cover - unreadable/locked file
        return False


def schema_missing_error() -> HTTPException:
    return HTTPException(status_code=503, detail=SCHEMA_MISSING_DETAIL)


def table_present(engine: Engine, table: str) -> bool:
    """Whether `table` exists; an unreadable/locked file counts as absent."""
    try:
        return sqlalchemy.inspect(engine).has_table(table)
    except Exception:  # pragma: no cover - unreadable/locked file
        return False


def columns_present(engine: Engine, table: str, *columns: str) -> bool:
    """Whether every one of `columns` exists on `table` (a table that has
    existed since the initial schema proves nothing by `has_table`; the check
    is for the column a later migration added). Unreadable counts as absent."""
    try:
        names = {column["name"] for column in sqlalchemy.inspect(engine).get_columns(table)}
    except Exception:  # pragma: no cover - unreadable/locked file
        return False
    return all(column in names for column in columns)


def schema_checked_db(
    flag_name: str, predicate: Callable[[Engine], bool]
) -> Callable[[Request], Iterator[OrmSession]]:
    """A DB-session dependency with "you never ran the migration" turned into a 503.

    Without this the first request against an unmigrated database is an
    `OperationalError` and a bare 500, which reads as a broken gateway. It is
    not broken; it is one documented command away from ready, and the error
    should say so.

    `predicate(engine)` is the feature's own proof its migration ran
    (`schema_is_present`, `table_present`, `columns_present`). The positive
    result is cached on `app.state` under `flag_name`: a schema cannot un-exist
    under a running process, and paying an `inspect()` round trip per request
    for a fact that changes once ever is pure waste. The flag names are part
    of the test surface (`app.state.<feature>_schema_verified = False` resets
    the check), so each router keeps the name it always had.

    One factory replaces the six copies that grew one router at a time
    (CLEANUP_PLAN step 3.3).
    """

    def dependency(request: Request) -> Iterator[OrmSession]:
        if not getattr(request.app.state, flag_name, False):
            if not predicate(request.app.state.db_engine):
                raise schema_missing_error()
            setattr(request.app.state, flag_name, True)
        yield from db_dependency(request)

    return dependency
