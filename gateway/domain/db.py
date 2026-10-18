"""The workspace database: one SQLite engine, one session factory, one dependency."""

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

SCHEMA_MISSING_DETAIL = (
    "the workspace database has no schema yet; run `uv run alembic upgrade head` "
    "in services/research-gateway"
)


def database_url(db_path: str) -> str:
    """`sqlite:///<abs path>` for a configured `RESEARCH_GATEWAY_DB_PATH`."""
    resolved = Path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved}"


def make_engine(db_path: str) -> Engine:
    """Build the engine for `db_path`, with SQLite's footguns disarmed."""
    engine = create_engine(
        database_url(db_path),
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _configure_connection(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            _enter_wal(cursor)
        finally:
            cursor.close()

    return engine


def _enter_wal(cursor: Any) -> None:
    """WAL + synchronous=NORMAL: a commit appends to the log instead of fsyncing a rollback journal.

    The event pipeline commits to this database on every streamed event, on the thread that serves the
    dashboard; with the default journal each commit cost ~12 ms of fsync on the production disk and a
    busy stream froze every request behind it. WAL is a property of the file, so this is a no-op once
    set; ``synchronous`` is per connection. A crash of this process loses nothing; an operating-system
    crash or power cut may roll back the last few commits but cannot corrupt the file. Anything that
    copies the database must use SQLite's backup API (see ``_backup_before_migrate``), not a file copy.
    """
    try:
        mode = cursor.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            logger.warning("workspace database stays in journal_mode=%s (WAL unavailable)", mode)
        cursor.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        # Switching modes needs a moment with no other writer; try again on the next connection.
        logger.warning("could not enable WAL on the workspace database; continuing as is", exc_info=True)


def make_sessionmaker(engine: Engine) -> sessionmaker[OrmSession]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def db_dependency(request: Request) -> Iterator[OrmSession]:
    """FastAPI dependency yielding one workspace DB session per request."""
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
    """A DB-session dependency with "you never ran the migration" turned into a 503."""

    def dependency(request: Request) -> Iterator[OrmSession]:
        if not getattr(request.app.state, flag_name, False):
            if not predicate(request.app.state.db_engine):
                raise schema_missing_error()
            setattr(request.app.state, flag_name, True)
        yield from db_dependency(request)

    return dependency
