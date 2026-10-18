"""The workspace database engine: WAL so per-event commits do not fsync a rollback journal."""

import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
from domain.db import make_engine


def test_workspace_engine_uses_wal_and_normal_sync_with_foreign_keys(tmp_path):
    engine = make_engine(str(tmp_path / "research.db"))
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA synchronous")).scalar() == 1  # NORMAL
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    engine.dispose()


def test_an_existing_rollback_journal_database_is_converted_and_keeps_its_rows(tmp_path):
    import sqlite3

    path = tmp_path / "research.db"
    legacy = sqlite3.connect(path)
    legacy.execute("CREATE TABLE t (v TEXT)")
    legacy.execute("INSERT INTO t VALUES ('before')")
    legacy.commit()
    assert legacy.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    legacy.close()

    engine = make_engine(str(path))
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("SELECT v FROM t")).scalar() == "before"
    engine.dispose()
