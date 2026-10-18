"""The pre-migrate backup: taken only when the schema will change, and pruned."""

import importlib.util
import sqlite3
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _plugin_api():
    spec = importlib.util.spec_from_file_location(
        "astation_plugin_api_under_test", PLUGIN_DIR / "dashboard" / "plugin_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stamped_db(path: Path, revision: str | None) -> Path:
    conn = sqlite3.connect(path)
    if revision is not None:
        conn.execute("CREATE TABLE alembic_version (version_num TEXT)")
        conn.execute("INSERT INTO alembic_version VALUES (?)", (revision,))
    conn.commit()
    conn.close()
    return path


def test_db_revisions_reads_the_stamp(tmp_path):
    api = _plugin_api()
    assert api._db_revisions(_stamped_db(tmp_path / "a.db", "abc123")) == {"abc123"}


def test_db_revisions_is_empty_without_a_stamp(tmp_path):
    api = _plugin_api()
    assert api._db_revisions(_stamped_db(tmp_path / "a.db", None)) == set()
    assert api._db_revisions(tmp_path / "missing.db") == set()


def test_backup_keeps_only_the_newest_copies(tmp_path):
    api = _plugin_api()
    db = _stamped_db(tmp_path / "research.db", "abc123")
    for day in range(1, 6):
        (tmp_path / f"research.db.bak-2026-09-0{day}-000000-pre-migrate").write_bytes(b"old")
    unrelated = tmp_path / "research.db.empty-pre-import-2026-09-19-0334"
    unrelated.write_bytes(b"keep")

    backup = api._backup_before_migrate(db)

    copies = sorted(p.name for p in tmp_path.glob("research.db.bak-*-pre-migrate"))
    assert backup.name in copies
    copy = sqlite3.connect(backup)
    try:
        assert copy.execute("SELECT version_num FROM alembic_version").fetchall() == [("abc123",)]
    finally:
        copy.close()
    assert len(copies) == api._BACKUPS_KEPT
    assert copies[:2] == [
        "research.db.bak-2026-09-04-000000-pre-migrate",
        "research.db.bak-2026-09-05-000000-pre-migrate",
    ]
    assert unrelated.exists()


def test_backup_includes_rows_still_in_the_wal(tmp_path):
    """The database runs in WAL mode: committed rows can live only in -wal, which a file copy drops."""
    api = _plugin_api()
    db = _stamped_db(tmp_path / "research.db", "abc123")
    live = sqlite3.connect(db)
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("PRAGMA wal_autocheckpoint=0")
    live.execute("CREATE TABLE t (v TEXT)")
    live.execute("INSERT INTO t VALUES ('committed-but-uncheckpointed')")
    live.commit()
    assert (tmp_path / "research.db-wal").stat().st_size > 0

    backup = api._backup_before_migrate(db)
    live.close()

    copy = sqlite3.connect(backup)
    try:
        assert copy.execute("SELECT v FROM t").fetchall() == [("committed-but-uncheckpointed",)]
    finally:
        copy.close()
