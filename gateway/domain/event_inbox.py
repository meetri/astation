"""Private durable inbox for unbound host events, separate from chat history.

No TTL/deletion on disconnect: only successful canonical capture acknowledges
an entry. A new server incarnation cannot bind an old live handle. Old-epoch
entries remain available for explicitly authorized recovery, never profile guesses.

Every event is written, bound and acknowledged here, so each event costs three commits on the hot path.
The store runs in WAL mode with ``synchronous=NORMAL`` on one long-lived connection: a commit
is an append to the log instead of a rollback-journal round trip with several fsyncs (measured ~12 ms
against ~0.04 ms on the production disk). Durability is unchanged against a crash of this process;
after an operating-system crash or power loss the last few commits may be rolled back, but the file
cannot be corrupted. WAL also creates ``-wal``/``-shm`` siblings, which SQLite gives the main file's
mode, so the 0600 restriction below covers them.
"""
from contextlib import contextmanager
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class EventInboxFull(RuntimeError):
    """The configured private-data quota is full; existing rows are untouched."""


class EventInbox:
    def __init__(self, path, *, max_rows=10_000, max_bytes=64 * 1024 * 1024,
                 page_size=100):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self.path = path
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.page_size = page_size
        # One connection, serialized: sqlite3 connections are not safe for concurrent use, and the
        # event pipeline may call from a worker thread while the loop reads counts for health.
        self._lock = threading.RLock()
        self._db = None
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS pending (id TEXT PRIMARY KEY, epoch TEXT NOT NULL, envelope TEXT NOT NULL, raw_type TEXT)')

    def _open(self):
        db = sqlite3.connect(self.path, check_same_thread=False)
        try:
            mode = db.execute('PRAGMA journal_mode=WAL').fetchone()[0]
            if str(mode).lower() != 'wal':
                # e.g. a filesystem without shared-memory support: still correct, just slower.
                logger.warning('event inbox could not enter WAL mode (journal_mode=%s)', mode)
            db.execute('PRAGMA synchronous=NORMAL')
        except Exception:
            db.close()
            raise
        return db

    @contextmanager
    def connect(self):
        """One transaction on the shared connection: commits on success, rolls back on error."""
        with self._lock:
            if self._db is None:
                self._db = self._open()
            try:
                with self._db:
                    yield self._db
            except sqlite3.Error:
                # A connection that raised a database error is not trusted again.
                self.close()
                raise

    def close(self):
        with self._lock:
            db, self._db = self._db, None
        if db is not None:
            db.close()

    def put(self, epoch, envelope, raw_type):
        value = json.dumps(envelope)
        with self.connect() as db:
            if db.execute('SELECT 1 FROM pending WHERE id=?', (envelope['event_id'],)).fetchone():
                return
            count, used = db.execute(
                'SELECT COUNT(*), COALESCE(SUM(length(CAST(envelope AS BLOB))),0) FROM pending'
            ).fetchone()
            size = len(value.encode('utf-8'))
            if count >= self.max_rows or used + size > self.max_bytes:
                raise EventInboxFull('event inbox quota exceeded')
            db.execute('INSERT OR IGNORE INTO pending VALUES (?,?,?,?)',
                       (envelope['event_id'], epoch, value, raw_type))

    def bind(self, envelope):
        """Persist an already-proven owner before any canonical side effect."""
        bound = dict(envelope, _bound_identity=True)
        with self.connect() as db:
            db.execute('UPDATE pending SET envelope=? WHERE id=?',
                       (json.dumps(bound), envelope['event_id']))

    def pending(self, epoch, *, after_id=None, limit=None):
        limit = min(limit or self.page_size, self.page_size)
        with self.connect() as db:
            if after_id is None:
                rows = db.execute(
                    "SELECT envelope,raw_type FROM pending WHERE (epoch=? OR json_extract(envelope, '$._bound_identity')=1) ORDER BY rowid LIMIT ?",
                    (epoch, limit),
                ).fetchall()
            else:
                marker = db.execute('SELECT rowid FROM pending WHERE id=?', (after_id,)).fetchone()
                after_rowid = marker[0] if marker else 0
                rows = db.execute(
                    "SELECT envelope,raw_type FROM pending WHERE (epoch=? OR json_extract(envelope, '$._bound_identity')=1) AND rowid>? "
                    'ORDER BY rowid LIMIT ?', (epoch, after_rowid, limit)
                ).fetchall()
        return [(json.loads(value), kind) for value, kind in rows]

    def counts(self, epoch=None):
        with self.connect() as db:
            pending, size = db.execute(
                'SELECT COUNT(*), COALESCE(SUM(length(CAST(envelope AS BLOB))),0) FROM pending'
            ).fetchone()
            current = (db.execute('SELECT COUNT(*) FROM pending WHERE epoch=?', (epoch,)).fetchone()[0]
                       if epoch is not None else 0)
        return {'pending': pending, 'current_epoch': current,
                'stranded_old_epoch': pending - current, 'bytes': size}

    def acknowledge(self, event_id):
        with self.connect() as db:
            db.execute('DELETE FROM pending WHERE id=?', (event_id,))
