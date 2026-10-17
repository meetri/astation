"""Private durable inbox for unbound host events, separate from chat history.

No TTL/deletion on disconnect: only successful canonical capture acknowledges
an entry. A new server incarnation cannot bind an old live handle. Old-epoch
entries remain available for explicitly authorized recovery, never profile guesses.
"""
from contextlib import contextmanager
import json
import os
import sqlite3
from pathlib import Path


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
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS pending (id TEXT PRIMARY KEY, epoch TEXT NOT NULL, envelope TEXT NOT NULL, raw_type TEXT)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
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
