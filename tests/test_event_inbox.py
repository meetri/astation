import os
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from domain.event_inbox import EventInbox, EventInboxFull


def envelope(event_id, text='x'):
    return {'event_id': event_id, 'payload': {'text': text}}


def test_existing_file_permissions_are_restricted(tmp_path):
    path = tmp_path / 'inbox.sqlite'
    path.touch(mode=0o666)
    os.chmod(path, 0o666)
    EventInbox(path)
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_row_and_byte_quotas_reject_without_deleting_pending_data(tmp_path):
    rows = EventInbox(tmp_path / 'rows.sqlite', max_rows=1, max_bytes=10_000)
    rows.put('old', envelope('one'), 'message.complete')
    with pytest.raises(EventInboxFull):
        rows.put('new', envelope('two'), 'message.complete')
    assert rows.counts() == {'pending': 1, 'current_epoch': 0, 'stranded_old_epoch': 1, 'bytes': rows.counts()['bytes']}

    sized = EventInbox(tmp_path / 'bytes.sqlite', max_rows=10, max_bytes=1)
    with pytest.raises(EventInboxFull):
        sized.put('now', envelope('large', 'PRIVATE'), 'message.complete')
    assert sized.counts()['pending'] == 0


def test_pending_reads_are_paginated_and_counts_include_old_epochs(tmp_path):
    inbox = EventInbox(tmp_path / 'inbox.sqlite', page_size=2)
    for i in range(5):
        inbox.put('current' if i > 0 else 'old', envelope(str(i)), 'message.complete')
    assert [row[0]['event_id'] for row in inbox.pending('current')] == ['1', '2']
    assert [row[0]['event_id'] for row in inbox.pending('current', after_id='2')] == ['3', '4']
    assert inbox.counts('current')['stranded_old_epoch'] == 1


def test_plugin_health_exposes_counts_but_no_private_payload(tmp_path, monkeypatch):
    path = tmp_path / 'inbox.sqlite'
    inbox = EventInbox(path)
    inbox.put('old', envelope('private-id', 'PRIVATE TEXT'), 'message.complete')
    inbox.put('current', envelope('current-id', 'MORE PRIVATE'), 'message.complete')
    spec = importlib.util.spec_from_file_location(
        'plugin_api_inbox_health', Path(__file__).parents[1] / 'dashboard' / 'plugin_api.py'
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    adapter = SimpleNamespace(event_recovery_path=path, event_identity_epoch='current')
    monkeypatch.setattr(api, '_hermes_app', lambda: SimpleNamespace(
        state=SimpleNamespace(hermes_adapter=adapter)
    ))
    result = api._event_inbox_stats()
    assert result['pending'] == 2
    assert result['stranded_old_epoch'] == 1
    assert 'PRIVATE' not in str(result) and 'private-id' not in str(result)


def test_inbox_runs_in_wal_with_normal_sync_and_keeps_siblings_private(tmp_path):
    path = tmp_path / 'inbox.sqlite'
    inbox = EventInbox(path)
    inbox.put('e', envelope('one', 'PRIVATE'), 'message.complete')
    with inbox.connect() as db:
        assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        assert db.execute('PRAGMA synchronous').fetchone()[0] == 1  # NORMAL
    for sibling in (tmp_path / 'inbox.sqlite-wal', tmp_path / 'inbox.sqlite-shm'):
        assert sibling.exists()
        assert os.stat(sibling).st_mode & 0o777 == 0o600


def test_inbox_reuses_one_connection(tmp_path):
    inbox = EventInbox(tmp_path / 'inbox.sqlite')
    with inbox.connect() as first:
        pass
    with inbox.connect() as second:
        pass
    assert first is second


def test_inbox_is_safe_to_use_from_worker_threads(tmp_path):
    import threading
    inbox = EventInbox(tmp_path / 'inbox.sqlite')
    errors = []

    def work(worker):
        try:
            for i in range(50):
                key = f'{worker}-{i}'
                inbox.put('e', envelope(key), 'message.complete')
                inbox.bind(envelope(key))
                inbox.acknowledge(key)
                inbox.counts('e')
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert inbox.counts('e')['pending'] == 0


def test_inbox_survives_close_and_reopens_lazily(tmp_path):
    path = tmp_path / 'inbox.sqlite'
    inbox = EventInbox(path)
    inbox.put('e', envelope('kept'), 'message.complete')
    inbox.close()
    inbox.close()  # idempotent
    assert inbox.counts('e')['pending'] == 1
    other = EventInbox(path)  # a second handle (health stats) sees the same committed rows
    assert other.counts('e')['pending'] == 1
    other.close()


def test_inbox_discards_a_connection_that_raised_a_database_error(tmp_path):
    import sqlite3
    inbox = EventInbox(tmp_path / 'inbox.sqlite')
    with pytest.raises(sqlite3.OperationalError):
        with inbox.connect() as db:
            db.execute('SELECT * FROM table_that_does_not_exist')
    inbox.put('e', envelope('after'), 'message.complete')  # reopens a fresh connection
    assert inbox.counts('e')['pending'] == 1
