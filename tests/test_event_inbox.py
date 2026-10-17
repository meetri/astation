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
