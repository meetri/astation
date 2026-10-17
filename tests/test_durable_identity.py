"""Real file-backed capture: unknown completions must survive reconnect and delay."""
import asyncio
from types import SimpleNamespace
from test_shared_profile_binding import state, raw
from api.bootstrap import wire_event_stream
from domain.event_stream import EventBroadcaster
from domain.run_recorder import RunRecorder


def test_authoritative_owner_without_resume_cache(state, monkeypatch):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    # The in-process server is authoritative, not a client-provided profile string.
    state.hermes_adapter.resolve_event_identity = lambda stored, live: ('same-id', live, 'gpt-astra')
    b = wire_event_stream(state, state.hermes_adapter, state.live_handle_cache,
                         SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    b._forward_one(raw(text='cold completion'))
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [r['text'] for r in rows] == ['cold completion']
    assert b._history[-1][1]['payload']['chat_row_id'] == rows[0]['id']
    assert state.chat_store.page(profile='default', stored_session_id='same-id') == []


def test_host_registry_exact_live_key_and_profile_home(monkeypatch):
    import sys, threading
    monkeypatch.setitem(sys.modules, 'hermes_constants', SimpleNamespace(
        profile_name_for_home=lambda home: home.rsplit('/', 1)[-1],
        get_routing_process_hermes_home=lambda: '/isolated/profiles/default',
    ))
    from domain.host_identity import resolve_host_identity
    host = SimpleNamespace(_sessions_lock=threading.Lock(), _sessions={
        'live-a': {'session_key': 'same-id', 'profile_home': '/isolated/profiles/gpt-astra'},
        'live-b': {'session_key': 'same-id', 'profile_home': '/isolated/profiles/default'},
    }, _session_lookup_key=lambda s, **kw: s['session_key'])
    assert resolve_host_identity(host, 'same-id', 'live-a') == ('same-id', 'live-a', 'gpt-astra')
    assert resolve_host_identity(host, 'same-id', None) is None
    assert resolve_host_identity(host, 'wrong', 'live-a') is None
    assert resolve_host_identity(host, 'same-id', 'unknown') is None


def test_unbound_survives_restart_delay_and_socket_reconnect(state, monkeypatch, tmp_path):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'server-boot-1'
    adapter.resolve_event_identity = lambda stored, live: None
    def build():
        return wire_event_stream(state, adapter, state.live_handle_cache,
            SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    b = build()
    b._quarantine_max_age = 0
    b._forward_one(raw(text='retained on disk'))
    asyncio.run(b.close())
    adapter.connection_generation += 1
    b = build()
    adapter.resolve_event_identity = lambda stored, live: ('same-id', live, 'gpt-astra')
    b._replay_quarantine()  # also called by the timer; no new upstream event needed
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [r['text'] for r in rows] == ['retained on disk']
    assert b._history[-1][1]['payload']['chat_row_id'] == rows[0]['id']
    b._replay_quarantine()
    assert len(state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')) == 1


def test_retry_after_capture_before_ack_is_idempotent(state, monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'server-boot-1'
    adapter.resolve_event_identity = lambda stored, live: None
    b = wire_event_stream(state, adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    b._forward_one(raw(text='retained on disk'))
    adapter.resolve_event_identity = lambda stored, live: ('same-id', live, 'gpt-astra')
    ack = b._inbox.acknowledge
    def crash(_):
        raise OSError('simulated crash before inbox acknowledgment')
    b._inbox.acknowledge = crash
    b._replay_quarantine()
    assert b._inbox.counts()['pending'] == 1
    original = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')[0]['id']
    state.chat_store.capture_submitted_user_row(profile='gpt-astra', stored_session_id='same-id', text='next user')
    b._inbox.acknowledge = ack
    b._replay_quarantine()
    b._replay_quarantine()  # next bounded timer pass wraps the pagination cursor
    assert b._inbox.counts()['pending'] == 0
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [r['text'] for r in rows] == ['retained on disk', 'next user']
    assert b._history[-1][1]['payload']['chat_row_id'] == original
    with state.db_sessions() as db:
        from sqlalchemy import select
        from domain.models import RunEvent
        assert len(list(db.scalars(select(RunEvent)))) == 1


def test_host_resolver_is_authoritative_even_when_event_has_no_ids(state, monkeypatch, tmp_path):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'server-boot-1'
    calls = []
    adapter.resolve_event_identity = lambda stored, live: calls.append((stored, live)) or None
    b = wire_event_stream(state, adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    b._forward_one(raw(text='PRIVATE', live=None, stored=None))
    assert calls == [(None, None)]
    assert not b._history
    assert state.chat_store.page(profile='default', stored_session_id='same-id') == []
    assert b._inbox.counts()['pending'] == 1


def test_fresh_identical_no_id_completions_are_not_text_deduplicated(state, monkeypatch):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.resolve_event_identity = lambda stored, live: ('same-id', 'live-owner', 'gpt-astra')
    b = wire_event_stream(state, adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    b._forward_one(raw(text='same answer', live=None, stored=None))
    b._forward_one(raw(text='same answer', live=None, stored=None))
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [row['text'] for row in rows] == ['same answer', 'same answer']
    assert rows[0]['id'] != rows[1]['id']


def test_resolver_failure_signals_generic_resync_and_keeps_durable_row(state, monkeypatch, tmp_path):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'server-boot-1'
    adapter.resolve_event_identity = lambda stored, live: None
    b = wire_event_stream(state, adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    b._forward_one(raw(text='PRIVATE'))
    controls = []
    b._fan_out = lambda frame, generation: controls.append(frame)
    def broken(*_):
        raise RuntimeError('PRIVATE resolver detail')
    adapter.resolve_event_identity = broken
    b._replay_quarantine()
    assert b._inbox.counts()['pending'] == 1
    assert controls[-1]['type'] == 'stream.reset'
    assert 'PRIVATE' not in str(controls[-1])
    assert controls[-1]['payload']['_reason'] == 'identity_resolution_failed'


def test_inbox_write_failure_signals_generic_resync(state, monkeypatch, tmp_path):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'server-boot-1'
    adapter.resolve_event_identity = lambda stored, live: None
    b = wire_event_stream(state, adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    controls = []
    b._fan_out = lambda frame, generation: controls.append(frame)
    b._inbox.put = lambda *_: (_ for _ in ()).throw(OSError('PRIVATE disk detail'))
    b._forward_one(raw(text='PRIVATE'))
    assert controls[-1]['payload']['_reason'] == 'identity_inbox_unavailable'
    assert 'PRIVATE' not in str(controls[-1])
