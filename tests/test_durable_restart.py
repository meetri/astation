"""A proven owner must survive a host restart before durable capture."""
from types import SimpleNamespace
from test_durable_identity import state, raw
from domain.event_stream import EventBroadcaster
from api.bootstrap import wire_event_stream
from domain.run_recorder import RunRecorder


def test_proven_identity_survives_host_restart_before_capture(state, monkeypatch, tmp_path):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'old-host'
    adapter.resolve_event_identity = lambda stored, live: ('same-id', 'live-a', 'gpt-astra')
    recorder = RunRecorder(state.db_sessions)
    noop = SimpleNamespace(handle_completed=lambda e: None)
    first = wire_event_stream(state, adapter, state.live_handle_cache, noop, recorder)
    real_capture = state.chat_store.capture_event
    state.chat_store.capture_event = lambda *args: (_ for _ in ()).throw(OSError('simulated DB unavailable'))
    first._forward_one(raw(text='proven before crash', stored=None))
    assert first._inbox.counts()['pending'] == 1
    state.chat_store.capture_event = real_capture
    adapter.event_identity_epoch = 'new-host'
    # The old registry no longer exists; never infer ownership from new handles.
    adapter.resolve_event_identity = lambda stored, live: None
    second = wire_event_stream(state, adapter, state.live_handle_cache, noop, recorder)
    second._replay_quarantine()
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [r['text'] for r in rows] == ['proven before crash']
    assert not state.chat_store.page(profile='default', stored_session_id='same-id')
    assert second._inbox.counts()['pending'] == 0


def test_run_persistence_failure_is_not_acknowledged(state, monkeypatch, tmp_path):
    from sqlalchemy import select
    from domain.models import RunEvent
    import domain.run_recorder as recorder_module
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'host'
    adapter.resolve_event_identity = lambda stored, live: ('same-id', 'live-a', 'gpt-astra')
    b = wire_event_stream(state, adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    persist = recorder_module.persist_run_event
    monkeypatch.setattr(recorder_module, 'persist_run_event', lambda *a, **kw: (_ for _ in ()).throw(OSError('DB transient')))
    b._forward_one(raw(text='retained despite run store failure'))
    assert b._inbox.counts()['pending'] == 1
    monkeypatch.setattr(recorder_module, 'persist_run_event', persist)
    b._replay_quarantine()
    assert b._inbox.counts()['pending'] == 0
    assert len(state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')) == 1
    with state.db_sessions() as db:
        assert len(list(db.scalars(select(RunEvent)))) == 1
