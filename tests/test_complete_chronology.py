from types import SimpleNamespace
from test_durable_identity import state, raw
from api.bootstrap import wire_event_stream
from domain.event_stream import EventBroadcaster
from domain.run_recorder import RunRecorder


def test_complete_canonical_tool_marker_subagent_chronology(state, monkeypatch, tmp_path):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    adapter = state.hermes_adapter
    adapter.event_recovery_path = tmp_path / 'pending.sqlite'
    adapter.event_identity_epoch = 'diagnostic'
    adapter.resolve_event_identity = lambda stored, live: ('same-id', 'live-a', 'gpt-astra')
    b = wire_event_stream(state, adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    state.chat_store.capture_submitted_user_row(profile='gpt-astra', stored_session_id='same-id', text='prompt')
    for kind, payload in [
        ('tool.start', {'tool_id': 't1', 'name': 'terminal', 'args': {'command': 'diagnostic fixture only'}}),
        ('tool.complete', {'tool_id': 't1', 'name': 'terminal', 'result': 'fixture output'}),
        ('subagent.start', {'subagent_id': 'child', 'goal': 'fixture child'}),
        ('subagent.complete', {'subagent_id': 'child', 'summary': 'fixture summary'}),
    ]:
        b._forward_one({'type': kind, 'payload': {'session_id': 'live-a', **payload}})
    b._forward_one({'type': 'status.update', 'payload': {'session_id': 'live-a', 'kind': 'compacted', 'text': 'Context compacted'}})
    b._forward_one(raw(text='reply without upstream completion ID', stored=None))
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [r['role'] for r in rows] == ['user', 'tool', 'subagent', 'subagent', 'marker', 'assistant']
    assert len({r['id'] for r in rows}) == len(rows)
    assert [r['seq'] for r in rows] == sorted(r['seq'] for r in rows)
    assert state.chat_store.page(profile='default', stored_session_id='same-id') == []
    assert b._inbox.counts()['pending'] == 0
