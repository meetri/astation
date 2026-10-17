"""Loss-aware quarantine tests, no upstream or user data access."""
import asyncio

import pytest

from test_shared_profile_binding import state, raw
from domain.event_stream import EventBroadcaster
from domain.live_handles import LiveHandleCache


def setup_stream(state, **kwargs):
    owner = LiveHandleCache(state.hermes_adapter)
    state.shared_profile_handle_caches['gpt-astra'] = owner
    seen, controls = [], []
    stream = EventBroadcaster(state.hermes_adapter, state.live_handle_cache,
        profile_caches=lambda: state.shared_profile_handle_caches,
        on_canonical_event=lambda c, e: seen.append(e), **kwargs)
    original = stream._fan_out
    def collect(frame, generation):
        controls.append(frame)
        original(frame, generation)
    stream._fan_out = collect
    return stream, owner, seen, controls


def test_binding_recovery_without_another_event(state):
    async def run():
        stream, owner, seen, _ = setup_stream(state)
        stream._forward_one(raw(text='retained'))
        task = asyncio.create_task(stream._watch_generation(.001))
        try:
            owner.put('same-id', 'live-gpt-astra')
            for _ in range(50):
                if seen:
                    break
                await asyncio.sleep(.002)
            assert [e['payload']['text'] for e in seen] == ['retained']
            owner.put('same-id', 'live-gpt-astra')
            await asyncio.sleep(.005)
            assert len(seen) == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    asyncio.run(run())


@pytest.mark.parametrize('limit', ['frames', 'bytes', 'age'])
def test_bounded_quarantine_signals_resync_without_private_metadata(state, limit):
    options = {'quarantine_max_frames': 1} if limit == 'frames' else (
        {'quarantine_max_bytes': 1} if limit == 'bytes' else {'quarantine_max_age': 0})
    stream, owner, seen, controls = setup_stream(state, **options)
    stream._forward_one(raw(text='SECRET'))
    stream._forward_one(raw(text='SECRET2'))
    stream._replay_quarantine()
    assert seen == []
    resets = [f for f in controls if f['type'] == 'stream.reset']
    assert resets and resets[-1]['payload']['_must_resync'] is True
    assert all('SECRET' not in str(f) and 'same-id' not in str(f) for f in controls)
    assert len(stream._quarantine) <= 1 if limit == 'frames' else not stream._quarantine


def test_old_generation_quarantine_cannot_rebind_to_new_owner(state):
    stream, owner, seen, controls = setup_stream(state)
    stream._forward_one(raw(text='old private'))
    state.hermes_adapter.connection_generation += 1
    state.live_handle_cache.put('same-id', 'live-gpt-astra')
    stream._forward_one(raw('message.delta', text='current'))
    assert [f['payload']['text'] for f in seen] == ['current']
    assert any(f['type'] == 'stream.reset' for f in controls)


def test_conflict_does_not_become_trusted_when_one_cache_disappears(state):
    stream, owner, seen, controls = setup_stream(state)
    owner.put('same-id', 'live-gpt-astra')
    state.live_handle_cache.put('same-id', 'live-gpt-astra')
    stream._forward_one(raw(text='ambiguous private'))
    state.live_handle_cache.clear()
    stream._forward_one(raw('message.delta', text='new valid'))
    assert [f['payload']['text'] for f in seen] == ['new valid']
    assert any(f['type'] == 'stream.reset' for f in controls)


def test_replay_preserves_order_identity_and_ignores_claimed_profile(state):
    stream, owner, seen, controls = setup_stream(state)
    for kind, text in [('message.delta', 'first'), ('message.complete', 'last')]:
        stream._forward_one(raw(kind, text=text, _profile='default', profile='default'))
    assert seen == [] and stream._last_seq == 0
    owner.put('same-id', 'live-gpt-astra')
    stream._replay_quarantine()
    stream._replay_quarantine()
    assert [f['payload']['text'] for f in seen] == ['first', 'last']
    assert [f['seq'] for f in seen] == [1, 2]
    assert all(f['payload']['_profile'] == 'gpt-astra' for f in seen)


def test_gap_is_advertised_to_later_subscribers(state, monkeypatch):
    stream, owner, seen, controls = setup_stream(state, quarantine_max_bytes=1)
    stream._forward_one(raw(text='lost'))
    assert not stream.replay_available(0)
    monkeypatch.setattr(stream, 'start', lambda: None)
    async def subscribe():
        async with stream.subscribe(after=0) as queue:
            assert queue.get_nowait()['type'] == 'stream.reset'
    asyncio.run(subscribe())


def test_recovered_completion_is_captured_once_before_fanout(state, monkeypatch):
    from api.bootstrap import wire_event_stream
    from domain.run_recorder import RunRecorder
    from types import SimpleNamespace
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    owner = LiveHandleCache(state.hermes_adapter)
    state.shared_profile_handle_caches['gpt-astra'] = owner
    stream = wire_event_stream(state, state.hermes_adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    stream._forward_one(raw(text='recovered'))
    assert state.chat_store.page(profile='gpt-astra', stored_session_id='same-id') == []
    owner.put('same-id', 'live-gpt-astra')
    stream._replay_quarantine()
    stream._replay_quarantine()
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert len(rows) == 1 and rows[0]['text'] == 'recovered'
    assert stream._history[-1][1]['payload']['chat_row_id'] == rows[0]['id']
    assert state.chat_store.page(profile='default', stored_session_id='same-id') == []


def test_stale_resume_response_does_not_install_current_binding(state):
    from domain.hermes_runtime import _resume_for_live_id
    from adapters.hermes import HermesConnectionError
    async def stale(stored_id, **kwargs):
        state.hermes_adapter.connection_generation += 1
        return {'session_id': 'live-gpt-astra'}
    state.hermes_adapter.resume_session = stale
    with pytest.raises(HermesConnectionError):
        asyncio.run(_resume_for_live_id(state.hermes_adapter, 'same-id', state.live_handle_cache))
    assert state.live_handle_cache.get('same-id') is None


def test_cached_binding_from_another_adapter_is_not_ownership(state):
    from types import SimpleNamespace
    stream, owner, seen, _ = setup_stream(state)
    other = LiveHandleCache(SimpleNamespace(connection_generation=1))
    other.put('same-id', 'live-gpt-astra')
    state.shared_profile_handle_caches['other'] = other
    stream._forward_one(raw(text='private'))
    assert seen == []


def test_rpc_response_from_replaced_connection_is_not_accepted():
    import itertools
    from adapters.hermes import HermesConnectionError
    from adapters.hermes.client import HermesAdapter
    class Socket:
        async def send(self, text):
            adapter._connection_generation += 1
            next(iter(adapter._pending.values())).set_result({'result': {'session_id': 'stale'}})
    adapter = HermesAdapter.__new__(HermesAdapter)
    adapter._connection_generation = 1
    adapter._ws = Socket()
    adapter._ws_live = True
    adapter._id_counter = itertools.count(1)
    adapter._pending = {}
    with pytest.raises(HermesConnectionError) as exc:
        asyncio.run(adapter.session_create('test'))
    assert exc.value.request_was_sent is True


def test_transport_queue_keeps_original_generation(state):
    from adapters.hermes.client import HermesAdapter
    adapter = HermesAdapter.__new__(HermesAdapter)
    adapter._connection_generation = 1
    adapter._event_queue = asyncio.Queue()
    event = raw(text='old')
    # A raw server field must never be accepted as transport provenance.
    event['_adapter_generation'] = 99
    adapter._enqueue_event(event)
    queued = adapter._event_queue.get_nowait()
    assert queued['_adapter_generation'] == 1
    stream, owner, seen, controls = setup_stream(state)
    state.hermes_adapter.connection_generation = 2
    owner.put('same-id', 'live-gpt-astra')
    stream._forward_one(queued)
    assert seen == []
    assert any(f['type'] == 'stream.reset' for f in controls)


def test_quarantine_copies_mutable_input(state):
    stream, owner, seen, _ = setup_stream(state)
    event = raw(text='original', nested={'value': 'original'})
    stream._forward_one(event)
    event['params']['nested']['value'] = 'changed'
    owner.put('same-id', 'live-gpt-astra')
    stream._replay_quarantine()
    assert seen[0]['payload']['nested']['value'] == 'original'
