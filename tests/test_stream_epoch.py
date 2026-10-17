from test_durable_identity import EventBroadcaster
from test_event_stream_replay import _FakeAdapter, _envelope


def test_restart_cursor_requires_same_epoch_even_when_sequences_overlap():
    old = EventBroadcaster(_FakeAdapter())
    old.inject(_envelope('old'), profile='default')
    new = EventBroadcaster(_FakeAdapter())
    new.inject(_envelope('new'), profile='default')
    assert old.stream_epoch != new.stream_epoch
    assert new.resume_cursor(1, old.stream_epoch) == (None, True)
    assert new.resume_cursor(1, None) == (None, True)
    assert new.resume_cursor(1, new.stream_epoch) == (1, False)
    assert new.resume_cursor(999, new.stream_epoch) == (None, True)
    assert new.resume_cursor(None, None) == (None, False)
    assert new._history[-1][1]['stream_epoch'] == new.stream_epoch


def test_future_cursor_is_not_a_successful_replay():
    stream = EventBroadcaster(_FakeAdapter())
    assert not stream.replay_available(999)


import json
import pytest
from types import SimpleNamespace
from api.events_ws import stream_events


@pytest.mark.asyncio
@pytest.mark.parametrize('matching', [False, True])
async def test_websocket_ready_exposes_validated_epoch_before_events(matching):
    adapter = _FakeAdapter()
    b = EventBroadcaster(adapter)
    b.inject(_envelope('one'), profile='default')
    class Socket:
        query_params = {'after': '1', 'epoch': b.stream_epoch if matching else 'previous-process'}
        frames = []
        async def accept(self): pass
        async def send_json(self, frame): self.frames.append(frame)
        async def send_text(self, text): self.frames.append(json.loads(text))
        async def receive(self): return {'type': 'websocket.disconnect'}
    socket = Socket()
    try:
        await stream_events(socket, SimpleNamespace(hermes_adapter=adapter, event_broadcaster=b))
        ready = socket.frames[0]
        assert ready['type'] == 'stream.ready'
        assert ready['payload']['stream_epoch'] == b.stream_epoch
        assert ready['payload']['cursor_reset'] is (not matching)
        assert ready['payload']['replay'] is matching
        assert ready['payload']['last_seq'] == 1
    finally:
        await b.close()
