"""Release-blocking delivery probes: no user DB or upstream network access.

Unknown private frames must not be published under a guessed owner. Once a
trusted explicit binding arrives, however, a legitimate completion must be
recoverable rather than discarded forever. These tests intentionally remain
red until that recovery contract exists.
"""
import pytest

from test_shared_profile_binding import state, raw
from domain.event_stream import EventBroadcaster
from domain.live_handles import LiveHandleCache


@pytest.mark.parametrize('scenario', ['external-session', 'cold-start', 'reconnect'])
def test_legitimate_completion_survives_binding_gap(state, scenario):
    adapter = state.hermes_adapter
    owner = LiveHandleCache(adapter)
    state.shared_profile_handle_caches['gpt-astra'] = owner
    if scenario == 'reconnect':
        owner.put('same-id', 'live-gpt-astra')
    seen = []
    broadcaster = EventBroadcaster(
        adapter, state.live_handle_cache,
        profile_caches=lambda: state.shared_profile_handle_caches,
        on_canonical_event=lambda canonical, envelope: seen.append(envelope),
    )
    if scenario == 'reconnect':
        adapter.connection_generation += 1
    # External completion or a frame racing the successful resume response.
    broadcaster._forward_one(raw(text='legitimate completion'))
    assert seen == [], 'unbound content must not escape under a guessed owner'
    # Simulate the successful profile-specific resume result. This is the same
    # public cache write the real _resume_for_live_id path uses.
    owner.put('same-id', 'live-gpt-astra')
    broadcaster._forward_one(raw('message.delta', text='next turn'))
    assert any(frame['payload'].get('text') == 'next turn' for frame in seen)
    completions = [frame for frame in seen
                   if frame['payload'].get('text') == 'legitimate completion']
    assert len(completions) == 1, 'legitimate completion was permanently dropped'
    assert completions[0]['payload']['_profile'] == 'gpt-astra'
