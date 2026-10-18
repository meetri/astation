"""The per-event persistence stage runs off the event loop without changing what clients see.

Every streamed event used to commit to SQLite on the asyncio loop that serves the dashboard, so a burst
froze every request behind it. These tests pin the three things that move must not break: the loop stays
responsive, event order and `seq` stay monotonic (including against `inject()`), and anything that needs
the loop (resync frames, task scheduling) still lands on it.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from test_shared_profile_binding import raw, state  # noqa: F401  (state is a fixture)

from api.bootstrap import wire_event_stream
from domain import capture_executor
from domain.capture_executor import call_on_loop, run_capture
from domain.profile_connection import ProfileConnection, ProfileConnectionManager
from domain.run_recorder import RunRecorder

SLOW_S = 0.15


class FakeEvents:
    """An adapter event feed the test pushes into and can end."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    async def __call__(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event


def wire(state, feed):
    adapter = state.hermes_adapter
    adapter.events = feed
    adapter.resolve_event_identity = lambda stored, live: ('same-id', live, 'gpt-astra')
    return wire_event_stream(state, adapter, state.live_handle_cache,
                             SimpleNamespace(handle_completed=lambda e: None),
                             RunRecorder(state.db_sessions))


def slow_capture(state, seen):
    real = state.chat_store.capture_event

    def capture(profile, canonical, envelope):
        seen.append(threading.current_thread().name)
        time.sleep(SLOW_S)  # stands in for a commit waiting on a busy disk
        return real(profile, canonical, envelope)

    state.chat_store.capture_event = capture


async def worst_loop_gap(duration_s):
    """Longest time the loop went without getting to run a 10 ms ticker."""
    worst, last, end = 0.0, time.perf_counter(), time.perf_counter() + duration_s
    while time.perf_counter() < end:
        await asyncio.sleep(0.01)
        now = time.perf_counter()
        worst, last = max(worst, now - last), now
    return worst


async def settle(broadcaster, feed, count):
    for _ in range(300):
        if len(broadcaster._history) >= count and feed.queue.empty():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'only {len(broadcaster._history)} of {count} frames were published')


@pytest.mark.asyncio
async def test_loop_stays_responsive_while_capture_blocks(state):
    feed, seen = FakeEvents(), []
    slow_capture(state, seen)
    broadcaster = wire(state, feed)
    for i in range(4):
        feed.queue.put_nowait(raw(text=f'reply {i}'))

    gap = await worst_loop_gap(4 * SLOW_S + 0.2)
    await settle(broadcaster, feed, 4)

    assert gap < SLOW_S / 2  # inline, the loop would have been dark for the full SLOW_S
    assert seen and all(name.startswith('astation-capture') for name in seen)
    await broadcaster.close()


@pytest.mark.asyncio
async def test_kill_switch_restores_inline_behaviour(state, monkeypatch):
    monkeypatch.setenv('ASTATION_CAPTURE_THREAD', '0')
    feed, seen = FakeEvents(), []
    slow_capture(state, seen)
    broadcaster = wire(state, feed)
    for i in range(2):
        feed.queue.put_nowait(raw(text=f'reply {i}'))

    gap = await worst_loop_gap(2 * SLOW_S + 0.2)
    await settle(broadcaster, feed, 2)

    assert gap >= SLOW_S * 0.9  # the loop was blocked, as before the change
    assert set(seen) == {threading.main_thread().name}
    await broadcaster.close()


@pytest.mark.asyncio
async def test_order_and_seq_hold_with_inject_interleaved(state):
    feed = FakeEvents()
    broadcaster = wire(state, feed)
    total = 30
    for i in range(total):
        feed.queue.put_nowait(raw(text=f'reply {i:02d}'))
        if i % 3 == 0:  # another profile's frame arriving on the loop mid-stream
            broadcaster.inject({'type': 'message.delta', 'event_id': f'inj-{i}',
                                'payload': {'text': f'inj {i}'}}, profile='other')
            await asyncio.sleep(0)

    expected = total + len(range(0, total, 3))
    await settle(broadcaster, feed, expected)

    seqs = [seq for seq, _frame, _text in broadcaster._history]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [r['text'] for r in rows] == [f'reply {i:02d}' for i in range(total)]
    await broadcaster.close()


@pytest.mark.asyncio
async def test_resync_requested_from_the_capture_thread_lands_on_the_loop(state):
    broadcaster = wire(state, FakeEvents())
    threads = []
    real_fan_out = broadcaster._fan_out
    broadcaster._fan_out = lambda *a, **k: (threads.append(threading.current_thread()), real_fan_out(*a, **k))

    await run_capture(broadcaster._identity_resync, 'identity_conflict')

    assert broadcaster._identity_gap is True
    assert threads and all(t is threading.main_thread() for t in threads)
    await broadcaster.close()


@pytest.mark.asyncio
async def test_call_on_loop_runs_inline_on_the_loop_and_queues_from_other_threads():
    ran = []
    where = []

    await run_capture(lambda: None)  # binds the loop

    call_on_loop(ran.append, 'inline')
    assert ran == ['inline']  # already on the loop: no hop

    def from_worker():
        where.append(threading.current_thread().name)
        call_on_loop(lambda: ran.append(threading.current_thread()))

    await run_capture(from_worker)
    await asyncio.sleep(0)
    assert where[0].startswith('astation-capture')
    assert ran[-1] is threading.main_thread()


@pytest.mark.asyncio
async def test_staged_stage_runs_in_order_and_serially(state):
    """One capture thread: stages never overlap, so recorder/chat-store state stays single-writer."""
    active, overlaps, order = 0, [], []
    lock = threading.Lock()

    def stage(n):
        nonlocal active
        with lock:
            active += 1
            overlaps.append(active)
        time.sleep(0.01)
        order.append(n)
        with lock:
            active -= 1

    await asyncio.gather(*(run_capture(stage, n) for n in range(8)))
    assert max(overlaps) == 1
    assert order == list(range(8))


@pytest.mark.asyncio
async def test_context_variables_reach_the_capture_thread():
    import contextvars
    var = contextvars.ContextVar('v', default='unset')
    var.set('from-loop')
    assert await run_capture(var.get) == 'from-loop'


@pytest.mark.asyncio
async def test_profile_pump_captures_off_loop_and_publishes_on_loop(state):
    published, threads = [], []
    captured = []

    def hook(canonical, envelope, profile):
        captured.append(threading.current_thread().name)
        time.sleep(SLOW_S)

    broadcaster = SimpleNamespace(
        inject=lambda envelope, profile: (published.append(envelope), threads.append(threading.current_thread())))
    manager = ProfileConnectionManager(base_adapter=state.hermes_adapter, broadcaster=broadcaster,
                                       chat_store=state.chat_store, on_canonical_event=hook)
    conn = ProfileConnection(profile='gpt-astra', adapter=state.hermes_adapter, is_default=False,
                             live_handle_cache=state.live_handle_cache)

    work = asyncio.create_task(manager._forward_one_staged(conn, raw(text='from another profile')))
    gap = await worst_loop_gap(SLOW_S + 0.1)
    await work

    assert gap < SLOW_S / 2
    assert captured and captured[0].startswith('astation-capture')
    assert len(published) == 1 and threads == [threading.main_thread()]
    assert [r['text'] for r in state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')] \
        == ['from another profile']


def test_executor_switch_defaults_on(monkeypatch):
    monkeypatch.delenv('ASTATION_CAPTURE_THREAD', raising=False)
    assert capture_executor.enabled() is True
    for off in ('0', 'false', 'OFF', ' no '):
        monkeypatch.setenv('ASTATION_CAPTURE_THREAD', off)
        assert capture_executor.enabled() is False
