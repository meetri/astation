"""`?after=<seq>` replay on `/ws/events`: seq numbering, replay, and reset-on-gap."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Import with `gateway/` on the path, then undo both the path and module-cache
# changes: other test files (test_capture.py) rely on `events` staying unimportable.
_gateway_dir = str(Path(__file__).resolve().parents[1] / "gateway")
_modules_before = set(sys.modules)
sys.path.insert(0, _gateway_dir)
try:
    from domain.event_stream import STREAM_RESET_EVENT_TYPE, EventBroadcaster
finally:
    sys.path.remove(_gateway_dir)
    for _name in list(sys.modules):
        if _name not in _modules_before:
            del sys.modules[_name]


class _FakeAdapter:
    """Just enough of `HermesAdapter` for the broadcaster's pump to idle quietly."""

    connection_generation = None
    is_connected = True

    async def events(self):
        while True:
            await asyncio.sleep(3600)
            yield {}  # pragma: no cover - never reached; makes this an async generator


@pytest_asyncio.fixture
async def broadcaster():
    instance = EventBroadcaster(_FakeAdapter(), history_max_frames=5)
    try:
        yield instance
    finally:
        await instance.close()


def _envelope(text: str) -> dict:
    return {
        "event_id": f"evt_{text}",
        "project_id": None,
        "session_id": None,
        "run_id": None,
        "type": "message.completed",
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {"text": text},
    }


async def _drain(queue: asyncio.Queue, count: int) -> list[dict]:
    frames = []
    for _ in range(count):
        frame, _text = await asyncio.wait_for(queue.get(), timeout=1)
        frames.append(frame)
    return frames


@pytest.mark.asyncio
async def test_seq_is_monotonic(broadcaster):
    for i in range(3):
        broadcaster.inject(_envelope(str(i)), profile="default")
    seqs = [seq for seq, _frame, _text in broadcaster._history]
    assert seqs == sorted(seqs)
    assert seqs == list(range(1, 4))


@pytest.mark.asyncio
async def test_replay_after_n(broadcaster):
    for i in range(5):
        broadcaster.inject(_envelope(str(i)), profile="default")

    async with broadcaster.subscribe(encoded=True, after=2) as queue:
        frames = await _drain(queue, 3)

    assert [f["seq"] for f in frames] == [3, 4, 5]
    assert [f["payload"]["text"] for f in frames] == ["2", "3", "4"]


@pytest.mark.asyncio
async def test_reset_when_after_too_old(broadcaster):
    # Buffer only holds 5 frames; push past that so seq 1 falls out of history.
    for i in range(8):
        broadcaster.inject(_envelope(str(i)), profile="default")

    async with broadcaster.subscribe(encoded=True, after=1) as queue:
        frame, _text = await asyncio.wait_for(queue.get(), timeout=1)

    assert frame["type"] == STREAM_RESET_EVENT_TYPE
    assert queue.empty()


@pytest.mark.asyncio
async def test_no_after_is_unchanged(broadcaster):
    for i in range(3):
        broadcaster.inject(_envelope(str(i)), profile="default")

    async with broadcaster.subscribe(encoded=True) as queue:
        # Nothing buffered is replayed; the socket only sees frames sent while subscribed.
        assert queue.empty()
        broadcaster.inject(_envelope("live"), profile="default")
        frame, _text = await asyncio.wait_for(queue.get(), timeout=1)

    assert frame["payload"]["text"] == "live"
    assert frame["seq"] == 4
