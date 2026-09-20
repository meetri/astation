"""The `/ws/events` stream, shared by the standalone app and the Hermes plugin."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from adapters.hermes import HermesAdapter, HermesError
from domain.event_stream import (
    _DESYNCHRONIZED_MESSAGE,
    _UPSTREAM_HEALTH_POLL_S,
    _UPSTREAM_LOST_MESSAGE,
    STREAM_DESYNCHRONIZED_EVENT_TYPE,
    EventBroadcaster,
    stream_ready_frame,
)
from domain.hermes_runtime import _ensure_connected

logger = logging.getLogger(__name__)


async def stream_events(websocket: WebSocket, app_state: Any) -> None:
    """Stream this subscriber's events until the client or the upstream goes."""
    await websocket.accept()
    adapter: HermesAdapter = app_state.hermes_adapter

    try:
        await _ensure_connected(app_state, adapter)
    except HermesError as exc:
        await websocket.send_json({"type": "error", "payload": {"message": str(exc)}})
        await websocket.close(code=1011)
        return

    broadcaster: EventBroadcaster = app_state.event_broadcaster
    async with broadcaster.subscribe(encoded=True) as queue:
        await websocket.send_json(
            stream_ready_frame(getattr(adapter, "connection_generation", None))
        )

        forward = asyncio.create_task(_forward_events(websocket, queue))
        disconnected = asyncio.create_task(_watch_for_disconnect(websocket))
        upstream_lost = asyncio.create_task(_watch_upstream(adapter))
        try:
            done, pending = await asyncio.wait(
                {forward, disconnected, upstream_lost},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                with contextlib.suppress(WebSocketDisconnect):
                    task.result()
            if upstream_lost in done and disconnected not in done:
                await _report_upstream_loss(websocket)
            elif forward in done and disconnected not in done:
                await _report_desynchronized(websocket)
        finally:
            logger.info("ws_events: connection finished")


async def _forward_events(websocket: WebSocket, queue: asyncio.Queue[Any]) -> None:
    """Send this subscriber's own copy of every canonical event."""
    while True:
        frame, text = await queue.get()
        await websocket.send_text(text)
        if frame.get("type") == STREAM_DESYNCHRONIZED_EVENT_TYPE:
            return


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    """Return as soon as the client goes away."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _watch_upstream(
    adapter: HermesAdapter, poll_seconds: float = _UPSTREAM_HEALTH_POLL_S
) -> None:
    """Return as soon as the gateway's own Hermes connection is no longer live."""
    while adapter.is_connected:
        await asyncio.sleep(poll_seconds)


async def _report_upstream_loss(websocket: WebSocket) -> None:
    """Tell this client the upstream died, then close."""
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "payload": {"message": _UPSTREAM_LOST_MESSAGE}})
        await websocket.close(code=1011)


async def _report_desynchronized(websocket: WebSocket) -> None:
    """Close a socket the gateway has stopped buffering for."""
    with contextlib.suppress(Exception):
        await websocket.send_json(
            {"type": "error", "payload": {"message": _DESYNCHRONIZED_MESSAGE}}
        )
        await websocket.close(code=1011)
