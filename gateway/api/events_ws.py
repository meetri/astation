"""The `/ws/events` stream, shared by the standalone app and the Hermes plugin.

Extracted from `api/main.py` unchanged so there is exactly ONE implementation of
the event stream. The two callers differ only in how they authenticate:

  * `api/main.py` (the standalone gateway) checks the gateway's own HTTP Basic
    credential, which its native client can set on a WebSocket upgrade.
  * The Hermes plugin cannot. Hermes's four auth gates are all
    `@app.middleware("http")` and Starlette never runs HTTP middleware for a
    websocket scope, so a plugin-mounted socket is otherwise reachable with no
    credential at all. The plugin calls Hermes's own `_ws_auth_ok` /
    `_ws_request_is_allowed` by hand and fails closed.

Everything after the auth decision -- subscribe, the `stream.ready` frame, the
three racing tasks, and the two ways this socket can be closed from our side --
is identical, and lives here.
"""

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
    """Stream this subscriber's events until the client or the upstream goes.

    The caller has already decided the client is authorized and has NOT yet
    accepted the socket. `app_state` carries the gateway services; under the
    plugin that is Hermes's own `app.state`, which the plugin populates.
    """
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
        # B-28: say "you are subscribed" before anything else, so the client
        # can distinguish a live-but-quiet stream from one that never came
        # back. Sent from inside `subscribe()` so no upstream frame can
        # overtake it or be lost behind it. It names the Hermes connection
        # generation, which is the generation every live handle this
        # client goes on to hold belongs to.
        await websocket.send_json(
            stream_ready_frame(getattr(adapter, "connection_generation", None))
        )

        forward = asyncio.create_task(_forward_events(websocket, queue))
        disconnected = asyncio.create_task(_watch_for_disconnect(websocket))
        # B-27: this socket staying open says nothing about the gateway's own
        # socket to Hermes. Watch that too, or an upstream death is silence.
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
                # Surface a genuine bug; a client hanging up is not one.
                with contextlib.suppress(WebSocketDisconnect):
                    task.result()
            if upstream_lost in done and disconnected not in done:
                await _report_upstream_loss(websocket)
            elif forward in done and disconnected not in done:
                # B-33: `_forward_events` only returns when it has sent the
                # `stream.desynchronized` frame. Nothing more will ever be
                # delivered on this subscription, so leaving the socket open
                # would leave the client believing it is live.
                await _report_desynchronized(websocket)
        finally:
            # Reached on every exit path, which is the point: leaving this
            # block is what unsubscribes (`subscribe()`'s own finally).
            logger.info("ws_events: connection finished")


async def _forward_events(websocket: WebSocket, queue: asyncio.Queue[Any]) -> None:
    """Send this subscriber's own copy of every canonical event.

    The feed is the broadcaster's `encoded=True` one: each item is
    `(frame, text)`, `text` already being the wire encoding, computed once
    for every client in `EventBroadcaster._fan_out` (CLEANUP_PLAN 3.9) --
    so this sends text and never re-serializes.

    Normally never returns -- the stream is endless. It returns in exactly one
    case: after forwarding the `stream.desynchronized` frame (B-33), which is
    the last frame this subscription will ever produce. `ws_events` turns that
    return into the close that makes the loss visible to the client.
    """
    while True:
        frame, text = await queue.get()
        await websocket.send_text(text)
        if frame.get("type") == STREAM_DESYNCHRONIZED_EVENT_TYPE:
            return


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    """Return as soon as the client goes away.

    Nothing is expected *from* the client on this socket, but the
    client->server side still has to be drained: an ASGI app learns about a
    closed peer through `receive()` and nowhere else. Without this, a handler
    that only sends never finishes, and the subscription it holds leaks for
    the lifetime of the process.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _watch_upstream(
    adapter: HermesAdapter, poll_seconds: float = _UPSTREAM_HEALTH_POLL_S
) -> None:
    """Return as soon as the gateway's own Hermes connection is no longer live.

    The client's socket and the upstream socket are two independent
    connections -- on this deployment they are not even to the same machine --
    so the client one staying open proves nothing about the other. When the
    upstream dies, `EventBroadcaster` simply stops having anything to forward
    and every connected app goes permanently quiet mid-reply with no error on
    either side (B-27).

    Polled rather than pushed because `HermesAdapter` has no "connection
    lost" callback to hook and `is_connected` is the flag every other
    recovery path in this service already keys on -- one boolean read per
    second per client, against the cost of an infinite "Thinking…".

    `is_connected` only goes True->False on a real death, so there is no
    false positive: `connect()` (which briefly clears it) runs only from
    `_ensure_connected()`, and only when it is already False.
    """
    while adapter.is_connected:
        await asyncio.sleep(poll_seconds)


async def _report_upstream_loss(websocket: WebSocket) -> None:
    """Tell this client the upstream died, then close (B-27).

    Reuses the exact frame shape the "could not reach Hermes at connect time"
    path already sends -- `{"type": "error", "payload": {"message": ...}}`
    followed by close 1011 -- so the client needs no new case: it surfaces as
    `.upstreamError`, which already resolves an in-flight reply bubble and
    raises the banner.

    Every failure here is suppressed on purpose. The peer may be going away
    at the same instant; failing to deliver the bad news must not turn into an
    exception out of the handler, which would skip the unsubscribe.
    """
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "payload": {"message": _UPSTREAM_LOST_MESSAGE}})
        await websocket.close(code=1011)


async def _report_desynchronized(websocket: WebSocket) -> None:
    """Close a socket the gateway has stopped buffering for (B-33).

    The `stream.desynchronized` frame has already gone out by the time this
    runs -- that is what ended `_forward_events`. This adds the
    `{"type": "error", ...}` + close 1011 pair on purpose: it is the exact
    signal every build already on the phone maps to a dropped stream, so a
    client that has never heard of `stream.desynchronized` still resolves its
    in-flight bubble, raises the banner and runs B-20's reload rather than
    sitting on a socket that will never speak again. New clients get the
    machine-readable frame *and* this; old ones get a recovery that works.

    Suppressed like `_report_upstream_loss` for the same reason: the peer may
    be leaving at this instant, and failing to deliver the bad news must not
    escape the handler and skip the unsubscribe.
    """
    with contextlib.suppress(Exception):
        await websocket.send_json(
            {"type": "error", "payload": {"message": _DESYNCHRONIZED_MESSAGE}}
        )
        await websocket.close(code=1011)
