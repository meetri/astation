"""The gateway -> client event stream: fan-out, frame bounding, control frames."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from adapters.hermes import HermesAdapter
from domain.live_handles import LiveHandleCache
from events import (
    DELIBERATELY_DROPPED_RAW_EVENTS,
    EventContext,
    normalize_event,
    raw_event_type,
)

logger = logging.getLogger(__name__)

BACKGROUND_COMPLETED_EVENT_TYPE = "background.completed"


_SEQ_UNASSIGNED = 0

_UPSTREAM_HEALTH_POLL_S = 1.0
_UPSTREAM_LOST_MESSAGE = (
    "the gateway lost its connection to Hermes; live updates stopped and any "
    "reply in progress was not delivered"
)

STREAM_READY_EVENT_TYPE = "stream.ready"
# Gateway-made frames carry seq 0: the counter is the client's replay cursor over Hermes's stream.
_SEQ_GATEWAY_SYNTHETIC = 0

STORED_SESSION_ID_FIELD = "_stored_session_id"
LIVE_SESSION_ID_FIELD = "_live_session_id"
CONNECTION_GENERATION_FIELD = "_connection_generation"
PROFILE_FIELD = "_profile"
DEFAULT_PROFILE_NAME = "default"

_SESSION_IDENTITY_FIELDS: tuple[str, ...] = (
    STORED_SESSION_ID_FIELD,
    LIVE_SESSION_ID_FIELD,
    CONNECTION_GENERATION_FIELD,
    PROFILE_FIELD,
)

STREAM_RESYNC_EVENT_TYPE = "stream.resync"


# Both caps are enforced; a subscriber past either is cut off rather than having frames dropped.
_MAX_SUBSCRIBER_QUEUED_FRAMES = 2048
_MAX_SUBSCRIBER_QUEUED_BYTES = 8 * 1024 * 1024

STREAM_DESYNCHRONIZED_EVENT_TYPE = "stream.desynchronized"
_DESYNCHRONIZED_MESSAGE = (
    "this client fell too far behind the event stream; the gateway stopped "
    "buffering for it and the transcript must be reloaded"
)

SUBMIT_STATUS_STREAMING = "streaming"
SUBMIT_STATUS_REDIRECTED = "redirected"
SUBMIT_STATUS_STEERED = "steered"
SUBMIT_STATUS_QUEUED = "queued"
SUBMIT_STATUS_UNKNOWN = "unknown"

KNOWN_SUBMIT_STATUSES: frozenset[str] = frozenset(
    {
        SUBMIT_STATUS_STREAMING,
        SUBMIT_STATUS_REDIRECTED,
        SUBMIT_STATUS_STEERED,
        SUBMIT_STATUS_QUEUED,
    }
)


# A frame past a client's own limit closes the whole socket, so this stays under the 1 MiB default.
_MAX_CLIENT_FRAME_BYTES = 900_000


_MAX_PLACEHOLDER_DROPPED_KEYS = 32
_MAX_PLACEHOLDER_KEY_CHARS = 64
_MAX_PLACEHOLDER_ENVELOPE_CHARS = 128

_TRUNCATION_REASON = (
    "event payload exceeded the gateway->app WebSocket frame budget "
    "and was dropped; receiving it would have closed the event socket"
)


def _frame_size(payload: dict[str, Any]) -> int:
    """Serialized size of one canonical event, in the bytes the WS will carry."""
    # json.dumps defaults over-estimate what Starlette writes, which keeps this check conservative.
    return len(json.dumps(payload).encode("utf-8"))


def _encode_frame(frame: dict[str, Any]) -> str:
    """The exact text `WebSocket.send_json` would put on the wire for `frame`."""
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False)


def _bounded_str(value: Any, limit: int) -> str:
    """A string of at most `limit` characters, whatever `value` was."""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _bounded_optional_str(value: Any, limit: int) -> str | None:
    """`_bounded_str`, except a null stays a null (P2-2)."""
    return None if value is None else _bounded_str(value, limit)


def _kept_session_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """The B-29 attribution keys, bounded, ready to survive a degraded frame."""
    body = payload.get("payload")
    if not isinstance(body, dict):
        return {}
    kept: dict[str, Any] = {}
    for field_name in _SESSION_IDENTITY_FIELDS:
        if field_name not in body:
            continue
        value = body[field_name]
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            kept[field_name] = value
        else:
            kept[field_name] = _bounded_str(value, _MAX_PLACEHOLDER_ENVELOPE_CHARS)
    return kept


def _oversized_event_placeholder(payload: dict[str, Any], size: int) -> dict[str, Any]:
    """Replace an unsendable event's payload with a marker of the same shape."""
    dropped = payload.get("payload")
    kept = _kept_session_identity(payload)
    keys = sorted(k for k in dropped if k not in kept) if isinstance(dropped, dict) else []
    return {
        **payload,
        "payload": {
            **kept,
            "_truncated": True,
            "_reason": _TRUNCATION_REASON,
            "_original_bytes": size,
            "_limit_bytes": _MAX_CLIENT_FRAME_BYTES,
            "_dropped_key_count": len(keys),
            "_dropped_keys": [
                _bounded_str(key, _MAX_PLACEHOLDER_KEY_CHARS)
                for key in keys[:_MAX_PLACEHOLDER_DROPPED_KEYS]
            ],
        },
    }


def _minimal_event_placeholder(payload: dict[str, Any], size: int) -> dict[str, Any]:
    """Last-resort marker whose size depends on nothing upstream controls."""
    seq = payload.get("seq")
    return {
        "event_id": _bounded_str(payload.get("event_id"), _MAX_PLACEHOLDER_ENVELOPE_CHARS),
        "project_id": _bounded_optional_str(
            payload.get("project_id"), _MAX_PLACEHOLDER_ENVELOPE_CHARS
        ),
        "session_id": _bounded_optional_str(
            payload.get("session_id"), _MAX_PLACEHOLDER_ENVELOPE_CHARS
        ),
        "run_id": _bounded_optional_str(payload.get("run_id"), _MAX_PLACEHOLDER_ENVELOPE_CHARS),
        "seq": seq if isinstance(seq, int) else -1,
        "type": _bounded_str(payload.get("type"), _MAX_PLACEHOLDER_ENVELOPE_CHARS),
        "timestamp": _bounded_str(payload.get("timestamp"), _MAX_PLACEHOLDER_ENVELOPE_CHARS),
        "payload": {
            **_kept_session_identity(payload),
            "_truncated": True,
            "_reason": _TRUNCATION_REASON,
            "_original_bytes": size,
            "_limit_bytes": _MAX_CLIENT_FRAME_BYTES,
        },
    }


def _bounded_client_frame_with_size(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    """`(frame_to_send, its serialized size)` -- the frame budget, measured once."""
    size = _frame_size(payload)
    if size <= _MAX_CLIENT_FRAME_BYTES:
        return payload, size

    logger.warning(
        "dropping oversized %s payload (%d B > %d B budget)",
        payload.get("type"),
        size,
        _MAX_CLIENT_FRAME_BYTES,
    )
    for candidate in (
        _oversized_event_placeholder(payload, size),
        _minimal_event_placeholder(payload, size),
    ):
        candidate_size = _frame_size(candidate)
        if candidate_size <= _MAX_CLIENT_FRAME_BYTES:
            return candidate, candidate_size

    logger.error(
        "minimal oversize placeholder still exceeds the %d B budget; dropping frame",
        _MAX_CLIENT_FRAME_BYTES,
    )
    return None, size


def _bounded_client_frame(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The frame to actually send: `payload` if it fits, else a marker that does."""
    frame, _size = _bounded_client_frame_with_size(payload)
    return frame


def _gateway_frame(
    event_type: str, payload: dict[str, Any], *, connection_generation: int | None
) -> dict[str, Any]:
    """A frame the gateway generated itself, shaped like every other frame."""
    return {
        "event_id": f"evt_{uuid.uuid4().hex}",
        "project_id": None,
        "session_id": None,
        "run_id": None,
        "seq": _SEQ_GATEWAY_SYNTHETIC,
        "type": event_type,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "payload": {
            "_gateway_generated": True,
            CONNECTION_GENERATION_FIELD: connection_generation,
            **payload,
        },
    }


def stream_ready_frame(connection_generation: int | None = None) -> dict[str, Any]:
    """"You are subscribed" -- the first frame on every socket."""
    return _gateway_frame(STREAM_READY_EVENT_TYPE, {}, connection_generation=connection_generation)


def resync_required_frame(
    connection_generation: int | None, previous_generation: int | None
) -> dict[str, Any]:
    """"Every live handle you hold is void" -- the B-29 generation signal."""
    return _gateway_frame(
        STREAM_RESYNC_EVENT_TYPE,
        {
            "_reason": "hermes_connection_replaced",
            "_live_handles_invalid": True,
            "_previous_connection_generation": previous_generation,
        },
        connection_generation=connection_generation,
    )


def desynchronized_frame(
    *,
    connection_generation: int | None,
    last_delivered_seq: int | None,
    first_dropped_seq: Any,
    reason: str,
    limit_frames: int = _MAX_SUBSCRIBER_QUEUED_FRAMES,
    limit_bytes: int = _MAX_SUBSCRIBER_QUEUED_BYTES,
) -> dict[str, Any]:
    """"You fell behind and I stopped buffering for you" -- B-33's signal."""
    return _gateway_frame(
        STREAM_DESYNCHRONIZED_EVENT_TYPE,
        {
            "_reason": reason,
            "_message": _DESYNCHRONIZED_MESSAGE,
            "_must_resync": True,
            "_last_delivered_seq": last_delivered_seq,
            "_first_dropped_seq": first_dropped_seq,
            "_limit_frames": limit_frames,
            "_limit_bytes": limit_bytes,
        },
        connection_generation=connection_generation,
    )


def session_identity_from_payload(
    raw_type: str | None, payload: dict[str, Any]
) -> tuple[str | None, str | None]:
    """`(stored_id, live_id)` as a raw Hermes payload names them -- the B-29 rule."""
    raw_session = payload.get("session_id")
    raw_stored = payload.get("stored_session_id")
    live_id: str | None = None
    stored_id: str | None = None

    if isinstance(raw_stored, str) and raw_stored:
        stored_id = raw_stored
        if isinstance(raw_session, str) and raw_session:
            live_id = raw_session
    elif raw_type == "session.title":

        # This event alone carries the STORED id; every other one carries the live handle.
        if isinstance(raw_session, str) and raw_session:
            stored_id = raw_session
    elif isinstance(raw_session, str) and raw_session:
        live_id = raw_session
    return stored_id, live_id


class _Subscriber:
    """One client's private feed, with a hard bound on what it can hold."""

    __slots__ = (
        "_queued_sizes",
        "desynchronized",
        "encoded",
        "last_queued_seq",
        "max_bytes",
        "max_frames",
        "queue",
        "queued_bytes",
    )

    def __init__(
        self,
        max_frames: int = _MAX_SUBSCRIBER_QUEUED_FRAMES,
        max_bytes: int = _MAX_SUBSCRIBER_QUEUED_BYTES,
        *,
        encoded: bool = False,
    ) -> None:
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.encoded = encoded
        # One slot above the frame cap, reserved for the desynchronized frame the cap triggers.
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_frames + 1)
        self.desynchronized = False
        self.queued_bytes = 0
        self.last_queued_seq: int | None = None
        self._queued_sizes: deque[int] = deque()

    def _release_consumed(self) -> None:
        consumed = len(self._queued_sizes) - self.queue.qsize()
        for _ in range(max(0, consumed)):
            self.queued_bytes -= self._queued_sizes.popleft()

    def offer(
        self,
        frame: dict[str, Any],
        size: int,
        generation: int | None,
        text: str | None = None,
    ) -> bool:
        """Queue `frame` for this client, or desynchronize it. True if queued."""
        if self.desynchronized:
            return False
        self._release_consumed()
        over_frames = self.queue.qsize() >= self.max_frames
        over_bytes = self.queued_bytes + size > self.max_bytes
        if over_frames or over_bytes:
            self._desynchronize(frame, over_frames, generation)
            return False
        self.queued_bytes += size
        self._queued_sizes.append(size)
        seq = frame.get("seq")
        if isinstance(seq, int):
            self.last_queued_seq = seq
        self._put(frame, text)
        return True

    def _put(self, frame: dict[str, Any], text: str | None) -> None:
        if self.encoded:
            self.queue.put_nowait((frame, text if text is not None else _encode_frame(frame)))
        else:
            self.queue.put_nowait(frame)

    def _desynchronize(
        self, undelivered: dict[str, Any], over_frames: bool, generation: int | None
    ) -> None:
        self.desynchronized = True
        reason = "subscriber_backlog_frames" if over_frames else "subscriber_backlog_bytes"
        logger.warning(
            "event subscriber fell behind (%s: %d frames / %d B queued); "
            "marking it desynchronized from seq %s and telling it to resync",
            reason,
            self.queue.qsize(),
            self.queued_bytes,
            undelivered.get("seq"),
        )
        self._put(
            desynchronized_frame(
                connection_generation=generation,
                last_delivered_seq=self.last_queued_seq,
                first_dropped_seq=undelivered.get("seq"),
                reason=reason,
                limit_frames=self.max_frames,
                limit_bytes=self.max_bytes,
            ),
            None,
        )


class EventBroadcaster:
    """Fan the Hermes event stream(s) out to every connected client."""

    def __init__(
        self,
        adapter: HermesAdapter,
        live_handle_cache: LiveHandleCache | None = None,
        *,
        max_queued_frames: int = _MAX_SUBSCRIBER_QUEUED_FRAMES,
        max_queued_bytes: int = _MAX_SUBSCRIBER_QUEUED_BYTES,
        generation_poll_seconds: float = _UPSTREAM_HEALTH_POLL_S,
        on_background_complete: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        on_generation_change: Callable[[int | None, int | None], None] | None = None,
        on_canonical_event: Callable[[Any, dict[str, Any]], None] | None = None,
        profile: str = DEFAULT_PROFILE_NAME,
        profile_caches: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._adapter = adapter
        self._profile = profile
        self._profile_caches = profile_caches
        self._on_background_complete = on_background_complete
        self._on_generation_change = on_generation_change
        self._on_canonical_event = on_canonical_event
        self._generation_poll_seconds = generation_poll_seconds
        self._subscribers: set[_Subscriber] = set()
        self._pump: asyncio.Task[None] | None = None
        self._generation_watch: asyncio.Task[None] | None = None
        self._seq = itertools.count(1)
        self._max_queued_frames = max_queued_frames
        self._max_queued_bytes = max_queued_bytes
        self._live_handle_cache = live_handle_cache
        self._generation_seen: int | None = self._connection_generation()
        self._unknown_types_logged: set[str] = set()

    @property
    def subscriber_count(self) -> int:
        """Live subscriptions. A client that hung up must not still be here."""
        return len(self._subscribers)

    def start(self) -> None:
        """Begin draining the upstream stream now, subscribers or not."""
        # Only one task may drain the adapter; two consumers would split the stream between them.
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._run())
        if self._generation_watch is None or self._generation_watch.done():
            self._generation_watch = asyncio.create_task(
                self._watch_generation(self._generation_poll_seconds)
            )

    @asynccontextmanager
    async def subscribe(self, *, encoded: bool = False) -> AsyncIterator[asyncio.Queue[Any]]:
        """Register a private feed, guaranteed to be unregistered on exit."""
        subscriber = _Subscriber(self._max_queued_frames, self._max_queued_bytes, encoded=encoded)
        self._subscribers.add(subscriber)
        self.start()
        try:
            yield subscriber.queue
        finally:
            self._subscribers.discard(subscriber)

    async def close(self) -> None:
        for task in (self._pump, self._generation_watch):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._pump = None
        self._generation_watch = None
        self._subscribers.clear()


    def _connection_generation(self) -> int | None:
        """The adapter's connection generation, or None if it has no such idea."""
        generation = getattr(self._adapter, "connection_generation", None)
        return generation if isinstance(generation, int) else None

    def _note_connection_generation(self) -> int | None:
        """Emit `stream.resync` if the Hermes connection has been replaced."""
        generation = self._connection_generation()
        if generation is None or generation == self._generation_seen:
            return generation
        previous = self._generation_seen
        self._generation_seen = generation
        logger.info(
            "Hermes connection generation moved %s -> %s; telling %d subscriber(s) to resync",
            previous,
            generation,
            len(self._subscribers),
        )
        self._fan_out(resync_required_frame(generation, previous), generation)
        if self._on_generation_change is not None:
            try:
                self._on_generation_change(generation, previous)
            except Exception:  # pragma: no cover - defensive
                logger.exception("generation-change hook failed; continuing")
        return generation

    async def _watch_generation(self, poll_seconds: float = _UPSTREAM_HEALTH_POLL_S) -> None:
        while True:
            await asyncio.sleep(poll_seconds)
            try:
                self._note_connection_generation()
            except Exception:  # pragma: no cover - defensive
                logger.exception("connection-generation watcher failed; continuing")

    def _fan_out(self, frame: dict[str, Any], generation: int | None) -> None:
        """Hand one already-bounded frame to every subscriber that can take it."""
        text = _encode_frame(frame)
        size = len(text.encode("utf-8"))
        for subscriber in list(self._subscribers):
            subscriber.offer(frame, size, generation, text)

    def _note_unnormalized(self, raw_event: dict[str, Any]) -> None:
        """Make an event type we don't handle discoverable, exactly once."""
        raw_type = raw_event_type(raw_event)
        if raw_type is None:
            raw_type = "<no method/type key>"
        if raw_type in self._unknown_types_logged:
            return
        self._unknown_types_logged.add(raw_type)
        if raw_type in DELIBERATELY_DROPPED_RAW_EVENTS:
            logger.info(
                "not forwarding %r (deliberate): %s",
                raw_type,
                DELIBERATELY_DROPPED_RAW_EVENTS[raw_type],
            )
        else:
            logger.warning(
                "unmapped Hermes event type %r -- not forwarded to clients. "
                "Add it to RAW_TO_CANONICAL_TYPE or to "
                "DELIBERATELY_DROPPED_RAW_EVENTS with a reason (B-14). "
                "Logged once per type per process.",
                raw_type,
            )

    async def _run(self) -> None:
        async for raw_event in self._adapter.events():
            try:
                self._forward_one(raw_event)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "failed to forward a %r event; dropping it and continuing",
                    raw_event_type(raw_event) if isinstance(raw_event, dict) else None,
                )

    def _stamp_session_identity(
        self, raw_type: str | None, payload: dict[str, Any], generation: int | None
    ) -> None:
        """Add the STORED id (and the live handle, and the generation) to a frame."""
        stored_id, live_id = session_identity_from_payload(raw_type, payload)
        stored_id, live_id, profile = self._resolve_identity(stored_id, live_id)

        payload[STORED_SESSION_ID_FIELD] = stored_id
        payload[LIVE_SESSION_ID_FIELD] = live_id
        payload[CONNECTION_GENERATION_FIELD] = generation
        payload[PROFILE_FIELD] = profile

    def _caches(self) -> list[tuple[str, Any]]:
        """`(profile, cache)` for every profile this connection serves."""
        caches: list[tuple[str, Any]] = []
        if self._live_handle_cache is not None:
            caches.append((self._profile, self._live_handle_cache))
        if self._profile_caches is not None:
            try:
                extra = self._profile_caches()
            except Exception:  # pragma: no cover - defensive
                extra = {}
            for name, cache in (extra or {}).items():
                if cache is not None and cache is not self._live_handle_cache:
                    caches.append((name, cache))
        return caches

    def _resolve_identity(
        self, stored_id: str | None, live_id: str | None
    ) -> tuple[str | None, str | None, str]:
        """Complete the id pair, and name the profile the session belongs to."""
        caches = self._caches()

        if stored_id is not None and live_id is not None:
            for name, cache in caches:
                if cache.stored_for_live(live_id) == stored_id or cache.get(stored_id) == live_id:
                    cache.observe_live_mapping(stored_id, live_id)
                    return stored_id, live_id, name
            if self._live_handle_cache is not None:
                self._live_handle_cache.observe_live_mapping(stored_id, live_id)
            return stored_id, live_id, self._profile

        if live_id is not None:
            for name, cache in caches:
                resolved = cache.stored_for_live(live_id)
                if resolved is not None:
                    return resolved, live_id, name
            return None, live_id, self._profile

        if stored_id is not None:
            for name, cache in caches:
                resolved = cache.get(stored_id)
                if resolved is not None:
                    return stored_id, resolved, name
            return stored_id, None, self._profile

        return None, None, self._profile

    def inject(self, envelope: dict[str, Any], *, profile: str) -> None:
        """Fan out a frame that came off ANOTHER profile's connection."""
        # The recorder hook is the caller's job here; running it again double-records the turn.
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            payload = {}
            envelope["payload"] = payload
        payload[PROFILE_FIELD] = profile
        frame, _ = _bounded_client_frame_with_size(envelope)
        if frame is None:
            return
        frame["seq"] = next(self._seq)
        self._fan_out(frame, self._generation_seen)

    def _forward_one(self, raw_event: dict[str, Any]) -> None:
        """Normalize, size-check and fan out exactly one upstream event."""
        generation = self._note_connection_generation()
        context = EventContext(
            project_id=None,
            session_id=None,
            run_id=None,
            seq=_SEQ_UNASSIGNED,
        )
        canonical = normalize_event(raw_event, context)
        if canonical is None:
            self._note_unnormalized(raw_event)
            return
        envelope = canonical.to_dict()
        self._stamp_session_identity(raw_event_type(raw_event), envelope["payload"], generation)
        if self._on_canonical_event is not None:
            try:
                self._on_canonical_event(canonical, envelope)
            except Exception:  # pragma: no cover - defensive
                logger.exception("canonical-event hook failed; forwarding the frame unattributed")
        injected: dict[str, Any] | None = None
        if (
            canonical.type == BACKGROUND_COMPLETED_EVENT_TYPE
            and self._on_background_complete is not None
        ):
            try:
                injected = self._on_background_complete(envelope)
            except Exception:  # pragma: no cover - defensive
                logger.exception("background-complete hook failed; forwarding the frame unledgered")
        frame, size = _bounded_client_frame_with_size(envelope)
        if frame is None:
            return

        # Spent last, after every branch that could discard the frame, so a client sees no gap.
        frame["seq"] = next(self._seq)
        del size
        self._fan_out(frame, generation)

        if injected is not None:
            injected_frame = _bounded_client_frame(injected)
            if injected_frame is not None:
                injected_frame["seq"] = _SEQ_GATEWAY_SYNTHETIC
                self._fan_out(injected_frame, generation)
