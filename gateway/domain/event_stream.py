"""The gateway -> client event stream: fan-out, frame bounding, control frames.

Everything that turns Hermes's raw event stream into the frames `/ws/events`
carries, moved here from `api/main.py` (CLEANUP_PLAN step 3.1) so nothing
below `api/` has to import the app module to reach it:

* `EventBroadcaster` -- the ONE task that drains the default connection's
  adapter, plus `inject()` for every other profile's frames (B-136), each
  subscriber on its own bounded `_Subscriber` queue (B-33), one shared `seq`.
* The frame budget (B-02/B-16): `_frame_size`, `_bounded_client_frame*` and
  the two placeholder tiers an oversized event degrades into.
* The gateway's own control frames (`stream.ready` B-28, `stream.resync`
  B-29, `stream.desynchronized` B-33) and the payload keys that identify
  which session a frame belongs to.
* The `prompt.submit` outcome vocabulary (B-05o).

`api/main.py` re-exports the public names so existing callers and tests keep
resolving them there.
"""

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

#: The canonical type `events/canonical.py` maps `background.complete` to.
#: Defined here (not in `api/background.py`, which re-exports it) because the
#: broadcaster's `_on_background_complete` hook keys on it and `domain/` must
#: not import from `api/`.
BACKGROUND_COMPLETED_EVENT_TYPE = "background.completed"

# P2-2: attribution on the canonical envelope is REAL now. The Phase-0
# placeholder ids (`proj_phase0`/`sess_phase0`/`run_phase0`) are gone: the
# normalizer produces an unattributed envelope (explicit nulls) and the
# `RunRecorder` hook (`api/runs.py`) resolves project/session from the
# stored-id mapping and attaches the frame to its Run row. A null means "not
# attributed", which the app degrades to "" (`GatewayEvent.init(from:)`) --
# an honest nothing instead of a placeholder that looked like something.

# Sentinel `seq` handed to the normalizer before the broadcaster knows whether
# this event will be forwarded at all. Never reaches a client: every frame that
# leaves `EventBroadcaster._run` has a real, contiguous `seq` stamped over it.
_SEQ_UNASSIGNED = 0

# --- B-27: the client has to be told the upstream died -------------------
#
# `EventBroadcaster` forwards what Hermes pushes and nothing else, so when the
# gateway's *own* socket to Hermes dies (Hermes restarts, the box it runs on
# blips, an idle reap) the app's `/ws/events` socket stays perfectly healthy
# and simply goes quiet. `EventBus` therefore keeps reporting `.live`, B-20's
# interrupted-reply recovery never fires, and a reply that was mid-stream
# spins on `ThinkingRow` forever. The two sockets are independent -- Hermes is
# on a different box from the gateway on this topology -- so an upstream-only
# drop is the *most* likely kind, not an edge case.
#
# `_watch_upstream` closes that hole: it samples `adapter.is_connected` and,
# the moment it goes False, the handler sends the same `{"type": "error"}`
# frame the "gateway can't reach Hermes" path already uses and closes 1011.
# The app maps that to `.upstreamError` -> the bubble resolves, the banner
# appears -- and its reconnect drives `_ensure_connected()` here, which
# rebuilds the upstream socket. So the loop self-heals rather than needing an
# HTTP request to happen to come along.
#
# There is no false-positive window. `is_connected` only goes True->False on a
# real death (the recv loop ending, a failed send, a failed heartbeat);
# `connect()` tears a socket down only from inside `_ensure_connected()`,
# which is reached only when `is_connected` is *already* False. So every
# observation of False means events were genuinely lost.
_UPSTREAM_HEALTH_POLL_S = 1.0
_UPSTREAM_LOST_MESSAGE = (
    "the gateway lost its connection to Hermes; live updates stopped and any "
    "reply in progress was not delivered"
)

# --- B-28: "the socket reconnected" has to be observable ------------------
#
# The app's `EventBus` only leaves `.connecting` for `.live` when a frame
# arrives, and this endpoint's first frame is whatever Hermes pushes next --
# which on a quiet system is *never*. So after a drop the app reconnected
# successfully and then sat in `.connecting` indefinitely, and B-20's
# automatic "reload the transcript on non-`.live` -> `.live`" recovery, which
# is the whole recovery for a turn that finished while the socket was down,
# never ran. The user had to notice the Reload button.
#
# One synthetic frame sent immediately after subscribing fixes it. It is
# deliberately shaped as an ordinary canonical envelope so no client needs a
# special case (an older build decodes it as an unknown type and ignores it),
# and it deliberately carries **`seq` 0**: `seq` is the client's replay cursor
# over the *upstream* stream (B-14) and a gateway-generated frame must not
# spend one. Sent inside `subscribe()`, so nothing Hermes pushes can slip in
# ahead of it or be lost behind it.
STREAM_READY_EVENT_TYPE = "stream.ready"
_SEQ_GATEWAY_SYNTHETIC = 0

# --- B-29: the client must never be able to silently mis-match a frame -----
#
# Hermes stamps almost every event with the **live handle**, which is
# process-local and is re-minted whenever the gateway reconnects. The app
# holds a copy of that handle and drops every frame whose id does not match it
# exactly (the B-17 filter, which is what stops one session's tokens being
# typed into another's transcript). So after any gateway<->Hermes reconnect the
# same session is one handle to the gateway and another to the app, and every
# frame is dropped *silently*: a turn started from Hermes's own TUI renders
# nothing at all, and a reply the user is waiting on thinks forever.
#
# The contract implemented here has two halves, and it is deliberately both
# rather than either -- the first makes matching stable, the second covers the
# window in which the first cannot answer:
#
#   1. **Every forwarded frame carries the STORED id.** `payload`
#      `_stored_session_id` is the durable id (`20260829_182532_991e3f`) --
#      the one the app persists and the one that never changes. It is the
#      field a client should match on. `_live_session_id` is the live handle
#      the frame was actually stamped with, kept alongside it so nothing is
#      lost, and `_connection_generation` says which Hermes connection both
#      belong to.
#   2. **A generation change is announced.** When the gateway's Hermes
#      connection is replaced, every live handle in the world is void; the
#      gateway emits a `stream.resync` frame saying so, so a client that is
#      holding one knows to re-resolve rather than to keep filtering on a
#      handle that can no longer match anything.
#
# `_stored_session_id` is `None` only when the gateway genuinely cannot
# attribute the frame -- a live handle it has never resolved on this
# connection. That is not silent: it is an explicit null, and it can only
# happen either before the client has opened that session (in which case the
# client has nothing to match it to anyway) or in the window right after a
# reconnect, which `stream.resync` has already flagged.
STORED_SESSION_ID_FIELD = "_stored_session_id"
LIVE_SESSION_ID_FIELD = "_live_session_id"
CONNECTION_GENERATION_FIELD = "_connection_generation"
# B-136: WHICH Hermes profile's connection a frame came off. A stored session
# id is only unique within a profile, so a client matching on
# `_stored_session_id` alone could attribute one agent's frame to another
# agent's open chat once every profile's stream shares `/ws/events`. Stamped
# on every forwarded frame -- `"default"` for the broadcaster's own upstream,
# the profile name for frames injected by `ProfileConnectionManager`.
PROFILE_FIELD = "_profile"
DEFAULT_PROFILE_NAME = "default"

# Gateway-owned payload keys that identify *which session* a frame belongs to.
# They survive the B-02 oversize degradation (see `_oversized_event_placeholder`):
# a frame whose payload had to be dropped must still be attributable, or the
# guard against killing the socket would reintroduce the silent mis-match this
# whole contract exists to remove.
_SESSION_IDENTITY_FIELDS: tuple[str, ...] = (
    STORED_SESSION_ID_FIELD,
    LIVE_SESSION_ID_FIELD,
    CONNECTION_GENERATION_FIELD,
    PROFILE_FIELD,
)

# Emitted when the gateway's Hermes connection generation moves, i.e. every
# live handle any client is holding has just become unusable. Shaped like every
# other frame (a client that has never heard of it decodes an unknown type and
# ignores it) and, like `stream.ready`, carries `seq` 0 because Hermes did not
# send it.
STREAM_RESYNC_EVENT_TYPE = "stream.resync"

# --- B-33: a subscriber that stops reading must not grow the gateway -------
#
# `EventBroadcaster` gave each subscriber an unbounded `asyncio.Queue`. That
# was a deliberate choice -- dropping the oldest frame would silently delete a
# `message.delta` out of the middle of a reply and corrupt the transcript the
# client is assembling -- but "unbounded" and "small in practice" are different
# claims and only the second was ever true. A backgrounded phone whose socket
# the OS has not yet reaped is neither disconnected nor reading, and a
# reasoning-heavy turn is ~11,600 frames, so one wedged client grows the
# gateway for as long as that lasts.
#
# The resolution keeps the reason the queue was unbounded and drops the
# unboundedness: nothing is ever removed from the middle. When a subscriber
# falls past the cap it is marked **desynchronized** -- the gateway stops
# buffering for it entirely and hands it one `stream.desynchronized` frame
# saying so. Losing the stream visibly and recoverably beats both unbounded
# memory and a quietly corrupted transcript, and the app already has the
# recovery: `stream.desynchronized` is followed by the `{"type": "error"}` +
# close 1011 pair that every existing build maps to a dropped stream, which
# runs B-20's reload-the-transcript path.
#
# Both caps are enforced. The frame cap bounds the queue length; the byte cap
# is what actually bounds *memory*, since B-02 permits a single frame of up to
# `_MAX_CLIENT_FRAME_BYTES`. 2,048 frames is orders of magnitude more jitter
# headroom than a reading client ever needs (a healthy subscriber's queue sits
# at 0-2 frames) while capping a wedged one at 8 MiB.
_MAX_SUBSCRIBER_QUEUED_FRAMES = 2048
_MAX_SUBSCRIBER_QUEUED_BYTES = 8 * 1024 * 1024

STREAM_DESYNCHRONIZED_EVENT_TYPE = "stream.desynchronized"
_DESYNCHRONIZED_MESSAGE = (
    "this client fell too far behind the event stream; the gateway stopped "
    "buffering for it and the transcript must be reloaded"
)

# --- prompt.submit outcomes (B-05o) --------------------------------------
#
# `prompt.submit` has FOUR distinct success answers, all of them HTTP 200 by
# the time they reach the app, and only ONE of them means "a fresh turn is
# now streaming back". They come from Hermes's own `tui_gateway/server.py`,
# and `redirected` was additionally observed live on 2026-08-29 (submitting
# to `astation Phase 0 spike DEBUG` while that session was busy). See
# `docs/PROTOCOL_VERIFIED.md`, "prompt.submit outcomes".
SUBMIT_STATUS_STREAMING = "streaming"  # new turn started; output will stream
SUBMIT_STATUS_REDIRECTED = "redirected"  # applied as a CORRECTION to the in-flight turn
SUBMIT_STATUS_STEERED = "steered"  # injected into the running turn at the next boundary
SUBMIT_STATUS_QUEUED = "queued"  # accepted; runs AFTER the current turn finishes
# Anything else -- an absent status, a null, a non-string, or a status string
# this gateway has never seen. Deliberately NOT treated as success: the whole
# point of B-05o is that "the POST returned 200" is not evidence the user's
# message will ever be answered.
SUBMIT_STATUS_UNKNOWN = "unknown"

KNOWN_SUBMIT_STATUSES: frozenset[str] = frozenset(
    {
        SUBMIT_STATUS_STREAMING,
        SUBMIT_STATUS_REDIRECTED,
        SUBMIT_STATUS_STEERED,
        SUBMIT_STATUS_QUEUED,
    }
)

# --- gateway -> app WebSocket frame budget (B-02) -------------------------
#
# Measured, not assumed (2026-08-29):
#
# * The *server* side applies no limit at all to a server->client send.
#   uvicorn's `ws_max_size` (default 16,777,216) is passed to `websockets`'
#   `ServerProtocol(max_size=...)`, and both the websockets docs and its
#   source describe that as "maximum size of INCOMING messages"; the sansio
#   send path checks nothing. Probed against a real uvicorn + Starlette
#   WebSocket on a spare port: a 64 MiB frame (67,108,913 B) was sent and
#   received intact, i.e. past `ws_max_size` itself. There is nothing to
#   raise on this side.
# * The *receiver* is what bites. A python `websockets` client at its default
#   1 MiB closed the socket with `1009 (message too big) frame exceeds limit
#   of 1048576 bytes` for every frame at/above 1 MiB in that same probe. Our
#   real client is `URLSessionWebSocketTask`, whose `maximumMessageSize`
#   measured **1,048,576** on this machine (`swift` one-liner) and which
#   `GatewayEventStream.swift` never raises.
#
# So the gateway must not emit a frame the app cannot receive: exceeding the
# app's limit does not drop one event, it kills the whole event socket --
# exactly the B-12 failure mode, on the other side of the gateway.
#
# 900,000 B leaves ~15% headroom under the 1,048,576 B default. It is
# deliberately the *tighter* of the two ceilings: `GatewayEventStream.swift`
# now raises its own `maximumMessageSize` to 16 MiB, but old builds are
# already installed on the phone (and any other client would arrive at its
# library's default), so the gateway holds the line that keeps every client
# alive. Raise this only once no client at the 1 MiB default can still
# connect.
#
# Nothing real comes close, which is why the tighter bound costs nothing:
#   * a complete live turn including a `tool.complete` for a ~230 KB tool
#     output had a largest forwarded frame of 2,526 B (the `message.completed`;
#     the `tool.complete` itself was 659 B -- the output is not inlined);
#   * 15,181 transcript entries across all 43 sessions on the live instance
#     max at 79,905 B, p99 33,144 B, median 560 B.
# But nothing *structurally* bounds a payload -- the Hermes->gateway socket
# accepts 64 MiB frames -- so an oversized event is degraded into a
# well-formed marker event here rather than being allowed to take the socket
# down.
_MAX_CLIENT_FRAME_BYTES = 900_000


# --- B-16: the guard's own replacement must fit too -----------------------
#
# The oversize marker below is what gets sent *instead of* a frame that would
# kill the socket, so a marker that is itself over budget defeats the entire
# guard. Every field in it except `_dropped_keys` is a fixed-size envelope
# value; `_dropped_keys` is derived from the dropped payload's keys and is
# therefore attacker/upstream-shaped -- unbounded in both count and per-key
# length. Nothing in real Hermes traffic gets near it (what makes a payload
# huge is always one long *value*, and every observed payload has single-digit
# key counts), but "not reachable today" is not a bound.
#
# So the marker is bounded by construction, in three tiers, and
# `_bounded_client_frame()` measures each tier and falls to the next until one
# fits. 32 keys x 64 chars is ~2.5 KB against a 900,000 B budget -- three
# orders of magnitude of headroom, while still naming enough keys to be useful
# for the real case (a handful of short keys, listed in full).
_MAX_PLACEHOLDER_DROPPED_KEYS = 32
_MAX_PLACEHOLDER_KEY_CHARS = 64
# Envelope strings (`event_id`, ids, `type`, `timestamp`) are gateway-generated
# and short in practice; the minimal tier truncates them anyway so its size is
# a function of these constants alone and not of anything upstream sends.
_MAX_PLACEHOLDER_ENVELOPE_CHARS = 128

_TRUNCATION_REASON = (
    "event payload exceeded the gateway->app WebSocket frame budget "
    "and was dropped; receiving it would have closed the event socket"
)


def _frame_size(payload: dict[str, Any]) -> int:
    """Serialized size of one canonical event, in the bytes the WS will carry.

    Deliberately an *over*-estimate of what `WebSocket.send_json` actually
    writes: Starlette serializes with `separators=(",", ":")` and
    `ensure_ascii=False`, while this uses `json.dumps`'s defaults, which add
    a space after every `,` and `:` and escape every non-ASCII character to
    `\\uXXXX`. Both differences can only make this number larger, never
    smaller, so a frame that passes this check cannot exceed the peer's limit
    on the wire.
    """
    return len(json.dumps(payload).encode("utf-8"))


def _encode_frame(frame: dict[str, Any]) -> str:
    """The exact text `WebSocket.send_json` would put on the wire for `frame`.

    Starlette serializes with `separators=(",", ":")` and `ensure_ascii=False`;
    this is that encoding, done ONCE per frame in `EventBroadcaster._fan_out`
    (CLEANUP_PLAN 3.9) so N websocket clients cost one `json.dumps`, not N+1
    (one to measure, one per `send_json`). `_frame_size` above stays the
    over-estimating budget check for the bounding step; this is the delivery
    encoding, and the byte count it yields is what each subscriber's backlog
    accounting uses, so that accounting is now exact rather than pessimistic.
    """
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False)


def _bounded_str(value: Any, limit: int) -> str:
    """A string of at most `limit` characters, whatever `value` was.

    Non-strings are coerced first, so a malformed envelope value (a dict, a
    huge int) cannot smuggle unbounded bytes into a frame whose whole purpose
    is to be small.
    """
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _bounded_optional_str(value: Any, limit: int) -> str | None:
    """`_bounded_str`, except a null stays a null (P2-2).

    The attribution ids are nullable now, and "not attributed" must not be
    coerced into the four-character string "None" -- the app treats a null
    envelope field as absent, but a non-empty string as an id.
    """
    return None if value is None else _bounded_str(value, limit)


def _kept_session_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """The B-29 attribution keys, bounded, ready to survive a degraded frame.

    A frame whose payload had to be thrown away is still a frame about a
    particular session, and a client that cannot tell which one can only drop
    it -- which is the silent mis-match B-29 is about, reintroduced by the
    guard that exists to keep the socket alive. So these three keys ride
    through every tier of the oversize degradation.

    `_stored_session_id` / `_live_session_id` are ids and are short by
    construction, but they came off the wire, so they are truncated like
    everything else here; `_connection_generation` is this process's own
    counter and is kept verbatim when it is an int.
    """
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
    """Replace an unsendable event's payload with a marker of the same shape.

    Keeps the canonical envelope (`event_id`/`seq`/`type`/`timestamp`) intact
    so the client's sequence tracking is unbroken and it can see *that* an
    event of this type happened, and names the keys that were dropped so the
    loss is explicit rather than silent.

    The session-identity keys (B-29) are **kept**, not dropped: attribution is
    what lets the client decide whether this event is even about the
    conversation it is showing.

    The key list is capped in count and in per-key length (B-16), and
    `_dropped_key_count` always reports the true total, so "how much was lost"
    survives even when the list itself had to be cut.
    """
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
    """Last-resort marker whose size depends on nothing upstream controls.

    Reached only if the envelope *itself* is what blew the budget (the
    placeholder above copies `**payload`, so a pathological `event_id` or
    `type` would ride along). Every string is re-truncated to
    `_MAX_PLACEHOLDER_ENVELOPE_CHARS` and the key list is gone entirely, which
    makes the serialized size a function of module constants only -- a few
    hundred bytes, provably under any budget worth having.
    """
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
        # `seq` is this process's own `itertools.count`, never upstream data;
        # kept verbatim when it is an int so the replay cursor stays usable.
        "seq": seq if isinstance(seq, int) else -1,
        "type": _bounded_str(payload.get("type"), _MAX_PLACEHOLDER_ENVELOPE_CHARS),
        "timestamp": _bounded_str(payload.get("timestamp"), _MAX_PLACEHOLDER_ENVELOPE_CHARS),
        "payload": {
            # B-29 attribution survives even the last-resort tier: three short,
            # already-bounded values against a 900,000 B budget.
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
    """`(frame_to_send, its serialized size)` -- the frame budget, measured once.

    Returns something guaranteed to serialize under `_MAX_CLIENT_FRAME_BYTES`,
    or `None` if even the minimal marker would not (unreachable, see below).
    Each tier is *measured* rather than assumed to be smaller than the last --
    that assumption is exactly the bug B-16 records.

    The size comes back with the frame because the caller needs it anyway:
    B-33's per-subscriber memory bound is a byte bound, and re-serializing
    every frame once per subscriber to find out how big it is would be paying
    for the accounting twice.
    """
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

    # Unreachable: the minimal tier's size is fixed by module constants, and
    # `test_minimal_placeholder_is_bounded_by_construction` pins it. If it ever
    # is reached, sending nothing is still better than sending a frame that
    # closes the socket -- but say so loudly, because it means a constant above
    # was raised past the budget.
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
    """A frame the gateway generated itself, shaped like every other frame.

    Three of these exist (`stream.ready`, `stream.resync`,
    `stream.desynchronized`) and they share one shape on purpose: the app
    decodes `/ws/events` with a single `Codable` struct that requires every
    envelope key, so a frame missing one is dropped by the decoder -- which
    for a frame whose entire job is to be *noticed* would be silence again.

    Two invariants:

    * **`seq` is 0 and never drawn from the broadcaster's counter.** That
      counter is the client's replay cursor over the *upstream* stream
      (B-14); spending one of its numbers on a frame Hermes never sent would
      put a permanent hole in the one value a resume protocol has to trust.
    * **`_`-prefixed payload keys are the gateway's namespace** (the B-02 /
      B-14 convention), so `_gateway_generated` is a claim upstream cannot
      forge.
    """
    return {
        "event_id": f"evt_{uuid.uuid4().hex}",
        # Null attribution (P2-2): a gateway-generated stream-control frame
        # belongs to no project, session or run, and saying so beats a
        # placeholder. The keys stay present -- the shape contract above.
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
    """ "You are subscribed" -- the first frame on every socket (B-28).

    The app's `EventBus` only reports `.live` once a frame arrives, and on a
    quiet system the next Hermes frame may never come, so a successful
    reconnect was indistinguishable from one that never completed and B-20's
    automatic reload-on-reconnect never fired.

    It also carries `_connection_generation` (B-29): that is the generation
    every live handle the client goes on to hold belongs to, so a client can
    tell a later `stream.resync` apart from the connection it is already on.
    """
    return _gateway_frame(STREAM_READY_EVENT_TYPE, {}, connection_generation=connection_generation)


def resync_required_frame(
    connection_generation: int | None, previous_generation: int | None
) -> dict[str, Any]:
    """ "Every live handle you hold is void" -- the B-29 generation signal.

    Emitted when the gateway's Hermes connection is replaced. Hermes live
    handles are process-local and are re-minted on every connection, so the
    instant this counter moves, the handle the app is filtering its frames on
    can no longer match anything the gateway will send -- and the app's own
    filter would drop the whole stream without a word.

    A client holding a live handle must re-resolve it (`POST /resume`) and
    reload the transcript. A client matching on `_stored_session_id` -- which
    is what it should be doing -- does not strictly need this frame, but it is
    still the signal that the gateway may briefly be unable to attribute
    frames for a session it has not re-resumed yet.
    """
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
    """ "You fell behind and I stopped buffering for you" -- B-33's signal.

    The last frame a desynchronized subscriber ever receives. Everything after
    `_last_delivered_seq` is gone for this client and will not be replayed, so
    the only correct response is to reload the transcript (`POST /resume` /
    `GET /messages`) -- the B-20 resync path.

    The gap is named rather than implied: `_last_delivered_seq` is the final
    frame this client actually got and `_first_dropped_seq` is the one it did
    not, so a client that tracks `seq` can see exactly what it missed instead
    of inferring a loss from a jump.
    """
    return _gateway_frame(
        STREAM_DESYNCHRONIZED_EVENT_TYPE,
        {
            "_reason": reason,
            "_message": _DESYNCHRONIZED_MESSAGE,
            "_must_resync": True,
            "_last_delivered_seq": last_delivered_seq,
            "_first_dropped_seq": first_dropped_seq,
            # The caps this subscriber was actually held to, not the module
            # defaults: a diagnostic that reports a number the run did not use
            # sends whoever reads it after the wrong thing.
            "_limit_frames": limit_frames,
            "_limit_bytes": limit_bytes,
        },
        connection_generation=connection_generation,
    )


def session_identity_from_payload(
    raw_type: str | None, payload: dict[str, Any]
) -> tuple[str | None, str | None]:
    """`(stored_id, live_id)` as a raw Hermes payload names them -- the B-29 rule.

    `payload.session_id` does not mean the same thing on every event
    (`docs/PROTOCOL_VERIFIED.md`): on almost all of them it is the *live*
    handle; on `session.title` it is the *stored* id; and `session.info`
    carries both, under `session_id` (live) and `stored_session_id`. This is
    the one place that reads those two keys. Shared by
    `EventBroadcaster._stamp_session_identity` (the default connection) and
    `ProfileConnectionManager._stamp_identity` (every other profile), which
    differ only in which cache they then consult.
    """
    raw_session = payload.get("session_id")
    raw_stored = payload.get("stored_session_id")
    live_id: str | None = None
    stored_id: str | None = None

    if isinstance(raw_stored, str) and raw_stored:
        # `session.info`: carries both, and is the one event that can
        # teach us a mapping we never resolved ourselves -- e.g. a session
        # someone started in Hermes's own TUI.
        stored_id = raw_stored
        if isinstance(raw_session, str) and raw_session:
            live_id = raw_session
    elif raw_type == "session.title":
        # The documented exception: this event's `session_id` is STORED.
        if isinstance(raw_session, str) and raw_session:
            stored_id = raw_session
    elif isinstance(raw_session, str) and raw_session:
        live_id = raw_session
    return stored_id, live_id


class _Subscriber:
    """One client's private feed, with a hard bound on what it can hold (B-33).

    The queue used to be unbounded, for a good reason: this is a transcript
    being assembled token by token, and silently discarding a `message.delta`
    from the middle of it produces a *wrong* reply on the user's screen with
    nothing anywhere saying so. Bounding it by evicting the oldest item would
    have done exactly that.

    So nothing is ever evicted. Instead the subscriber has two caps -- a frame
    count and a byte total -- and the first frame that would breach either
    ends the subscription's usefulness explicitly: `desynchronized` is set,
    every later frame is refused (so memory stops growing at the cap), and one
    `stream.desynchronized` frame is queued telling the client it must reload.
    Visible, recoverable loss instead of unbounded memory or a corrupted
    transcript.

    The queue is sized one slot *above* the frame cap and the cap is checked
    before the put, so there is always room for that signal: the frame that
    says "I could not fit any more" must never itself be the one that does not
    fit.
    """

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
        # `encoded` subscribers (the websocket handler) receive
        # `(frame, text)` pairs, `text` being the wire encoding computed once
        # for every subscriber in `_fan_out`; everyone else gets the dict.
        self.encoded = encoded
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_frames + 1)
        self.desynchronized = False
        self.queued_bytes = 0
        self.last_queued_seq: int | None = None
        # Sizes of the frames still in `queue`, oldest first. The consumer
        # holds the queue directly and never tells us it read something, so
        # the count it has taken is recovered from `qsize()`: this producer is
        # the only one putting, so `len(_queued_sizes) - qsize()` is exactly
        # how many have been consumed since the last look.
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
            # Deliberately not buffering: this client has already been told
            # its stream is broken and will reload. Holding frames for it is
            # exactly the unbounded growth B-33 is about.
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
        # Fits by construction: `maxsize` is `max_frames + 1` and the cap was
        # checked before this put.
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
    """Fan the Hermes event stream(s) out to every connected client.

    Drains the default connection's adapter itself (`_run`); every other
    profile's connection is drained by `ProfileConnectionManager`, which
    hands its frames in through `inject()` (B-136). Either way each frame
    leaves tagged `_profile`, bounded, and stamped with one shared `seq`.

    `HermesAdapter.events()` is backed by a **single shared `asyncio.Queue`**,
    and `Queue.get()` hands each item to exactly one waiter. So when N tasks
    iterate `adapter.events()` concurrently, they do not each see the whole
    stream -- they *split* it. Two clients each render roughly half the
    `message.delta` tokens of every reply.

    That is not hypothetical: it was observed end-to-end. A reply that Hermes
    completed as `'1\\n2\\n3\\n...12'` arrived at the client as
    `'\\n2\\n3\\n...12'` and as `'1\\n\\n3\\n...12'` on different runs -- a
    different token missing each time, because stale `/ws/events` handlers
    were still parked on the same queue stealing frames (see `ws_events`,
    which now also notices when a client goes away).

    So exactly one task iterates the adapter, and every subscriber gets its
    own queue. The per-stream `seq` counter lives here too, for the same
    reason: it must be monotonic across the whole upstream stream, not
    restarted per client.

    `seq` counts **forwarded** frames, not raw upstream ones (B-14). A client
    uses it as a replay cursor -- "give me everything after 184" -- so a gap
    in it is indistinguishable from a lost frame, and the client is right to
    treat it as one. It used to be spent before the normalizer had even been
    consulted, which meant every deliberately-dropped upstream event punched a
    hole in it: with `reasoning.delta` unmapped, a single real turn burned
    ~950 sequence numbers on frames nobody would ever receive. It is now
    allocated as the last step before fan-out, so what a subscriber sees is
    contiguous.
    """

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
    ) -> None:
        self._adapter = adapter
        # B-136: the profile whose connection `adapter` is. Stamped as
        # `_profile` on every frame this broadcaster forwards from its own
        # upstream; frames from other profiles' connections arrive through
        # `inject()` carrying their own name.
        self._profile = profile
        # P2-1/P2-2 hooks, wired by `lifespan`. Optional so a broadcaster
        # built without them (tests, a second mount) still forwards
        # everything -- the hooks add durability, attribution and injection,
        # they never gate forwarding.
        #
        # `on_background_complete` runs for every forwarded
        # `background.completed` frame: it persists the result (the frame is
        # the ONLY copy of it anywhere -- B-42), may backfill the frame's
        # `_stored_session_id` from the submit-time ledger record, and may
        # return one synthesized message frame to inject right after it.
        #
        # `on_generation_change` runs when the Hermes connection is replaced:
        # running background tasks are orphaned and their sessions re-resumed
        # (P2-0b: the completion event is delivered only to attached
        # connections, so the re-resume is what rescues a still-running task),
        # and open runs are marked interrupted (P2-2).
        #
        # `on_canonical_event` (P2-2, `RunRecorder.handle_event`) runs for
        # EVERY forwarded frame, `(canonical, envelope)`: it writes the real
        # project/session/run attribution onto the envelope and persists the
        # run-relevant types into `run_events`. Synchronous by contract and
        # guarded -- persistence must never block or crash the fan-out.
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
        # B-29: how a live handle on an event is turned back into the STORED
        # id the app actually holds. Optional so a test (or any caller that
        # has no cache) still gets well-formed frames -- they simply carry a
        # null `_stored_session_id`, which is the honest answer.
        self._live_handle_cache = live_handle_cache
        # B-29: which Hermes connection the frames we are forwarding belong
        # to. Seeded from the adapter so construction alone never looks like a
        # reconnect; every later move emits `stream.resync`.
        self._generation_seen: int | None = self._connection_generation()
        # B-14: raw event names we have already complained about. Per type,
        # not per frame -- an unmapped `reasoning.delta` would otherwise emit
        # 945 identical warnings in a single turn, which is how a real signal
        # gets tuned out. Per broadcaster instance so the first turn after a
        # restart still reports what it saw.
        self._unknown_types_logged: set[str] = set()

    @property
    def subscriber_count(self) -> int:
        """Live subscriptions. A client that hung up must not still be here."""
        return len(self._subscribers)

    def start(self) -> None:
        """Begin draining the upstream stream now, subscribers or not.

        `/ws/events` is a *live* feed, and the adapter's event queue is
        unbounded. If nothing consumes it until the first client subscribes,
        everything Hermes pushed in the meantime sits there and is flushed at
        that client the instant it connects. Verified live: a brand-new
        socket was handed a complete reply to a turn that had finished 20
        seconds before the client existed -- which the app then types into
        whatever transcript is open, duplicating a reply the user already has.

        Draining from process start means a subscriber only ever sees what
        arrives *after* it subscribes, and the queue cannot grow without
        bound while nobody is connected.

        Also starts the connection-generation watcher (B-29). That one has to
        be a poller rather than a check inside the pump, because the case it
        exists for is precisely the one where the pump has nothing to do: the
        upstream socket is replaced and Hermes then says nothing for a while,
        so a client would go on filtering frames against a handle that can
        never match again with no event to trigger the discovery.

        Idempotent, and also the restart path if either task ever dies.
        """
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._run())
        if self._generation_watch is None or self._generation_watch.done():
            self._generation_watch = asyncio.create_task(
                self._watch_generation(self._generation_poll_seconds)
            )

    @asynccontextmanager
    async def subscribe(self, *, encoded: bool = False) -> AsyncIterator[asyncio.Queue[Any]]:
        """Register a private feed, guaranteed to be unregistered on exit.

        The queue handed back is bounded (B-33): a subscriber that stops
        reading is cut off with a `stream.desynchronized` frame rather than
        being allowed to buffer the whole event stream in the gateway's heap.

        `encoded=True` (the websocket handler) yields `(frame, text)` pairs
        where `text` is the wire encoding computed once per frame for every
        client; the default yields the frame dict.
        """
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

    # --- B-29: connection generation ------------------------------------

    def _connection_generation(self) -> int | None:
        """The adapter's connection generation, or None if it has no such idea.

        Same tolerance as `LiveHandleCache._generation`: an adapter that does
        not expose the counter simply cannot make the guarantee, so frames
        carry a null generation rather than a made-up one.
        """
        generation = getattr(self._adapter, "connection_generation", None)
        return generation if isinstance(generation, int) else None

    def _note_connection_generation(self) -> int | None:
        """Emit `stream.resync` if the Hermes connection has been replaced.

        The whole of B-29 in one place: the moment this counter moves, every
        live handle every client is holding is void, and a client filtering on
        one would silently drop the entire stream. Telling it beats letting it
        find out by never rendering anything again.
        """
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
            # After the resync frame: subscribers hear "your handles are void"
            # before any consequence of the reconnect (orphaning, rescue
            # traffic) can surface. Guarded like the pump body -- a ledger
            # failure must not kill generation tracking.
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
        """Hand one already-bounded frame to every subscriber that can take it.

        Encoded exactly once here (CLEANUP_PLAN 3.9): the wire text goes to
        every encoded subscriber as-is, and its byte length is the size every
        subscriber's backlog budget is charged.
        """
        text = _encode_frame(frame)
        size = len(text.encode("utf-8"))
        for subscriber in list(self._subscribers):
            subscriber.offer(frame, size, generation, text)

    def _note_unnormalized(self, raw_event: dict[str, Any]) -> None:
        """Make an event type we don't handle discoverable, exactly once (B-14).

        A drop listed in `DELIBERATELY_DROPPED_RAW_EVENTS` is a decision and
        stays quiet at info level. Anything else is a type nobody has looked at
        -- `reasoning.delta` was one of those for the whole of Phase 0, and the
        only reason it was ever found is that someone happened to tcpdump a
        turn. It gets a warning, once per type, so the next one shows up in the
        log instead of vanishing.
        """
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
            # One malformed frame must not take the pump down. This task is
            # the ONLY thing draining the upstream stream: if it dies, every
            # connected client goes permanently silent with no error anywhere
            # -- an `asyncio.Task` exception is only reported when the task is
            # garbage-collected, and `start()` is not called again until some
            # *new* client subscribes. That is precisely the silent-stall
            # failure mode B-14 was about, reintroduced one level up. Per
            # frame, so a single bad event costs exactly that event.
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
        """Add the STORED id (and the live handle, and the generation) to a frame.

        This is the server half of B-29, and it exists because of the trap
        `docs/PROTOCOL_VERIFIED.md` records: **`payload.session_id` does not
        mean the same thing on every event.** On almost all of them it is the
        *live* handle (`07ff0d50`), which is process-local and is re-minted on
        every gateway<->Hermes reconnect; on `session.title` it is the
        *stored* id (`20260829_182702_185869`); and `session.info` carries
        both, under `session_id` (live) and `stored_session_id`. A client that
        filters frames by comparing `payload.session_id` to the handle it is
        holding therefore drops the whole stream, silently, after any
        reconnect -- and mis-reads `session.title` even without one.

        So the gateway answers the question itself, once, in the one place
        that knows both id spaces, and writes the answer into
        `_stored_session_id`. That key means the same thing on every frame.

        Attribution is deliberately **read-only against the submit path**: a
        `session.info` teaches `LiveHandleCache` a live->stored mapping for
        labelling frames (`observe_live_mapping`) but never a stored->live one,
        because that second map is what `prompt.submit` resolves against and a
        wrong entry there would send the user's message into the wrong
        research session.
        """
        stored_id, live_id = session_identity_from_payload(raw_type, payload)

        cache = self._live_handle_cache
        if cache is not None:
            if stored_id is not None and live_id is not None:
                cache.observe_live_mapping(stored_id, live_id)
            elif live_id is not None:
                stored_id = cache.stored_for_live(live_id)
            elif stored_id is not None:
                live_id = cache.get(stored_id)

        # Assigned, never `setdefault`: `_`-prefixed keys are the gateway's
        # namespace (B-02/B-14), so upstream must not be able to forge one.
        payload[STORED_SESSION_ID_FIELD] = stored_id
        payload[LIVE_SESSION_ID_FIELD] = live_id
        payload[CONNECTION_GENERATION_FIELD] = generation
        payload[PROFILE_FIELD] = self._profile

    def inject(self, envelope: dict[str, Any], *, profile: str) -> None:
        """Fan out a frame that came off ANOTHER profile's connection (B-136).

        `ProfileConnectionManager` drains every non-default profile's
        dedicated adapter itself (`domain/profile_connection.py::_forward_one`)
        and, until this seam existed, fed only the chat store and the run
        recorder -- so a `kimi25` turn was recorded, transcribed, and never
        streamed: the Run Inspector filled up second by second while the chat
        showed nothing until a reload. This is the missing last hop.

        `envelope` is already a normalized, identity-stamped canonical
        envelope (the caller ran `normalize_event` and its own
        `_stamp_identity`, and the recorder has already written run
        attribution onto it). What this method adds is exactly what every
        default-profile frame gets on its way out of `_forward_one`, and
        nothing the caller has already done:

        * **`_profile`**, assigned rather than `setdefault` -- the gateway's
          namespace, so neither upstream nor a mis-wired caller can mislabel
          a frame. A stored id is only unique WITHIN a profile, so this key
          is what lets a client tell one agent's `20260906_...` session from
          another agent's session of the same name.
        * the B-02 size bound, so one oversized `tool.completed` cannot kill
          every client's socket;
        * a `seq` from the same counter every other forwarded frame draws
          from, spent last, so what a subscriber sees stays contiguous
          (B-14) regardless of how many upstreams feed it;
        * the B-33 bounded per-subscriber offer.

        Deliberately NOT run here: `_on_canonical_event` (the recorder --
        the caller already invoked it with the right profile, and running it
        again would open every non-default turn twice, the exact
        double-record B-145's regression test guards against) and
        `_on_background_complete` (the ledger only knows the default
        connection's tasks). Hooks are the caller's job; fan-out is this
        method's.

        Synchronous and I/O-free, like `_forward_one`, so it is safe to call
        from inside another connection's pump loop.
        """
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
        # B-29: if the upstream socket was replaced, say so *before* the first
        # frame from the new connection goes out, so a client never sees a
        # frame stamped with a handle it has not been told to re-resolve.
        generation = self._note_connection_generation()
        context = EventContext(
            # Unattributed at birth (P2-2): the normalizer cannot know the
            # project/session/run, and explicit nulls are the honest envelope
            # until the RunRecorder hook below writes the real answer.
            project_id=None,
            session_id=None,
            run_id=None,
            # Placeholder. The real `seq` is stamped at the bottom of this
            # method, once the frame is known to be going out -- see B-14.
            seq=_SEQ_UNASSIGNED,
        )
        canonical = normalize_event(raw_event, context)
        if canonical is None:
            self._note_unnormalized(raw_event)
            return
        envelope = canonical.to_dict()
        # B-29, before the size check: the identity keys are part of the frame
        # that gets measured, and they survive the degradation if it happens.
        self._stamp_session_identity(raw_event_type(raw_event), envelope["payload"], generation)
        # P2-2: real attribution + selective persistence. After identity
        # stamping (the recorder keys off `_stored_session_id`), before the
        # size check (so the attributed envelope is what gets measured) and
        # before the background hook (which copies envelope attribution onto
        # its injected frame). Guarded twice -- the recorder never raises by
        # contract, and the pump survives even if that contract breaks.
        if self._on_canonical_event is not None:
            try:
                self._on_canonical_event(canonical, envelope)
            except Exception:  # pragma: no cover - defensive
                logger.exception("canonical-event hook failed; forwarding the frame unattributed")
        # P2-1/B-42: a background.completed frame is a task's only
        # announcement and its payload the only copy of the result, so the
        # ledger hook runs before fan-out -- it persists the result, can
        # backfill `_stored_session_id` from the submit-time record (the
        # frame carries only a live handle), and may hand back one
        # synthesized message frame to inject after this one. After stamping,
        # so the hook sees the cache's attribution; before the size check, so
        # a backfilled id is part of what gets measured and degraded-but-kept
        # (`_kept_session_identity`). Failure inside the hook never costs the
        # frame its fan-out.
        injected: dict[str, Any] | None = None
        if (
            canonical.type == BACKGROUND_COMPLETED_EVENT_TYPE
            and self._on_background_complete is not None
        ):
            try:
                injected = self._on_background_complete(envelope)
            except Exception:  # pragma: no cover - defensive
                logger.exception("background-complete hook failed; forwarding the frame unledgered")
        # B-02: one frame the client cannot receive kills the whole event
        # socket (1009), not just that event. Measured once here, for all
        # subscribers, since they all get the same bytes.
        frame, size = _bounded_client_frame_with_size(envelope)
        if frame is None:
            # Unreachable (see `_bounded_client_frame_with_size`), and already
            # logged at error level there.
            return
        # B-14: `seq` is the client's replay cursor, so it may only be
        # spent on a frame the client actually receives. Assigned here,
        # last, after every decision that could still discard this frame --
        # dropped types and unsendable frames now cost nothing, and what
        # arrives at a subscriber is contiguous.
        frame["seq"] = next(self._seq)
        # Bounded per subscriber (B-33). Nothing is ever removed from the
        # middle -- that would silently corrupt the transcript being
        # assembled -- so a client that falls past the cap is cut off with
        # an explicit `stream.desynchronized` frame instead. `_fan_out`
        # encodes once for every subscriber (CLEANUP_PLAN 3.9); `size` from
        # the bounding step above was the over-estimate, the exact wire bytes
        # are what the backlog budget is charged.
        del size
        self._fan_out(frame, generation)

        if injected is not None:
            # The synthesized background-result message (P2-1), right behind
            # the completion frame it was derived from. Bounded like any
            # frame (its text is an agent-written result of arbitrary size,
            # B-02) and stamped `seq` 0 like every gateway-synthesized frame
            # (`stream.ready`/`stream.resync`): `seq` is the replay cursor
            # over the *upstream* stream and Hermes never sent this.
            injected_frame = _bounded_client_frame(injected)
            if injected_frame is not None:
                injected_frame["seq"] = _SEQ_GATEWAY_SYNTHETIC
                self._fan_out(injected_frame, generation)
