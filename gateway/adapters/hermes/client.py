"""HermesAdapter — client for the Hermes TUI Gateway wire protocol.

Implements exactly the flow verified in `docs/PROTOCOL_VERIFIED.md` (task
P0-2 in `TASKS.md`): password login -> cookie jar -> mint a fresh single-use
ws-ticket immediately before each connect -> connect
`ws://host:port/api/ws?ticket=...` -> newline-delimited JSON-RPC 2.0 in both
directions. Nothing here invents an alternative auth mechanism (no static
bearer token, no long-lived cached ticket) — see PROTOCOL_VERIFIED.md if that
ever looks tempting, it was explicitly ruled out against the real instance.

This module makes no live network calls at import time or construction time
— a `HermesAdapter()` only talks to Hermes once `login()` / `mint_ticket()` /
`connect()` / `request()` are awaited.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets
import websockets.exceptions

from config.settings import Settings, get_settings

from .exceptions import (
    HermesAuthError,
    HermesConnectionError,
    HermesProtocolError,
    HermesRPCError,
)
from .models import APPROVAL_CHOICES, REWIND_FIELDS, RewindRequest

logger = logging.getLogger(__name__)

# Cookies set by a successful `/auth/password-login`, per PROTOCOL_VERIFIED.md.
_SESSION_COOKIE_NAMES = ("hermes_session_at", "hermes_session_rt", "hermes_session_provider")

_LOGIN_PATH = "/auth/password-login"
_TICKET_PATH = "/api/auth/ws-ticket"
_WS_PATH = "/api/ws"

# Hermes's own dashboard file API (PV "Phase 3 probe", 2026-08-30). These two
# routes are UNDOCUMENTED upstream -- found by probing, verified live
# (byte-exact sha256, native Range/206, 401 without the session cookie, 403
# outside `/opt/data`) -- and must be treated as fragile: nothing guarantees
# their shape survives a Hermes upgrade. `api/sandbox.py` health-checks them
# for exactly that reason.
_FILES_LIST_PATH = "/api/files"
_FILES_DOWNLOAD_PATH = "/api/files/download"

# Hermes's dashboard text-file API, used by the editor (P7, EDITOR_DESIGN.md
# section 1; PV "File text read/write for the editor", measured live
# 2026-09-06). Same fragility caveat as the two routes above -- and one
# sharper one: NEITHER of these confines to the managed root (reading
# `/etc/passwd` and writing `/tmp/x` both succeeded live), so the gateway's
# own path validation in `api/sandbox_text.py` is the only wall.
_FS_READ_TEXT_PATH = "/api/fs/read-text"
_FS_WRITE_TEXT_PATH = "/api/fs/write-text"
# `POST /api/files/mkdir` `{"path"}` -- makes the directory AND its parents
# (`target.mkdir(parents=True, exist_ok=True)`, verified in Hermes's own
# `web_server.py`). This is what lets the gateway create a per-project
# instructions folder before `fs_write_text` (which 400s on a missing parent)
# writes `HERMES.md` into it. Same no-confinement caveat as the two routes
# above: `api/projects.py` validates the path against the sandbox root first.
_FILES_MKDIR_PATH = "/api/files/mkdir"

_READY_EVENT_METHOD = "gateway.ready"
_READY_TIMEOUT_S = 10.0
_REQUEST_TIMEOUT_S = 30.0
# Keepalive so an idle socket (nobody touching the phone for a while) keeps
# producing inbound traffic and isn't reaped upstream.
#
# The method has to be one the live instance actually implements. Verified
# against the real Hermes 2026-08-29: `session.active_list` answers with
# `{"sessions": [...]}` (it is in PROTOCOL_VERIFIED.md's confirmed catalog),
# while `gateway.ping`, `ping`, `gateway.heartbeat` and `heartbeat` all come
# back `[-32601] unknown method`, and this instance's `gateway.ready` payload
# carries only `change_events` and `skin` -- no heartbeat feature flag. A
# keepalive aimed at a method that does not exist is worse than none: every
# beat would look like a transport failure.
_HEARTBEAT_INTERVAL_S = 25.0
_KEEPALIVE_METHOD = "session.active_list"

# `websockets` defaults to a 1 MiB frame limit and *closes the connection*
# (1009) when the peer exceeds it. `session.resume` returns the whole saved
# transcript in one frame, and a real research session blows past 1 MiB:
# measured 1,639,756 bytes for "Configure hindsight and lcm plugins across
# agents" on the live instance. Symptom was "Hermes WebSocket closed while a
# request was pending" on *some* sessions -- and message count does not
# predict it (a 165-message session measured 454 KB, a 126-message one
# 1.6 MB) because tool output dominates the payload.
#
# Bounded rather than None: transcripts only grow, and an unlimited frame
# size would let one enormous session exhaust the gateway's memory instead
# of failing one request. 64 MiB is ~40x the largest transcript observed.
_MAX_WS_FRAME_BYTES = 64 * 1024 * 1024

#: How many upstream events may sit unread in `events()`'s queue before new
#: ones are dropped. Sized for a burst, not a backlog: a turn is ~15 frames
#: (HANDOFF), so 10,000 is minutes of streaming with the drainer alive.
_MAX_QUEUED_UPSTREAM_EVENTS = 10_000

# --- Server -> client requests (B-197, Hermes 0.21.3) ---------------------
#
# The four human-in-the-loop prompts USED to be event notifications
# (`clarify.request`) answered by a matching RPC (`clarify.respond`). On
# 0.21.3 they are JSON-RPC requests sent FROM the server TO this client:
#
#   {"jsonrpc": "2.0", "id": "srq-<12 hex>", "method": "clarify",
#    "params": {"session_id": "...", "questions": [...]}}
#
# and are answered with `request.answer {id, result}` (or by writing a bare
# response frame with the same id). Hermes's own `tui_gateway/server_requests.py`
# states the change outright: "no paired `*.request` notification /
# `*.respond` method, no per-kind `*.expire`". The event catalogue on 0.21.3
# contains none of `clarify.request`, `approval.request`, `sudo.request`,
# `secret.request`, `sudo.expire`, `secret.expire`.
#
# Rather than teach every layer above a second shape, the adapter translates
# the request back into the raw event name the normalizer already knows, so
# `events/canonical.py`, the run recorder, persistence and the app's own
# decoder are untouched by the migration.
SERVER_REQUEST_RAW_TYPES: dict[str, str] = {
    "clarify": "clarify.request",
    "approval": "approval.request",
    "sudo": "sudo.request",
    "secret": "secret.request",
}

#: The open request's own id, stamped into the payload so the answer routes
#: can address it. `_`-prefixed: the gateway's namespace, never
#: forgeable from upstream because it is assigned, not merged.
SERVER_REQUEST_ID_FIELD = "_srq_id"

#: Hermes withdraws any open request with one of these, replacing the old
#: per-kind `sudo.expire` / `secret.expire` events.
REQUEST_CANCEL_EVENT = "request.cancel"

#: How many `srq-id -> request_id` pairs to remember so a `request.cancel`
#: can name the prompt the client is showing. One per outstanding prompt in
#: practice; the cap only bounds a pathological run.
_MAX_TRACKED_SERVER_REQUESTS = 256


class HermesAdapter:
    """One Hermes session's worth of transport: HTTP login/ticket + one WS connection.

    Public interface other agents/code should call:
        await login()                          -- HTTP, sets session cookies
        await connect()                         -- mints a ticket, opens the WS,
                                                    waits for gateway.ready
        await request(method, params)           -- raw JSON-RPC call/response
        async for event in adapter.events(): ..  -- raw server-pushed events
        await close()                           -- tears down HTTP + WS

        Typed RPC methods (thin wrappers over request()):
            session_list, session_create, session_history, prompt_submit,
            session_interrupt, resume_session, approval_respond,
            clarify_respond, sudo_respond, secret_respond

        Sandbox file routes (plain HTTP, no WS needed -- P3-1a):
            files_list, files_download

    Usable as an async context manager: `async with HermesAdapter() as h: ...`
    guarantees `close()` runs.
    """

    # Verified live 2026-08-29: `session.resume` is the method that loads a
    # *saved* transcript. `session.activate` only switches between sessions
    # that are already live in the gateway process and rejects a stored id
    # with [4001] session not found. See PROTOCOL_VERIFIED.md "Session
    # resume & the two id spaces".
    _RESUME_METHOD = "session.resume"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._http = httpx.AsyncClient(base_url=self._settings.hermes_base_url)
        self._logged_in = False

        # Typed as `Any` rather than a concrete websockets client-connection
        # class: the installed `websockets` (17.x) makes `websockets.connect`
        # resolve to its newer asyncio implementation whose connection type
        # differs from the legacy `WebSocketClientProtocol` (now deprecated),
        # and this adapter only relies on the common `send()` /
        # async-iteration / `close()` surface either version provides.
        self._ws: Any | None = None
        # Whether `self._ws` is actually *usable*. `self._ws is not None` is not
        # enough: when Hermes goes away (restart, Mac sleep, Wi-Fi blip) the
        # receive loop just ends and the connection object stays assigned. A
        # gateway that only checked `self._ws is not None` would think it was
        # still connected forever and every later call would fail with "failed
        # to send ..." until the process was restarted.
        self._ws_live = False
        self._recv_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        # `srq-<id>` -> the `request_id` clients were shown for it, so a
        # `request.cancel` can name the card to tear down. Insertion-ordered
        # and capped (`_MAX_TRACKED_SERVER_REQUESTS`).
        self._server_request_ids: dict[str, str] = {}
        #: In-flight declines, held so the loop cannot collect them mid-send.
        self._decline_tasks: set[asyncio.Task[None]] = set()
        # Bounded (CLEANUP_PLAN 3.9): nothing legitimately leaves this many
        # frames unread -- the broadcaster and every profile pump drain it
        # continuously -- so a full queue means the drainer is gone, and the
        # right outcome is a loud drop, not a heap that grows until the
        # process dies. Dropped frames are counted and logged once per
        # connection.
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=_MAX_QUEUED_UPSTREAM_EVENTS
        )
        self.dropped_events = 0
        self._drop_logged_generation: int | None = None
        self._id_counter = itertools.count(1)
        self._ready = asyncio.Event()
        # Which connection we are on. 0 == never connected; every `connect()`
        # bumps it. Anything derived from a live Hermes connection (above all
        # a *live session handle*, which is process-local to the Hermes
        # gateway and not stable across connections) can be tagged with the
        # generation it came from and become unusable by construction the
        # moment this counter moves. See `connection_generation`.
        self._connection_generation = 0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        # Deliberately does not touch self._http.cookies or any ticket/password
        # material, so a stray `print(adapter)`/`repr(adapter)` in a debugger
        # or log call can never leak a credential.
        state = "connected" if self.is_connected else ("logged_in" if self._logged_in else "fresh")
        return f"HermesAdapter(host={self._settings.hermes_host!r}, state={state!r})"

    @property
    def is_logged_in(self) -> bool:
        """Whether `login()` has completed successfully."""
        return self._logged_in

    @property
    def is_connected(self) -> bool:
        """Whether `connect()` has completed and the WebSocket is still open.

        Public counterpart to the internal `_ws` check other modules (e.g. the
        FastAPI service) need in order to lazily/idempotently drive
        `login()`/`connect()` without reaching into private attributes.

        Goes back to False as soon as the receive loop ends, which is the only
        signal that Hermes hung up. Without that, a single dropped upstream
        socket wedges the gateway permanently: `_ensure_connected()` would
        no-op forever and every request would 502 until uvicorn was restarted.
        """
        return self._ws is not None and self._ws_live

    @property
    def connection_generation(self) -> int:
        """Monotonic counter identifying *which* Hermes connection we are on.

        0 before the first `connect()`; incremented by every `connect()` call
        (including a reconnect after a drop, and including one that then fails
        to complete -- the bump happens as soon as the previous socket is torn
        down, because everything derived from that socket is already dead at
        that point).

        Exists so a caller can cache something that is only valid for one
        connection -- specifically a *live session handle* -- and have the
        cache entry become unreachable automatically rather than by
        remembering to invalidate it. Hermes's live handles are process-local
        and were verified to differ across connections for the same stored id
        (`5bfd9de6`, then `d8779141`, then `7373be85`); a handle reused across
        a reconnect fails with a bare `[4001] session not found`. Tag a cached
        handle with the generation it was resolved on and refuse it when this
        value has moved: that is a structural guarantee, not a hope. See
        `api.main.LiveHandleCache`.
        """
        return self._connection_generation

    async def __aenter__(self) -> HermesAdapter:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Auth + connection lifecycle
    # ------------------------------------------------------------------

    async def login(self) -> None:
        """POST /auth/password-login and store the resulting session cookies.

        Body is exactly `{"username", "password", "provider": "basic", "next":
        ""}` per PROTOCOL_VERIFIED.md. `httpx.AsyncClient` merges `Set-Cookie`
        response headers into its own persistent cookie jar automatically, so
        the cookies are "stored" simply by reusing `self._http` for every
        later request (mint_ticket) — they are never copied out into a
        loggable attribute on this object.
        """
        payload = {
            "username": self._settings.hermes_username,
            "password": self._settings.hermes_password.get_secret_value(),
            "provider": "basic",
            "next": "",
        }
        try:
            resp = await self._http.post(_LOGIN_PATH, json=payload)
        except httpx.HTTPError as exc:
            raise HermesConnectionError("password-login request failed") from exc
        finally:
            del payload  # drop the only reference holding the raw password

        if resp.status_code == 401:
            raise HermesAuthError("Hermes rejected the configured credentials (401)")
        if resp.status_code == 422:
            raise HermesAuthError(
                "password-login request was malformed (422) -- check the "
                "'provider' field is exactly 'basic'"
            )
        if resp.status_code >= 400:
            raise HermesAuthError(f"password-login failed with HTTP {resp.status_code}")

        missing = [name for name in _SESSION_COOKIE_NAMES if name not in self._http.cookies]
        if missing:
            raise HermesProtocolError(
                f"password-login returned {resp.status_code} but did not set "
                f"expected cookie(s): {missing}"
            )
        self._logged_in = True

    async def _ws_auth_query(self) -> str:
        """The credential query string for the `/api/ws` upgrade.

        A seam, not a setting. Hermes accepts different WebSocket credentials
        depending on how its dashboard was started: a single-use `?ticket=`
        when the auth gate is engaged (any non-loopback bind), and a plain
        `?token=` session token when it is not. This class always mints a
        ticket, which is right for every remote caller. The in-process plugin
        subclasses this to present whichever one the dashboard it lives inside
        will actually accept -- see `plugin/dashboard/plugin_api.py`.
        """
        return f"ticket={await self.mint_ticket()}"

    async def mint_ticket(self) -> str:
        """POST /api/auth/ws-ticket (cookie-authenticated).

        Returns a fresh ticket. Tickets are single-use with a 30-second TTL
        (per PROTOCOL_VERIFIED.md) -- this method is called fresh by
        `connect()` immediately before every connect attempt; the returned
        value is never cached on `self` or reused across calls.
        """
        if not self._logged_in:
            raise HermesAuthError("mint_ticket() called before a successful login()")
        try:
            resp = await self._http.post(_TICKET_PATH)
        except httpx.HTTPError as exc:
            raise HermesConnectionError("ws-ticket request failed") from exc

        if resp.status_code == 401:
            # The session cookie has expired. Drop the logged-in flag so the
            # next `login()`/`connect()` cycle re-authenticates with the
            # configured password instead of retrying a dead cookie forever.
            self._logged_in = False
            raise HermesAuthError("session cookie was rejected minting a ws-ticket (401)")
        if resp.status_code >= 400:
            raise HermesAuthError(f"ws-ticket request failed with HTTP {resp.status_code}")

        try:
            ticket = resp.json()["ticket"]
        except (ValueError, KeyError, TypeError) as exc:
            raise HermesProtocolError(
                "ws-ticket response did not contain a 'ticket' field"
            ) from exc
        if not isinstance(ticket, str) or not ticket:
            raise HermesProtocolError("ws-ticket response contained an empty/invalid ticket")
        return ticket

    async def connect(self) -> None:
        """Mint a fresh ticket, open the WS, and wait for `gateway.ready`.

        Per PROTOCOL_VERIFIED.md the server sends `gateway.ready` immediately
        on accept; this method blocks until that event has been observed
        (with a timeout) so callers never race a `request()` against a socket
        that isn't actually live on the server side yet.

        Safe to call again on the same adapter to *re*-connect: any previous
        socket (live or dead) is torn down first, so a long-lived service can
        recover from Hermes restarting without rebuilding the adapter. The
        HTTP client and its session cookies survive, so a reconnect normally
        costs one ws-ticket mint and no re-login.

        Bumps `connection_generation` *before* the new socket is opened. The
        old socket is already gone by then, so every live handle resolved on
        it is already worthless; moving the counter first means a
        generation-tagged cache can never hand back a handle from the dead
        connection, even if this connect attempt goes on to fail.
        """
        await self._teardown_ws()
        self._connection_generation += 1
        auth_query = await self._ws_auth_query()
        scheme = "wss" if self._settings.hermes_scheme == "https" else "ws"
        url = f"{scheme}://{self._settings.hermes_host}:{self._settings.hermes_port}{_WS_PATH}?{auth_query}"
        # The ticket is single-use and only ever needed in this URL. Drop the
        # local references once the connect attempt has been made so it
        # isn't sitting around in a variable a later debugging session might
        # print.
        try:
            self._ws = await websockets.connect(url, max_size=_MAX_WS_FRAME_BYTES)
        except Exception as exc:
            raise HermesConnectionError("failed to open the Hermes WebSocket") from exc
        finally:
            del url, auth_query

        self._ws_live = True
        self._ready.clear()
        self._recv_task = asyncio.create_task(self._recv_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=_READY_TIMEOUT_S)
        except TimeoutError as exc:
            # Only the socket, not the HTTP client: tearing down `self._http`
            # here would close the cookie jar too, so a single ready-timeout
            # would make every later login()/connect() attempt fail on a
            # closed client instead of simply retrying.
            await self._teardown_ws()
            raise HermesConnectionError(
                f"did not observe a '{_READY_EVENT_METHOD}' event within {_READY_TIMEOUT_S}s of connecting"
            ) from exc

        # Only once the socket is confirmed live on the server side.
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Poke Hermes periodically so an idle socket isn't reaped.

        Only a *transport* failure means the connection is gone. A JSON-RPC
        error reply is the opposite: the frame went out and an answer came
        back, so the socket demonstrably works and the beat must keep going.
        Treating an RPC error as death would tear down a perfectly healthy
        connection -- including one with a reply mid-stream, since the next
        request's `_ensure_connected()` would close the socket to "recover"
        it.

        On a real transport failure, mark the socket dead and stop: that
        flips `is_connected` so the next caller rebuilds the connection
        instead of discovering the corpse mid-request.
        """
        while True:
            try:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                await self.request(_KEEPALIVE_METHOD)
            except asyncio.CancelledError:
                raise
            except HermesRPCError:
                # Answered, just not happily. The socket is alive; keep beating.
                continue
            except Exception:
                self._ws_live = False
                return

    async def _teardown_ws(self) -> None:
        """Close the WebSocket half only, leaving the HTTP client usable.

        Separate from `close()` so `connect()` can replace a dead socket
        without destroying the session cookies it would need to reconnect.
        """
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        self._ws_live = False

    async def close(self) -> None:
        """Tear down the WebSocket (if any) and the HTTP client."""
        await self._teardown_ws()
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Sandbox file routes (P3-1a) -- cookie-authenticated plain HTTP on the
    # same host:port as the WS endpoint, per PV "Phase 3 probe". These need
    # no WebSocket: they ride `self._http`'s persistent cookie jar, the same
    # one `login()` fills and `mint_ticket()` uses.
    # ------------------------------------------------------------------

    async def files_list(self, path: str) -> httpx.Response:
        """`GET /api/files?path=<dir>` -- list a sandbox directory.

        Measured shape (PV "Phase 3 probe", verbatim): a JSON object
        `{"path", "parent", "entries": [{"name", "path", "is_directory",
        "size", "mtime", "mime_type"}, ...], "root", "locked_root",
        "can_change_path"}`. `mime_type` comes free per file entry, so no
        client-side sniffing is ever needed.

        Returns the fully-read `httpx.Response` rather than a parsed dict:
        the caller (`api/sandbox.py`) owns the status mapping (upstream 403
        "Path outside managed files root" / 404 are *client* answers to pass
        through, not adapter failures) and the shape health-check -- this
        route is UNDOCUMENTED upstream, so the adapter refuses to bake in
        assumptions beyond "it answers HTTP".

        Auth: logs in lazily if needed, and on a 401 (expired session
        cookie) drops the logged-in flag, re-logs-in with the configured
        password, and retries ONCE -- the same pattern `mint_ticket()` uses,
        because a long-lived gateway process otherwise starts silently
        401ing when the cookie expires.
        """
        return await self._files_get(_FILES_LIST_PATH, path, stream=False)

    async def files_download(self, path: str, *, range_header: str | None = None) -> httpx.Response:
        """`GET /api/files/download?path=<abs>` -- raw sandbox file bytes, STREAMED.

        Measured (PV "Phase 3 probe" + "Phase 3 build probes"): byte-exact
        (sha256-verified through 819,200 B), correct
        `Content-Type`/`Content-Length`, native `Range` support (`206` +
        `Content-Range`), 401 without the session cookie, 403 outside
        `/opt/data`.

        The response is opened with `stream=True` and its body is NOT read
        here -- the caller iterates `aiter_bytes()` and **must** call
        `await response.aclose()` when done (`api/sandbox.py` does this in
        the streaming generator's `finally`). Never buffer the whole body:
        that decision was made in P3-1a, not left open.

        `range_header` is passed through verbatim when given (e.g.
        `"bytes=100000-199999"`), so scrubbing an audio file or PDF costs
        only the requested slice. Same lazy-login + 401-retry-once behaviour
        as `files_list()`.
        """
        headers = {"Range": range_header} if range_header else None
        return await self._files_get(_FILES_DOWNLOAD_PATH, path, headers=headers, stream=True)

    async def fs_read_text(self, path: str) -> httpx.Response:
        """`GET /api/fs/read-text?path=<abs>` -- a sandbox file decoded as text.

        Measured live 2026-09-06 (PV "File text read/write for the editor"):

        * 200 `{"binary": bool, "byteSize": int, "language": str,
          "mimeType": str, "path": str, "text": str, "truncated": bool}`.
          `text` is UTF-8 decoded with `errors="replace"`, so a non-UTF-8
          file comes back with U+FFFD in it rather than failing; `truncated`
          is true past 512 KiB (`_FS_TEXT_PREVIEW_MAX_BYTES`) and the text is
          CUT at that point -- a truncated read can never be written back
          whole, which is why the editor refuses to edit one.
        * 404 `{"detail": "File not found"}`; 400 `Path points to a
          directory`.
        * **No confinement.** Reading `/etc/passwd` returned 200 live, and a
          relative path resolves against Hermes's own cwd. Nothing here
          checks the path: `api/sandbox_text.py` validates against the
          sandbox root BEFORE calling this, and that check is the only wall.

        Returns the fully-read `httpx.Response`; the caller owns the status
        mapping, same division of labour as `files_list()`. Lazy login and
        the 401-retry-once handling are shared with the other file routes.
        """
        return await self._files_get(_FS_READ_TEXT_PATH, path, stream=False)

    async def fs_write_text(self, path: str, content: str) -> httpx.Response:
        """`POST /api/fs/write-text` `{"path": <abs>, "content": <str>}` -- write a text file.

        Measured live 2026-09-06 (PV "File text read/write for the editor"):

        * 200 `{"ok": true, "path": str, "byteSize": int}`. The body key is
          `content` -- `text` is a 422 upstream.
        * 400 `Parent directory does not exist` / `Path points to a
          directory`; 413 `Content too large` above 8 MiB
          (`_FS_TEXT_WRITE_MAX_BYTES`).
        * The write is staged to a sibling temp file and renamed, so an
          interrupted write leaves the old file intact (read from source).
        * **No confinement.** Writing `/tmp/x` returned 200 live. Exactly as
          for `fs_read_text()`: the gateway's path validation is the only
          thing standing between an HTTP body and an arbitrary write on the
          Hermes host. Never call this with a path that has not been through
          `api.sandbox._validate_sandbox_path`.

        The content is never logged here or anywhere on the gateway.
        """
        return await self._files_post(_FS_WRITE_TEXT_PATH, {"path": path, "content": content})

    async def files_mkdir(self, path: str) -> httpx.Response:
        """`POST /api/files/mkdir` `{"path": <abs>}` -- create a directory and
        its parents.

        Measured live 2026-09-07: the
        handler is `target.mkdir(parents=True, exist_ok=True)`, so a nested
        path is made in one call and an existing directory is a no-op 200.
        409 if a *file* already sits at the path; 403 if the directory is not
        writable.

        Rides the same login cookie jar as `fs_write_text` (`_files_post`),
        so it authenticates exactly the way the editor's writes already do.
        **No confinement** upstream -- `api/projects.py` validates the path
        against the sandbox root before calling this, the same wall
        `api/sandbox_text.py` relies on.
        """
        return await self._files_post(_FILES_MKDIR_PATH, {"path": path})

    async def _files_get(
        self,
        url_path: str,
        file_path: str,
        *,
        headers: dict[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        """Shared GET for the file routes (`?path=` query): see `_files_request`."""
        return await self._files_request(
            "GET", url_path, params={"path": file_path}, headers=headers, stream=stream
        )

    async def _files_post(self, url_path: str, body: dict[str, Any]) -> httpx.Response:
        """Shared JSON POST for the file routes: see `_files_request`.

        `body` goes up as the JSON request body verbatim -- the caller shapes
        it to the measured wire contract (`fs_write_text`'s `{"path",
        "content"}`). Never streamed: the measured responses are small JSON.
        """
        return await self._files_request("POST", url_path, json_body=body, stream=False)

    async def _files_request(
        self,
        method: str,
        url_path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        """One HTTP request on the cookie jar: lazy login + one 401 retry.

        The retry mirrors `mint_ticket()`'s expired-cookie handling, but has
        to *complete* the recovery itself rather than just flagging it:
        `mint_ticket()` can rely on the next `login()`/`connect()` cycle,
        while a file request is the whole operation -- so on a 401 the flag
        is dropped, `login()` runs again with the configured password, and
        the request is re-sent exactly once. A second 401 after a *fresh
        successful* login is not a cookie-expiry problem (Hermes changed the
        route's auth, or the account lost access) and raises `HermesAuthError`
        so it fails loud instead of looping.

        A POST is re-sent on the 401 retry exactly like a GET: a 401 means
        Hermes refused the request before looking at the body, so nothing
        was written and the retry is not a double write.
        """
        attempted_relogin = False
        while True:
            if not self._logged_in:
                await self.login()
            request = self._http.build_request(
                method, url_path, params=params, json=json_body, headers=headers
            )
            try:
                response = await self._http.send(request, stream=stream)
            except httpx.HTTPError as exc:
                raise HermesConnectionError(f"{method} {url_path} request failed") from exc

            if response.status_code != 401:
                return response

            # Expired/rejected session cookie. Close the (possibly streamed)
            # error body before touching the client again.
            await response.aclose()
            self._logged_in = False
            if attempted_relogin:
                raise HermesAuthError(
                    f"{method} {url_path} still returned 401 after a fresh "
                    "successful login -- the files route's auth no longer "
                    "accepts this session (Hermes upgrade?)"
                )
            attempted_relogin = True

    # ------------------------------------------------------------------
    # Wire-level receive loop
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        ws = self._ws
        try:
            async for raw_message in ws:
                self._dispatch_message(raw_message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # The receive loop ending *is* the connection ending -- there is no
            # other notification. Mark the socket dead so `is_connected` turns
            # False and the next `connect()` re-establishes it, rather than
            # leaving a corpse that every later `request()` tries to send on.
            self._ws_live = False
            # `request()` inserts into `_pending` *before* the send and pops
            # its own entry if the send fails, so everything still in here was
            # successfully written to the socket: Hermes received these calls
            # and only their replies were lost. Say so, or `_with_reconnect()`
            # will replay a `prompt.submit` that already reached a real
            # session.
            closed_exc = HermesConnectionError(
                "Hermes WebSocket closed while a request was pending",
                request_was_sent=True,
            )
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(closed_exc)
            self._pending.clear()

    def _dispatch_message(self, raw_message: str | bytes) -> None:
        """Split one incoming WS message into newline-delimited JSON lines.

        PROTOCOL_VERIFIED.md describes the wire format as "newline-delimited
        JSON-RPC 2.0 ... one JSON object per line -- no additional framing"
        and separately notes the server may coalesce/batch streamed
        `message.delta` tokens. Rather than assume a WS message == exactly
        one JSON-RPC frame, this splits on '\\n' defensively so a single WS
        message containing several coalesced newline-delimited objects is
        still handled correctly; the common case (one object, no trailing
        newline) is just a one-element split.
        """
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8", errors="replace")
        for line in raw_message.split("\n"):
            line = line.strip()
            if line:
                self._dispatch_line(line)

    def _dispatch_line(self, line: str) -> None:
        try:
            frame = json.loads(line)
        except ValueError:
            # Never include the raw line in the log: unclear what a malformed
            # frame might contain, and the credential-hygiene requirement
            # errs toward silence over a chance leak. Surface it as a
            # synthetic event instead so a caller/normalizer can at least
            # count/observe that it happened.
            self._enqueue_event({"method": "_adapter.malformed_frame", "params": {}})
            return
        if not isinstance(frame, dict):
            self._enqueue_event({"method": "_adapter.malformed_frame", "params": {}})
            return

        frame_id = frame.get("id")
        is_response = frame_id is not None and ("result" in frame or "error" in frame)
        if is_response and frame_id in self._pending:
            future = self._pending.pop(frame_id)
            if not future.done():
                future.set_result(frame)
            return

        # a server -> client REQUEST. Hermes mints string ids
        # (`srq-<12 hex>`) precisely so they cannot collide with a client's
        # integer ones, so a string id on a frame carrying a `method` is
        # never an answer to something we sent.
        if not is_response and isinstance(frame_id, str) and isinstance(frame.get("method"), str):
            self._dispatch_server_request(frame_id, frame["method"], frame.get("params"))
            return

        # Anything else is a server-initiated event: gateway.ready,
        # message.delta, tool.start/progress/complete, approval.request,
        # clarify.request, sudo.request/expire, secret.request/expire, etc.
        #
        # CORRECTION (found live against the real instance, P0-9): every
        # server-pushed event's *outer* JSON-RPC envelope has a literal
        # `"method": "event"` -- it is NOT the actual event name as
        # PROTOCOL_VERIFIED.md's prose ("newline-delimited JSON-RPC 2.0")
        # implied. The real event name lives at `params.type`, and the
        # event's own payload is nested one level deeper at `params.payload`
        # (some events, e.g. `message.start`, carry only `session_id` and no
        # `payload` at all). Confirmed via a raw-frame capture:
        #   {"jsonrpc": "2.0", "method": "event",
        #    "params": {"type": "gateway.ready", "payload": {...}}}
        # Unwrap that envelope here so every downstream consumer of
        # `events()` (this adapter's own `_ready` gate, the profile pumps, the
        # event normalizer) can keep treating `event["method"]` as the
        # real event name (`gateway.ready`, `message.delta`, `tool.start`,
        # ...) exactly as the event catalog in PROTOCOL_VERIFIED.md
        # documents, without each of them re-deriving this unwrap
        # themselves.
        if frame.get("method") == "event":
            outer_params = frame.get("params")
            outer_params = outer_params if isinstance(outer_params, dict) else {}
            event_type = outer_params.get("type")

            payload = outer_params.get("payload")
            normalized_params: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {}
            if "session_id" in outer_params and "session_id" not in normalized_params:
                normalized_params["session_id"] = outer_params["session_id"]

            frame = {"method": event_type, "params": normalized_params}

            if event_type == REQUEST_CANCEL_EVENT:
                translated = self._translate_request_cancel(normalized_params)
                if translated is None:
                    return
                frame = translated

        if frame.get("method") == _READY_EVENT_METHOD:
            self._ready.set()
        self._enqueue_event(frame)

    def _dispatch_server_request(self, srq_id: str, method: str, params: Any) -> None:
        """Turn one server -> client request into the raw event it replaced.

        A method we cannot render is DECLINED rather than ignored. Hermes
        routes each request to the single transport that owns the session
        (`tui_gateway/server.py::write_json`), so nobody else is waiting to
        answer it, and an unanswered request blocks the agent for the whole
        clarify timeout -- which the operator may have configured as unlimited.
        An error response is what Hermes itself documents a client without a
        handler for that method doing, and it frees the turn immediately.
        """
        payload = dict(params) if isinstance(params, dict) else {}
        raw_type = SERVER_REQUEST_RAW_TYPES.get(method)
        if raw_type is None:
            self._decline_server_request(srq_id, method)
            return

        payload[SERVER_REQUEST_ID_FIELD] = srq_id
        # Approval carries its own `request_id` (the approval queue's, which
        # `approval.respond` still takes); the other three have none, and the
        # open request's id is what answers them.
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = srq_id
            payload["request_id"] = srq_id
        self._remember_server_request(srq_id, request_id)
        self._enqueue_event({"method": raw_type, "params": payload})

    def _remember_server_request(self, srq_id: str, request_id: str) -> None:
        self._server_request_ids[srq_id] = request_id
        while len(self._server_request_ids) > _MAX_TRACKED_SERVER_REQUESTS:
            self._server_request_ids.pop(next(iter(self._server_request_ids)))

    def _translate_request_cancel(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """`request.cancel {id, method, reason}` -> the raw `*.cancel` name.

        Returns `None` for a request kind this gateway never surfaced, so a
        withdrawn `vault.unlock_prompt` does not reach clients as a
        resolution for a card they were never shown.
        """
        method = params.get("method")
        raw_type = SERVER_REQUEST_RAW_TYPES.get(method) if isinstance(method, str) else None
        if raw_type is None:
            return None
        srq_id = params.get("id")
        payload = dict(params)
        if isinstance(srq_id, str):
            payload[SERVER_REQUEST_ID_FIELD] = srq_id
            # The id the client is showing, which for an approval is not the
            # open request's id.
            payload["request_id"] = self._server_request_ids.pop(srq_id, srq_id)
        return {"method": raw_type.replace(".request", ".cancel"), "params": payload}

    def _decline_server_request(self, srq_id: str, method: str) -> None:
        """Answer one unrenderable request with a JSON-RPC error, best effort."""
        logger.warning(
            "declining Hermes server request %r: this gateway has no handler for it, "
            "and leaving it unanswered would block the turn until its timeout",
            method,
        )
        frame = {
            "jsonrpc": "2.0",
            "id": srq_id,
            "error": {"code": -32601, "message": f"no handler for {method!r}"},
        }
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - dispatch always runs on the loop
            return
        # Referenced so the task cannot be garbage-collected mid-flight
        # (RUF006): a decline that vanishes is a turn that blocks.
        task = loop.create_task(self._write_frame(frame))
        self._decline_tasks.add(task)
        task.add_done_callback(self._decline_tasks.discard)

    async def _write_frame(self, frame: dict[str, Any]) -> None:
        """Write one frame with no correlated answer. Never raises: this runs
        detached from any request, and a failed decline must not take the
        receive loop's task tree down with it."""
        try:
            if self._ws is not None:
                await self._ws.send(json.dumps(frame) + "\n")
        except Exception:  # pragma: no cover - defensive
            logger.debug("failed to write an unsolicited frame", exc_info=True)

    def _enqueue_event(self, frame: dict[str, Any]) -> None:
        """Queue one event for `events()`, dropping (loudly) if nobody drains."""
        try:
            self._event_queue.put_nowait(frame)
        except asyncio.QueueFull:
            self.dropped_events += 1
            if self._drop_logged_generation != self._connection_generation:
                self._drop_logged_generation = self._connection_generation
                logger.warning(
                    "upstream event queue full (%d frames unread): dropping events; "
                    "nothing is draining this adapter's events()",
                    self._event_queue.maxsize,
                )

    # ------------------------------------------------------------------
    # Generic JSON-RPC request/response + event stream
    # ------------------------------------------------------------------

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one JSON-RPC 2.0 request and await the correlated response.

        Correlates strictly by "id". Raises `HermesRPCError` if the server
        responds with a JSON-RPC `error` object, `HermesProtocolError` on
        timeout, `HermesConnectionError` if not connected or the send fails.
        """
        if not self.is_connected:
            raise HermesConnectionError(
                "request() called before connect(), or after the Hermes "
                "WebSocket closed -- call connect() again"
            )

        request_id = next(self._id_counter)
        envelope = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._ws.send(json.dumps(envelope) + "\n")
        except Exception as exc:
            self._pending.pop(request_id, None)
            # Mark the socket dead so `is_connected` flips to False and the
            # caller's `_ensure_connected()` actually reconnects. Without this
            # the adapter reports "connected" over a corpse forever: the recv
            # loop may never observe a close frame for a socket that died
            # mid-send, so every later request fails the same way and only a
            # process restart clears it.
            self._ws_live = False
            raise HermesConnectionError(f"failed to send '{method}' request") from exc

        try:
            frame = await asyncio.wait_for(future, timeout=_REQUEST_TIMEOUT_S)
        except TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise HermesProtocolError(f"'{method}' timed out waiting for a response") from exc

        if "error" in frame and frame["error"] is not None:
            err = frame["error"] or {}
            raise HermesRPCError(
                method, err.get("code"), err.get("message", "unknown error"), err.get("data")
            )
        return frame.get("result") or {}

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield normalized-but-still-raw JSON-RPC event dicts as they arrive.

        Covers message.delta, message.complete, tool.start/progress/complete,
        approval.request, clarify.request, sudo.request, secret.request,
        sudo.expire, secret.expire, gateway.ready, and anything else the
        server pushes that isn't a correlated response to a `request()` call.
        Each frame is yielded exactly as received (after newline-splitting in
        `_dispatch_message`) -- no attempt is made to merge, split, or
        de-batch `message.delta` bursts; if the server coalesces several
        tokens into one frame, that frame is yielded once, as-is. Callers
        that want canonical workspace events (P0-3's event normalizer) map
        over this stream themselves.
        """
        while True:
            event = await self._event_queue.get()
            yield event

    # ------------------------------------------------------------------
    # Typed RPC methods
    # ------------------------------------------------------------------

    async def session_list(
        self, *, profile: str | None = None, **extra_params: Any
    ) -> dict[str, Any]:
        """Verified live against the real instance: returns
        `{"sessions": [{"id", "title", "preview", "started_at",
        "message_count", "source"}, ...]}`.

        `profile` scopes the listing to one Hermes **profile** (the operator's
        "agent"/"channel"): measured live 2026-09-01 (PV "Profiles"), `{}`
        returned 130 sessions and `{"profile": "mlx"}` returned 8. The key is
        exactly `profile` -- `profile_name` is silently ignored and returns
        everything, which is why this method sends the param only when one was
        asked for rather than always sending a default. Omitted (the default)
        is the unscoped, all-sessions listing this method has always returned.
        """
        params: dict[str, Any] = dict(extra_params)
        if profile is not None:
            params["profile"] = profile
        return await self.request("session.list", params)

    async def session_title(
        self, live_session_id: str, title: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Rename a session. **`session_id` here is the LIVE handle.**

        Verified live 2026-09-01 (PV "`session.title` and `cli.exec -p`") on a
        throwaway spike session -- and this is the two-id-space trap in its
        purest form::

            session.title {"session_id": <STORED>, "title": ...} -> [4001] session not found
            session.title {"session_id": <LIVE>,   "title": ...} -> {"pending": false, "title": ...}

        The `session.title` **event** reports the STORED id; the **request**
        takes the live handle, like `prompt.submit` / `session.history` /
        `session.interrupt`. Resolve it through `_with_live_handle()` --
        never pass a stored id here.

        Returns `{"pending": bool, "title": str}`; the rename was confirmed by
        read-back in `session.list`.
        """
        return await self.request(
            "session.title",
            {"session_id": live_session_id, "title": title, **extra_params},
        )

    async def profiles_list(self) -> dict[str, Any]:
        """Every Hermes profile (the operator's "agents"/"channels"), one call.

        Verified live 2026-09-01 (PV "Profiles"): returns `{"profiles":
        [{name, path, is_default, model, provider, description, display_name,
        skill_count, last_session: {id, title, preview, started_at,
        last_active, message_count}, worker_session, canonical_session,
        ui_meta_revisions, has_avatar}, ...]}` -- four profiles on the live
        instance (`default`, `gpu-a`, `gpu-b`, `mlx`).

        A **lens**, not a picker: `session.create` accepts `profile` and
        silently ignores it, so nothing here
        can create a session inside a chosen profile. Browsing and filtering
        (`session_list(profile=...)`) is the honest use.
        """
        return await self.request("profiles.list", {})

    async def profiles_describe(self, name: str) -> dict[str, Any]:
        """One profile in full: `{name, description, soul, model, skills,
        toolsets, toolsets_pinned, mcp_servers}`.

        Verified live 2026-09-01 (PV "Profiles"). `name` is required --
        without it Hermes answers `[4063] name required`.
        """
        return await self.request("profiles.describe", {"name": name})

    async def model_options(self, *, refresh: bool = False) -> dict[str, Any]:
        """The provider/model catalog: `{"providers": [{slug, name,
        is_current, is_user_defined, models: [...]}, ...]}`.

        Verified live 2026-09-01 (PV "Profiles"): 8 providers, `custom`
        being `is_current`. Re-measured 2026-09-06 (`docs/AGENT_MODEL_DESIGN.md`
        §7): each provider entry also carries `authenticated`, `source`,
        `featured_models`, `capabilities {model: {fast, reasoning}}`, for
        `openrouter` a `pricing {model: {input "$2.00", output "$10.00",
        cache "$0.20", free}}` map (per **1M** tokens, as **strings**), and
        for `custom` an `api_url`. No context lengths anywhere in it.

        **This IS a picker's list now -- per profile, not per session.**
        `session.create` still ignores a `model` override (measured with a
        deliberately bogus value) and `model.default` is still
        `[-32601] unknown method`, so a per-*session* switch remains
        unreachable. But `profiles_configure()` below is verified live: the
        unit of model choice is the profile ("agent"), and this catalog is
        what a profile's model is validated against (`api/profile_admin.py`).

        `refresh=True` sends `{"refresh": true}`: Hermes then probes EVERY
        custom provider and busts its model cache (`hermes_cli/inventory.py`
        `build_model_options_payload`); a plain call probes only the current
        custom provider, so a user-defined local provider added to
        `config.yaml` lists no models until one refresh has happened.
        """
        return await self.request("model.options", {"refresh": True} if refresh else {})

    async def profiles_create(self, **params: Any) -> dict[str, Any]:
        """Create a Hermes profile: `profiles.create {name, description?,
        provider?, model?, ...}`.

        Verified live 2026-09-06 on a throwaway profile
: `{name, description, provider,
        model}` answers `{ok, name, path, model_set: true, mirrored: {env,
        auth, voice}}`, and `profiles.list` shows the pinned model straight
        away. The new profile inherits the launch profile's `.env` /
        `auth.json` (`mirrored`), which is why an OpenRouter model works on it
        at once. When `provider`/`model` are omitted the profile inherits the
        launch profile's model (Hermes's `mirror_credentials` behaviour).

        `params` are forwarded as given, so the caller owns the closed schema
        (`api/profile_admin.py::ProfileCreate`) -- nothing here may widen it.
        """
        return await self.request("profiles.create", dict(params))

    async def profiles_configure(self, name: str, **params: Any) -> dict[str, Any]:
        """Write one or more sections of a profile: `profiles.configure
        {name, model?, provider?, description?, ...}`.

        Verified live 2026-09-06 on a throwaway profile
: `{name, provider, model}` answers
        `{ok, applied: {model: true}}`, and both `profiles.list` and
        `hermes -p <name> config get model` read the new value back. **Works
        from the default connection** -- the write is by profile *directory*,
        not by which dashboard process received it -- so no per-profile
        connection is needed for it. `applied` is per section
        (`{model: bool, description: bool, ...}`): a caller must read it
        rather than trust `ok`, because a section Hermes did not apply is
        reported there and nowhere else.

        Scope of a model write, read from source and to be borne in mind: the
        per-profile gateway reads `config.yaml` with an mtime-keyed cache, so
        the new model applies to the **next new session** on that profile;
        an already-open session keeps the model it was spawned with.

        It does NOT take `base_url` (that is a separate, unprobed
        `model.base_url` write), and nothing above this forwards one.
        """
        return await self.request("profiles.configure", {"name": name, **params})

    async def config_get(self, key: str) -> dict[str, Any]:
        """Read one Hermes config key: `{"key"}` in, `{"value"}` out.

        Verified live 2026-08-30 / 2026-09-01 (PV "`config.get`" and
        "Profiles"). **There is a narrow allowlist and no enumeration**:
        `profile`, `approvals.mode` and `diagnostics.share_nous` are real
        keys, while `model`, `model.provider`, `models`, `model_catalog`,
        `model.context_length` and `profiles` all answer
        `[4002] unknown config key`. Read-only by deliberate choice at every
        layer above this one -- `config.set` is not wrapped here.
        """
        return await self.request("config.get", {"key": key})

    async def session_active_list(self) -> dict[str, Any]:
        """Sessions live in the Hermes *process* right now, with a status each.

        Verified live 2026-09-01 (PROTOCOL_VERIFIED.md, "session.active_list
        & session.status"): returns `{"sessions": [{"current": bool,
        "id": <LIVE handle>, "last_active": <epoch float>,
        "message_count": int, "model": str, "preview": str,
        "session_key": <STORED id>, "started_at": <epoch float>,
        "status": "idle", "title": str}, ...]}`. `session_key` is the STORED
        id (two-id-spaces rule). `status` spellings measured live 2026-09-03
        (PV "Three-eyes follow-up"): `"idle"` for a quiet session,
        `"starting"` transiently right after a `session.resume` (settled to
        `idle` within ~60 s for all 12 observed), `"working"` mid-turn. Treat
        anything but `"idle"` as "possibly still working". A resumed session
        stays in this list for the life of the connection until
        `session_close()`; the list was empty on a fresh connect. This is also
        the connection keepalive method (`_KEEPALIVE_METHOD`).
        """
        return await self.request("session.active_list", {})

    async def session_create(self, title: str, **extra_params: Any) -> dict[str, Any]:
        """Create a brand-new session.

        Verified live 2026-08-29: returns both ids explicitly::

            {"session_id": "07ff0d50",                    # LIVE handle
             "stored_session_id": "20260829_182702_185869", # durable id
             "message_count": 0, "messages": [], "info": {...}}

        Use `session_id` for subsequent calls; persist `stored_session_id`
        as the durable mapping (see PROTOCOL_VERIFIED.md).
        """
        return await self.request("session.create", {"title": title, **extra_params})

    async def resume_session(self, stored_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Resume a *saved* session, given its durable (stored) id.

        Verified live 2026-08-29. Takes a **stored** id (the `id` from
        `session_list()`, e.g. "20260829_182532_991e3f") and returns::

            {"session_id": "5bfd9de6",              # LIVE handle -- use this
             "resumed":    "20260829_182532_991e3f", # the stored id echoed back
             "message_count": 2,
             "messages": [ ...transcript rows... ]}

        **A transcript row's only guaranteed key is `role`** -- notably NOT
        `text`. Measured over all 47 live sessions / 16,572 messages, 58% are
        `role: "tool"` rows carrying `args` / `context` / `name` and no
        `text`, `row_id` or `timestamp` whatsoever. The full table is
        in `docs/PROTOCOL_VERIFIED.md`, "Transcript message shape"; do not
        write a decoder for these rows from an example transcript.

        The returned `session_id` is a **live** handle, in a different id
        space from the stored id -- see `live_id_from_resume()` and
        PROTOCOL_VERIFIED.md. Every subsequent call for this session
        (`session_history`, `prompt_submit`, `session_interrupt`,
        `session.activate`) must use that live handle, NOT the stored id.

        Idempotent: resuming an already-resumed stored id returns the same
        live handle rather than leaking a second live session.
        """
        return await self.request(
            self._RESUME_METHOD, {"session_id": stored_session_id, **extra_params}
        )

    @staticmethod
    def live_id_from_resume(resume_result: dict[str, Any]) -> str:
        """Extract the live handle from a `resume_session()` result.

        Exists so call sites can't accidentally keep using the stored id:
        the two ids look different but are both opaque strings, so passing
        the wrong one fails at runtime with a bare [4001] rather than at the
        type level.
        """
        live_id = resume_result.get("session_id")
        if not isinstance(live_id, str) or not live_id:
            raise HermesProtocolError(
                "session.resume result has no usable 'session_id' (live handle); "
                f"got keys {sorted(resume_result)!r}"
            )
        return live_id

    async def session_history(self, live_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Fetch history for a **live** session handle.

        Verified live 2026-08-29: returns `{"count": int, "messages": [...]}`,
        whose elements are the same non-uniform transcript rows
        `resume_session()` returns -- see that method and B-34.
        Requires a live handle (from `session_create()`'s `session_id` or
        `live_id_from_resume()`); passing a stored id fails with
        [4001] session not found.
        """
        return await self.request(
            "session.history", {"session_id": live_session_id, **extra_params}
        )

    async def prompt_submit(
        self,
        session_id: str,
        text: str,
        *,
        rewind: RewindRequest | None = None,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """Submit a prompt to a session.

        CRITICAL: `truncate_before_row_id`, `truncate_before_user_ordinal`,
        `confirm_truncate`, and `confirm_empty_truncate` are rewind-only
        fields (see `docs/PROTOCOL_VERIFIED.md`'s "Rewind / edit semantics"
        section) that perform a destructive rewrite of session history on the
        real server. This method NEVER sends them unless the caller passes an
        explicit `RewindRequest` via the separate `rewind=` keyword. If any
        of those field names show up in `**extra_params` instead (e.g. via a
        shared kwargs dict or a copy-pasted default), this raises `ValueError`
        rather than silently forwarding them -- an ordinary prompt submission
        must never carry them by accident.
        """
        leaked = REWIND_FIELDS & extra_params.keys()
        if leaked:
            raise ValueError(
                f"prompt_submit() received rewind-only field(s) {sorted(leaked)} "
                "outside the `rewind=` parameter. Ordinary prompt submissions must "
                "never carry truncate/confirm fields -- pass a RewindRequest via "
                "`rewind=` if a rewind is genuinely intended."
            )
        params: dict[str, Any] = {"session_id": session_id, "text": text, **extra_params}
        if rewind is not None:
            params.update(rewind.to_params())
        return await self.request("prompt.submit", params)

    async def prompt_background(
        self, session_id: str, text: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Submit a fire-and-forget background task. `session_id` is a **LIVE handle**.

        Verified live 2026-08-30 (PV "Phase 2 probe"): returns
        `{"task_id": "bg_..."}` immediately. Everything else about the wire
        contract is negative space, measured across five tasks:

        * **No events while it runs** -- no deltas, no tool frames, nothing;
          `delegation.status` / `agents.list` / `process.list` are all empty
          mid-run and `session.resume` reports `idle`. There is no status
          surface to poll.
        * **Completion is exactly one event**, `background.complete`
          `{task_id, text}`, stamped with the live handle, delivered only to
          connections attached to the session at that instant (PV "Phase 2a
          probes": lost 3/3 on a bare reconnect, delivered 2/2 after a
          pre-completion `session.resume` on the new connection).
        * **Hermes persists no trace**: no transcript row, and a
          background-only session is never saved (`[4007]` on resume).

        So callers MUST record `task_id` (against the STORED session id) at
        submit time -- the gateway's `background_tasks` ledger is the only
        durable record the task ever existed.
        """
        return await self.request(
            "prompt.background", {"session_id": session_id, "text": text, **extra_params}
        )

    async def session_interrupt(self, session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Stop the turn running on a session. `session_id` is a **LIVE handle**.

        Verified live 2026-08-30, including mid-turn while tokens were
        streaming: `{"session_id": <LIVE>}` -> `{"status": "interrupted"}`. A
        stored id here fails `[4001] session not found` like every other
        live-handle method.
        """
        return await self.request("session.interrupt", {"session_id": session_id, **extra_params})

    async def session_close(self, session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Drop a session from the Hermes *process*. `session_id` is a **LIVE handle**.

        Verified live 2026-09-03 (PV "Three-eyes follow-up"): `{"session_id":
        <LIVE>}` -> `{"closed": true}`, and the session leaves
        `session.active_list` at once. Measured on a throwaway spike only.

        This is a process-local live-session control (Hermes's own grouping
        with `session.active_list` / `session.activate`), **not a delete**: the
        stored session and its transcript are untouched and `session.resume`
        mints a fresh handle next time. The one caller that needs it is
        `DELETE /api/sessions/{id}`: a resume issued before the CLI delete
        leaves a **zombie** handle in `active_list` (still answering
        `session.history` with `count: 0`) unless it is closed afterwards.
        """
        return await self.request("session.close", {"session_id": session_id, **extra_params})

    async def session_branch(self, live_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Fork a session: copy its transcript into a brand-new session.

        **`session_id` here is the LIVE handle**, like `prompt.submit` /
        `session.history` / `session.title`. Verified live 2026-09-02
        (PV "Session branching"):

            session.branch {"session_id": <STORED>} -> [4001] session not found
            session.branch {"session_id": <LIVE>}   -> the new branch

        An **empty** session (no turns yet) answers
        `[4008] nothing to branch -- send a message first`, so nothing above
        this may offer branching on a session with no messages.

        Returns::

            {"session_id": "dbdcc7be",                       # LIVE of the NEW branch
             "stored_session_id": "20260902_065211_885302",  # STORED of the new branch
             "title": "<parent title> #2",                   # auto-suffixed
             "parent": "20260901_073831_245633",             # the SOURCE's STORED id
             "message_count": 2,
             "messages": [ ...the copied transcript... ]}

        **There is NO truncation: the branch is a complete copy.** Measured
        2026-09-02 across five sources (2, 4 and 4-row transcripts, settled and
        interrupted): `branch["messages"] == session.history(parent)["messages"]`
        row for row, `row_id`s included, final assistant reply included.

        The earlier "one message SHORTER (3 -> 2)" reading was a comparison
        between two different counters. Hermes's `message_count` -- the one
        `session.list`, `session.resume` and `session.history` all report --
        can exceed `len(messages)` by one on a session whose last turn was
        interrupted; the branch's own `message_count` always equals its
        `len(messages)`. Compare `len(messages)` to `len(messages)` and the
        delta is zero. See `docs/PROTOCOL_VERIFIED.md`.

        The `parent` link is what lets the gateway file the fork into the same
        project as its source without a second lookup.

        **`name` names the fork; `title` does not.** Measured back to back on
        one session: `{"name": "X"}` -> `{"title": "X"}`, while
        `{"title": "X"}` is accepted and silently ignored (the fork is still
        `"<parent> #N"`). Every other session method here spells this `title`,
        so pass it as `name` -- `api/instance.py`'s
        `SESSION_BRANCH_TITLE_PARAM` is the single place that translation
        happens.
        """
        return await self.request("session.branch", {"session_id": live_session_id, **extra_params})

    # ------------------------------------------------------------------
    # Slash commands (P2-4). All three shapes measured live 2026-08-30 --
    # PV "Phase 2 probe" (`commands.catalog` / `command.resolve`) and
    # PV "Phase 2a probes" (`command.dispatch`).
    # ------------------------------------------------------------------

    async def commands_catalog(self) -> dict[str, Any]:
        """The whole command catalog, one ~34 KB result, seven keys.

        Measured: `pairs` (178 `[name, description]` pairs -- usage strings
        are embedded in the description, not a separate field), `canon`
        (alias -> canonical), `sub` (command -> subcommand names),
        `categories` (6 `{name, pairs}` groups), `skills`
        (`"/skill-name"` -> `{"usage": int, "origin": ...}`, 86 entries),
        `skill_count`, `warning`. Callers must preserve `categories` and the
        per-skill `usage` -- the app's command sheet sections by the former
        and sorts skills by the latter (P2-9).
        """
        return await self.request("commands.catalog", {})

    async def command_resolve(self, name: str) -> dict[str, Any]:
        """Resolve one CORE command name -> `{canonical, description, category}`.

        `name` is the ONLY parameter Hermes reads (measured: `text` /
        `command` / `line` all fail `[4011] unknown command: None` -- the
        error names the value it parsed from `name`, which is None for any
        other field). Leading slash optional; aliases canonicalize
        (`/reset` -> `new`); NO prefix matching (`/stat` -> `[4011]`), so
        autocomplete must be built from the catalog. `canonical` comes back
        WITHOUT the slash. Knows only the 178 core pairs -- a skill name
        fails `[4011]` here (resolve and dispatch cover disjoint classes).
        """
        return await self.request("command.resolve", {"name": name})

    async def session_compress(self, live_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Compress a session's context **in place** (P6-4, measured 2026-09-04).

        Takes the **LIVE** handle (a stored id answers `[4001]`). Same session,
        same stored id, same live handle afterwards -- Hermes runs
        `compression.in_place: true` and the context engine is `hermes-lcm`.
        Answers in ~50 ms when there is nothing to do. Measured result keys:
        `status` ("compressed"), `removed`, `before_messages`, `after_messages`,
        `before_tokens`, `after_tokens`, `summary` (`{noop, aborted,
        refused_would_grow, fallback_used, headline, token_line, note}`),
        `usage` (the full usage dict, `context_used/max/percent`,
        `compressions`), `info`, `messages` (the post-compress transcript).

        **Every extra parameter is silently ignored** (`focus_topic`, `arg`,
        `args`, `text`, `focus`, ... all accepted, none applied), so this is the
        *plain* compress only; the partial form (`here N`) goes through
        `command_dispatch("/compress", args=..., session_id=live)`. Under LCM
        only raw messages outside the fresh tail (32 messages / 24k tokens on
        the operator's host) are eligible, so a short session is an honest no-op
        (`summary.noop`). Events: `status.update` kind `compressing` ->
        `compacting` -> `compacted` -> `session.info` -> `status` "ready".
        See PV "Compress + the LCM context engine".
        """
        return await self.request(
            "session.compress", {"session_id": live_session_id, **extra_params}
        )

    async def command_dispatch(
        self,
        name: str,
        *,
        args: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a SKILL command's expansion -- or, for `/compress`, EXECUTE it.

        **2026-09-04 correction (P6-4, measured):** `/compress` is dispatchable
        and *runs*: `{"name": "/compress", "args": "here 2", "session_id":
        <LIVE>}` answered `{"type": "exec", "output": "Compressed: 23 -> 22
        messages\nApprox request size: ~66,322 -> ~66,271 tokens"}`. `args` is
        the field that carries the raw argument (`here N`, `--keep N`, a focus
        topic); `arg` is ignored; `name: "/compress here 2"` is `[4018]`. It is
        refused `[4009] session busy -- /interrupt the current turn before
        /compress` while a turn runs. `args: "--preview"` alone is a
        `[-32603] internal error` on this build. Both optional parameters are
        sent only when given, so every existing skill-lookup call is unchanged.

        The 2026-08-30 measurement below still holds for skill names:

        Measured (PV "Phase 2a probes"): dispatching a skill name returns
        `{"type": "skill", "name", "display", "message"}` immediately, where
        `message` is the full skill prompt for the CLIENT to submit as its
        next `prompt.submit` -- no agent turn starts, nothing is written to
        any session (180 s observed: zero events, history count unchanged).

        Core catalog commands are NOT dispatchable at all: every one fails
        `[4018] not a quick/plugin/bundle/skill command: <name>` regardless
        of param shape, and an unknown name returns the same `[4018]` -- the
        error does not distinguish "exists but core" from "does not exist".
        `session_id` changes nothing for a skill lookup, so it is not sent
        unless a caller passes it (the `/compress` route must).
        """
        params: dict[str, Any] = {"name": name}
        if args is not None:
            params["args"] = args
        if session_id is not None:
            params["session_id"] = session_id
        return await self.request("command.dispatch", params)

    # ------------------------------------------------------------------
    # The four human-in-the-loop prompts
    #
    # `approval.request` / `clarify.request` / `sudo.request` /
    # `secret.request` are all resolved by calling the matching `*.respond`
    # RPC with the request's ORIGINAL `request_id`. They do NOT share a shape,
    # and getting the field name wrong is invisible -- three of these four
    # methods used to send the wrong parameter and Hermes answered 200 to two
    # of them anyway. Each verb below is the one Hermes actually reads;
    # `docs/PROTOCOL_VERIFIED.md` ("The four human-in-the-loop prompts")
    # records which are measured and which are source-read.
    #
    # Two of the four carry a **credential**, not a decision:
    # `sudo.respond` takes the user's sudo *password* and `secret.respond`
    # takes the secret the agent asked the user to supply. Both fall under
    # `docs/ARCHITECTURE.md` §14 -- request-scoped, never logged, never
    # stored, never in an event payload, never in a URL, never echoed back.
    # Neither this module nor `request()` logs any parameter, which is what
    # makes that guarantee hold here.
    # ------------------------------------------------------------------

    async def approval_respond(
        self,
        live_session_id: str,
        choice: str,
        request_id: str | None = None,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """Answer an `approval.request`. **Verified live 2026-08-30.**

        Unlike the other three, this one is *session*-keyed: Hermes resolves
        the session before it ever looks at the choice, so `live_session_id`
        (a LIVE handle, not a stored id) is mandatory and a request_id alone
        fails `[4001] session not found`.

        `choice` is one of `APPROVAL_CHOICES` -- `once` / `session` /
        `always` / `deny`. It is a string, not a boolean; Hermes defaults a
        missing one to `deny`, so an unvalidated value silently denies.

        `request_id` is optional on the wire (Hermes resolves the oldest
        pending approval without it) and every caller here should still pass
        it: the client may be answering a card that has since been superseded.

        Returns Hermes's `{"resolved": <int>}` -- **a count, not a boolean**.
        `{"resolved": 0}` means nothing was pending and is a no-op, not a
        success. Callers must check it; see `api/prompts.py`.
        """
        if choice not in APPROVAL_CHOICES:
            raise ValueError(
                f"approval choice {choice!r} is not one of {list(APPROVAL_CHOICES)}; "
                "Hermes treats an unrecognized/missing choice as 'deny', so a "
                "typo here would silently deny the user's approval"
            )
        params: dict[str, Any] = {"session_id": live_session_id, "choice": choice}
        if request_id is not None:
            params["request_id"] = request_id
        return await self.request("approval.respond", {**params, **extra_params})

    async def request_answer(
        self, request_id: str, result: dict[str, Any], **extra_params: Any
    ) -> dict[str, Any]:
        """Answer one open server -> client request (B-197, Hermes 0.21.3).

        `request_id` is the open request's own id (`srq-<12 hex>`), which the
        adapter stamped onto the prompt's payload as both `_srq_id` and
        `request_id` before forwarding it.

        Returns `{"status": "ok"}` or `{"status": "expired"}`. As with the
        RPCs this replaced, `expired` is a normal result and not an error: the
        request already timed out, was withdrawn, or someone else answered.
        """
        return await self.request(
            "request.answer", {"id": request_id, "result": result, **extra_params}
        )

    async def clarify_lock(
        self, request_id: str, question_id: str, answer: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Lock ONE answer of a batch clarify.

        A batch asks several questions in one request. Answers stay editable
        until every question is locked, and the lock that empties `remaining`
        resolves the whole request -- so a caller that locks every question
        never needs `request_answer` at all.

        Returns `{"status": "ok"|"expired", "remaining": [qid, ...]}`.
        """
        return await self.request(
            "clarify.lock",
            {
                "request_id": request_id,
                "question_id": question_id,
                "answer": answer,
                **extra_params,
            },
        )

    async def _answer_or_legacy(
        self,
        request_id: str,
        result: dict[str, Any],
        legacy_method: str,
        legacy_params: dict[str, Any],
        **extra_params: Any,
    ) -> dict[str, Any]:
        """`request.answer` on 0.21.3, the pre-0.21.3 `*.respond` RPC below it.

        The fallback is not politeness: this repo is shared, the two shapes
        are a version apart, and the failure mode of guessing wrong is the one
        B-39 already cost us -- Hermes answering `{"status": "ok"}` while
        throwing the user's answer away. `[-32601] unknown method` is the one
        error that means "this build does not have that RPC", so it is the
        only one that falls through.
        """
        try:
            return await self.request_answer(request_id, result, **extra_params)
        except HermesRPCError as exc:
            if exc.code != -32601:
                raise
            logger.info(
                "request.answer is unknown to this Hermes; falling back to %s", legacy_method
            )
        return await self.request(
            legacy_method, {"request_id": request_id, **legacy_params, **extra_params}
        )

    async def clarify_respond(
        self,
        request_id: str,
        answer: str = "",
        *,
        answers: dict[str, str] | None = None,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """Answer a clarify prompt, single question or batch.

        `answers` is the batch form: `{question_id: answer}` for the whole
        set. `answer` is the single-question form, and `""` means skip.

        On 0.21.3 this is `request.answer {id, result}`; `clarify.respond` no
        longer exists on that build and is only reached through the version
        fallback. What the old docstring recorded still matters, because the
        result KEY is the same trap one level in: the field is `answer`, and
        sending `response` was not an error -- Hermes replied
        `{"status": "ok"}` and the clarify tool completed with
        `user_response: ""`, resuming the turn having discarded what the user
        typed. That was B-39, observed twice on one session.
        """
        result: dict[str, Any] = (
            {"answers": dict(answers)} if answers is not None else {"answer": answer}
        )
        return await self._answer_or_legacy(
            request_id, result, "clarify.respond", {"answer": answer}, **extra_params
        )

    async def sudo_respond(
        self, request_id: str, password: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Answer a `sudo.request` with the user's sudo password.

        **UNVERIFIED against the live instance.** The field name `password`
        is read from the deployed Hermes's own source
        (`/opt/hermes/tui_gateway/methods_prompt.py`, `_respond(rid, params,
        "password")`), not measured -- provoking a real `sudo.request` means
        producing a real credential on the operator's machine. Treat a `200` from
        this call the way B-39 teaches: a wrong field name here would answer
        `{"status": "ok"}` and drop the password.

        **`sudo.request` is not "approve running sudo" -- it is "type your
        sudo password".** `password` is a §14 credential: it is never logged,
        never persisted, never put in an event payload, never in a URL and
        never echoed in a response body, on any layer above this one.

        On 0.21.3 this is `request.answer {id, result: {"value": ...}}` --
        the result key is `value` for both credential prompts, and
        `sudo.respond`'s `password` field survives only behind the version
        fallback. A withdrawn prompt now arrives as one `request.cancel`
        rather than a per-kind `sudo.expire`.

        Returns `{"status": "ok"}` / `{"status": "expired"}` (confirmed live
        for an unknown request_id, probed with a fake placeholder).
        """
        return await self._answer_or_legacy(
            request_id,
            {"value": password},
            "sudo.respond",
            {"password": password},
            **extra_params,
        )

    async def secret_respond(
        self, request_id: str, value: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Answer a `secret.request` with the secret the agent asked for.

        Field name `value` is source-read, like `sudo.respond` above, and
        **unverified against the live instance** for the same reason:
        answering one requires a real credential. The `{"status": "expired"}`
        reply for an unknown `request_id` *was* confirmed live using an
        obviously-fake placeholder.

        `value` is a §14 credential -- see `sudo_respond`. Never logged here
        or anywhere else in this module.
        """
        return await self._answer_or_legacy(
            request_id, {"value": value}, "secret.respond", {"value": value}, **extra_params
        )

    async def cli_exec(self, argv: list[str], **extra_params: Any) -> dict[str, Any]:
        """Run the `hermes` **CLI binary** with `argv`, out of band of any turn.

        **`argv` is the only field that is read.** Measured 2026-08-30 (PV
        "Phase 3 probe", Attempt 2): seven string-shaped field names --
        `command`, `cmd`, `line`, `cli`, `cmdline`, `text`, `input` -- are all
        silently ignored and fall through to invoking a bare interactive
        `hermes`, which answers `{"blocked": true, ...}`. A wrong field name
        here does not error; it "succeeds" while doing nothing you asked.

        Returns `{"blocked": bool, "code": int, "output": str}` (and a `hint`
        when blocked). **`code` is the CLI's own exit status and a non-zero
        value is a FAILURE** -- the RPC itself still returns 200, so a caller
        that only checks for a `HermesError` will report a silent success for
        a command that did not run.

        This is **not** a host shell: it spawns the `hermes` binary itself, so
        only that CLI's own subcommands are reachable (`profile`, `sessions`,
        `config`, `model`, ...). Verified live 2026-09-01 (PV "`session.title`
        and `cli.exec -p`") that `-p <profile>` genuinely routes.

        **`argv` becomes a process argument list.** Every element is a real
        argv token, so anything a caller interpolates in is parsed by the
        CLI's own argparse -- a value beginning with `-` is read as an OPTION,
        not as a positional. Callers building an argv out of client-supplied
        material must validate the shape first; `api/instance.py`'s
        `_validate_stored_session_id_for_argv()` is that gate for the one
        destructive user of this method (`hermes sessions delete`).
        """
        return await self.request("cli.exec", {"argv": list(argv), **extra_params})
