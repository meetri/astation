"""HermesAdapter — client for the Hermes TUI Gateway wire protocol."""

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

_SESSION_COOKIE_NAMES = ("hermes_session_at", "hermes_session_rt", "hermes_session_provider")

_LOGIN_PATH = "/auth/password-login"
_TICKET_PATH = "/api/auth/ws-ticket"
_WS_PATH = "/api/ws"

_FILES_LIST_PATH = "/api/files"
_FILES_DOWNLOAD_PATH = "/api/files/download"

_FS_READ_TEXT_PATH = "/api/fs/read-text"
_FS_WRITE_TEXT_PATH = "/api/fs/write-text"
_FILES_MKDIR_PATH = "/api/files/mkdir"

_READY_EVENT_METHOD = "gateway.ready"
_READY_TIMEOUT_S = 10.0
_REQUEST_TIMEOUT_S = 30.0
_HEARTBEAT_INTERVAL_S = 25.0
_KEEPALIVE_METHOD = "session.active_list"

_MAX_WS_FRAME_BYTES = 64 * 1024 * 1024

_MAX_QUEUED_UPSTREAM_EVENTS = 10_000

SERVER_REQUEST_RAW_TYPES: dict[str, str] = {
    "clarify": "clarify.request",
    "approval": "approval.request",
    "sudo": "sudo.request",
    "secret": "secret.request",
}

SERVER_REQUEST_ID_FIELD = "_srq_id"

REQUEST_CANCEL_EVENT = "request.cancel"

_MAX_TRACKED_SERVER_REQUESTS = 256


class HermesAdapter:
    """One Hermes session's worth of transport: HTTP login/ticket + one WS connection."""

    _RESUME_METHOD = "session.resume"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._http = httpx.AsyncClient(base_url=self._settings.hermes_base_url)
        self._logged_in = False

        self._ws: Any | None = None
        self._ws_live = False
        self._recv_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._server_request_ids: dict[str, str] = {}
        self._decline_tasks: set[asyncio.Task[None]] = set()
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=_MAX_QUEUED_UPSTREAM_EVENTS
        )
        self.dropped_events = 0
        self._drop_logged_generation: int | None = None
        self._id_counter = itertools.count(1)
        self._ready = asyncio.Event()
        self._connection_generation = 0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        state = "connected" if self.is_connected else ("logged_in" if self._logged_in else "fresh")
        return f"HermesAdapter(host={self._settings.hermes_host!r}, state={state!r})"

    @property
    def is_logged_in(self) -> bool:
        """Whether `login()` has completed successfully."""
        return self._logged_in

    @property
    def is_connected(self) -> bool:
        """Whether `connect()` has completed and the WebSocket is still open."""
        return self._ws is not None and self._ws_live

    @property
    def connection_generation(self) -> int:
        """Monotonic counter identifying *which* Hermes connection we are on."""
        return self._connection_generation

    async def __aenter__(self) -> HermesAdapter:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


    async def login(self) -> None:
        """POST /auth/password-login and store the resulting session cookies."""
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
            del payload

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
        """The credential query string for the `/api/ws` upgrade."""
        return f"ticket={await self.mint_ticket()}"

    async def mint_ticket(self) -> str:
        """POST /api/auth/ws-ticket (cookie-authenticated)."""
        if not self._logged_in:
            raise HermesAuthError("mint_ticket() called before a successful login()")
        try:
            resp = await self._http.post(_TICKET_PATH)
        except httpx.HTTPError as exc:
            raise HermesConnectionError("ws-ticket request failed") from exc

        if resp.status_code == 401:
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
        """Mint a fresh ticket, open the WS, and wait for `gateway.ready`."""
        await self._teardown_ws()
        self._connection_generation += 1
        auth_query = await self._ws_auth_query()
        scheme = "wss" if self._settings.hermes_scheme == "https" else "ws"
        url = f"{scheme}://{self._settings.hermes_host}:{self._settings.hermes_port}{_WS_PATH}?{auth_query}"
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
            await self._teardown_ws()
            raise HermesConnectionError(
                f"did not observe a '{_READY_EVENT_METHOD}' event within {_READY_TIMEOUT_S}s of connecting"
            ) from exc

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Poke Hermes periodically so an idle socket isn't reaped."""
        while True:
            try:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                await self.request(_KEEPALIVE_METHOD)
            except asyncio.CancelledError:
                raise
            except HermesRPCError:
                continue
            except Exception:
                self._ws_live = False
                return

    async def _teardown_ws(self) -> None:
        """Close the WebSocket half only, leaving the HTTP client usable."""
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


    async def files_list(self, path: str) -> httpx.Response:
        """`GET /api/files?path=<dir>` -- list a sandbox directory."""
        return await self._files_get(_FILES_LIST_PATH, path, stream=False)

    async def files_download(self, path: str, *, range_header: str | None = None) -> httpx.Response:
        """`GET /api/files/download?path=<abs>` -- raw sandbox file bytes, STREAMED."""
        headers = {"Range": range_header} if range_header else None
        return await self._files_get(_FILES_DOWNLOAD_PATH, path, headers=headers, stream=True)

    async def fs_read_text(self, path: str) -> httpx.Response:
        """`GET /api/fs/read-text?path=<abs>` -- a sandbox file decoded as text."""
        return await self._files_get(_FS_READ_TEXT_PATH, path, stream=False)

    async def fs_write_text(self, path: str, content: str) -> httpx.Response:
        """`POST /api/fs/write-text` `{"path": <abs>, "content": <str>}` -- write a text file."""
        return await self._files_post(_FS_WRITE_TEXT_PATH, {"path": path, "content": content})

    async def files_mkdir(self, path: str) -> httpx.Response:
        """`POST /api/files/mkdir` `{"path": <abs>}` -- create a directory and
        its parents.
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
        """Shared JSON POST for the file routes: see `_files_request`."""
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
        """One HTTP request on the cookie jar: lazy login + one 401 retry."""
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

            await response.aclose()
            self._logged_in = False
            if attempted_relogin:
                raise HermesAuthError(
                    f"{method} {url_path} still returned 401 after a fresh "
                    "successful login -- the files route's auth no longer "
                    "accepts this session (Hermes upgrade?)"
                )
            attempted_relogin = True


    async def _recv_loop(self) -> None:
        assert self._ws is not None
        ws = self._ws
        try:
            async for raw_message in ws:
                self._dispatch_message(raw_message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._ws_live = False
            closed_exc = HermesConnectionError(
                "Hermes WebSocket closed while a request was pending",
                request_was_sent=True,
            )
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(closed_exc)
            self._pending.clear()

    def _dispatch_message(self, raw_message: str | bytes) -> None:
        """Split one incoming WS message into newline-delimited JSON lines."""
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

        if not is_response and isinstance(frame_id, str) and isinstance(frame.get("method"), str):
            self._dispatch_server_request(frame_id, frame["method"], frame.get("params"))
            return

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
        """Turn one server -> client request into the raw event it replaced."""
        payload = dict(params) if isinstance(params, dict) else {}
        raw_type = SERVER_REQUEST_RAW_TYPES.get(method)
        if raw_type is None:
            self._decline_server_request(srq_id, method)
            return

        payload[SERVER_REQUEST_ID_FIELD] = srq_id
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
        """`request.cancel {id, method, reason}` -> the raw `*.cancel` name."""
        method = params.get("method")
        raw_type = SERVER_REQUEST_RAW_TYPES.get(method) if isinstance(method, str) else None
        if raw_type is None:
            return None
        srq_id = params.get("id")
        payload = dict(params)
        if isinstance(srq_id, str):
            payload[SERVER_REQUEST_ID_FIELD] = srq_id
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


    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one JSON-RPC 2.0 request and await the correlated response."""
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
        """Yield normalized-but-still-raw JSON-RPC event dicts as they arrive."""
        while True:
            event = await self._event_queue.get()
            yield event


    async def session_list(
        self, *, profile: str | None = None, **extra_params: Any
    ) -> dict[str, Any]:
        """Verified live against the real instance: returns
        `{"sessions": [{"id", "title", "preview", "started_at",
        "message_count", "source"}, ...]}`.
        """
        params: dict[str, Any] = dict(extra_params)
        if profile is not None:
            params["profile"] = profile
        return await self.request("session.list", params)

    async def session_title(
        self, live_session_id: str, title: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Rename a session. **`session_id` here is the LIVE handle.**"""
        return await self.request(
            "session.title",
            {"session_id": live_session_id, "title": title, **extra_params},
        )

    async def profiles_list(self) -> dict[str, Any]:
        """Every Hermes profile (the operator's "agents"/"channels"), one call."""
        return await self.request("profiles.list", {})

    async def profiles_describe(self, name: str) -> dict[str, Any]:
        """One profile in full: `{name, description, soul, model, skills,
        toolsets, toolsets_pinned, mcp_servers}`.
        """
        return await self.request("profiles.describe", {"name": name})

    async def model_options(self, *, refresh: bool = False) -> dict[str, Any]:
        """The provider/model catalog: `{"providers": [{slug, name,
        is_current, is_user_defined, models: [...]}, ...]}`.
        """
        return await self.request("model.options", {"refresh": True} if refresh else {})

    async def profiles_create(self, **params: Any) -> dict[str, Any]:
        """Create a Hermes profile: `profiles.create {name, description?,
        provider?, model?, ...}`.
        """
        return await self.request("profiles.create", dict(params))

    async def profiles_configure(self, name: str, **params: Any) -> dict[str, Any]:
        """Write one or more sections of a profile: `profiles.configure
        {name, model?, provider?, description?, ...}`.
        """
        return await self.request("profiles.configure", {"name": name, **params})

    async def config_get(self, key: str) -> dict[str, Any]:
        """Read one Hermes config key: `{"key"}` in, `{"value"}` out."""
        return await self.request("config.get", {"key": key})

    async def session_active_list(self) -> dict[str, Any]:
        """Sessions live in the Hermes *process* right now, with a status each."""
        return await self.request("session.active_list", {})

    async def session_create(self, title: str, **extra_params: Any) -> dict[str, Any]:
        """Create a brand-new session."""
        return await self.request("session.create", {"title": title, **extra_params})

    async def resume_session(self, stored_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Resume a *saved* session, given its durable (stored) id."""
        return await self.request(
            self._RESUME_METHOD, {"session_id": stored_session_id, **extra_params}
        )

    @staticmethod
    def live_id_from_resume(resume_result: dict[str, Any]) -> str:
        """Extract the live handle from a `resume_session()` result."""
        live_id = resume_result.get("session_id")
        if not isinstance(live_id, str) or not live_id:
            raise HermesProtocolError(
                "session.resume result has no usable 'session_id' (live handle); "
                f"got keys {sorted(resume_result)!r}"
            )
        return live_id

    async def session_history(self, live_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Fetch history for a **live** session handle."""
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
        """Submit a prompt to a session."""
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
        """Submit a fire-and-forget background task. `session_id` is a **LIVE handle**."""
        return await self.request(
            "prompt.background", {"session_id": session_id, "text": text, **extra_params}
        )

    async def session_interrupt(self, session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Stop the turn running on a session. `session_id` is a **LIVE handle**."""
        return await self.request("session.interrupt", {"session_id": session_id, **extra_params})

    async def session_close(self, session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Drop a session from the Hermes *process*. `session_id` is a **LIVE handle**."""
        return await self.request("session.close", {"session_id": session_id, **extra_params})

    async def session_branch(self, live_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Fork a session: copy its transcript into a brand-new session."""
        return await self.request("session.branch", {"session_id": live_session_id, **extra_params})


    async def commands_catalog(self) -> dict[str, Any]:
        """The whole command catalog, one ~34 KB result, seven keys."""
        return await self.request("commands.catalog", {})

    async def command_resolve(self, name: str) -> dict[str, Any]:
        """Resolve one CORE command name -> `{canonical, description, category}`."""
        return await self.request("command.resolve", {"name": name})

    async def session_compress(self, live_session_id: str, **extra_params: Any) -> dict[str, Any]:
        """Compress a session's context **in place** (P6-4, measured 2026-09-04)."""
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
        """Fetch a SKILL command's expansion -- or, for `/compress`, EXECUTE it."""
        params: dict[str, Any] = {"name": name}
        if args is not None:
            params["args"] = args
        if session_id is not None:
            params["session_id"] = session_id
        return await self.request("command.dispatch", params)


    async def approval_respond(
        self,
        live_session_id: str,
        choice: str,
        request_id: str | None = None,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """Answer an `approval.request`. **Verified live 2026-08-30.**"""
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
        """Answer one open server -> client request (B-197, Hermes 0.21.3)."""
        return await self.request(
            "request.answer", {"id": request_id, "result": result, **extra_params}
        )

    async def clarify_lock(
        self, request_id: str, question_id: str, answer: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Lock ONE answer of a batch clarify."""
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
        """`request.answer` on 0.21.3, the pre-0.21.3 `*.respond` RPC below it."""
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
        """Answer a clarify prompt, single question or batch."""
        result: dict[str, Any] = (
            {"answers": dict(answers)} if answers is not None else {"answer": answer}
        )
        return await self._answer_or_legacy(
            request_id, result, "clarify.respond", {"answer": answer}, **extra_params
        )

    async def sudo_respond(
        self, request_id: str, password: str, **extra_params: Any
    ) -> dict[str, Any]:
        """Answer a `sudo.request` with the user's sudo password."""
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
        """Answer a `secret.request` with the secret the agent asked for."""
        return await self._answer_or_legacy(
            request_id, {"value": value}, "secret.respond", {"value": value}, **extra_params
        )

    async def cli_exec(self, argv: list[str], **extra_params: Any) -> dict[str, Any]:
        """Run the `hermes` **CLI binary** with `argv`, out of band of any turn."""
        return await self.request("cli.exec", {"argv": list(argv), **extra_params})
