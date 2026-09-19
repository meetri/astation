"""Exception hierarchy for `HermesAdapter`.

None of these exceptions ever include the password, a cookie value, or a
ticket value in their message — only status codes, method names, and
Hermes-supplied error codes/messages that are themselves not credential
material (per `docs/PROTOCOL_VERIFIED.md`, error bodies like `{"detail":
"Invalid credentials"}` don't echo the submitted password).
"""

from __future__ import annotations

from typing import Any


class HermesError(Exception):
    """Base class for all HermesAdapter errors."""


class HermesAuthError(HermesError):
    """Login (`/auth/password-login`) or ticket minting (`/api/auth/ws-ticket`)
    failed, or a session cookie was rejected."""


class HermesConnectionError(HermesError):
    """The HTTP or WebSocket transport failed: request-send failure, WS
    connect failure, or the socket closing while a request was pending.

    `request_was_sent` says whether the JSON-RPC frame this error belongs to
    had already been **written to the socket** when the transport died. It is
    the difference between an error that is safe to retry and one that is not,
    and it must be answered by the place that raises, not guessed by the
    caller:

    * ``False`` (the default) -- the frame never left us. `request()` raises
      this before the send (not connected) or when the send itself failed, so
      Hermes never saw the call and replaying it is free.
    * ``True`` -- the frame *was* delivered and only the **reply** was lost,
      which is what `_recv_loop` reports when the socket closes with requests
      still pending. Hermes may well have acted on it. Replaying a
      `prompt.submit` in that state submits the user's message to a real
      research session twice -- and if the first copy is still running, the
      duplicate comes back `redirected`, i.e. applied as a *correction* to the
      turn already in flight. `api.main._with_reconnect()` refuses to retry
      these.
    """

    def __init__(self, *args: Any, request_was_sent: bool = False) -> None:
        super().__init__(*args)
        self.request_was_sent = request_was_sent


class HermesProtocolError(HermesError):
    """The server sent something that doesn't match the documented protocol
    (unparseable frame, missing expected field, timed-out response)."""


class HermesRPCError(HermesError):
    """The server returned a JSON-RPC `error` object for a `request()` call."""

    def __init__(self, method: str, code: Any, message: str, data: Any = None) -> None:
        super().__init__(f"{method} failed: [{code}] {message}")
        self.method = method
        self.code = code
        self.message = message
        self.data = data
