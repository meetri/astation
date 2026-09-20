"""Exception hierarchy for `HermesAdapter`."""

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
