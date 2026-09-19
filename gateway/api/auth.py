"""HTTP Basic auth for the Research Gateway's own surface.

This is the **gateway's own** credential (`RESEARCH_GATEWAY_USERNAME` /
`RESEARCH_GATEWAY_PASSWORD`), which is deliberately distinct from
`HERMES_USERNAME`/`HERMES_PASSWORD` -- those are what the gateway uses
*upstream* against Hermes and must never be accepted as an inbound
credential here.

Scope:

* Every `/api/*` route is protected by `require_basic_auth` (attached once on
  the router in `api/main.py`, so a newly-added route cannot forget it).
* `WS /ws/events` is protected by `websocket_client_is_authorized()`. A
  browser cannot set `Authorization` on a WebSocket upgrade, but our client is
  a native `URLSessionWebSocketTask`, which can -- so the same Basic header is
  read and validated on the upgrade request and no ticket system is needed.
* `GET /health` is intentionally left unauthenticated: it is a liveness probe
  and leaks nothing.

Comparisons use `secrets.compare_digest` for **both** the username and the
password, and both are always evaluated (no short-circuit `and`) so response
timing cannot be used to learn that a username was correct while the password
was not -- or that a username exists at all.
"""

from __future__ import annotations

import base64
import binascii
import secrets

from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config.settings import Settings, get_settings

# Sent on every 401 so a client (and `curl -u`) knows to retry with Basic.
_WWW_AUTHENTICATE = {"WWW-Authenticate": 'Basic realm="research-gateway"'}

# `auto_error=False`: FastAPI's own 401 for a missing header omits the
# `WWW-Authenticate` challenge shape we want and gives a different body than
# the wrong-credentials case. Handling it here keeps "no header" and "bad
# header" indistinguishable to a caller.
_basic_scheme = HTTPBasic(auto_error=False, realm="research-gateway")


def _unauthorized() -> HTTPException:
    """One 401 for every failure mode: missing, malformed, or wrong."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers=_WWW_AUTHENTICATE,
    )


def _credentials_match(settings: Settings, username: str | None, password: str | None) -> bool:
    """Constant-time check of a candidate username/password pair.

    Both comparisons always run (bitwise `&`, not `and`) so the time taken
    does not reveal whether the username alone was right. If the gateway
    credential is not configured at all, this returns False -- the service
    fails *closed* rather than accepting everything.
    """
    expected_user = settings.research_gateway_username
    expected_password = settings.research_gateway_password.get_secret_value()
    if not expected_user or not expected_password:
        return False

    user_ok = secrets.compare_digest(
        (username or "").encode("utf-8"), expected_user.encode("utf-8")
    )
    password_ok = secrets.compare_digest(
        (password or "").encode("utf-8"), expected_password.encode("utf-8")
    )
    return bool(user_ok & password_ok)


async def require_basic_auth(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(_basic_scheme),
) -> str:
    """FastAPI dependency guarding every `/api/*` route.

    Returns the authenticated username so a route could log/attribute it;
    raises 401 with a `WWW-Authenticate: Basic` challenge otherwise. The
    credential itself is never logged or echoed back.
    """
    settings = get_settings()
    if credentials is None or not _credentials_match(
        settings, credentials.username, credentials.password
    ):
        raise _unauthorized()
    # Non-secret: the *username* only, and only for request attribution.
    request.state.gateway_user = credentials.username
    return credentials.username


def _decode_basic_header(header_value: str | None) -> tuple[str, str] | None:
    """Parse `Authorization: Basic <b64(user:pass)>` -> `(user, pass)`.

    Returns None for anything unparseable (absent header, wrong scheme, bad
    base64, no colon) so the caller treats every malformed case identically.
    """
    if not header_value:
        return None
    scheme, _, encoded = header_value.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    username, separator, password = decoded.partition(":")
    if not separator:
        return None
    return username, password


def websocket_client_is_authorized(websocket: WebSocket) -> bool:
    """Validate the Basic header on a WebSocket upgrade request.

    A native `URLSessionWebSocketTask` *can* set `Authorization` on the
    upgrade (browsers cannot), and we control the client -- so the same
    credential path is reused here instead of inventing a query-string ticket
    that would end up in server logs and browser history.

    The caller must reject with close code 1008 (policy violation) before
    accepting the socket or streaming anything.
    """
    parsed = _decode_basic_header(websocket.headers.get("authorization"))
    settings = get_settings()
    if parsed is None:
        # Still burn an equivalent comparison so an absent header and a wrong
        # one are not trivially distinguishable by timing.
        _credentials_match(settings, "", "")
        return False
    username, password = parsed
    return _credentials_match(settings, username, password)
