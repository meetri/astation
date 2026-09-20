"""HTTP Basic auth for the Research Gateway's own surface."""

from __future__ import annotations

import base64
import binascii
import secrets

from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config.settings import Settings, get_settings

_WWW_AUTHENTICATE = {"WWW-Authenticate": 'Basic realm="research-gateway"'}

_basic_scheme = HTTPBasic(auto_error=False, realm="research-gateway")


def _unauthorized() -> HTTPException:
    """One 401 for every failure mode: missing, malformed, or wrong."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers=_WWW_AUTHENTICATE,
    )


def _credentials_match(settings: Settings, username: str | None, password: str | None) -> bool:
    """Constant-time check of a candidate username/password pair."""
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
    """FastAPI dependency guarding every `/api/*` route."""
    settings = get_settings()
    if credentials is None or not _credentials_match(
        settings, credentials.username, credentials.password
    ):
        raise _unauthorized()
    request.state.gateway_user = credentials.username
    return credentials.username


def _decode_basic_header(header_value: str | None) -> tuple[str, str] | None:
    """Parse `Authorization: Basic <b64(user:pass)>` -> `(user, pass)`."""
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
    """Validate the Basic header on a WebSocket upgrade request."""
    parsed = _decode_basic_header(websocket.headers.get("authorization"))
    settings = get_settings()
    if parsed is None:
        _credentials_match(settings, "", "")
        return False
    username, password = parsed
    return _credentials_match(settings, username, password)
