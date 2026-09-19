"""Compatibility shim: the runtime edge lives in `domain/hermes_runtime.py`.

Moved there in CLEANUP_PLAN step 3.5 so the stateful services under `domain/`
can reconnect and resolve live handles without importing a router package.
Every name is re-exported here so `from api.hermes_runtime import ...` keeps
resolving; new code should import from `domain.hermes_runtime`.
"""

from domain.hermes_runtime import (  # noqa: F401
    _HERMES_LIVE_SESSION_NOT_FOUND_CODE,
    _HERMES_SESSION_NOT_FOUND_CODES,
    _HERMES_SESSION_NOT_FOUND_TEXT,
    _HERMES_STORED_SESSION_NOT_FOUND_CODE,
    _ensure_connected,
    _http_error_from_hermes,
    _is_session_not_found,
    _resume_for_live_id,
    _rpc_error_code,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)
