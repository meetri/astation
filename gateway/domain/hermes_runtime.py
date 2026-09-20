"""Connecting to Hermes, and retrying safely when the socket was already dead.

Extracted from `api/main.py` unchanged when the Projects routes arrived
(`api/projects.py`), because those routes also need to reach Hermes -- the
per-project session list is enriched with `session.list` metadata -- and a
second copy of the reconnect rules is precisely the kind of duplication B-15
turned into a double-submitted prompt. It lived at `api/hermes_runtime.py`
until CLEANUP_PLAN step 3.5 moved the stateful services (`BackgroundLedger`,
`SnapshotSweeper`, `AttachmentOrchestrator`, ...) under `domain/`; they all
resolve live handles and reconnect through here, and `domain/` must not import
from `api/`, so this is the runtime edge's home now. `api.hermes_runtime` and
`api.main` re-export every name, so `api.main._ensure_connected` still
resolves for anything that reads it there.

Nothing in here knows about the workspace database. It is the runtime edge and
nothing else.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException

from adapters.hermes import (
    HermesAdapter,
    HermesConnectionError,
    HermesError,
    HermesRPCError,
)
from domain.live_handles import LiveHandleCache

logger = logging.getLogger(__name__)


async def _ensure_connected(app_state: Any, adapter: HermesAdapter) -> None:
    """Idempotently get `adapter` through `login()` + `connect()` exactly once.

    One shared adapter per profile, reused across requests; the default
    profile's adapter is `app.state.adapter`, the others live in
    `domain/profile_connection.py` and go through `resolve_profile_adapter`.

    Also the *re*-connect path. `adapter.is_connected` goes back to False when
    Hermes hangs up (restart, Mac sleep, Wi-Fi blip), so the next request
    re-establishes the socket instead of the gateway staying wedged until
    uvicorn is restarted. The lock keeps concurrent requests from racing two
    `connect()` calls against each other.

    **`app_state` is the caller's `request.app.state` / `websocket.app.state`,
    never the module-global `app.state` (B-22).** The lock and the adapter must
    come from the *same* application object or the serialization is imaginary:
    every route reads its adapter from the request's app, so reaching for the
    module-global app's lock protects a different application's connect. Today
    there is exactly one app object and the two are the same lock, but the
    moment this router is mounted on a sub-app -- or a test builds a second
    `FastAPI()` -- concurrent requests would race two `connect()` calls on the
    same adapter, and the second connect tears down the socket the first just
    handed to its caller (and bumps `connection_generation` twice, stranding
    every live handle in `LiveHandleCache`).
    """
    if adapter.is_connected:
        return
    async with app_state.hermes_connect_lock:
        # Re-check under the lock: whoever held it may have just connected.
        if adapter.is_connected:
            return
        if not adapter.is_logged_in:
            await adapter.login()
        await adapter.connect()


async def _with_reconnect(
    app_state: Any, adapter: HermesAdapter, operation: Callable[[], Awaitable[Any]]
) -> Any:
    """Run `operation()`, rebuilding the Hermes socket and retrying once if it was dead.

    `app_state` is threaded through to `_ensure_connected()` for the reason
    given there (B-22): the connect lock has to belong to the same application
    object the adapter came from.

    A socket can die between requests (Hermes restart, idle reap, Wi-Fi blip)
    with nothing noticing until the next send fails, so the first request after
    the drop would otherwise always surface an error the system then silently
    heals from. One retry turns that into a normal response.

    **Only an operation Hermes never received is retried.** That is not true of
    every `HermesConnectionError`, which is what B-15 was: the adapter raises
    the same class for "the socket was already dead, nothing was sent" *and*
    for "the frame went out, then the socket closed before the reply came
    back". Replaying the second kind resubmits a prompt Hermes has already
    started acting on -- the user's message is sent to a real research session
    twice, and if the first copy is still streaming the duplicate returns
    `redirected`, i.e. it is applied as a *correction* to that in-flight turn.

    `HermesConnectionError.request_was_sent` is how the adapter tells the two
    apart, decided where the truth is known (`HermesAdapter._recv_loop` fails
    pending futures; `request()` pops its own entry when the send fails). A
    delivered-but-unanswered call propagates untouched. The common case -- a
    socket that died between requests, so the failure happens before the send
    -- still retries, which is the whole point of B-11.

    A delivered request whose reply is lost to a *timeout* raises
    `HermesProtocolError` and was never retried here either, for the same
    reason.
    """
    await _ensure_connected(app_state, adapter)
    try:
        return await operation()
    except HermesConnectionError as exc:
        if getattr(exc, "request_was_sent", False):
            raise
        await _ensure_connected(app_state, adapter)
        return await operation()


def resolve_profile_adapter(app_state: Any, profile: str | None) -> HermesAdapter:
    """The `HermesAdapter` a session-scoped route should talk to (B-136).

    `profile` is `None`, `""`, or `"default"` for the connection every route
    used before multi-profile support existed -- `app_state.hermes_adapter`,
    unchanged. Anything else is looked up on
    `app_state.profile_connection_manager` (built at `lifespan` time,
    `api/main.py`; see `domain/profile_connection.py`), which owns one
    `HermesAdapter` per non-default profile, auto-provisioned from
    `profiles.list` once `RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S` is
    turned on.

    A stored session id is only unique *within* a profile
    (`docs/CHAT_HISTORY_DESIGN.md` §4), so a route must never guess which
    connection to use -- the caller passes `profile` explicitly (a `Session`
    row's own `profile` column for a filed session, or a value the caller
    already has from browsing `GET /api/sessions?profile=...` for an unfiled
    one), and this is the one place that turns it into a connection.

    Raises a 503 naming the profile if it isn't connected right now (never
    ever silently falls back to the default connection -- that would submit
    a message to the wrong agent).
    """
    if not profile or profile == "default":
        return app_state.hermes_adapter
    manager = getattr(app_state, "profile_connection_manager", None)
    connection = manager.get_connection(profile) if manager is not None else None
    if connection is not None:
        return connection.adapter
    # No per-profile connection. Inside Hermes that is the NORMAL case and not
    # an error: `session.create` and `session.resume` both accept a `profile`
    # on 0.21.3, so one connection reaches every profile and no isolated
    # dashboard (nor the Docker socket needed to launch one) is required.
    # Measured: a bare resume of another profile's session answers `[4007]
    # session not found`, which is why the profile has to be PASSED -- see
    # `_resume_for_live_id`.
    return app_state.hermes_adapter


def profile_is_observable(app_state: Any, profile: str | None) -> bool:
    """Can we ask a connection "is this profile's session still running?"

    NOT the same question as `resolve_profile_adapter`. One connection can
    *drive* every profile — `session.create` and `session.resume` both take a
    `profile` — but `session.active_list` answers only for the profile the
    connection itself belongs to. **Measured on 0.21.3:** a running session in
    another profile is absent from `active_list`, with or without a `profile`
    parameter.

    That distinction is load-bearing. "Absent from active_list" is exactly
    what the stale-run reconciliation reads as "the turn is over", so asking
    the shared connection about another profile would close genuinely-running
    turns (B-136, and the reason `Run.profile` exists). Anything that infers
    liveness must call this first and SKIP a profile it cannot observe.
    """
    if not profile or profile == "default":
        return True
    manager = getattr(app_state, "profile_connection_manager", None)
    connection = manager.get_connection(profile) if manager is not None else None
    return connection is not None


def resolve_live_handle_cache(app_state: Any, profile: str | None) -> Any:
    """The `LiveHandleCache` a session-scoped route should read/write (B-136).

    `LiveHandleCache` used to be injected as a `cache_factory=` parameter
    because it lived in `api.main`, which imports this module; it lives in
    `domain/live_handles.py` now (CLEANUP_PLAN step 3.1) and is imported
    directly.

    **Why a non-default profile cannot share `app_state.live_handle_cache`:**
    that cache is keyed on `(connection_generation, stored_id)`, read off
    *one specific adapter object* it was constructed with
    (`LiveHandleCache.__init__`). A non-default profile's connection is a
    completely different `HermesAdapter` instance with its own, independent
    generation counter -- sharing the cache would key entries against the
    wrong adapter's generation, and could return a live handle resolved
    against a dashboard process that isn't the one now serving that profile.
    So each non-default connection gets its own cache, lazily built here and
    stored on the `ProfileConnection` itself
    (`domain.profile_connection.ProfileConnection.live_handle_cache`) --
    see that field's docstring for why a stale one can never outlive a
    reconnect.
    """
    if not profile or profile == "default":
        return app_state.live_handle_cache
    manager = getattr(app_state, "profile_connection_manager", None)
    connection = manager.get_connection(profile) if manager is not None else None
    if connection is not None:
        if connection.live_handle_cache is None:
            connection.live_handle_cache = LiveHandleCache(connection.adapter)
        return connection.live_handle_cache

    # One shared connection serving every profile. Each profile still gets its
    # OWN cache: a stored id is unique only within a profile
    #, so a single cache keyed on
    # (generation, stored_id) could alias two different sessions that happen to
    # share an id and hand a route the wrong live handle. The caches all bind
    # to the same adapter, so they share its generation counter and a
    # reconnect invalidates every one of them together.
    adapter = app_state.hermes_adapter
    caches = getattr(app_state, "shared_profile_handle_caches", None)
    if caches is None:
        caches = {}
        app_state.shared_profile_handle_caches = caches
    cache = caches.get(profile)
    # A `LiveHandleCache` is bound to ONE adapter object: it reads that
    # object's generation counter to key its entries, so a cache built against
    # a different adapter can hand back a handle resolved on a connection this
    # adapter never made -- a latent `[4001] session not found`. The default
    # cache is rebuilt with its adapter by `lifespan`; these are built lazily
    # and would otherwise outlive a swapped adapter, so check identity.
    if cache is None or cache.adapter is not adapter:
        cache = LiveHandleCache(adapter)
        caches[profile] = cache
    return cache


def _validate_stored_session_id(stored_session_id: str) -> str:
    """Reject an obviously-unusable stored id before any Hermes round-trip."""
    cleaned = stored_session_id.strip()
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="stored_session_id must be a non-empty Hermes stored session id",
        )
    return cleaned


# --- Hermes's "that session isn't here" error codes ----------------
#
# Both are real, both were measured against the live instance on 2026-08-29,
# and they are NOT interchangeable -- they belong to the two id spaces:
#
#   [4001] session not found -- a **live-handle** method was given a handle
#          this Hermes process does not have. Probed live: `session.history`,
#          `prompt.submit`, `session.activate` and `session.interrupt` all
#          answer 4001 for a garbage live handle *and* for a stored id passed
#          where a live handle belongs. This is also what a cached handle
#          turns into when Hermes restarts under a still-open socket, which is
#          the self-heal path in `_with_live_handle()`.
#   [4007] session not found -- `session.resume` was given a **stored** id it
#          cannot load: an unknown/typo'd id, or a session Hermes never
#          persisted. Probed live: a brand-new `session.create` that has not
#          been prompted yet has `message_count: 0`, does not appear in
#          `session.list`, and resuming its stored id fails [4007] (its live
#          handle still works). See `docs/PROTOCOL_VERIFIED.md`.
#
# A caller asking about a session that doesn't exist is a client error (404),
# not a gateway/upstream failure (502) -- see `_http_error_from_hermes()`.
_HERMES_LIVE_SESSION_NOT_FOUND_CODE = 4001
_HERMES_STORED_SESSION_NOT_FOUND_CODE = 4007
_HERMES_SESSION_NOT_FOUND_CODES: frozenset[int] = frozenset(
    {_HERMES_LIVE_SESSION_NOT_FOUND_CODE, _HERMES_STORED_SESSION_NOT_FOUND_CODE}
)

# Narrow text fallback, for a *third* code this gateway has not catalogued
# yet. Deliberately the whole observed phrase and not the bare words "not
# found": every 4001/4007 seen live carries exactly `session not found`, while
# "not found" on its own also matches ordinary upstream failures -- a tool
# reporting a missing file, a model name Hermes cannot resolve -- and reporting
# one of those as `404 "Hermes has no session with stored id ..."` names the
# wrong thing entirely and sends whoever debugs it after a session that is
# fine. Kept only because the code list is knowledge, not a guarantee: a
# Hermes upgrade could add a fourth code, and a session that has genuinely
# gone missing must still 404 and must still self-heal a stale live handle.
_HERMES_SESSION_NOT_FOUND_TEXT = "session not found"


def _rpc_error_code(exc: HermesRPCError) -> int | None:
    """`exc.code` as an int, or None if it isn't one.

    `HermesRPCError.code` is typed `Any` because JSON-RPC lets a server put
    anything there; every code this instance sends is an int, but a string
    `"4001"` must not silently fall through to the text fallback.
    """
    code = exc.code
    if isinstance(code, bool):  # bool is an int subclass; not a code
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str):
        try:
            return int(code.strip())
        except ValueError:
            return None
    return None


def _is_session_not_found(exc: HermesError) -> bool:
    """Whether `exc` is Hermes's "that session isn't here" answer (B-23).

    Decided on the **numeric code** -- `[4001]` for an unknown live handle,
    `[4007]` for a stored id `session.resume` cannot load -- both confirmed
    live and documented above and in `docs/PROTOCOL_VERIFIED.md`. That is the
    whole of the intended rule; the message check below is a fallback for a
    code we have not seen yet, scoped to Hermes's exact `session not found`
    phrase.

    This used to accept *any* RPC error whose message contained the words "not
    found", which quietly turned unrelated upstream failures into
    `404 "Hermes has no session with stored id ..."` and bought them one
    pointless `session.resume` retry in `_with_live_handle()`. It also meant
    `[4007]` was only ever handled correctly by accident, because its message
    happens to read `session not found`. Widening 404 to errors that are not
    about sessions is not something to do implicitly -- a new code gets added
    to `_HERMES_SESSION_NOT_FOUND_CODES` with the observation that justifies
    it.

    Two callers depend on this and both want the same answer: 404-vs-502 in
    `_http_error_from_hermes()`, and the one-shot re-resolve of a stale cached
    live handle in `_with_live_handle()`.
    """
    if not isinstance(exc, HermesRPCError):
        return False
    code = _rpc_error_code(exc)
    if code is not None and code in _HERMES_SESSION_NOT_FOUND_CODES:
        return True
    # Fallback only. Reached for an uncatalogued code, a non-numeric code, or
    # a code we would otherwise call unrelated -- and only when Hermes said
    # this exact thing about a session.
    return _HERMES_SESSION_NOT_FOUND_TEXT in str(exc.message).lower()


def _http_error_from_hermes(exc: HermesError, stored_session_id: str) -> HTTPException:
    """Map any `HermesError` onto a structured HTTP error -- never a bare 500.

    An unknown/typo'd stored session id is a *client* mistake, so Hermes's
    `[4001] session not found` becomes a 404 naming the id that failed.
    Everything else (auth failure, unreachable host, protocol mismatch, some
    other RPC error) is an upstream problem and becomes a 502 carrying the
    adapter's own message. The Hermes exception messages deliberately contain
    no credential material (see `adapters/hermes/exceptions.py`), so echoing
    them into `detail` is safe.
    """
    if _is_session_not_found(exc):
        return HTTPException(
            status_code=404,
            detail=(f"Hermes has no session with stored id {stored_session_id!r} ({exc})"),
        )
    return HTTPException(status_code=502, detail=str(exc))


async def _resume_for_live_id(
    adapter: HermesAdapter,
    stored_session_id: str,
    cache: LiveHandleCache | None = None,
    profile: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Given a STORED session id, return `(live_handle, raw_resume_result)`.

    This is the *only* place in the service that issues the resolution
    documented in `docs/PROTOCOL_VERIFIED.md` ("Session resume & the two id
    spaces"), so no route re-implements it:

        look up the stored id -> `session.resume {"session_id": <stored>}`
        -> use the `session_id` it returns (the LIVE handle)

    Always performs a real `session.resume`, so it also returns the
    transcript that call loads (`message_count` / `messages`) -- that is what
    `POST /resume` is for, and why that route does not read from the cache.
    The freshly resolved handle *is* written to `cache` (when one is given)
    so a subsequent send/fetch on the same session does not have to pay for
    the transcript again. `session.resume` is idempotent within a connection
    window, so this never leaks a second live session.

    Callers that only need the handle should go through `_with_live_handle()`
    instead: on a cache hit it skips this round trip entirely, which is the
    whole point of B-01 (the 1.6 MB transcript was being re-downloaded on
    every message fetch and every prompt submit).
    """
    # A stored id is only unique WITHIN a profile, and a bare resume searches
    # the connection's own profile store only -- measured: resuming another
    # profile's session without this answers `[4007] session not found`. Naming
    # the profile lets ONE connection reach every profile, which is what
    # removed the need for a per-profile isolated dashboard (and the Docker
    # socket that launching one required).
    extra = {"profile": profile} if profile and profile != "default" else {}
    result = await adapter.resume_session(stored_session_id, **extra)
    live_id = HermesAdapter.live_id_from_resume(result)
    if cache is not None:
        cache.put(stored_session_id, live_id)
    return live_id, result


async def _with_live_handle(
    adapter: HermesAdapter,
    cache: LiveHandleCache | None,
    stored_session_id: str,
    operation: Callable[[str], Awaitable[Any]],
    profile: str | None = None,
) -> tuple[str, Any]:
    """Run `operation(live_handle)`, resolving the handle from cache if possible.

    Returns `(live_handle_used, operation_result)`.

    Two failure modes are handled, and only these two:

    * **Wrong connection.** Impossible by construction -- `LiveHandleCache`
      keys on the adapter's `connection_generation`, so a handle from a
      previous socket is unreachable, and `_with_reconnect()` reconnects
      (bumping that counter) before this ever runs.
    * **Hermes forgot the session under a still-open socket** (its own
      restart). The cached handle looks fine to us but answers
      `[4001] session not found`. Then, and only then, the entry is dropped
      and the operation is retried **once** against a freshly resumed handle.
      A second `[4001]` is a genuine "no such session" and propagates to the
      404 mapping.

    Retrying is safe for the operations that use this, including
    `prompt.submit`: `[4001]` means Hermes rejected the call outright, so
    there is no half-applied turn to double-submit. (A call that *was*
    delivered but whose reply was lost surfaces as a timeout instead, which
    is deliberately never retried -- see `_with_reconnect`.)
    """
    cached = cache.get(stored_session_id) if cache is not None else None
    if cached is not None:
        try:
            return cached, await operation(cached)
        except HermesError as exc:
            if not _is_session_not_found(exc):
                raise
            # Stale handle: self-heal rather than surfacing a 404 for a
            # session that does exist.
            logger.info(
                "cached live handle for %s was rejected by Hermes; re-resolving",
                stored_session_id,
            )
            cache.discard(stored_session_id)  # type: ignore[union-attr]

    live_id, _result = await _resume_for_live_id(
        adapter, stored_session_id, cache, profile=profile
    )
    return live_id, await operation(live_id)
