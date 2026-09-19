"""`LiveHandleCache`: stored session id -> Hermes live handle, per connection.

Moved here from `api/main.py` (CLEANUP_PLAN step 3.1) so `domain/hermes_runtime.py`,
`api/prompts.py`, `api/snapshots.py` and `domain/profile_connection.py` can
import it directly instead of taking it as an injected factory or a local
import to dodge the cycle through the app module. `api/main.py` re-exports it.

The two-id-spaces rule (`CLAUDE.md`): stored ids persist, live handles are
process-local to one Hermes connection and change across reconnects. This
class is the only place a stored->live mapping is cached, and its key includes
the connection generation so a handle from a dead connection is unreachable
by construction.
"""

from __future__ import annotations

from typing import Any


class LiveHandleCache:
    """Stored session id -> Hermes live handle, valid for exactly one connection.

    Why this needs care (B-01 vs B-06). `session.resume` is the only way to
    turn a durable stored id into the live handle every real operation needs
    -- and it returns the **entire transcript** in the same reply. Re-running
    it per request meant a 1.6 MB download on every message fetch and every
    single chat message sent. But a live handle is process-local to the
    Hermes gateway and is *not* stable across connections (the same stored id
    resolved to `5bfd9de6`, `d8779141`, and `7373be85` on three different
    connections), so a naively cached handle is a latent
    `[4001] session not found`.

    The safety here is structural, not disciplinary: **the cache key is
    `(connection_generation, stored_id)`**, and `connection_generation` is
    bumped by `HermesAdapter.connect()`. A handle resolved on a previous
    connection is therefore not merely stale, it is unreachable -- there is no
    lookup that can return it. A generation change additionally sweeps the
    whole map, so nothing from a dead connection lingers in memory either.

    That still leaves the case the generation counter cannot see: Hermes
    itself restarting (or forgetting the session) underneath a socket that is
    still open. That surfaces as `[4001]` on a call made with a cached
    handle, and `_with_live_handle()` self-heals it by dropping the entry and
    re-resolving exactly once.

    In-memory only, and deliberately: this must not survive a process restart
    and a live handle must never reach the database (`domain/models.py` keeps
    `runtime_live_session_id` as an ephemeral column for the same reason).

    **Two maps, on purpose (B-29).** `_handles` is stored -> live and is what
    `prompt.submit` / `session.history` resolve against; `_stored_by_live` is
    live -> stored and is used only to *label* outgoing events with the
    durable id the app can match on. Every `put()` writes both, but
    `observe_live_mapping()` -- the path that learns from a `session.info`
    event rather than from our own `session.resume` -- writes only the second.
    A wrong entry in the labelling map mislabels a frame; a wrong entry in the
    submit map sends the user's message into someone else's research session.
    Those are not the same risk and they do not share a code path.
    """

    # Bounded so a long-running gateway that touches many sessions cannot
    # grow this without limit. Entries are two short strings; 256 is far more
    # than the 43 sessions on the live instance.
    _MAX_ENTRIES = 256

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._handles: dict[tuple[int, str], str] = {}
        # B-29 attribution only -- never read by anything that submits.
        self._stored_by_live: dict[tuple[int, str], str] = {}
        self._generation_seen: int | None = None

    def _generation(self) -> int | None:
        """Current connection generation, or None if the adapter has none.

        An adapter that does not expose `connection_generation` cannot make
        the by-construction guarantee above, so caching is simply disabled
        rather than done unsafely.
        """
        generation = getattr(self._adapter, "connection_generation", None)
        return generation if isinstance(generation, int) else None

    def _sweep(self, generation: int) -> None:
        """Drop everything from older connections when the generation moves."""
        if self._generation_seen != generation:
            self._generation_seen = generation
            self._handles = {
                key: value for key, value in self._handles.items() if key[0] == generation
            }
            self._stored_by_live = {
                key: value for key, value in self._stored_by_live.items() if key[0] == generation
            }

    @staticmethod
    def _evict_oldest(entries: dict[tuple[int, str], str], limit: int) -> None:
        if len(entries) >= limit:
            # dicts preserve insertion order, so this evicts the oldest entry.
            entries.pop(next(iter(entries)))

    def get(self, stored_id: str) -> str | None:
        generation = self._generation()
        if generation is None:
            return None
        self._sweep(generation)
        return self._handles.get((generation, stored_id))

    def put(self, stored_id: str, live_id: str) -> None:
        generation = self._generation()
        if generation is None:
            return
        self._sweep(generation)
        self._evict_oldest(self._handles, self._MAX_ENTRIES)
        self._handles[(generation, stored_id)] = live_id
        self._remember_live_mapping(generation, stored_id, live_id)

    def observe_live_mapping(self, stored_id: str, live_id: str) -> None:
        """Learn a live->stored pairing for *labelling* events only (B-29).

        Fed by `session.info`, which is the one Hermes event carrying both ids
        -- so a session started somewhere else entirely (Hermes's own TUI, a
        second client) becomes attributable without this gateway ever having
        resumed it. Deliberately does **not** touch the stored->live map that
        `prompt.submit` resolves against: labelling a frame wrong is a display
        bug, resolving a submit wrong puts the user's text in another session.
        """
        generation = self._generation()
        if generation is None:
            return
        self._sweep(generation)
        self._remember_live_mapping(generation, stored_id, live_id)

    def _remember_live_mapping(self, generation: int, stored_id: str, live_id: str) -> None:
        self._evict_oldest(self._stored_by_live, self._MAX_ENTRIES)
        self._stored_by_live[(generation, live_id)] = stored_id

    def stored_for_live(self, live_id: str) -> str | None:
        """The durable id behind a live handle on *this* connection, if known.

        `None` is a real answer, not a failure: it means no route has resolved
        that handle and no `session.info` has named it on this connection, so
        the gateway genuinely does not know. The caller stamps the null rather
        than guessing -- see `EventBroadcaster._stamp_session_identity`.
        """
        generation = self._generation()
        if generation is None:
            return None
        self._sweep(generation)
        return self._stored_by_live.get((generation, live_id))

    def discard(self, stored_id: str) -> None:
        """Forget one entry -- the self-heal path for a handle Hermes rejected.

        Only the stored->live direction is dropped. The live->stored pairing
        stays: a frame Hermes already stamped with that handle still came from
        that session, and being able to say so is what keeps the loss visible.
        """
        generation = self._generation()
        if generation is None:
            return
        self._handles.pop((generation, stored_id), None)

    def clear(self) -> None:
        self._handles.clear()
        self._stored_by_live.clear()
        self._generation_seen = None

    def __len__(self) -> int:
        return len(self._handles)
