"""`LiveHandleCache`: stored session id -> Hermes live handle, per connection."""

from __future__ import annotations

from typing import Any


class LiveHandleCache:
    """Stored session id -> Hermes live handle, valid for exactly one connection."""

    _MAX_ENTRIES = 256

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self.adapter = adapter
        self._handles: dict[tuple[int, str], str] = {}
        self._stored_by_live: dict[tuple[int, str], str] = {}
        self._generation_seen: int | None = None

    def _generation(self) -> int | None:
        """Current connection generation, or None if the adapter has none."""
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
        """Learn a live->stored pairing for *labelling* events only."""
        generation = self._generation()
        if generation is None:
            return
        self._sweep(generation)
        self._remember_live_mapping(generation, stored_id, live_id)

    def _remember_live_mapping(self, generation: int, stored_id: str, live_id: str) -> None:
        self._evict_oldest(self._stored_by_live, self._MAX_ENTRIES)
        self._stored_by_live[(generation, live_id)] = stored_id

    def stored_for_live(self, live_id: str) -> str | None:
        """The durable id behind a live handle on *this* connection, if known."""
        generation = self._generation()
        if generation is None:
            return None
        self._sweep(generation)
        return self._stored_by_live.get((generation, live_id))

    def discard(self, stored_id: str) -> None:
        """Forget one entry -- the self-heal path for a handle Hermes rejected."""
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
