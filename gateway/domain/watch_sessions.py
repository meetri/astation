"""Sessions the gateway opened as a lazy WATCH, not as a conversation."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

WATCH_TTL_S = 6 * 60 * 60
MAX_WATCHED = 500


class WatchedSessions:
    def __init__(self, ttl_s: float = WATCH_TTL_S, capacity: int = MAX_WATCHED) -> None:
        self._entries: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._ttl = ttl_s
        self._capacity = capacity
        self._lock = threading.Lock()

    def watch(self, stored_id: str, profile: str = "default") -> None:
        if not stored_id:
            return
        key = (profile, stored_id)
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = time.time()
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def unwatch(self, stored_id: str, profile: str = "default") -> None:
        with self._lock:
            self._entries.pop((profile, stored_id), None)

    def is_watched(self, stored_id: str | None, profile: str = "default") -> bool:
        if not stored_id:
            return False
        key = (profile, stored_id)
        with self._lock:
            seen = self._entries.get(key)
            if seen is None:
                return False
            if time.time() - seen >= self._ttl:
                self._entries.pop(key, None)
                return False
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


watched = WatchedSessions()
