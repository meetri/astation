"""File-backed rewrite prompts, editable from the phone."""

from __future__ import annotations

import logging
import posixpath
import time
from collections.abc import Awaitable, Callable
from typing import Any

from domain.sandbox_fs import sandbox_fs_for
from domain.sandbox_paths import is_under_root

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_SUBDIR = "astation/prompts"

# The file name describes the prompt, not the wire style: "explain" reads code.md.
PROMPT_FILE_FOR_STYLE: dict[str, str] = {
    "listen": "listen.md",
    "document": "document.md",
    "explain": "code.md",
}

# Not in PROMPT_FILE_FOR_STYLE: it is not a rewrite style, and /api/rewrite rejects it.
HANDOFF_PROMPT_FILE = "handoff.md"

# full and raw are absent on purpose: full is the style prompt, raw never reaches one.
PROMPT_FILE_FOR_DEPTH: dict[str, str] = {
    "brief": "depth-brief.md",
    "medium": "depth-medium.md",
}

MAX_PROMPT_CHARS = 32_768


def prompt_dir(settings: Any) -> str:
    """The directory the prompt files live in."""
    configured = (getattr(settings, "rewrite_prompt_dir", "") or "").strip()
    if configured:
        return configured
    root = (getattr(settings, "hermes_sandbox_root", "") or "/opt/data").rstrip("/")
    return posixpath.join(root, DEFAULT_PROMPT_SUBDIR)


class PromptFileStore:
    """Reads prompt files through an injected reader, with a TTL cache."""

    def __init__(
        self,
        reader: Callable[[str], Awaitable[str | None]],
        *,
        directory: str,
        ttl_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reader = reader
        self._directory = directory
        self._ttl_s = max(0.0, ttl_s)
        self._clock = clock
        self._cache: dict[str, tuple[float, str | None]] = {}

    @property
    def directory(self) -> str:
        return self._directory

    def invalidate(self) -> None:
        """Drop every cached entry. Not called in the request path; here so a
        test (or a future explicit 'reload prompts' action) need not wait out
        the TTL."""
        self._cache.clear()

    async def text(self, name: str | None) -> str | None:
        """The prompt in `name`, or `None` to fall back to the setting."""
        if not name:
            return None
        now = self._clock()
        cached = self._cache.get(name)
        if cached is not None and cached[0] > now:
            return cached[1]
        value = await self._read(name)
        self._cache[name] = (now + self._ttl_s, value)
        return value

    async def _read(self, name: str) -> str | None:
        path = posixpath.join(self._directory, name)
        try:
            raw = await self._reader(path)
        except Exception:  # pragma: no cover - the reader is already guarded
            logger.exception("prompt file %s could not be read; using the configured prompt", path)
            return None
        if raw is None:
            return None
        text = raw.strip()
        # An empty file means "use the setting", so clearing one in the editor reverts.
        if not text:
            return None
        if len(text) > MAX_PROMPT_CHARS:
            logger.warning(
                "prompt file %s is %d chars (cap %d); using the configured prompt instead",
                path,
                len(text),
                MAX_PROMPT_CHARS,
            )
            return None
        return text


def hermes_prompt_reader(
    app_state: Any, *, sandbox_root: str
) -> Callable[[str], Awaitable[str | None]]:
    """A reader that fetches prompt files through the sandbox backend."""

    async def read(path: str) -> str | None:
        # Hermes's own fs/read-text confines nothing: measured, it reads /etc/passwd.
        if not is_under_root(posixpath.normpath(path), sandbox_root):
            logger.warning(
                "prompt directory %r is outside the sandbox root %r; "
                "file-backed prompts are ignored (the app's editor could not "
                "write there either)",
                path,
                sandbox_root,
            )
            return None
        # Resolved per call, never captured: tests replace the adapter after startup.
        adapter = getattr(app_state, "hermes_adapter", None)
        if adapter is None:
            return None
        try:
            response = await sandbox_fs_for(app_state, adapter).fs_read_text(path)
        except Exception:
            return None
        if getattr(response, "status_code", None) != 200:
            return None
        try:
            body = response.json()
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        if body.get("truncated") is True or body.get("binary") is True:
            return None
        text = body.get("text")
        return text if isinstance(text, str) else None

    return read


__all__ = [
    "DEFAULT_PROMPT_SUBDIR",
    "HANDOFF_PROMPT_FILE",
    "MAX_PROMPT_CHARS",
    "PROMPT_FILE_FOR_DEPTH",
    "PROMPT_FILE_FOR_STYLE",
    "PromptFileStore",
    "hermes_prompt_reader",
    "prompt_dir",
]
