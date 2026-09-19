"""File-backed rewrite prompts, editable from the phone (owner, 2026-09-07).

The owner's ask: *"Can these prompts be saved on the file system and then I
can use the app's editing capabilities to edit it"* -- so the speech prompts
stop being a redeploy (or a Settings text field) and become files you open in
the app's own source editor, change, and save.

**Why the files live in the Hermes sandbox and not in the gateway's data dir.**
The app's editor writes through `PUT /api/sandbox/text`, which confines every
write to `Settings.hermes_sandbox_root` and performs it via Hermes's
`fs/write-text` -- the gateway container does not mount that filesystem at all
(measured against a deployed pair: Hermes maps its own data directory to
`hermes_sandbox_root`, default `/opt/data`, while the gateway mounts only its
own `deploy/data`). Putting the prompts anywhere else would mean teaching the
editor a second backing store. Putting them here
costs no app change whatsoever: the sandbox browser lists them, the viewer
opens them, the editor edits them, and the conflict dialog and
changed-on-disk banner work exactly as they do for any other file.

**Resolution order, per prompt** -- the file is an OVERRIDE, never a
requirement:

1. the file, if it exists and has non-whitespace content;
2. otherwise the configured setting (`REWRITE_PROMPT` and friends, including
   anything the runtime overlay set from Settings > Providers);
3. otherwise the built-in default in `config/settings.py`.

So a fresh install behaves exactly as it does today, deleting a file reverts
that one prompt to the setting, and nothing here can leave the rewrite route
without a prompt.

**Nothing in here may fail a rewrite.** Every read is best-effort: an
unreachable Hermes, a 404, a directory, a truncated or binary file, or a
shape the route did not expect all resolve to `None`, which means "use the
setting". That is the same discipline the speech path follows end to end --
every failure class ends in words being spoken.

**Caching.** A document plays as dozens of chunks and each one resolves a
style prompt and a depth suffix, so an uncached read would be a Hermes round
trip per chunk per prompt. Entries (including misses) are held for
`REWRITE_PROMPT_FILE_TTL_S` seconds, so an edit saved from the phone takes
effect within that window without a restart.

One thing worth knowing: these files are inside the agent's own sandbox, so
the agent can read and write them like any other file there. That is the
price of them being editable by the app's existing editor.
"""

from __future__ import annotations

import logging
import posixpath
import time
from collections.abc import Awaitable, Callable
from typing import Any

from domain.sandbox_fs import sandbox_fs_for
from domain.sandbox_paths import is_under_root

logger = logging.getLogger(__name__)

#: Directory holding the prompt files, relative to the sandbox root, when
#: `REWRITE_PROMPT_DIR` is empty.
DEFAULT_PROMPT_SUBDIR = "astation/prompts"

#: `style` -> file name. Same three styles `api/rewrite.py` knows; the file
#: names say what they are for rather than repeating the wire value, because
#: the owner reads these in a file browser (`code.md`, not `explain.md`).
PROMPT_FILE_FOR_STYLE: dict[str, str] = {
    "listen": "listen.md",
    "document": "document.md",
    "explain": "code.md",
}

#: The "Continue in a new session" prompt (`api/handoff.py`), as a file beside
#: the speech prompts. Not in `PROMPT_FILE_FOR_STYLE`: it is not a rewrite
#: style (`POST /api/rewrite` must keep refusing it with a 422), it just
#: shares the directory, the store and the fallback rule.
HANDOFF_PROMPT_FILE = "handoff.md"

#: `depth` -> file name for that depth's suffix. `full` and `raw` are absent
#: on purpose: `full` IS the style prompt's own instruction (there is no
#: suffix to edit) and `raw` never reaches a prompt at all.
PROMPT_FILE_FOR_DEPTH: dict[str, str] = {
    "brief": "depth-brief.md",
    "medium": "depth-medium.md",
}

#: Refuse anything larger than this many characters as a prompt. A prompt is
#: ~1-2 KB; a file this big is a mistake (the wrong file saved over one of
#: these), and sending it as a system prompt would burn the model's context.
MAX_PROMPT_CHARS = 32_768


def prompt_dir(settings: Any) -> str:
    """The directory the prompt files live in.

    `REWRITE_PROMPT_DIR` when set; otherwise `<sandbox root>/astation/
    prompts`, so the default follows the sandbox root instead of hardcoding
    `/opt/data` a second time.
    """
    configured = (getattr(settings, "rewrite_prompt_dir", "") or "").strip()
    if configured:
        return configured
    root = (getattr(settings, "hermes_sandbox_root", "") or "/opt/data").rstrip("/")
    return posixpath.join(root, DEFAULT_PROMPT_SUBDIR)


class PromptFileStore:
    """Reads prompt files through an injected reader, with a TTL cache.

    `reader(path)` returns the file's text, or `None` for anything that is not
    a usable prompt (missing, a directory, truncated, binary, unreachable). It
    is injected rather than taking a `HermesAdapter` directly so the whole
    resolution order is testable without a socket -- the same reason
    `SpeechDeliveryResolver` takes a `RewriteAPI` on the app side.
    """

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
        #: name -> (expires_at, text|None). A miss is cached too, so a
        #: 40-chunk document does not pay 40 round trips to learn the same
        #: file is still absent.
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
            # Belt and braces: a prompt file must never be able to fail a
            # rewrite, whatever a future reader does.
            logger.exception("prompt file %s could not be read; using the configured prompt", path)
            return None
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            # An empty file is "use the setting", not "use an empty prompt" --
            # so clearing a file in the editor is a revert, not a footgun.
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
    """A reader that fetches prompt files through the sandbox backend.

    The backend is resolved from `app_state` on every call rather than
    captured: the test suite swaps `app.state.hermes_adapter` for a fake
    AFTER startup has run, and a captured adapter would keep talking to the
    real one (the same rule `api/bootstrap.py` documents for the profile
    manager).

    Confined to `sandbox_root` before the call, because Hermes's own
    `fs/read-text` confines nothing -- measured, it will happily read
    `/etc/passwd`. This is the same wall `api/sandbox_text.py` puts in front
    of the identical route, applied here for the identical reason. It stays
    even on the direct backend, which confines too: this check names the
    PROMPT root, which is narrower than the sandbox, and the warning it logs
    is the only signal an operator gets that file-backed prompts are being
    silently ignored.

    Returns `None` for every non-answer -- unreachable, non-200, non-JSON, a
    shape that is not the measured one, a directory, a truncated read, or a
    binary file -- so the caller falls back to the configured prompt.
    """

    async def read(path: str) -> str | None:
        if not is_under_root(posixpath.normpath(path), sandbox_root):
            logger.warning(
                "prompt directory %r is outside the sandbox root %r; "
                "file-backed prompts are ignored (the app's editor could not "
                "write there either)",
                path,
                sandbox_root,
            )
            return None
        adapter = getattr(app_state, "hermes_adapter", None)
        if adapter is None:
            return None
        try:
            response = await sandbox_fs_for(app_state, adapter).fs_read_text(path)
        except Exception:
            # Hermes down, socket dead, anything: use the configured prompt.
            return None
        if getattr(response, "status_code", None) != 200:
            return None
        try:
            body = response.json()
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        # A truncated read is a partial prompt and a binary file is not one at
        # all; both are the editor's own refusal rules, applied here too.
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
