"""The sandbox filesystem backend: Hermes's HTTP file API, or direct I/O.

PLUGIN_V2_PLAN section 4.4. Every sandbox read, write, listing and directory
create in this gateway goes through Hermes's undocumented `/api/files*` and
`/api/fs/*` routes today. Running as a Hermes *plugin* puts this code in the
same process as those routes and on the same filesystem, so the round trip
buys nothing: the gateway can `open()` the file.

**But only as a plugin.** The standalone sidecar (`deploy/docker-compose.yml`)
is a separate container that mounts its own data directory and nothing of
Hermes's -- the sandbox is not on its filesystem at all. It stays on HTTP, and
must, because it is the rollback path for the whole plugin cutover (plan
section 7 step 4). So this module is a *backend choice*, not a replacement:

* `HermesHttpSandboxFS` -- the adapter's routes. What the sidecar uses, and
  what the plugin falls back to if direct access is not configured.
* `DirectSandboxFS` -- `os` and `pathlib` in a worker thread. What the plugin
  uses.

Both present the adapter's five file methods and return `httpx.Response`, so
`api/sandbox.py`, `api/sandbox_text.py`, `domain/prompt_files.py`,
`domain/project_workspace.py` and `domain/artifact_ingest.py` keep their status
mapping, their shape checks and their health tripwire exactly as written. One
backend swap, no route rewrites, and the sidecar's behaviour is untouched by
construction.

**What direct I/O gives up, and what this module owes back.** Hermes's file
routes are not a dumb `open()`. They carry two guards the gateway has been
leaning on without owning:

1. **Real-path confinement.** `domain/sandbox_paths.validate_sandbox_path` is
   lexical by design -- it cannot see that `<root>/link` is a symlink to
   `/etc`. Hermes resolves every path and rejects anything landing outside its
   `locked_root`. That is the "two guards cover each other" note in
   `sandbox_paths`; going direct removes the second one. `_resolve` below
   resolves and re-checks containment, so the symlink escape stays closed.
2. **The sensitive-file denylist.** Hermes refuses to list, read or download
   `.env` / `.envrc` / `auth.json` / `credentials` / `.git-credentials` and the
   `mcp-tokens/` and `pairing/` trees (its own comment cites the credential-leak
   issue behind it). The sandbox root on the deploy host *is* `HERMES_HOME`, so
   those files are real and they are in the browsable tree. Direct I/O with no
   filter would expose every one of them through `GET /api/sandbox/files` and
   `GET /api/sandbox/text` -- a regression, not a speedup. `_is_sensitive_path`
   mirrors the upstream rule.

Both are read from Hermes 0.21.3's own source (`hermes_cli/web_routers/files.py`,
`hermes_cli/web_server_files.py`, tag `v2026.9.14`) rather than guessed, and
`tests/test_sandbox_fs.py` pins the behaviours that would fail silently.

**What direct I/O gains.** Besides the round trip: a real compare-and-swap.
Hermes's `write-text` has no conditional write, so `PUT /api/sandbox/text`
reads, compares a hash, then writes -- and anything that writes the file in
between (an agent's own `patch` tool) is silently lost under the editor's save
(`api/sandbox_text.py`, "Atomicity, honestly"). `fs_write_text(...,
if_match_sha256=...)` re-reads and compares *inside the same worker-thread
call* as the write, with no await point between them, which shrinks that window
from an HTTP round trip to the microseconds between two syscalls. It is not a
lock and this module does not claim to be one: a writer that lands inside that
window still wins. It is a window two orders of magnitude smaller, and the
HTTP backend keeps the old behaviour rather than pretending to offer the guard.

Download deliberately stays on HTTP in both backends -- see `files_download`.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import stat
from pathlib import Path
from typing import Any, Protocol

import anyio.to_thread
import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "DirectSandboxFS",
    "HermesHttpSandboxFS",
    "SandboxFS",
    "SandboxFSUnsupported",
    "sandbox_fs_for",
]


# ---------------------------------------------------------------------------
# Upstream constants, mirrored from Hermes 0.21.3 source
# ---------------------------------------------------------------------------

#: `hermes_cli/web_routers/files.py:_FS_TEXT_SOURCE_MAX_BYTES` -- above this a
#: read-text is refused outright (413) rather than previewed.
_TEXT_SOURCE_MAX_BYTES = 64 * 1024 * 1024

#: `_FS_TEXT_PREVIEW_MAX_BYTES` -- the read is CUT here and `truncated` is set.
#: The editor refuses to write back a truncated read, so this number is load
#: bearing: raising it here without raising it upstream would let the editor
#: open a file the sidecar's Hermes would then truncate on save.
_TEXT_PREVIEW_MAX_BYTES = 512 * 1024

#: `_FS_TEXT_WRITE_MAX_BYTES` -- write ceiling, 413 above it.
_TEXT_WRITE_MAX_BYTES = 8 * 1024 * 1024

#: `_FS_PREVIEW_LANGUAGE_BY_EXT`. The app's editor keys syntax highlighting off
#: this string, so it is copied verbatim rather than re-derived from mimetypes:
#: a different answer here than upstream would change highlighting the moment
#: the backend switched, which is exactly the kind of silent drift this whole
#: module is trying not to introduce.
_LANGUAGE_BY_EXT = {
    ".c": "c", ".conf": "ini", ".cpp": "cpp", ".css": "css", ".csv": "csv",
    ".go": "go", ".graphql": "graphql", ".h": "c", ".hpp": "cpp",
    ".html": "html", ".java": "java", ".js": "javascript", ".json": "json",
    ".jsx": "jsx", ".kt": "kotlin", ".lua": "lua", ".md": "markdown",
    ".mjs": "javascript", ".py": "python", ".rb": "ruby", ".rs": "rust",
    ".sh": "shell", ".sql": "sql", ".svg": "xml", ".toml": "toml",
    ".ts": "typescript", ".tsx": "tsx", ".txt": "text", ".xml": "xml",
    ".yaml": "yaml", ".yml": "yaml", ".zsh": "shell",
}

#: `_FS_MIME_TYPES` -- extensions where Hermes overrides `mimetypes`.
_MIME_OVERRIDES = {
    ".avi": "video/x-msvideo", ".bmp": "image/bmp", ".flac": "audio/flac",
    ".gif": "image/gif", ".jpeg": "image/jpeg", ".jpg": "image/jpeg",
    ".m4a": "audio/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
    ".mp3": "audio/mpeg", ".mp4": "video/mp4", ".ogg": "audio/ogg",
    ".opus": "audio/ogg; codecs=opus", ".png": "image/png",
    ".svg": "image/svg+xml", ".wav": "audio/wav", ".webm": "video/webm",
    ".webp": "image/webp",
}

#: `_SENSITIVE_MANAGED_FILE_BASENAMES` -- credential stores that must never be
#: listed, read or downloaded. Upstream's comment: these become live secrets in
#: the browsable tree the moment the managed root is HERMES_HOME, which on the
#: deploy host it is.
_SENSITIVE_BASENAMES = frozenset({
    "auth.json", "auth.lock", "credentials", "config.yaml",
    ".anthropic_oauth.json", "google_token.json", "google_oauth_pending.json",
    "google_oauth.json", "webhook_subscriptions.json", "bws_cache.json",
    "bws_cache.enc.json", ".git-credentials",
})

#: `_SENSITIVE_MANAGED_DIR_NAMES` -- whole subtrees of credential material.
#: Matched on ANY path component, because the browser can descend: a
#: basename-only check would still serve `mcp-tokens/<server>.json`.
_SENSITIVE_DIR_NAMES = frozenset({"mcp-tokens", "pairing"})


class SandboxFSUnsupported(RuntimeError):
    """A backend was asked for a guarantee it cannot make.

    Raised rather than degraded on purpose: the only current case is a
    conditional write against the HTTP backend, and silently dropping the
    condition would turn "refuse to clobber" into "clobber", which is the
    opposite of what the caller asked for.
    """


# ---------------------------------------------------------------------------
# Synthesized responses
# ---------------------------------------------------------------------------

#: Stand-in request on every synthesized response. `httpx.Response` needs one
#: before `.raise_for_status()` or `.request` is touched; no caller does today,
#: but an unset `request` raises a confusing `RuntimeError` if one ever starts.
_SYNTHETIC_REQUEST = httpx.Request("GET", "file:///sandbox-fs")


def _ok(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body, request=_SYNTHETIC_REQUEST)


def _err(status: int, detail: str) -> httpx.Response:
    """An error in Hermes's own `{"detail": ...}` shape.

    The wording matters as much as the status: `api/sandbox.py` forwards the
    upstream detail to the client, so a direct-backend message that reads
    differently from the HTTP one would change what the app displays purely
    because of where the gateway happens to be running.
    """
    return httpx.Response(status, json={"detail": detail}, request=_SYNTHETIC_REQUEST)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mime_type(path: Path) -> str:
    """Hermes's `_fs_mime_type`: its override table, then `mimetypes`."""
    suffix = path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _looks_binary(data: bytes) -> bool:
    """Hermes's `_fs_looks_binary`: a NUL, or >12% control bytes."""
    if not data:
        return False
    if b"\0" in data:
        return True
    suspicious = sum(1 for byte in data if byte < 32 and byte not in {9, 10, 13})
    return suspicious / len(data) > 0.12


def _is_sensitive_name(name: str) -> bool:
    """Hermes's `_is_sensitive_filename`: `.env` / `.env.*` / `.envrc` plus the
    credential-store basenames, case-insensitively."""
    lowered = name.lower()
    if lowered == ".env" or lowered.startswith(".env.") or lowered == ".envrc":
        return True
    return lowered in _SENSITIVE_BASENAMES


def _is_sensitive_path(path: Path) -> bool:
    """Hermes's `_is_sensitive_path`: sensitive basename, or any component a
    credential directory. Read-side guard -- list, read, download."""
    if _is_sensitive_name(path.name):
        return True
    return any(part.lower() in _SENSITIVE_DIR_NAMES for part in path.parts)


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


class SandboxFS(Protocol):
    """The five file operations, as `HermesAdapter` already shapes them.

    Every method returns a fully-read `httpx.Response` (except
    `files_download`, which streams), because that is what the existing
    callers parse -- including their error mapping and the undocumented-shape
    health tripwire in `api/sandbox.py`.
    """

    #: Whether `fs_write_text` can honour `if_match_sha256`. False on the HTTP
    #: backend, whose upstream route has no conditional write. Callers check
    #: this instead of catching `SandboxFSUnsupported` on the hot path.
    supports_conditional_write: bool

    #: For `GET /api/sandbox/health` and the plugin's `/health`: which backend
    #: is actually serving, since the two differ in confinement and speed and
    #: "which one am I on?" is otherwise unanswerable from outside.
    backend_name: str

    async def files_list(self, path: str) -> httpx.Response: ...

    async def files_download(
        self, path: str, *, range_header: str | None = None
    ) -> httpx.Response: ...

    async def fs_read_text(self, path: str) -> httpx.Response: ...

    async def fs_write_text(
        self, path: str, content: str, *, if_match_sha256: str | None = None
    ) -> httpx.Response: ...

    async def files_mkdir(self, path: str) -> httpx.Response: ...


# ---------------------------------------------------------------------------
# HTTP backend: what the sidecar keeps using
# ---------------------------------------------------------------------------


class HermesHttpSandboxFS:
    """The adapter's file routes, unchanged.

    A pass-through wrapper, not a reimplementation: the lazy login, the
    401-re-login-retry-once and the streaming download all stay where they
    are, in `adapters/hermes/client.py`. This class exists so the sidecar and
    the plugin select between two objects with one interface instead of
    branching at five call sites.
    """

    supports_conditional_write = False
    backend_name = "hermes-http"

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    async def files_list(self, path: str) -> httpx.Response:
        return await self._adapter.files_list(path)

    async def files_download(
        self, path: str, *, range_header: str | None = None
    ) -> httpx.Response:
        return await self._adapter.files_download(path, range_header=range_header)

    async def fs_read_text(self, path: str) -> httpx.Response:
        return await self._adapter.fs_read_text(path)

    async def fs_write_text(
        self, path: str, content: str, *, if_match_sha256: str | None = None
    ) -> httpx.Response:
        if if_match_sha256 is not None:
            raise SandboxFSUnsupported(
                "Hermes's write-text route has no conditional write; the "
                "caller must check supports_conditional_write and fall back "
                "to read-compare-write rather than dropping the condition"
            )
        return await self._adapter.fs_write_text(path, content)

    async def files_mkdir(self, path: str) -> httpx.Response:
        return await self._adapter.files_mkdir(path)


# ---------------------------------------------------------------------------
# Direct backend: what the plugin uses
# ---------------------------------------------------------------------------


class DirectSandboxFS:
    """`os`/`pathlib` under one root, in a worker thread.

    **Off the event loop, always.** The plugin's routes run on the dashboard's
    own uvicorn loop, shared with every Hermes session (plan section 10: "routes
    never block"). A sandbox walk is thousands of `stat` calls on a container
    bind mount; doing that inline would stall every other request in the
    process, not just this gateway's. Every filesystem touch below goes through
    `anyio.to_thread.run_sync`.

    **One root, resolved.** `root` is `Settings.hermes_sandbox_root`, and it is
    resolved once at construction so a symlinked root (a bind mount via a link)
    still compares equal to the resolved paths under it -- otherwise every
    request 403s and the cause is invisible.
    """

    supports_conditional_write = True
    backend_name = "direct"

    def __init__(self, root: str) -> None:
        self._root = Path(root).expanduser().resolve(strict=False)

    # -- path resolution ---------------------------------------------------

    def _resolve(self, raw_path: str) -> Path | httpx.Response:
        """Resolve inside the root, or an error response saying why.

        This is the guard that replaces Hermes's `locked_root`. The caller has
        already run the lexical check in `domain/sandbox_paths`; this adds what
        that one structurally cannot: symlinks are followed *before*
        containment is tested, so `<root>/link -> /etc` is rejected here even
        though it is spelled as a path inside the root.

        Returns a `Path` or an `httpx.Response` rather than raising, because
        the callers below are already in the business of turning responses
        into `HTTPException`s and a second error channel would mean two ways
        to express the same 403.
        """
        text = str(raw_path or "").strip()
        if not text or "\x00" in text:
            return _err(400, "Invalid path")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            return _err(400, "Path must be absolute")
        if ".." in candidate.parts:
            return _err(400, "Path cannot contain '..'")
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return _err(400, "Invalid path")
        if resolved != self._root and self._root not in resolved.parents:
            # Same wording and status as upstream, so a client cannot tell
            # which backend refused it -- see `_err`.
            return _err(403, "Path outside managed files root")
        return resolved

    # -- listing -----------------------------------------------------------

    def _entry(self, target: Path) -> dict[str, Any]:
        """Hermes's `_managed_file_entry`, minus the re-containment check.

        Upstream re-resolves and re-checks every entry against the root. That
        matters there because it also serves un-rooted installs; here the
        parent directory is already inside the resolved root and `scandir` does
        not leave it. A symlink entry is reported with `follow_symlinks=False`
        semantics for `is_directory` so a link to a directory outside the root
        lists as a file rather than as a descendable directory -- and
        descending into it fails `_resolve` anyway.
        """
        try:
            st = target.stat()
        except OSError:
            # A dangling symlink or a file deleted mid-walk. Upstream 500s the
            # whole listing; degrading one entry keeps the directory readable,
            # which is what a browse screen needs.
            st = None
        is_dir = bool(st is not None and stat.S_ISDIR(st.st_mode))
        return {
            "name": target.name or str(target),
            "path": str(target),
            "is_directory": is_dir,
            "size": None if (is_dir or st is None) else st.st_size,
            "mtime": None if st is None else st.st_mtime,
            "mime_type": None if is_dir else (
                mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            ),
        }

    def _list_sync(self, target: Path) -> httpx.Response:
        if not target.exists():
            return _err(404, "Path not found")
        if not target.is_dir():
            return _err(400, "Path is not a directory")
        try:
            with os.scandir(target) as scan:
                entries = [
                    self._entry(Path(entry.path))
                    for entry in scan
                    if not _is_sensitive_path(Path(entry.path))
                ]
        except PermissionError:
            return _err(403, "Directory is not readable")
        except OSError as exc:
            return _err(500, f"Could not read directory: {exc}")
        entries.sort(key=lambda item: (not item["is_directory"], str(item["name"]).lower()))
        root = str(self._root)
        parent = None
        if target.parent != target and target != self._root:
            parent = str(target.parent)
        return _ok({
            "path": str(target),
            "parent": parent,
            "entries": entries,
            "root": root,
            "locked_root": root,
            # False for the same reason upstream sets it False under a locked
            # root: this gateway serves one sandbox and the app must not offer
            # a "change root" affordance the backend would refuse.
            "can_change_path": False,
        })

    async def files_list(self, path: str) -> httpx.Response:
        resolved = self._resolve(path)
        if isinstance(resolved, httpx.Response):
            return resolved
        return await anyio.to_thread.run_sync(self._list_sync, resolved)

    # -- download ----------------------------------------------------------

    async def files_download(
        self, path: str, *, range_header: str | None = None
    ) -> httpx.Response:
        """Not implemented here on purpose -- download stays on HTTP.

        `GET /api/sandbox/download` streams arbitrary bytes with `Range`
        passthrough. Hermes's route already does that correctly and is the one
        file endpoint whose server-side confinement is worth keeping as a
        second wall for raw bytes (the plan's own note on section 4.4). There
        is also nothing to win: a single streamed response over loopback costs
        one round trip, not the thousands a walk costs.

        `api/sandbox.py` therefore keeps calling the adapter for downloads.
        This method exists so the protocol is honest about the gap rather than
        letting a future caller reach for it and silently get the wrong wall.
        """
        raise SandboxFSUnsupported(
            "downloads stay on Hermes's HTTP route, which streams and enforces "
            "its own confinement; call adapter.files_download directly"
        )

    # -- read text ---------------------------------------------------------

    def _read_text_sync(self, target: Path) -> httpx.Response:
        try:
            st = target.stat()
        except (FileNotFoundError, NotADirectoryError):
            return _err(404, "File not found")
        except PermissionError:
            return _err(403, "File is not readable")
        except OSError as exc:
            return _err(400, str(exc) or "Invalid path")
        if stat.S_ISDIR(st.st_mode):
            return _err(400, "Path points to a directory")
        if not stat.S_ISREG(st.st_mode):
            return _err(400, "Only regular files can be read")
        if _is_sensitive_path(target):
            return _err(403, "Access to sensitive files is not allowed")
        if st.st_size > _TEXT_SOURCE_MAX_BYTES:
            return _err(413, "File too large")
        try:
            with target.open("rb") as handle:
                data = handle.read(min(st.st_size, _TEXT_PREVIEW_MAX_BYTES))
        except PermissionError:
            return _err(403, "File is not readable")
        except OSError as exc:
            return _err(400, str(exc) or "File read failed")
        return _ok({
            "binary": _looks_binary(data[:4096]),
            "byteSize": st.st_size,
            "language": _LANGUAGE_BY_EXT.get(target.suffix.lower(), "text"),
            "mimeType": _mime_type(target),
            "path": str(target),
            "text": data.decode("utf-8", errors="replace"),
            "truncated": st.st_size > _TEXT_PREVIEW_MAX_BYTES,
            # Beyond the measured upstream shape. Additive: the read-text
            # shape check in `api/sandbox_text.py` asserts the keys it needs
            # are present, never that no others are, so the same body still
            # satisfies the HTTP-backend contract.
            "mtime": st.st_mtime,
        })

    async def fs_read_text(self, path: str) -> httpx.Response:
        resolved = self._resolve(path)
        if isinstance(resolved, httpx.Response):
            return resolved
        return await anyio.to_thread.run_sync(self._read_text_sync, resolved)

    # -- write text --------------------------------------------------------

    def _write_text_sync(
        self, target: Path, text: str, if_match_sha256: str | None
    ) -> httpx.Response:
        """Compare (optionally) and write, with no await point between them.

        The whole value of the conditional write is that these two steps share
        one thread hop: an `await` here would hand the loop back and reopen the
        window this is closing.
        """
        encoded = text.encode("utf-8")
        if len(encoded) > _TEXT_WRITE_MAX_BYTES:
            return _err(413, "Content too large")
        try:
            st: os.stat_result | None = target.stat()
        except FileNotFoundError:
            st = None
        except PermissionError:
            return _err(403, "File is not writable")
        except OSError as exc:
            return _err(400, str(exc) or "Invalid path")
        if st is not None and stat.S_ISDIR(st.st_mode):
            return _err(400, "Path points to a directory")
        if st is not None and not stat.S_ISREG(st.st_mode):
            return _err(400, "Only regular files can be written")
        if not target.parent.is_dir():
            return _err(400, "Parent directory does not exist")

        if if_match_sha256 is not None:
            current = "" if st is None else _decoded_preview(target, st)
            if current is None:
                return _err(400, "File read failed")
            current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if current_sha != if_match_sha256:
                # The body `api/sandbox_text.py` already returns for a failed
                # compare, built here so the route does not need a second
                # read to answer (and so the text it reports is the text that
                # actually lost the race, not a later one).
                return httpx.Response(
                    409,
                    json={
                        "detail": "file changed since it was read",
                        "current_sha256": current_sha,
                        "text": current,
                    },
                    request=_SYNTHETIC_REQUEST,
                )

        # Hermes's staging pattern: sibling temp file, then `os.replace`, so an
        # interrupted write leaves the original intact. The pid suffix keeps
        # two processes from colliding on the same temp name.
        tmp = target.with_name(f".{target.name}.trg-tmp-{os.getpid()}")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
        except PermissionError:
            tmp.unlink(missing_ok=True)
            return _err(403, "File is not writable")
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            return _err(500, f"Could not write file: {exc}")
        return _ok({"ok": True, "path": str(target), "byteSize": len(encoded)})

    async def fs_write_text(
        self, path: str, content: str, *, if_match_sha256: str | None = None
    ) -> httpx.Response:
        resolved = self._resolve(path)
        if isinstance(resolved, httpx.Response):
            return resolved
        if _is_sensitive_path(resolved):
            # Upstream's denylist is read-side only; applying it to writes as
            # well is deliberate. This gateway's editor has no business
            # rewriting `.env` or `auth.json`, and a write it cannot read back
            # would be a strange thing to allow.
            return _err(403, "Access to sensitive files is not allowed")
        return await anyio.to_thread.run_sync(
            self._write_text_sync, resolved, content, if_match_sha256
        )

    # -- mkdir -------------------------------------------------------------

    def _mkdir_sync(self, target: Path) -> httpx.Response:
        if target.exists() and not target.is_dir():
            return _err(409, "A file already exists at that path")
        try:
            target.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return _err(403, "Directory is not writable")
        except FileExistsError:
            return _err(409, "A file already exists at that path")
        except OSError as exc:
            return _err(500, f"Could not create directory: {exc}")
        root = str(self._root)
        return _ok({
            "ok": True,
            "path": str(target),
            "root": root,
            "locked_root": root,
            "can_change_path": False,
        })

    async def files_mkdir(self, path: str) -> httpx.Response:
        resolved = self._resolve(path)
        if isinstance(resolved, httpx.Response):
            return resolved
        return await anyio.to_thread.run_sync(self._mkdir_sync, resolved)


def _decoded_preview(target: Path, st: os.stat_result) -> str | None:
    """The text a `fs_read_text` would have returned, for the compare.

    Same cut and same `errors="replace"` decode, because the hash the app sent
    was computed over exactly that -- comparing against a differently-decoded
    or un-truncated read would fail every time on a file with one bad byte.
    """
    try:
        with target.open("rb") as handle:
            data = handle.read(min(st.st_size, _TEXT_PREVIEW_MAX_BYTES))
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def sandbox_fs_for(app_state: Any, adapter: Any) -> SandboxFS:
    """The backend for this request: whatever was installed, else HTTP.

    The plugin sets `app.state.sandbox_fs` at startup; the sidecar never does,
    so it gets the adapter path it has always had. Defaulting here rather than
    at startup means a gateway that comes up before the backend is chosen still
    serves files -- over HTTP, slower, correct -- instead of 500ing.
    """
    existing = getattr(app_state, "sandbox_fs", None)
    if existing is not None:
        return existing
    return HermesHttpSandboxFS(adapter)


def describe_backend(app_state: Any) -> dict[str, Any]:
    """What `/health` reports: which backend, and what it can do.

    Worth surfacing for the same reason `foreign_prompts: "hook"` is (plan
    round 7): the two backends differ in confinement and in whether a save can
    be clobbered, and the fallback is silent by design.
    """
    backend = getattr(app_state, "sandbox_fs", None)
    if backend is None:
        return {"backend": "hermes-http", "conditional_write": False}
    return {
        "backend": getattr(backend, "backend_name", type(backend).__name__),
        "conditional_write": bool(getattr(backend, "supports_conditional_write", False)),
    }
