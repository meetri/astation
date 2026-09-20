"""The sandbox filesystem backend: Hermes's HTTP file API, or direct I/O."""

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


_TEXT_SOURCE_MAX_BYTES = 64 * 1024 * 1024

_TEXT_PREVIEW_MAX_BYTES = 512 * 1024

_TEXT_WRITE_MAX_BYTES = 8 * 1024 * 1024

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

_MIME_OVERRIDES = {
    ".avi": "video/x-msvideo", ".bmp": "image/bmp", ".flac": "audio/flac",
    ".gif": "image/gif", ".jpeg": "image/jpeg", ".jpg": "image/jpeg",
    ".m4a": "audio/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
    ".mp3": "audio/mpeg", ".mp4": "video/mp4", ".ogg": "audio/ogg",
    ".opus": "audio/ogg; codecs=opus", ".png": "image/png",
    ".svg": "image/svg+xml", ".wav": "audio/wav", ".webm": "video/webm",
    ".webp": "image/webp",
}

_SENSITIVE_BASENAMES = frozenset({
    "auth.json", "auth.lock", "credentials", "config.yaml",
    ".anthropic_oauth.json", "google_token.json", "google_oauth_pending.json",
    "google_oauth.json", "webhook_subscriptions.json", "bws_cache.json",
    "bws_cache.enc.json", ".git-credentials",
})

_SENSITIVE_DIR_NAMES = frozenset({"mcp-tokens", "pairing"})


class SandboxFSUnsupported(RuntimeError):
    """A backend was asked for a guarantee it cannot make."""


_SYNTHETIC_REQUEST = httpx.Request("GET", "file:///sandbox-fs")


def _ok(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body, request=_SYNTHETIC_REQUEST)


def _err(status: int, detail: str) -> httpx.Response:
    """An error in Hermes's own `{"detail": ...}` shape."""
    return httpx.Response(status, json={"detail": detail}, request=_SYNTHETIC_REQUEST)


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


class SandboxFS(Protocol):
    """The five file operations, as `HermesAdapter` already shapes them."""

    supports_conditional_write: bool

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


class HermesHttpSandboxFS:
    """The adapter's file routes, unchanged."""

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


class DirectSandboxFS:
    """`os`/`pathlib` under one root, in a worker thread."""

    supports_conditional_write = True
    backend_name = "direct"

    def __init__(self, root: str) -> None:
        self._root = Path(root).expanduser().resolve(strict=False)


    def _resolve(self, raw_path: str) -> Path | httpx.Response:
        """Resolve inside the root, or an error response saying why."""
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
            return _err(403, "Path outside managed files root")
        return resolved


    def _entry(self, target: Path) -> dict[str, Any]:
        """Hermes's `_managed_file_entry`, minus the re-containment check."""
        try:
            st = target.stat()
        except OSError:
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
            "can_change_path": False,
        })

    async def files_list(self, path: str) -> httpx.Response:
        resolved = self._resolve(path)
        if isinstance(resolved, httpx.Response):
            return resolved
        return await anyio.to_thread.run_sync(self._list_sync, resolved)


    async def files_download(
        self, path: str, *, range_header: str | None = None
    ) -> httpx.Response:
        """Not implemented here on purpose -- download stays on HTTP."""
        raise SandboxFSUnsupported(
            "downloads stay on Hermes's HTTP route, which streams and enforces "
            "its own confinement; call adapter.files_download directly"
        )


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
            "mtime": st.st_mtime,
        })

    async def fs_read_text(self, path: str) -> httpx.Response:
        resolved = self._resolve(path)
        if isinstance(resolved, httpx.Response):
            return resolved
        return await anyio.to_thread.run_sync(self._read_text_sync, resolved)


    def _write_text_sync(
        self, target: Path, text: str, if_match_sha256: str | None
    ) -> httpx.Response:
        """Compare (optionally) and write, with no await point between them."""
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
                return httpx.Response(
                    409,
                    json={
                        "detail": "file changed since it was read",
                        "current_sha256": current_sha,
                        "text": current,
                    },
                    request=_SYNTHETIC_REQUEST,
                )

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
            return _err(403, "Access to sensitive files is not allowed")
        return await anyio.to_thread.run_sync(
            self._write_text_sync, resolved, content, if_match_sha256
        )


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
    """The text a `fs_read_text` would have returned, for the compare."""
    try:
        with target.open("rb") as handle:
            data = handle.read(min(st.st_size, _TEXT_PREVIEW_MAX_BYTES))
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def sandbox_fs_for(app_state: Any, adapter: Any) -> SandboxFS:
    """The backend for this request: whatever was installed, else HTTP."""
    existing = getattr(app_state, "sandbox_fs", None)
    if existing is not None:
        return existing
    return HermesHttpSandboxFS(adapter)


def describe_backend(app_state: Any) -> dict[str, Any]:
    """What `/health` reports: which backend, and what it can do."""
    backend = getattr(app_state, "sandbox_fs", None)
    if backend is None:
        return {"backend": "hermes-http", "conditional_write": False}
    return {
        "backend": getattr(backend, "backend_name", type(backend).__name__),
        "conditional_write": bool(getattr(backend, "supports_conditional_write", False)),
    }
