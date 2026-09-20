"""Grouping artifacts by what they ARE, for filtering a listing."""

from __future__ import annotations

import os

KINDS: tuple[str, ...] = ("image", "document", "code", "data", "audio", "video", "archive", "other")

_EXTENSIONS: dict[str, str] = {
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "webp": "image",
    "bmp": "image",
    "tiff": "image",
    "tif": "image",
    "heic": "image",
    "svg": "image",
    "pdf": "document",
    "md": "document",
    "markdown": "document",
    "txt": "document",
    "rtf": "document",
    "doc": "document",
    "docx": "document",
    "odt": "document",
    "qmd": "document",
    "tex": "document",
    "py": "code",
    "js": "code",
    "ts": "code",
    "tsx": "code",
    "jsx": "code",
    "swift": "code",
    "c": "code",
    "h": "code",
    "cpp": "code",
    "hpp": "code",
    "rs": "code",
    "go": "code",
    "rb": "code",
    "java": "code",
    "kt": "code",
    "sh": "code",
    "bash": "code",
    "zsh": "code",
    "sql": "code",
    "html": "code",
    "css": "code",
    "diff": "code",
    "patch": "code",
    "ipynb": "code",
    "json": "data",
    "yaml": "data",
    "yml": "data",
    "toml": "data",
    "csv": "data",
    "tsv": "data",
    "xml": "data",
    "parquet": "data",
    "npz": "data",
    "npy": "data",
    "db": "data",
    "sqlite": "data",
    "xlsx": "data",
    "wav": "audio",
    "mp3": "audio",
    "ogg": "audio",
    "oga": "audio",
    "m4a": "audio",
    "flac": "audio",
    "aac": "audio",
    "mp4": "video",
    "mov": "video",
    "webm": "video",
    "mkv": "video",
    "avi": "video",
    "zip": "archive",
    "tar": "archive",
    "gz": "archive",
    "tgz": "archive",
    "bz2": "archive",
    "xz": "archive",
    "7z": "archive",
}

_MIME_PREFIXES: tuple[tuple[str, str], ...] = (
    ("image/", "image"),
    ("audio/", "audio"),
    ("video/", "video"),
)

_MIME_EXACT: dict[str, str] = {
    "application/pdf": "document",
    "application/json": "data",
    "application/xml": "data",
    "text/csv": "data",
    "text/markdown": "document",
    "text/plain": "document",
    "application/zip": "archive",
    "application/gzip": "archive",
    "application/x-tar": "archive",
}


def extension_of(source_path: str | None) -> str:
    """The lowercased extension without its dot, or `""`."""
    if not source_path:
        return ""
    return os.path.splitext(source_path)[1].lstrip(".").lower()


def kind_for(mime_type: str | None, source_path: str | None = None) -> str:
    """Which `KINDS` bucket this artifact belongs to."""
    extension = extension_of(source_path)
    if extension in _EXTENSIONS:
        return _EXTENSIONS[extension]

    mime = (mime_type or "").strip().lower()
    if mime in _MIME_EXACT:
        return _MIME_EXACT[mime]
    for prefix, kind in _MIME_PREFIXES:
        if mime.startswith(prefix):
            return kind
    if mime.startswith("text/"):
        return "code"
    return "other"


def is_valid_kind(kind: str) -> bool:
    return kind in KINDS


def source_dir_of(source_path: str | None) -> str | None:
    """The directory an artifact was fetched from, or `None`."""
    cleaned = (source_path or "").strip()
    if not cleaned or not cleaned.startswith("/"):
        return None
    parent = os.path.dirname(cleaned.rstrip("/"))
    return parent if parent and parent != "/" else None


def folder_components(source_path: str | None) -> tuple[str, ...]:
    """The directory components of `source_path`, root first, no leading slash."""
    parent = source_dir_of(source_path)
    if parent is None:
        return ()
    return tuple(part for part in parent.strip("/").split("/") if part)


def is_within(source_path: str | None, prefix: str) -> bool:
    """Whether `source_path` sits at or under directory `prefix`."""
    cleaned = (source_path or "").strip()
    wanted = (prefix or "").strip().rstrip("/")
    if not cleaned or not wanted:
        return False
    return cleaned == wanted or cleaned.startswith(wanted + "/")
