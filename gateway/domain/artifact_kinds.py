"""Grouping artifacts by what they ARE, for filtering a listing.

The store records a `mime_type` and often a `source_path`, but neither is what
a person filters by: nobody looks for `application/vnd.oasis…`, they look for
"documents". This maps both signals onto a short vocabulary the app can show
as chips.

Two rules, and the order matters:

1. **The extension wins over the MIME type when they disagree.** The ingest
   path guesses a MIME from the filename and falls back to
   `application/octet-stream`, so a `.py` file routinely arrives as
   `text/x-python`, `text/plain` or octet-stream depending on the platform.
   The extension is the thing the user actually sees in the file browser.
2. **Unknown is its own answer, never a wrong one.** Anything unrecognised is
   `other`, which is filterable in its own right — guessing would quietly hide
   files from a filter that claims to be complete.
"""

from __future__ import annotations

import os

#: The filter vocabulary, in the order a UI should offer it.
KINDS: tuple[str, ...] = ("image", "document", "code", "data", "audio", "video", "archive", "other")

_EXTENSIONS: dict[str, str] = {
    # image
    "png": "image", "jpg": "image", "jpeg": "image", "gif": "image", "webp": "image",
    "bmp": "image", "tiff": "image", "tif": "image", "heic": "image", "svg": "image",
    # document
    "pdf": "document", "md": "document", "markdown": "document", "txt": "document",
    "rtf": "document", "doc": "document", "docx": "document", "odt": "document",
    "qmd": "document", "tex": "document",
    # code
    "py": "code", "js": "code", "ts": "code", "tsx": "code", "jsx": "code",
    "swift": "code", "c": "code", "h": "code", "cpp": "code", "hpp": "code",
    "rs": "code", "go": "code", "rb": "code", "java": "code", "kt": "code",
    "sh": "code", "bash": "code", "zsh": "code", "sql": "code", "html": "code",
    "css": "code", "diff": "code", "patch": "code", "ipynb": "code",
    # data
    "json": "data", "yaml": "data", "yml": "data", "toml": "data", "csv": "data",
    "tsv": "data", "xml": "data", "parquet": "data", "npz": "data", "npy": "data",
    "db": "data", "sqlite": "data", "xlsx": "data",
    # audio / video
    "wav": "audio", "mp3": "audio", "ogg": "audio", "oga": "audio", "m4a": "audio",
    "flac": "audio", "aac": "audio",
    "mp4": "video", "mov": "video", "webm": "video", "mkv": "video", "avi": "video",
    # archive
    "zip": "archive", "tar": "archive", "gz": "archive", "tgz": "archive",
    "bz2": "archive", "xz": "archive", "7z": "archive",
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
    """Which `KINDS` bucket this artifact belongs to.

    Extension first (see the module docstring), then the MIME type, then
    `other` — which is a real answer, not a failure.
    """
    extension = extension_of(source_path)
    if extension in _EXTENSIONS:
        return _EXTENSIONS[extension]

    mime = (mime_type or "").strip().lower()
    if mime in _MIME_EXACT:
        return _MIME_EXACT[mime]
    for prefix, kind in _MIME_PREFIXES:
        if mime.startswith(prefix):
            return kind
    # `text/*` that is not one of the exact matches above is source more often
    # than prose in this workspace (`text/x-python`, `text/x-shellscript`).
    if mime.startswith("text/"):
        return "code"
    return "other"


def is_valid_kind(kind: str) -> bool:
    return kind in KINDS
