"""Direct filesystem access, replacing Hermes's undocumented file routes."""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

MAX_TEXT_BYTES = 2_000_000


class SandboxConfinementError(Exception):
    """A path resolved outside the sandbox root."""


def _resolve_within(raw_path: str, root: str) -> Path:
    """Resolve `raw_path` and refuse anything outside `root`."""
    root_real = Path(os.path.realpath(root))
    target = Path(os.path.realpath(raw_path))
    if target != root_real and root_real not in target.parents:
        raise SandboxConfinementError(f"path outside the sandbox root: {raw_path}")
    return target


def _json_response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("GET", "http://plugin.local/"),
    )


def _error(status: int, detail: str) -> httpx.Response:
    return _json_response({"detail": detail}, status=status)


def list_directory(raw_path: str, root: str) -> httpx.Response:
    """`GET /api/files?path=` — the measured shape, verbatim."""
    try:
        target = _resolve_within(raw_path, root)
    except SandboxConfinementError as exc:
        return _error(403, str(exc))
    if not target.exists():
        return _error(404, f"not found: {raw_path}")
    if not target.is_dir():
        return _error(400, f"not a directory: {raw_path}")

    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_directory": child.is_dir(),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "mime_type": mimetypes.guess_type(child.name)[0],
            })
    except OSError as exc:
        return _error(500, f"could not read the directory: {exc}")

    root_real = str(Path(os.path.realpath(root)))
    return _json_response({
        "path": str(target),
        "parent": str(target.parent) if str(target) != root_real else None,
        "entries": entries,
        "root": root_real,
        "locked_root": root_real,
        "can_change_path": False,
    })


def read_text(raw_path: str, root: str) -> httpx.Response:
    """`GET /api/fs/read-text?path=` — the measured shape, verbatim."""
    try:
        target = _resolve_within(raw_path, root)
    except SandboxConfinementError as exc:
        return _error(403, str(exc))
    if not target.exists() or not target.is_file():
        return _error(404, f"not found: {raw_path}")
    try:
        data = target.read_bytes()
    except OSError as exc:
        return _error(500, f"could not read the file: {exc}")

    truncated = len(data) > MAX_TEXT_BYTES
    head = data[:MAX_TEXT_BYTES] if truncated else data
    try:
        text = head.decode("utf-8")
        binary = False
    except UnicodeDecodeError:
        text = ""
        binary = True

    return _json_response({
        "binary": binary,
        "byteSize": len(data),
        "language": target.suffix.lstrip(".") or "",
        "mimeType": mimetypes.guess_type(target.name)[0] or "application/octet-stream",
        "path": str(target),
        "text": text,
        "truncated": truncated,
    })


def write_text(raw_path: str, content: str, root: str) -> httpx.Response:
    """`POST /api/fs/write-text` — the measured shape, verbatim."""
    try:
        target = _resolve_within(raw_path, root)
    except SandboxConfinementError as exc:
        return _error(403, str(exc))
    encoded = content.encode("utf-8")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.trg-tmp")
        temp.write_bytes(encoded)
        os.replace(temp, target)
    except OSError as exc:
        return _error(500, f"could not write the file: {exc}")
    return _json_response({"ok": True, "path": str(target), "byteSize": len(encoded)})


def make_directory(raw_path: str, root: str) -> httpx.Response:
    """`POST /api/files/mkdir` — parents created, an existing dir is a 200."""
    try:
        target = _resolve_within(raw_path, root)
    except SandboxConfinementError as exc:
        return _error(403, str(exc))
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _error(500, f"could not create the directory: {exc}")
    return _json_response({"ok": True, "path": str(target)})


def deliver_attachment(
    sandbox_path: str, source_path: str, root: str, expected_checksum: str | None = None
) -> tuple[bool, str]:
    """Put an uploaded file into the sandbox directly."""
    import hashlib
    import shutil

    try:
        target = _resolve_within(sandbox_path, root)
    except SandboxConfinementError as exc:
        return False, str(exc)

    source = Path(source_path)
    if not source.is_file():
        return False, f"the uploaded bytes are missing from the store: {source_path}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.trg-tmp")
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    except OSError as exc:
        return False, f"could not write the file into the sandbox: {exc}"

    size = target.stat().st_size
    if expected_checksum:
        digest = hashlib.sha256()
        try:
            with open(target, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            return False, f"could not re-read the file to verify it: {exc}"
        actual = digest.hexdigest()
        if actual != expected_checksum:
            return False, (
                f"the file written to the sandbox does not match the upload "
                f"(expected {expected_checksum[:12]}…, got {actual[:12]}…)"
            )
    return True, f"written into the sandbox ({size} bytes, sha256 match)"
