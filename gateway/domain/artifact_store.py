"""The content-addressed artifact store (P3-1) -- `ArtifactStore` and the row shape."""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import posixpath
import uuid
from pathlib import Path
from typing import Any

from domain.artifact_kinds import extension_of, kind_for, source_dir_of
from domain.models import Artifact
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

# Exceeding this is the designed unavailable path, not a crash; it bounds disk use.
MAX_INGEST_BYTES = 256 * 1024 * 1024


STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"

_FALLBACK_MIME = "application/octet-stream"


def _guess_mime(source_path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(source_path)
    return guessed or _FALLBACK_MIME


class ArtifactTooLargeError(Exception):
    """The stream exceeded `MAX_INGEST_BYTES`; the partial temp file is gone."""


class ArtifactStore:
    """Files under the PINNED `<root>/<sha256[:2]>/<sha256>` layout."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # The shard/name layout is pinned: changing it migrates stored files, not just rows.
    def storage_key_for(self, checksum: str) -> str:
        return f"{checksum[:2]}/{checksum}"

    def path_for_key(self, storage_key: str) -> Path:
        """Resolve a row's storage key to a real path, re-confined to the root."""
        # Revalidated though the key is ours: a bad key must not escape the root.
        normalized = posixpath.normpath(storage_key)
        if normalized.startswith(("/", "..")) or "\x00" in normalized:
            raise ValueError(f"storage key {storage_key!r} escapes the artifact root")
        resolved = self.root / normalized
        return resolved

    async def write_stream(self, chunks: Any) -> tuple[str, int, str]:
        """Consume an async byte iterator into the store."""
        incoming = self.root / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        temp_path = incoming / uuid.uuid4().hex
        digest = hashlib.sha256()
        size = 0
        try:
            with open(temp_path, "wb") as handle:  # noqa: ASYNC230 — sync write; CLEANUP_PLAN §3
                async for chunk in chunks:
                    size += len(chunk)
                    if size > MAX_INGEST_BYTES:
                        raise ArtifactTooLargeError(
                            f"artifact exceeds the {MAX_INGEST_BYTES} B ingest cap"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
            checksum = digest.hexdigest()
            storage_key = self.storage_key_for(checksum)
            final_path = self.root / storage_key
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                temp_path.unlink(missing_ok=True)
            else:
                os.replace(temp_path, final_path)
            return checksum, size, storage_key
        # After a successful replace the temp name is gone; this covers the other paths.
        finally:
            temp_path.unlink(missing_ok=True)


def _artifact_json(artifact: Artifact) -> dict[str, Any]:
    """One artifact row for the API. `storage_key` stays server-side: it is a
    store-internal path fragment, and the only public address for content is
    `GET /api/artifacts/{id}/content` (§14, no filesystem paths on the wire)."""
    return {
        "id": artifact.id,
        "project_id": artifact.project_id,
        "producing_run_id": artifact.producing_run_id,
        "title": artifact.title,
        "mime_type": artifact.mime_type,
        "status": artifact.status,
        "size_bytes": artifact.size_bytes,
        "checksum": artifact.checksum,
        "source_path": artifact.source_path,
        "metadata": artifact.metadata_json,
        "created_at": iso_z(artifact.created_at),
        "kind": kind_for(artifact.mime_type, artifact.source_path),
        "extension": extension_of(artifact.source_path),
        # A star is per project, so it cannot be read off this row; filled in per scope.
        "bookmarked_at": None,
        "bookmarked": False,
        "archived_at": iso_z(artifact.archived_at) if artifact.archived_at else None,
        "archived": artifact.archived_at is not None,
        "source_dir": source_dir_of(artifact.source_path),
    }
