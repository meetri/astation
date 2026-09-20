"""The content-addressed artifact store (P3-1) -- `ArtifactStore` and the row shape.

Moved out of `api/artifacts.py` (CLEANUP_PLAN step 3.4). The REST surface and
the range/slice helpers stay there; the ingestors are `domain/artifact_ingest.py`.

## On-disk layout -- PINNED (P3-1)

    <RESEARCH_GATEWAY_ARTIFACT_ROOT>/<sha256[:2]>/<sha256>

Content-addressed by the full lowercase hex sha256 of the bytes, sharded by
its first two hex chars (256 buckets). One file per unique content: two
artifact rows with the same checksum share one stored file, and the store
never deletes on ingest. In-flight writes go to `<root>/incoming/<uuid>` and
are `os.replace`d into place, so a crash mid-fetch leaves junk only under
`incoming/`, never a half-written addressed file. **Changing this layout
later means migrating real files, not just rows -- do not change it
casually** (the docstring you are reading is the pin the task spec asked
for).

"""

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

#: Hard cap on one ingested artifact. Measured byte-exact through 819,200 B
#: (P3-0d); multi-MB is inference, and an unbounded fetch could fill the
#: gateway's disk from a runaway sandbox file. Exceeding the cap is the
#: designed `unavailable` failure path, not a crash.
MAX_INGEST_BYTES = 256 * 1024 * 1024


#: Row statuses (P3-1.0, `domain/models.py::Artifact`).
STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"

_FALLBACK_MIME = "application/octet-stream"


def _guess_mime(source_path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(source_path)
    return guessed or _FALLBACK_MIME


# ---------------------------------------------------------------------------
# The content-addressed store
# ---------------------------------------------------------------------------


class ArtifactTooLargeError(Exception):
    """The stream exceeded `MAX_INGEST_BYTES`; the partial temp file is gone."""


class ArtifactStore:
    """Files under the PINNED `<root>/<sha256[:2]>/<sha256>` layout.

    Write path: stream into `<root>/incoming/<uuid>`, hashing as bytes
    arrive, then `os.replace` into the addressed location. Content-addressed
    means writing the same bytes twice is a no-op (the second temp file is
    discarded), and nothing here ever deletes an addressed file.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def storage_key_for(self, checksum: str) -> str:
        return f"{checksum[:2]}/{checksum}"

    def path_for_key(self, storage_key: str) -> Path:
        """Resolve a row's storage key to a real path, re-confined to the root.

        The key came out of our own database, but §14 says no route serves an
        arbitrary filesystem path -- so it is re-validated lexically the same
        way the sandbox routes validate theirs, and a corrupted/hand-edited
        key fails loudly instead of opening `/etc/passwd`.
        """
        normalized = posixpath.normpath(storage_key)
        if normalized.startswith(("/", "..")) or "\x00" in normalized:
            raise ValueError(f"storage key {storage_key!r} escapes the artifact root")
        resolved = self.root / normalized
        return resolved

    async def write_stream(self, chunks: Any) -> tuple[str, int, str]:
        """Consume an async byte iterator into the store.

        Returns `(sha256_hex, size_bytes, storage_key)`. Raises
        `ArtifactTooLargeError` past `MAX_INGEST_BYTES` (temp file removed).
        File I/O is synchronous on purpose -- local disk, chunk-sized writes,
        the same "microseconds are not worth an async driver" call as
        `domain/db.py`.
        """
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
                # Content-addressed dedup at the byte level: same bytes are
                # already stored; the fresh copy is redundant.
                temp_path.unlink(missing_ok=True)
            else:
                os.replace(temp_path, final_path)
            return checksum, size, storage_key
        finally:
            temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Rows as JSON
# ---------------------------------------------------------------------------


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
        # Grouping for the listing filter. Derived rather than stored, so a
        # better classification applies to every existing row without a
        # migration (`domain/artifact_kinds.py`).
        "kind": kind_for(artifact.mime_type, artifact.source_path),
        "extension": extension_of(artifact.source_path),
        #: A star is per PROJECT since 2026-09-19 (`domain/bookmark_store.py`),
        #: so it cannot be read off the artifact row: `bookmarked` /
        #: `bookmarked_at` are filled in by `_with_bookmarks` for the scope the
        #: request named. Defaulted here so every row carries the keys and a
        #: client never treats absence as a special case.
        "bookmarked_at": None,
        "bookmarked": False,
        #: Hidden from the library's default listings, not deleted: the bytes,
        #: the provenance and every transcript chip that opens this row keep
        #: working. There is no delete for an artifact anywhere in this system.
        "archived_at": iso_z(artifact.archived_at) if artifact.archived_at else None,
        "archived": artifact.archived_at is not None,
        #: The directory this was fetched from, derived rather than stored so
        #: it cannot drift from the path it came from
        #: (`domain/artifact_kinds.py`). NULL for a row with no path and for a
        #: bare root-level name -- neither has a folder.
        "source_dir": source_dir_of(artifact.source_path),
    }
