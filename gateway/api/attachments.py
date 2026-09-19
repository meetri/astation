"""Composer attachments: upload, poll, list and the capability serve route (P3-3).

The chain itself -- prime -> verify -> attach, as a background task per row
-- is `AttachmentOrchestrator` in `domain/attachment_orchestrator.py`
(CLEANUP_PLAN step 3.5), re-exported here with the row states and the
priming-command builders.

## The capability serve URL

The priming turn's `curl` runs on the Hermes host WITHOUT the gateway's
Basic-auth credential (credentials never go into prompt text, transcripts,
or logs). So the serve route is mounted UNAUTHENTICATED on the app root and
keyed on a single-use-scope capability token (`secrets.token_urlsafe(32)`):

* the token grants exactly one attachment's bytes, nothing else;
* it is refused (410) once the row is terminal (attached/failed/orphaned),
  so the copy that lands in the session transcript goes dead as soon as the
  flow ends;
* an unknown token is a plain 404 with no detail.

The URL's host comes from `RESEARCH_GATEWAY_PUBLIC_BASE_URL` when set, else
from the upload request's own Host header -- correct whenever the phone and
the Hermes host both reach the gateway at the same address (P3-0c: verified
live, the GET arrived from the Hermes host). **Firewall gotcha, measured:**
macOS's Application Firewall silently blackholes a server binary
not on its allow list (TCP accepts, HTTP never answers). The gateway process
itself must be firewall-allowed or the priming curl times out with nothing
logged anywhere -- the verify timeout is what surfaces it.

Bytes live in the shared content-addressed `ArtifactStore` -- an attachment
re-uses the artifact root and its crash-safe write path, and attaching the
same file twice stores it once.
"""

from __future__ import annotations

import logging
import mimetypes
import posixpath
import re
import secrets
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from api.artifacts import _file_slice
from config.settings import get_settings
from domain.artifact_store import ArtifactStore, ArtifactTooLargeError
from domain.attachment_orchestrator import (  # noqa: F401  (re-exported; see module docstring)
    _CURL_MAX_TIME_S,
    _IN_FLIGHT_STATES,
    _UPLOAD_CHUNK_BYTES,
    STATE_ATTACHED,
    STATE_ATTACHING,
    STATE_FAILED,
    STATE_ORPHANED,
    STATE_PRIMING,
    STATE_UPLOADED,
    TERMINAL_STATES,
    VERIFY_POLL_INTERVAL_S,
    VERIFY_TIMEOUT_S,
    AttachmentOrchestrator,
    build_prime_command,
    build_prime_text,
)
from domain.db import schema_checked_db, table_present
from domain.hermes_runtime import (
    _validate_stored_session_id,
)
from domain.models import Attachment, new_id
from domain.sandbox_paths import validate_sandbox_path
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

#: Authenticated routes (mounted under `/api` in `api.main`).
attachments_router = APIRouter(tags=["attachments"])

#: The UNAUTHENTICATED capability serve route (mounted on the app root in
#: `api.main` -- see module docstring for why it must not require Basic auth).
attachment_serve_router = APIRouter(tags=["attachments"])

#: Upload size cap (task spec: ~25MB). Enforced while streaming the upload,
#: so an oversized body is refused at the cap, not after buffering.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

_FALLBACK_MIME = "application/octet-stream"


def sanitize_filename(raw: str | None) -> str:
    """A sandbox-safe, shell-inert filename from whatever the picker sent.

    The filename ends up inside the priming turn's command line and in a
    sandbox path, so this is a security boundary, not cosmetics: only
    `[A-Za-z0-9._-]` survives, path components are stripped, leading dots are
    de-fanged (no hidden files, no `..`), and the result is bounded and
    never empty.
    """
    base = posixpath.basename((raw or "").replace("\\", "/")).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    cleaned = cleaned.lstrip(".")
    if len(cleaned) > 80:
        stem, dot, ext = cleaned.rpartition(".")
        if dot and 0 < len(ext) <= 12:
            cleaned = stem[: 80 - len(ext) - 1].rstrip(".") + "." + ext
        else:
            cleaned = cleaned[:80]
    return cleaned or "file"


def resolve_mime(declared: str | None, filename: str) -> str:
    """The attachment's MIME type: the picker's declaration when it said
    something real, else a guess from the filename, else octet-stream."""
    declared = (declared or "").split(";")[0].strip().lower()
    if declared and declared != _FALLBACK_MIME:
        return declared
    guessed, _encoding = mimetypes.guess_type(filename)
    return guessed or declared or _FALLBACK_MIME


def kind_for_mime(mime_type: str) -> str:
    """`image` rides `image.attach` (the P3-0b/P3-0c measured RPC);
    everything else -- PDF and documents first, per the owner's multimedia
    priority -- is referenced by its verified sandbox path."""
    return "image" if mime_type.startswith("image/") else "document"


def reference_text_for(attachment: Attachment) -> str | None:
    """What the app appends to the user's message for a document attachment.

    Only a VERIFIED document gets one -- the path has been fetched back
    byte-exact, so telling the agent to read it is honest. Images get None:
    `image.attach` already referenced them into the conversation.
    """
    if attachment.kind != "document" or attachment.state != STATE_ATTACHED:
        return None
    return (
        f"[Attached file: {attachment.sandbox_path} "
        f"({attachment.mime_type}, {attachment.size_bytes} bytes) -- "
        "uploaded by the user; read it from that sandbox path.]"
    )


def _attachment_json(attachment: Attachment) -> dict[str, Any]:
    """One row for the API. `serve_token` and `storage_key` STAY server-side:
    the token is the capability itself and the key is store-internal (§14).
    `attach_result` is scrubbed of `serve_url` for the same reason -- it
    embeds the token (found leaking in the live rehearsal's own output)."""
    attach_result = attachment.attach_result_json
    if isinstance(attach_result, dict):
        attach_result = {k: v for k, v in attach_result.items() if k != "serve_url"}
        attach_result = attach_result or None
    return {
        "id": attachment.id,
        "stored_session_id": attachment.stored_session_id,
        "filename": attachment.filename,
        "mime_type": attachment.mime_type,
        "size_bytes": attachment.size_bytes,
        "checksum": attachment.checksum,
        "kind": attachment.kind,
        "sandbox_path": attachment.sandbox_path,
        "state": attachment.state,
        "detail": attachment.detail,
        "attach_result": attach_result,
        "reference_text": reference_text_for(attachment),
        "created_at": iso_z(attachment.created_at),
        "updated_at": iso_z(attachment.updated_at),
    }


# ---------------------------------------------------------------------------
# DB dependency (503 until the attachments migration has run)
# ---------------------------------------------------------------------------


#: A DB session, with "you never ran the attachments migration" as a 503
#: (`domain.db.schema_checked_db`, same pattern as `api.artifacts._artifacts_db`).
_attachments_db = schema_checked_db(
    "attachments_schema_verified", lambda engine: table_present(engine, "attachments")
)


# ---------------------------------------------------------------------------
# Routes: upload, poll, list
# ---------------------------------------------------------------------------


async def _upload_chunks(upload: UploadFile) -> AsyncIterator[bytes]:
    while True:
        chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


def _serve_base_url(request: Request) -> str:
    """The base URL the SANDBOX HOST will curl -- see settings docstring."""
    configured = get_settings().research_gateway_public_base_url.strip()
    if configured:
        return configured.rstrip("/")
    host = request.headers.get("host", "")
    if not host:
        raise HTTPException(
            status_code=422,
            detail=(
                "cannot build a serve URL: the request carried no Host header "
                "and RESEARCH_GATEWAY_PUBLIC_BASE_URL is not set"
            ),
        )
    return f"http://{host}"


@attachments_router.post("/sessions/{stored_session_id}/attachments", status_code=202)
async def upload_attachment(
    stored_session_id: str,
    file: UploadFile,
    request: Request,
    db: OrmSession = Depends(_attachments_db),
) -> dict:
    """Accept one composer attachment and start the asynchronous attach.

    Multipart upload, 25MB cap enforced while streaming (413 past it). The
    202 answer is immediate -- bytes stored, ledger row written, orchestration
    scheduled -- and carries the row the app then POLLS via
    `GET /api/attachments/{id}` until `state` is terminal. No Hermes call
    happens on this request path (a prompt turn takes minutes and must never
    sit in a synchronous UI path -- measured, PV "Phase 3 build probes").
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    settings = get_settings()
    # The attachment dir must itself be inside the sandbox root -- validated
    # here, before anything is stored, so a misconfiguration fails loudly on
    # the first upload rather than silently in the background task.
    attachment_dir = validate_sandbox_path(
        settings.hermes_attachment_dir, settings.hermes_sandbox_root
    )
    serve_base = _serve_base_url(request)

    filename = sanitize_filename(file.filename)
    mime_type = resolve_mime(file.content_type, filename)
    store: ArtifactStore = request.app.state.artifact_store

    async def _capped() -> AsyncIterator[bytes]:
        total = 0
        async for chunk in _upload_chunks(file):
            total += len(chunk)
            if total > MAX_ATTACHMENT_BYTES:
                raise ArtifactTooLargeError(f"attachment exceeds the {MAX_ATTACHMENT_BYTES} B cap")
            yield chunk

    try:
        checksum, size, storage_key = await store.write_stream(_capped())
    except ArtifactTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if size == 0:
        raise HTTPException(status_code=422, detail="attachment is empty (0 bytes)")

    token = secrets.token_urlsafe(32)
    # Mint the id up front (the ORM default only fires at flush) -- the
    # sandbox target path embeds it for per-attachment uniqueness.
    attachment = Attachment(
        id=new_id("attach"),
        stored_session_id=stored_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=size,
        checksum=checksum,
        storage_key=storage_key,
        kind=kind_for_mime(mime_type),
        serve_token=token,
        state=STATE_UPLOADED,
        detail="uploaded; scheduling the sandbox fetch",
    )
    # Unique target per attachment: dir + row id + sanitized name. Nothing
    # user-controlled escapes `sanitize_filename`'s character class.
    attachment.sandbox_path = f"{attachment_dir}/{attachment.id}_{filename}"
    attachment.attach_result_json = {"serve_url": f"{serve_base}/attachments/serve/{token}"}
    db.add(attachment)
    db.commit()

    orchestrator: AttachmentOrchestrator = request.app.state.attachment_orchestrator
    orchestrator.start_attach(attachment.id)
    return {"attachment": _attachment_json(attachment)}


@attachments_router.get("/attachments/{attachment_id}")
async def get_attachment(attachment_id: str, db: OrmSession = Depends(_attachments_db)) -> dict:
    """The app's poll target while it shows "preparing attachment"."""
    attachment = db.get(Attachment, attachment_id)
    if attachment is None:
        raise HTTPException(status_code=404, detail=f"no attachment with id {attachment_id!r}")
    return {"attachment": _attachment_json(attachment)}


@attachments_router.get("/sessions/{stored_session_id}/attachments")
async def list_session_attachments(
    stored_session_id: str, db: OrmSession = Depends(_attachments_db)
) -> dict:
    """This session's attachments, newest first (app resume/restore)."""
    stored_id = _validate_stored_session_id(stored_session_id)
    rows = db.execute(
        select(Attachment)
        .where(Attachment.stored_session_id == stored_id)
        .order_by(Attachment.created_at.desc(), Attachment.id)
        .limit(100)
    ).scalars()
    return {
        "stored_session_id": stored_id,
        "attachments": [_attachment_json(a) for a in rows],
    }


# ---------------------------------------------------------------------------
# The capability serve route (unauthenticated -- see module docstring)
# ---------------------------------------------------------------------------


@attachment_serve_router.get("/attachments/serve/{token}")
async def serve_attachment(token: str, request: Request):
    """Serve one attachment's bytes to the priming turn's `curl`.

    Keyed on the capability token alone: the curl runs on the Hermes host
    with no gateway credential. Unknown token -> bare 404; terminal row ->
    410 (the token in the transcript goes dead the moment the flow ends).
    Exact `Content-Type`/`Content-Length` so the fetched copy is byte-exact
    by construction.
    """
    factory = getattr(request.app.state, "db_sessions", None)
    if factory is None:  # pragma: no cover - lifespan always sets it
        raise HTTPException(status_code=404, detail="not found")
    with factory() as db:
        row = db.execute(
            select(Attachment).where(Attachment.serve_token == token)
        ).scalar_one_or_none()
        if row is None or not secrets.compare_digest(row.serve_token, token):
            raise HTTPException(status_code=404, detail="not found")
        if row.state in TERMINAL_STATES:
            raise HTTPException(
                status_code=410,
                detail="this attachment's serve link is no longer active",
            )
        storage_key = row.storage_key
        mime_type = row.mime_type
        size = row.size_bytes
        filename = row.filename
    store: ArtifactStore = request.app.state.artifact_store
    try:
        path = store.path_for_key(storage_key)
    except ValueError as exc:  # pragma: no cover - corrupt row
        raise HTTPException(status_code=404, detail="not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return StreamingResponse(
        _file_slice(path, 0, size - 1) if size else iter(()),
        media_type=mime_type,
        headers={
            "content-length": str(size),
            "content-disposition": f'attachment; filename="{filename}"',
        },
    )
