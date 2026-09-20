"""Editor text read/write (P7 phase 1): hash-guarded `GET`/`PUT /sandbox/text`.

The gateway half of `docs/EDITOR_DESIGN.md` section 10. Two routes over
Hermes's dashboard text-file API (`GET /api/fs/read-text`, `POST
/api/fs/write-text` -- measured live 2026-09-06, PV "File text read/write
for the editor"), no DB writes, no history yet.

* `GET /api/sandbox/text?path=<abs>` -- the file as decoded text plus the
  gateway's `sha256` of that text, Hermes's `language`/`mime_type`/
  `byte_size`/`truncated`/`binary`, and an `editable` verdict with a
  human reason when it is false.
* `PUT /api/sandbox/text` `{path, content, expected_sha256}` -- write the
  file, but only if the text Hermes holds RIGHT NOW still hashes to
  `expected_sha256`. Anything else is a 409 carrying the current text and
  its hash, so the app can offer reload / overwrite / see-diff (design
  section 4). Returns the GET shape re-read after the write, so the app's
  next save has a fresh hash without a second round trip.

**Reading can be widened; writing cannot.** `GET` validates with
`validate_view_path` -- the sandbox root plus any `RESEARCH_GATEWAY_VIEW_ROOTS`
entry -- and reports `editable: false` for anything only the latter admitted.
`PUT` still validates with `validate_sandbox_path`, so no setting can widen
what this gateway writes, and `domain/artifact_ingest.py` keeps walking the
strict root. Default is empty: same confinement as before.

**The path check here is the only wall.** Unlike `/api/files*` (which
Hermes confines to its `locked_root`), the `fs/*` routes confine nothing:
live, writing `/tmp/x` and reading `/etc/passwd` both returned 200, and a
relative path resolves against Hermes's cwd. Every route in this module
runs `domain.sandbox_paths.validate_sandbox_path` against
`Settings.hermes_sandbox_root` BEFORE any upstream call -- absolute,
normalized, no `..` escape, inside the root -- and forwards the normalized
path, never the raw one. This is not defense in depth; it is the defense.

**The hash is over the text, not the bytes.** Hermes decodes with
`errors="replace"`, so for a non-UTF-8 file the text is not the file. Both
sides of the guard (this module and the app) hash `text.encode("utf-8")`,
which is what the app actually holds and what a write would put back --
consistent even when it differs from the on-disk bytes. `byte_size` is
Hermes's on-disk `byteSize` and may differ from `len(text.encode())` for
exactly that reason.

**Why truncated and binary files are refused for editing.** Hermes cuts
`text` past 512 KiB (`truncated: true`) -- writing that back would
silently truncate the file on disk. A `binary: true` file has been
decoded with replacement characters -- writing it back would corrupt it.
Both open read-only with the reason; the app shows it.

**Atomicity, honestly.** A single write is as atomic as Hermes's
temp-file + rename. The compare-then-write
in `PUT` is NOT locked: an agent's `patch` tool can write the same file
between this route's read and its write, and that write is then lost
under the editor's. The design accepts this (section 4): Hermes offers no
mtime or compare-and-swap, the window is one HTTP round trip, and the
NEXT save catches it (the re-read hash will not match). Nothing in
phase 1 pretends otherwise.

File contents are never logged -- not on success, not on error.

Auth: mounted on the authenticated `/api` router in `api.main`, so Basic
auth is inherited by construction. Upstream, `fs_read_text`/`fs_write_text`
ride the same cookie jar and 401 -> re-login -> retry-once loop as the
browse routes.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from adapters.hermes import HermesAdapter, HermesError
from api.sandbox import _PASSTHROUGH_CLIENT_ERRORS, _raise_for_upstream
from config.settings import get_settings
from domain.sandbox_fs import SandboxFS, sandbox_fs_for
from domain.sandbox_paths import (
    is_under_root,
    parse_roots,
    validate_sandbox_path,
    validate_view_path,
)

logger = logging.getLogger(__name__)

sandbox_text_router = APIRouter(tags=["sandbox"])

#: Upstream client errors the text routes forward with their own status and
#: detail. The browse set (403/404/416) plus the two the `fs/*` routes were
#: measured to answer deliberately: 400 (`Path points to a directory`,
#: `Parent directory does not exist`) and 413 (`Content too large`, above
#: 8 MiB on write).
_TEXT_PASSTHROUGH = _PASSTHROUGH_CLIENT_ERRORS | frozenset({400, 413})

#: Hermes's read-text preview cap, measured (`_FS_TEXT_PREVIEW_MAX_BYTES`).
#: Quoted in the editable_reason so the app can show a real number.
_HERMES_READ_TEXT_CAP = "512 KiB"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TextWriteRequest(BaseModel):
    """Body of `PUT /sandbox/text`. Closed: an unknown key (including `text`,
    which is what Hermes itself 422s) is a 422 here, never silently dropped.

    `expected_sha256` is the hex digest the app computed over the text it
    was given by the last `GET` (or the last `PUT`'s response) -- the
    guard's whole premise is that the app hashes what it holds.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Absolute path inside the sandbox root.")
    content: str = Field(description="The full new text of the file.")
    expected_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase hex sha256 of the text the app last read.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _editability(
    truncated: bool, binary: bool, *, writable: bool = True
) -> tuple[bool, str | None]:
    """The editable verdict and its reason -- one place, so GET and PUT agree."""
    if not writable:
        return False, (
            "file is outside the sandbox root, so it opens read-only; "
            "RESEARCH_GATEWAY_VIEW_ROOTS widens reading only, never writing"
        )
    if binary:
        return False, (
            "file is binary: Hermes decoded it with replacement characters, "
            "so writing the text back would corrupt it"
        )
    if truncated:
        return False, (
            f"file is larger than Hermes's read-text cap ({_HERMES_READ_TEXT_CAP}); "
            "the text is cut and cannot be written back whole"
        )
    return True, None


def _read_text_shape_problems(body: Any) -> list[str]:
    """Deviations from the MEASURED read-text shape this module depends on.

    Only the keys the guard and the app rely on are required (`text`,
    `truncated`, `binary`); `byteSize`/`language`/`mimeType` are passed
    through when present and default when not, because nothing here breaks
    without them.
    """
    if not isinstance(body, dict):
        return [f"read-text body is {type(body).__name__}, not an object"]
    problems: list[str] = []
    for key, kind in (("text", str), ("truncated", bool), ("binary", bool)):
        if not isinstance(body.get(key), kind):
            problems.append(f"missing/mis-typed {key!r}")
    return problems


def _text_response(target: str, body: dict[str, Any], *, writable: bool = True) -> dict[str, Any]:
    """The GET shape, built from a shape-checked read-text body."""
    text: str = body["text"]
    truncated: bool = body["truncated"]
    binary: bool = body["binary"]
    editable, reason = _editability(truncated, binary, writable=writable)
    return {
        "path": target,
        "text": text,
        "sha256": _sha256_of_text(text),
        "byte_size": body.get("byteSize"),
        "language": body.get("language"),
        "mime_type": body.get("mimeType"),
        "truncated": truncated,
        "binary": binary,
        "editable": editable,
        "editable_reason": reason,
    }


async def _read_current(backend: SandboxFS, target: str) -> dict[str, Any]:
    """Read `target` through the sandbox backend and return the checked body.

    Raises the forwarded status (404 missing, 400 directory), 502 on transport
    failure or shape drift. `target` must already be validated.

    The shape check stays even on the direct backend. It was written for an
    undocumented upstream route, but it now also catches this gateway's own
    backend drifting from the shape the app parses -- the same failure, one
    layer closer.
    """
    try:
        response = await backend.fs_read_text(target)
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if response.status_code != 200:
        await _raise_for_upstream(response, target, passthrough=_TEXT_PASSTHROUGH)
    try:
        body = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Hermes read-text returned non-JSON for {target!r} -- the "
                "undocumented route may have changed shape"
            ),
        ) from exc
    problems = _read_text_shape_problems(body)
    if problems:
        # Loud, same reason as the browse health tripwire: undocumented
        # upstream route, drift must not fail quietly. No content is logged.
        logger.error(
            "Hermes read-text answered outside the measured shape for a "
            "sandbox path (%s) -- the editor routes may be broken",
            "; ".join(problems),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Hermes read-text answered outside the measured shape: {'; '.join(problems)}",
        )
    return body


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@sandbox_text_router.get("/sandbox/text")
async def read_sandbox_text(
    request: Request,
    path: str = Query(...),
) -> dict[str, Any]:
    """One sandbox file as text, with the hash the app must send back on save.

    `editable` is false -- with `editable_reason` saying why -- for a
    truncated or binary file, and for one outside the sandbox root that only
    `RESEARCH_GATEWAY_VIEW_ROOTS` let this route open; the app opens all
    three read-only. Upstream 404 (missing) and 400 (a directory) come back
    with Hermes's own words.
    """
    settings = get_settings()
    target = validate_view_path(
        path, settings.hermes_sandbox_root, parse_roots(settings.research_gateway_view_roots)
    )
    adapter: HermesAdapter = request.app.state.hermes_adapter
    body = await _read_current(sandbox_fs_for(request.app.state, adapter), target)
    return _text_response(
        target, body, writable=is_under_root(target, settings.hermes_sandbox_root)
    )


@sandbox_text_router.put("/sandbox/text")
async def write_sandbox_text(request: Request, payload: TextWriteRequest) -> Any:
    """Compare-then-write. Never writes when the compare fails.

    Order of checks, each one a stop:

    1. path validation (422 malformed / 403 outside the root) -- before
       anything reaches Hermes;
    2. read the current file: 404 if it does not exist (phase 1 does not
       create files), 400 if it is a directory;
    3. 409 `{"detail": "file is not editable", ...GET fields}` if the
       current file is truncated or binary -- the same verdict the GET gave;
    4. 409 `{"detail": "file changed since it was read", "current_sha256",
       "text"}` if the current text does not hash to `expected_sha256` --
       the app gets the current text so it can reload, overwrite (by
       re-sending with `current_sha256`), or diff;
    5. write (413 forwarded above 8 MiB), then re-read and return the GET
       shape with the fresh hash.

    The compare-then-write is not locked -- see the module docstring for
    what that means and why phase 1 accepts it.
    """
    settings = get_settings()
    target = validate_sandbox_path(payload.path, settings.hermes_sandbox_root)
    adapter: HermesAdapter = request.app.state.hermes_adapter
    backend = sandbox_fs_for(request.app.state, adapter)

    current = await _read_current(backend, target)
    current_view = _text_response(target, current)
    if not current_view["editable"]:
        return JSONResponse(
            status_code=409,
            content={"detail": "file is not editable", **current_view},
        )
    if current_view["sha256"] != payload.expected_sha256:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "file changed since it was read",
                "current_sha256": current_view["sha256"],
                "text": current_view["text"],
            },
        )

    # Step 4 again, and this time without a gap. The compare above ran before
    # an await, so a writer can still land between it and the write; a backend
    # that can re-compare inside the write itself closes that. The 409 it
    # returns is already in this route's conflict shape, so it forwards as-is.
    # On the HTTP backend there is no conditional write and the window stays
    # what the module docstring describes.
    condition = current_view["sha256"] if backend.supports_conditional_write else None
    try:
        if condition is None:
            response = await backend.fs_write_text(target, payload.content)
        else:
            response = await backend.fs_write_text(
                target, payload.content, if_match_sha256=condition
            )
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if response.status_code == 409:
        return JSONResponse(status_code=409, content=response.json())
    if response.status_code != 200:
        await _raise_for_upstream(response, target, passthrough=_TEXT_PASSTHROUGH)
    await response.aclose()

    # Re-read rather than trusting our own content: the hash the app saves
    # next time must be of what the backend actually holds (and `byte_size`,
    # `language`, `truncated` are its to report, not ours to guess).
    fresh = await _read_current(backend, target)
    return _text_response(target, fresh)
