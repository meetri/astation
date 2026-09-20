"""Editor text read/write (P7 phase 1): hash-guarded `GET`/`PUT /sandbox/text`."""

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

_TEXT_PASSTHROUGH = _PASSTHROUGH_CLIENT_ERRORS | frozenset({400, 413})

_HERMES_READ_TEXT_CAP = "512 KiB"


class TextWriteRequest(BaseModel):
    """Body of `PUT /sandbox/text`. Closed: an unknown key (including `text`,
    which is what Hermes itself 422s) is a 422 here, never silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Absolute path inside the sandbox root.")
    content: str = Field(description="The full new text of the file.")
    expected_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase hex sha256 of the text the app last read.",
    )


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
    """Deviations from the MEASURED read-text shape this module depends on."""
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
    """Read `target` through the sandbox backend and return the checked body."""
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


@sandbox_text_router.get("/sandbox/text")
async def read_sandbox_text(
    request: Request,
    path: str = Query(...),
) -> dict[str, Any]:
    """One sandbox file as text, with the hash the app must send back on save."""
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
    """Compare-then-write. Never writes when the compare fails."""
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

    fresh = await _read_current(backend, target)
    return _text_response(target, fresh)
