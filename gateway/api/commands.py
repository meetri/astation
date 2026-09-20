"""Slash-command routes (P2-4): catalog (cached) + resolve + dispatch."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from adapters.hermes import HermesAdapter, HermesError, HermesRPCError
from domain.hermes_runtime import _rpc_error_code, _with_reconnect

logger = logging.getLogger(__name__)

commands_router = APIRouter(tags=["commands"])

CATALOG_TTL_S = 300.0

_UNKNOWN_COMMAND_CODE = 4011
_NOT_DISPATCHABLE_CODE = 4018

_now = time.monotonic


class CommandName(BaseModel):
    """Body for resolve/dispatch: `name` is the only field Hermes reads."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _reject_blank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must contain non-whitespace characters")
        return value


def _http_502(exc: HermesError) -> HTTPException:
    return HTTPException(status_code=502, detail=str(exc))


@commands_router.get("/commands")
async def get_catalog(request: Request, refresh: bool = False) -> dict:
    """The command catalog, verbatim, cached ~5 min."""
    state = request.app.state
    cached: tuple[float, dict[str, Any]] | None = getattr(state, "commands_catalog_cache", None)
    now = _now()

    if cached is not None and not refresh and now - cached[0] < CATALOG_TTL_S:
        return {
            "catalog": cached[1],
            "cached": True,
            "age_seconds": now - cached[0],
            "stale": False,
        }

    adapter: HermesAdapter = state.hermes_adapter
    try:
        result = await _with_reconnect(state, adapter, adapter.commands_catalog)
    except HermesError as exc:
        if cached is not None:
            logger.warning(
                "commands.catalog refetch failed; serving the stale cached copy (age %.0fs): %s",
                now - cached[0],
                exc,
            )
            return {
                "catalog": cached[1],
                "cached": True,
                "age_seconds": now - cached[0],
                "stale": True,
            }
        raise _http_502(exc) from exc

    state.commands_catalog_cache = (now, result)
    return {"catalog": result, "cached": False, "age_seconds": 0.0, "stale": False}


@commands_router.post("/commands/resolve")
async def resolve_command(body: CommandName, request: Request) -> dict:
    """Resolve one CORE command name -- Hermes's answer, verbatim."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: adapter.command_resolve(body.name),
        )
    except HermesRPCError as exc:
        if _rpc_error_code(exc) == _UNKNOWN_COMMAND_CODE:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown core command {body.name!r}. command.resolve knows "
                    "only the core catalog (no prefix matching); skill commands "
                    "are resolved via the catalog's `skills` dict and "
                    "POST /api/commands/dispatch instead."
                ),
            ) from exc
        raise _http_502(exc) from exc
    except HermesError as exc:
        raise _http_502(exc) from exc
    return {"name": body.name, "resolved": result}


@commands_router.post("/commands/dispatch")
async def dispatch_command(body: CommandName, request: Request) -> dict:
    """Fetch a SKILL command's expansion. **Never executes anything** (P2-0a)."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: adapter.command_dispatch(body.name),
        )
    except HermesRPCError as exc:
        if _rpc_error_code(exc) == _NOT_DISPATCHABLE_CODE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{body.name!r} is not dispatchable. Hermes dispatches only "
                    "quick/plugin/bundle/skill commands, and its [4018] error "
                    "does not distinguish 'this is a core command' (which has "
                    "no remote execution surface -- use the catalog + resolve "
                    "for metadata) from 'no such command'. If this is a skill, "
                    "check its name against the catalog's `skills` dict."
                ),
            ) from exc
        raise _http_502(exc) from exc
    except HermesError as exc:
        raise _http_502(exc) from exc
    return {"name": body.name, "dispatched": result}
