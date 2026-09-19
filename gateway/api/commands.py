"""Slash-command routes (P2-4): catalog (cached) + resolve + dispatch.

Shaped by the P2-0a verdict (PV "Phase 2a probes", measured live 2026-08-30):
**there is no remote execution surface for slash commands.** `command.resolve`
knows only the 178 core commands and returns metadata; `command.dispatch`
knows only the quick/plugin/bundle/skill class and is an *expansion fetch* --
it returns the full skill prompt for the client to submit as an ordinary
turn, and never starts anything itself. The two classes are disjoint. So:

* **Core commands** (`/help`, `/status`, `/model`, ...) get catalog +
  resolve only. They are TUI verbs; Hermes offers no way to run one
  remotely, and this gateway does not pretend otherwise.
* **Skill commands** get dispatch: the returned `message` is the prompt the
  app then sends via the existing `POST /turns`. The dispatch route is
  therefore live (not the 501 the task reserved for "dispatch proved
  unusable") -- it is usable, as exactly what it is: a lookup.

The catalog is one ~34 KB result that changes only when skills/plugins are
installed, so it is cached here with a short TTL (5 min). The cached body is
Hermes's result **verbatim** -- in particular `categories` (the app sections
its command sheet by them) and the per-skill `usage` counts (the app sorts
skills by them, P2-9) pass through untouched. If Hermes is unreachable when
the TTL has lapsed, the stale copy is served (marked `stale: true`) rather
than failing the sheet: a command list from five minutes ago beats none.

Mounted on the authenticated `/api` router like everything else.
"""

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

#: Catalog cache TTL. The catalog moves when a skill/plugin is installed --
#: an hours-to-weeks timescale -- so five minutes trades staleness nobody
#: will notice for not paying 34 KB per composer keystroke session.
CATALOG_TTL_S = 300.0

#: `command.resolve`'s "unknown command" / `command.dispatch`'s "not
#: dispatchable" RPC error codes, measured live (PV "Phase 2 probe" /
#: "Phase 2a probes").
_UNKNOWN_COMMAND_CODE = 4011
_NOT_DISPATCHABLE_CODE = 4018

# Injectable monotonic clock (tests age the cache without sleeping).
_now = time.monotonic


class CommandName(BaseModel):
    """Body for resolve/dispatch: `name` is the only field Hermes reads.

    Closed schema like every other body in this service -- `command.resolve`
    silently parses `None` from any other field name (measured: the `[4011]`
    error echoes `None`), so accepting extra fields here would turn a client
    typo into a confusing upstream error instead of a 422.
    """

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
    """The command catalog, verbatim, cached ~5 min.

    Response: `{"catalog": <Hermes's commands.catalog result, untouched>,
    "cached": bool, "age_seconds": float, "stale": bool}`. `catalog` keeps
    all seven measured keys -- notably `categories` and `skills` (with each
    skill's `usage`), which the app surfaces depend on. `stale` is only ever
    true when the TTL lapsed AND the refetch failed; the copy served is then
    the last good one, with its real age. `?refresh=true` bypasses the TTL
    (but still serves stale on a failed fetch rather than 502ing while a
    copy exists).
    """
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
    """Resolve one CORE command name -- Hermes's answer, verbatim.

    Success: `{"canonical", "description", "category"}` (canonical comes
    back without the slash; aliases canonicalize; no prefix matching --
    measured). 404 for `[4011] unknown command` -- which includes every
    *skill* name, since resolve and dispatch cover disjoint classes; a
    client resolving a skill should use the catalog's `skills` dict and
    dispatch instead.
    """
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
    """Fetch a SKILL command's expansion. **Never executes anything** (P2-0a).

    Success: `{"dispatched": {"type": "skill", "name", "display",
    "message"}}` verbatim -- `message` is the full skill prompt, and it is
    the CLIENT's job to submit it as an ordinary turn
    (`POST /api/sessions/{id}/turns`); no agent turn starts here and nothing
    is written to any session (measured).

    400 for `[4018]`: the name is either a core command (Hermes cannot
    dispatch those at all -- there is no remote execution surface) or does
    not exist; the wire error genuinely does not distinguish the two, and
    neither does this route, honestly.
    """
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
