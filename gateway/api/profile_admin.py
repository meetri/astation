"""Agent (profile) model management: detail, catalog, model write, lifecycle (P6).

The operator's ask, verbatim: *"I need a way to
manage each profile. I want to be able to select which model is used for each
profile if it's local, openrouter, etc... which models I can choose from, how
much it costs, latency, etc."* Six routes, the §8 contract exactly:

    GET    /api/profiles/{name}            one agent: row + model facts + measured stats
    GET    /api/models/catalog             every configured provider's models, with facts
    PUT    /api/profiles/{name}/model      {provider, model} -> profiles.configure -> read back
    POST   /api/profiles                   create (201) -> profiles.create -> reconcile
    PATCH  /api/profiles/{name}            rename (cli.exec) and/or description (profiles.configure)
    DELETE /api/profiles/{name}?confirm=   delete (204) -> cli.exec ["profile","delete","-y",name]

`GET /api/profiles` (`api/instance.py`) is untouched: still `profiles.list`
verbatim plus `connected`.

**The unit of model choice is the profile, not the session.** Measured (PV
"Profiles"): `session.create` ignores `model`, and `/model` is a core TUI
command `command.dispatch` refuses. `profiles.configure` is the one write that
exists and it is verified live 2026-09-06 (§7). A model written here applies
to the profile's **next new session**; an open session keeps the model it was
spawned with. The app says so beside the picker.

**All Hermes calls here go over the default connection.** §7 measured that
`profiles.configure` works from the default connection -- the write is by
profile directory, not by which dashboard process took it -- and `cli.exec`
has always run on the default connection. So every call is
`_with_reconnect(request.app.state, request.app.state.hermes_adapter, ...)`,
and nothing here needs `resolve_profile_adapter`.

**A user-defined provider is mirrored into the profile's own config.** After a
successful `profiles.create` / `profiles.configure` naming a provider that
`model.options` marks `is_user_defined` (a `providers.<slug>` entry in the
default profile's config.yaml, e.g. the operator's second local llama.cpp box),
the route runs `hermes -p <profile> config set providers.<slug>.{name,base_url,
discover_models}` through `cli.exec`. Measured 2026-09-06: without it the write
lands but the agent's first turn answers `Unknown provider '<slug>'`, because a
profile's process loads its own config.yaml. `docs/AGENT_MODEL_DESIGN.md` §9.

**Validation before any write.** `PUT .../model` and `POST /profiles` check
`provider` against `model.options` and `model` against that provider's list,
and a miss is a 422 naming the provider and how many models it lists -- a
typo'd model id that reached `profiles.configure` would be written verbatim
into the profile's `config.yaml`, and the first sign would be the next
session failing to start. For OpenRouter the check also takes any id on
OpenRouter's public list (the app lets the operator paste one, 2026-09-10); a
public list that cannot be read is a 422 saying so, never a pass.

**Names that become process arguments are gated by a regex.** Rename and
delete have no RPC (§7) and go through `cli.exec` as
`["profile","rename",old,new]` / `["profile","delete","-y",name]`, where every
element is a real argv token parsed by the CLI's own argparse (see
`api/instance.py`'s `_ARGV_SAFE_STORED_ID` for the full reasoning). A name
matching `PROFILE_NAME_PATTERN` (`^[a-z0-9][a-z0-9-]{0,39}$`) cannot start
with `-`, so it cannot be read as an option. The same rule applies on create,
so a profile this gateway makes can always be renamed and deleted by it.
`cli.exec` answers 200 whatever the CLI did; only `blocked: false` and
`code == 0` is success, and anything else is a **502 carrying the CLI's own
output** -- the `delete_session` precedent.

**Lifecycle changes reconcile the connection manager right away.** A created
profile gets its own dashboard connection (`ProfileConnectionManager`,
`domain/profile_connection.py`) at the next reconciliation tick -- five
minutes by default -- so after create / rename / delete the route asks for a
pass now. Best-effort: a reconcile failure is logged and never fails the
route, because the Hermes-side change has already happened and reporting it
as failed would be a lie.

**Two things never fail these routes.** `model_facts` and `stats` are
decoration on the profile row: an OpenRouter fetch that fails yields
`model_facts: null` (or facts from Hermes alone), a database problem yields
`stats: null`, and the profile is still shown.

**A profile's model need not be in Hermes's curated list.** `model.options`
carries 44 OpenRouter ids; the operator's `kimi25` runs on `moonshotai/kimi-k2.5`,
which is not among them and works fine. So the detail
route's facts and spend fall back to the fetched OpenRouter index
(`ModelCatalogCache.cached_openrouter()`) for the OpenRouter provider; the
index itself never enters a payload.

Mounted on the authenticated `/api` router in `api.main`, so Basic auth is
inherited by construction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError
from api.projects import workspace_db
from domain.hermes_runtime import _with_reconnect
from domain.model_catalog import (
    ModelCatalogCache,
    build_catalog,
    find_provider,
    is_openrouter,
    model_facts_for,
    price_lookup_from_catalog,
    selectable_model_ids,
)
from domain.profile_stats import DEFAULT_WINDOW_DAYS, compute_profile_stats

logger = logging.getLogger(__name__)

profile_admin_router = APIRouter(tags=["profile-admin"])

#: The §8 slug rule. Lower-case, digits and hyphens, no leading hyphen, at
#: most 40 characters. **Also the argv gate** (module docstring): a value
#: that passes cannot be read as a CLI option.
PROFILE_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,39}$"
_PROFILE_NAME = re.compile(PROFILE_NAME_PATTERN)

#: The profile that cannot be deleted and whose rename is display-only (§8).
DEFAULT_PROFILE_NAME = "default"

#: The `hermes` CLI subcommands the two RPC-less lifecycle operations use
#: (§7: "Rename/delete: no RPC"). Constants so the tests asserting the exact
#: argv assert against the literal the route sends.
HERMES_PROFILE_RENAME_ARGV: tuple[str, ...] = ("profile", "rename")
HERMES_PROFILE_DELETE_ARGV: tuple[str, ...] = ("profile", "delete", "-y")

#: Description cap. Hermes's own descriptions are a sentence or two; the cap
#: keeps a runaway client from writing an essay into `config.yaml`.
MAX_DESCRIPTION_CHARS = 500

#: `hermes -p <profile> config set providers.<slug>.<key> <value>` -- how a
#: user-defined provider is copied into a profile's own config.yaml (measured
#: 2026-09-06: a profile whose config lacks the `providers:` entry fails its
#: first turn with `Unknown provider 'local-3080ti'`, because each profile
#: process loads ITS config, not the default profile's). The keys mirrored are
#: the ones `model.options` exposes; a keyed endpoint's `key_env` is not among
#: them and stays a hand edit.
HERMES_CONFIG_SET_ARGV: tuple[str, ...] = ("config", "set")
_PROVIDER_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_MIRRORED_PROVIDER_KEYS: tuple[str, ...] = ("name", "base_url", "discover_models")


# ---------------------------------------------------------------------------
# Bodies -- closed schemas, every one
# ---------------------------------------------------------------------------


class ModelSelection(BaseModel):
    """Body for `PUT /api/profiles/{name}/model`. `{provider, model}` and nothing else.

    `base_url` is deliberately not a field (§8: "not accepted in this
    round") -- it is a different, unprobed Hermes write (`model.base_url`),
    and `extra="forbid"` makes sending it a 422 rather than a silent drop.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)


class ProfileCreate(BaseModel):
    """Body for `POST /api/profiles`.

    `provider` and `model` come together or not at all: `profiles.create`
    pins a model only when both are given (§7), and one without the other
    would either be ignored or pair a provider with the launch profile's
    model id, neither of which the caller asked for.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=PROFILE_NAME_PATTERN)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_CHARS)
    provider: str | None = Field(default=None, min_length=1, max_length=100)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    clone_from: str | None = Field(default=None, pattern=PROFILE_NAME_PATTERN)

    @model_validator(mode="after")
    def _provider_and_model_together(self) -> ProfileCreate:
        if (self.provider is None) != (self.model is None):
            raise ValueError("provider and model must be given together, or neither")
        return self


class ProfileUpdate(BaseModel):
    """Body for `PATCH /api/profiles/{name}`: `name` and/or `description`, at least one."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, pattern=PROFILE_NAME_PATTERN)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_CHARS)

    @model_validator(mode="after")
    def _at_least_one(self) -> ProfileUpdate:
        if self.name is None and self.description is None:
            raise ValueError("nothing to change: give name and/or description")
        return self


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _validate_profile_name(name: str) -> str:
    """The regex gate for a name that will become a process argument (422 otherwise)."""
    if not _PROFILE_NAME.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail=(
                f"profile name {name!r} is not usable here: it must match "
                f"{PROFILE_NAME_PATTERN} (lower-case letters, digits and hyphens, "
                "no leading hyphen, at most 40 characters). This name becomes an "
                "argument to the `hermes` CLI, so the shape is enforced rather "
                "than passed through."
            ),
        )
    return name


async def _hermes(request: Request, operation: Any) -> Any:
    """`_with_reconnect` on the default adapter, with any Hermes failure a 502."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        return await _with_reconnect(request.app.state, adapter, operation)
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _profile_rows(request: Request) -> list[dict[str, Any]]:
    adapter: HermesAdapter = request.app.state.hermes_adapter
    result = await _hermes(request, adapter.profiles_list)
    rows = result.get("profiles") if isinstance(result, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _find_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("name") == name), None)


def _require_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    row = _find_row(rows, name)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Hermes has no profile named {name!r} (GET /api/profiles lists them)",
        )
    return row


def _catalog_cache(request: Request) -> ModelCatalogCache:
    cache = getattr(request.app.state, "model_catalog_cache", None)
    if cache is None:
        cache = ModelCatalogCache()
        request.app.state.model_catalog_cache = cache
    return cache


async def _model_options(request: Request, *, refresh: bool = False) -> dict[str, Any]:
    """`model.options`; with `refresh` Hermes probes EVERY custom provider and
    re-fills its own model cache (a normal call probes only the current one,
    so a newly added local provider lists no ids until someone refreshes)."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    call = (lambda: adapter.model_options(refresh=True)) if refresh else adapter.model_options
    result = await _hermes(request, call)
    return result if isinstance(result, dict) else {}


async def _catalog(
    request: Request, *, refresh: bool = False, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    if options is None:
        options = await _model_options(request, refresh=refresh)
    return await build_catalog(options, cache=_catalog_cache(request), refresh=refresh)


async def _catalog_best_effort(request: Request) -> dict[str, Any] | None:
    """The catalog, or None -- decoration for the detail route, never its failure."""
    try:
        return await _catalog(request)
    except HTTPException as exc:
        logger.info("model catalog unavailable for profile detail: %s", exc.detail)
        return None
    except Exception:
        logger.warning("model catalog failed for profile detail", exc_info=True)
        return None


async def _validate_selection(
    request: Request, options: dict[str, Any], provider: str, model: str
) -> None:
    """422 unless `provider` is configured and lists `model` (§8).

    "Lists" is `selectable_model_ids`: Hermes's own list, or for a local
    provider Hermes has not probed yet, what the endpoint's `/models`
    answers -- the same fallback `GET /api/models/catalog` shows, so the
    picker and this check never disagree. For OpenRouter it also takes any
    id on OpenRouter's public list: Hermes curates a few dozen, the app
    lets the operator paste the rest, and a profile on an
    uncurated id runs fine (`kimi25`, measured 2026-09-06).
    """
    ids = await selectable_model_ids(options, provider, cache=_catalog_cache(request))
    if ids is None:
        known = [p.get("slug") for p in options.get("providers", []) if isinstance(p, dict)]
        raise HTTPException(
            status_code=422,
            detail=(
                f"provider {provider!r} is not configured on this Hermes instance "
                f"(model.options lists {len(known)} provider(s): {', '.join(str(k) for k in known)})"
            ),
        )
    if model in ids:
        return
    if is_openrouter(options, provider):
        public = await _catalog_cache(request).openrouter()
        if public is not None and model in public:
            return
        why = (
            "OpenRouter's public model list could not be read to check it; try again in a minute"
            if public is None
            else f"it is not among the {len(public)} on OpenRouter's public list either; "
            "check the id at openrouter.ai/models"
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {model!r} is not one of the {len(ids)} model(s) provider "
                f"{provider!r} lists (Hermes's curated list), and {why}"
            ),
        )
    raise HTTPException(
        status_code=422,
        detail=(
            f"model {model!r} is not one of the {len(ids)} model(s) provider "
            f"{provider!r} lists (model.options, or the endpoint's own /models for a "
            f"local provider); GET /api/models/catalog shows them"
        ),
    )


def _connected_by_name(request: Request) -> dict[str, bool | None]:
    manager = getattr(request.app.state, "profile_connection_manager", None)
    if manager is None:
        return {}
    try:
        return {row["name"]: row["connected"] for row in manager.list_profiles()}
    except Exception:
        logger.warning("profile connection manager could not list connections", exc_info=True)
        return {}


async def _reconcile_connections(request: Request, why: str) -> None:
    """Ask the manager for a pass now. Logged, never raised (module docstring)."""
    manager = getattr(request.app.state, "profile_connection_manager", None)
    if manager is None:
        return
    try:
        await manager.reconcile()
    except Exception:
        logger.warning("profile connection reconcile after %s failed", why, exc_info=True)


def _cli_succeeded(result: Any) -> tuple[bool, Any, str]:
    """`(succeeded, code, said)` from a `cli.exec` result, the `delete_session` rules."""
    payload = result if isinstance(result, dict) else {}
    code = payload.get("code")
    output = payload.get("output")
    output_text = output if isinstance(output, str) else ""
    blocked = payload.get("blocked") is True
    succeeded = not blocked and isinstance(code, int) and not isinstance(code, bool) and code == 0
    hint = payload.get("hint")
    said = output_text or (str(hint) if hint else "") or "no output"
    return succeeded, code, said


async def _mirror_provider_into_profile(
    request: Request, options: dict[str, Any], name: str, provider: str
) -> None:
    """Copy a user-defined provider's definition into `name`'s own config.yaml.

    Built-in providers (`openrouter`, `anthropic`, ...) need nothing. A
    user-defined one (`is_user_defined` in `model.options`, with an `api_url`)
    exists only in whichever config.yaml declares it -- the default profile's,
    where the operator added it -- and a profile created or re-pointed from the
    app has its own config.yaml, loaded by its own process. Without this the
    model write lands and the agent's first turn fails with `Unknown provider`.
    Idempotent: `config set` on an existing key rewrites the same value.

    A CLI failure is a 502 with the CLI's output: the model is already
    written by then, and reporting a 201/200 for an agent that cannot run
    would be a lie; the row still exists and the next list shows it.
    """
    entry = find_provider(options, provider)
    if not isinstance(entry, dict) or entry.get("is_user_defined") is not True:
        return
    api_url = entry.get("api_url")
    if not isinstance(api_url, str) or not api_url:
        return
    if not _PROVIDER_SLUG.fullmatch(provider):
        logger.warning(
            "not mirroring provider %r into profile %r: slug is not argv-safe", provider, name
        )
        return
    values = {"name": provider, "base_url": api_url, "discover_models": "true"}
    for key in _MIRRORED_PROVIDER_KEYS:
        dotted = f"providers.{provider}.{key}"
        try:
            await _run_cli(
                request,
                ["-p", name, *HERMES_CONFIG_SET_ARGV, dotted, values[key]],
                f"-p {name} config set {dotted}",
            )
        except HTTPException as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"model written for {name!r}, but its config.yaml could not receive "
                    f"provider {provider!r} ({dotted}), so its next session would fail with "
                    f"`Unknown provider`: {exc.detail}"
                ),
            ) from exc
    logger.info("mirrored provider %r (%s) into profile %r", provider, api_url, name)


async def _run_cli(request: Request, argv: list[str], what: str) -> str:
    """`cli.exec argv`; the CLI's output on success, a 502 carrying it otherwise."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    logger.warning("issuing `hermes %s` (argv=%r)", " ".join(argv), argv)
    result = await _hermes(request, lambda: adapter.cli_exec(argv))
    succeeded, code, said = _cli_succeeded(result)
    if not succeeded:
        blocked = result.get("blocked") if isinstance(result, dict) else None
        raise HTTPException(
            status_code=502,
            detail=f"`hermes {what}` did not succeed (blocked={blocked!r}, code={code!r}): {said}",
        )
    return said if said != "no output" else ""


def _profile_detail(
    request: Request,
    db: OrmSession,
    row: dict[str, Any],
    catalog: dict[str, Any] | None,
) -> dict[str, Any]:
    """The §8 `GET /profiles/{name}` body from a `profiles.list` row."""
    name = row.get("name")
    provider = row.get("provider") if isinstance(row.get("provider"), str) else None
    model_id = row.get("model") if isinstance(row.get("model"), str) else None
    provider_entry = find_provider(catalog, provider) if (catalog and provider) else None
    api_url = provider_entry.get("api_url") if provider_entry else None
    # The private OpenRouter index: a profile may run on an OpenRouter model
    # Hermes's curated list omits (`kimi25`, measured 2026-09-06); the public
    # list still prices and describes it. Never part of any payload.
    openrouter_index = _catalog_cache(request).cached_openrouter()
    facts = (
        model_facts_for(catalog, provider, model_id, openrouter=openrouter_index)
        if catalog
        else None
    )

    stats: dict[str, Any] | None
    try:
        stats = compute_profile_stats(
            db,
            str(name),
            DEFAULT_WINDOW_DAYS,
            price_lookup=price_lookup_from_catalog(catalog, openrouter=openrouter_index),
        )
    except SQLAlchemyError:
        logger.warning("profile stats unavailable for %r", name, exc_info=True)
        stats = None

    return {
        "name": name,
        "display_name": row.get("display_name") or "",
        "description": row.get("description") or "",
        "is_default": row.get("is_default") is True,
        "connected": _connected_by_name(request).get(name),
        "skill_count": row.get("skill_count"),
        "path": row.get("path"),
        "last_session": row.get("last_session"),
        "model": {"provider": provider, "id": model_id, "api_url": api_url},
        "model_facts": facts,
        "stats": stats,
    }


async def _detail_for(request: Request, db: OrmSession, name: str) -> dict[str, Any]:
    rows = await _profile_rows(request)
    row = _require_row(rows, name)
    catalog = await _catalog_best_effort(request)
    return _profile_detail(request, db, row, catalog)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@profile_admin_router.get("/models/catalog")
async def get_model_catalog(
    request: Request,
    refresh: bool = Query(default=False),
) -> dict[str, Any]:
    """Every configured provider's models, with facts (§8 shape; `domain/model_catalog.py`).

    `model.options` is fetched live every time; the OpenRouter list and any
    local `/models` probe are cached an hour and `?refresh=1` re-fetches
    them. 502 only when `model.options` itself fails -- an external source
    that fails yields nulls, not an error.
    """
    return await _catalog(request, refresh=refresh)


@profile_admin_router.get("/profiles/{name}")
async def get_profile(
    name: str,
    request: Request,
    db: OrmSession = Depends(workspace_db),
) -> dict[str, Any]:
    """One agent: its `profiles.list` row, the catalog's facts about its model,
    and the gateway's measured stats for it (§8). 404 for an unknown profile;
    502 for a `profiles.list` failure; facts and stats never fail it."""
    return await _detail_for(request, db, name)


@profile_admin_router.put("/profiles/{name}/model")
async def set_profile_model(
    name: str,
    body: ModelSelection,
    request: Request,
    db: OrmSession = Depends(workspace_db),
) -> dict[str, Any]:
    """`{provider, model}` -> `profiles.configure {name, provider, model}` -> read back (§8).

    Order: the profile must exist (404), the selection must be in
    `model.options` (422), then exactly `{name, provider, model}` goes to
    `profiles.configure`. Hermes's `applied.model` is checked -- `ok` alone
    is not trusted -- and a write Hermes reports as not applied is a 502.
    The response is `GET /profiles/{name}`'s shape, re-read from
    `profiles.list` so the app shows what Hermes now holds, not what was sent.
    """
    rows = await _profile_rows(request)
    _require_row(rows, name)
    options = await _model_options(request)
    await _validate_selection(request, options, body.provider, body.model)

    adapter: HermesAdapter = request.app.state.hermes_adapter
    result = await _hermes(
        request, lambda: adapter.profiles_configure(name, provider=body.provider, model=body.model)
    )
    applied = result.get("applied") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("ok") is False
        or (isinstance(applied, dict) and applied.get("model") is False)
    ):
        raise HTTPException(
            status_code=502,
            detail=f"profiles.configure did not apply the model for {name!r}: {result!r}",
        )
    logger.info("profile %r model set to %s/%s", name, body.provider, body.model)
    await _mirror_provider_into_profile(request, options, name, body.provider)

    rows = await _profile_rows(request)
    row = _require_row(rows, name)
    catalog = await _catalog_best_effort(request)
    return _profile_detail(request, db, row, catalog)


@profile_admin_router.post("/profiles", status_code=201)
async def create_profile(
    body: ProfileCreate,
    request: Request,
    db: OrmSession = Depends(workspace_db),
) -> dict[str, Any]:
    """`profiles.create`, then reconcile connections, then the new row (§8, 201).

    409 if the name exists. `provider`+`model` are validated against
    `model.options` when given; when absent the new profile inherits the
    launch profile's model (Hermes's own behaviour, §7). Only the keys the
    caller sent are forwarded.
    """
    rows = await _profile_rows(request)
    if _find_row(rows, body.name) is not None:
        raise HTTPException(status_code=409, detail=f"a profile named {body.name!r} already exists")
    if body.clone_from is not None:
        _require_row(rows, body.clone_from)
    options: dict[str, Any] = {}
    if body.provider is not None and body.model is not None:
        options = await _model_options(request)
        await _validate_selection(request, options, body.provider, body.model)

    params: dict[str, Any] = {"name": body.name}
    if body.description is not None:
        params["description"] = body.description
    if body.provider is not None:
        params["provider"] = body.provider
        params["model"] = body.model
    if body.clone_from is not None:
        params["clone_from"] = body.clone_from

    adapter: HermesAdapter = request.app.state.hermes_adapter
    result = await _hermes(request, lambda: adapter.profiles_create(**params))
    if not isinstance(result, dict) or result.get("ok") is False:
        raise HTTPException(status_code=502, detail=f"profiles.create did not succeed: {result!r}")
    logger.warning("CREATED Hermes profile %r (model_set=%r)", body.name, result.get("model_set"))
    if body.provider is not None:
        await _mirror_provider_into_profile(request, options, body.name, body.provider)

    await _reconcile_connections(request, f"creating profile {body.name!r}")
    return await _detail_for(request, db, body.name)


@profile_admin_router.patch("/profiles/{name}")
async def update_profile(
    name: str,
    body: ProfileUpdate,
    request: Request,
    db: OrmSession = Depends(workspace_db),
) -> dict[str, Any]:
    """Rename via `cli.exec ["profile","rename",old,new]` and/or description via
    `profiles.configure {name, description}` (§8).

    Description is written first, under the old name, so a rename that then
    fails leaves a profile that still exists under a name the caller knows.
    Renaming `default` is a display-name change on Hermes's side: the CLI is
    still called, the profile keeps the id `default`, the row is re-read by
    `default`, and the response says `renamed_display_only: true`. Any other
    rename reconciles connections and returns the row under its new name.
    """
    _validate_profile_name(name)
    rows = await _profile_rows(request)
    _require_row(rows, name)
    adapter: HermesAdapter = request.app.state.hermes_adapter

    if body.description is not None:
        result = await _hermes(
            request, lambda: adapter.profiles_configure(name, description=body.description)
        )
        applied = result.get("applied") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("ok") is False
            or (isinstance(applied, dict) and applied.get("description") is False)
        ):
            raise HTTPException(
                status_code=502,
                detail=f"profiles.configure did not apply the description for {name!r}: {result!r}",
            )

    renamed_display_only = False
    final_name = name
    if body.name is not None and body.name != name:
        if _find_row(rows, body.name) is not None:
            raise HTTPException(
                status_code=409, detail=f"a profile named {body.name!r} already exists"
            )
        argv = [*HERMES_PROFILE_RENAME_ARGV, name, body.name]
        await _run_cli(request, argv, f"profile rename {name} {body.name}")
        if name == DEFAULT_PROFILE_NAME:
            renamed_display_only = True  # `default` keeps its id; only its display name moved
        else:
            final_name = body.name
        logger.warning(
            "RENAMED Hermes profile %r -> %r (display_only=%r)",
            name,
            body.name,
            renamed_display_only,
        )
        await _reconcile_connections(request, f"renaming profile {name!r}")

    detail = await _detail_for(request, db, final_name)
    detail["renamed_display_only"] = renamed_display_only
    return detail


@profile_admin_router.delete("/profiles/{name}", status_code=204)
async def delete_profile(
    name: str,
    request: Request,
    confirm: str | None = Query(default=None),
) -> Response:
    """`cli.exec ["profile","delete","-y",name]` -> reconcile -> 204 (§8).

    Refuses `default` (409) and refuses without `?confirm=<name>` (422) --
    the confirmation is the name itself, typed by the caller, so a mis-tap
    on a list row cannot delete anything. 404 for an unknown profile. A
    non-zero CLI `code` is a 502 carrying the CLI's output. The profile's own
    sessions die with its directory; the app says so before asking.
    """
    _validate_profile_name(name)
    if name == DEFAULT_PROFILE_NAME:
        raise HTTPException(status_code=409, detail="the default profile cannot be deleted")
    if confirm != name:
        raise HTTPException(
            status_code=422,
            detail=f"deleting profile {name!r} requires ?confirm={name} (got {confirm!r})",
        )
    rows = await _profile_rows(request)
    row = _require_row(rows, name)
    if row.get("is_default") is True:
        raise HTTPException(status_code=409, detail="the default profile cannot be deleted")

    argv = [*HERMES_PROFILE_DELETE_ARGV, name]
    await _run_cli(request, argv, f"profile delete -y {name}")
    logger.warning("DELETED Hermes profile %r and its sessions", name)
    await _reconcile_connections(request, f"deleting profile {name!r}")
    return Response(status_code=204)


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "HERMES_CONFIG_SET_ARGV",
    "HERMES_PROFILE_DELETE_ARGV",
    "HERMES_PROFILE_RENAME_ARGV",
    "MAX_DESCRIPTION_CHARS",
    "PROFILE_NAME_PATTERN",
    "ModelSelection",
    "ProfileCreate",
    "ProfileUpdate",
    "create_profile",
    "delete_profile",
    "get_model_catalog",
    "get_profile",
    "profile_admin_router",
    "set_profile_model",
    "update_profile",
]
