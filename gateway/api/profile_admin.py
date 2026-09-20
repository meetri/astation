"""Agent (profile) model management: detail, catalog, model write, lifecycle (P6)."""

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

# Also the argv gate: a leading '-' would be read as a CLI option, not a name.
PROFILE_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,39}$"
_PROFILE_NAME = re.compile(PROFILE_NAME_PATTERN)

DEFAULT_PROFILE_NAME = "default"

HERMES_PROFILE_RENAME_ARGV: tuple[str, ...] = ("profile", "rename")
HERMES_PROFILE_DELETE_ARGV: tuple[str, ...] = ("profile", "delete", "-y")

MAX_DESCRIPTION_CHARS = 500

HERMES_CONFIG_SET_ARGV: tuple[str, ...] = ("config", "set")
_PROVIDER_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
# key_env is deliberately absent: a keyed endpoint stays a hand edit.
_MIRRORED_PROVIDER_KEYS: tuple[str, ...] = ("name", "base_url", "discover_models")


class ModelSelection(BaseModel):
    """Body for `PUT /api/profiles/{name}/model`. `{provider, model}` and nothing else."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)


class ProfileCreate(BaseModel):
    """Body for `POST /api/profiles`."""

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
    """422 unless `provider` is configured and lists `model` (§8)."""
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


# Without this the model write lands but the agent's first turn fails: unknown provider.
async def _mirror_provider_into_profile(
    request: Request, options: dict[str, Any], name: str, provider: str
) -> None:
    """Copy a user-defined provider's definition into `name`'s own config.yaml."""
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


@profile_admin_router.get("/models/catalog")
async def get_model_catalog(
    request: Request,
    refresh: bool = Query(default=False),
) -> dict[str, Any]:
    """Every configured provider's models, with facts (§8 shape; `domain/model_catalog.py`)."""
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
    """`{provider, model}` -> `profiles.configure {name, provider, model}` -> read back (§8)."""
    rows = await _profile_rows(request)
    _require_row(rows, name)
    options = await _model_options(request)
    await _validate_selection(request, options, body.provider, body.model)

    adapter: HermesAdapter = request.app.state.hermes_adapter
    result = await _hermes(
        request, lambda: adapter.profiles_configure(name, provider=body.provider, model=body.model)
    )
    # ok alone is not proof; Hermes reports the per-field outcome under applied.
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
    """`profiles.create`, then reconcile connections, then the new row (§8, 201)."""
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
    """
    _validate_profile_name(name)
    rows = await _profile_rows(request)
    _require_row(rows, name)
    adapter: HermesAdapter = request.app.state.hermes_adapter

    # Description first, under the old name, so a failed rename leaves a known name.
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
        # Renaming default moves only its display name; the profile id stays "default".
        if name == DEFAULT_PROFILE_NAME:
            renamed_display_only = True
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
    """`cli.exec ["profile","delete","-y",name]` -> reconcile -> 204 (§8)."""
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
    # A second gate: the default profile can be listed under some other name.
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
