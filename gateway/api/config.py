"""Runtime provider configuration: `/api/config/providers` (P5-9)."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from config import runtime_config
from config.runtime_config import (
    CAPABILITIES,
    CONFIG_KEYS_BY_NAME,
    ConfigKey,
    ConfigValueError,
    keys_for_capability,
)
from config.settings import ENV_FILE, Settings, get_settings

logger = logging.getLogger(__name__)

config_router = APIRouter(tags=["config"])

PROBE_TIMEOUT_S = 10.0

_PROBE_DETAIL_CHARS = 300

MAX_PROBE_MODELS = 500
MAX_MODEL_ID_CHARS = 200

PROBE_OK = "ok"
PROBE_UNREACHABLE = "unreachable"
PROBE_UNAUTHORIZED = "unauthorized"
PROBE_HTTP_ERROR = "http_error"
PROBE_NOT_JSON = "not_json"
PROBE_NO_MODELS = "no_models"

_PROBE_URL_SPEC: ConfigKey = CONFIG_KEYS_BY_NAME["rewrite_base_url"]

PROBE_OUTCOMES: tuple[str, ...] = (
    PROBE_OK,
    PROBE_UNREACHABLE,
    PROBE_UNAUTHORIZED,
    PROBE_HTTP_ERROR,
    PROBE_NOT_JSON,
    PROBE_NO_MODELS,
)


class ProviderConfigUpdate(BaseModel):
    """Body for `PUT /api/config/providers`."""

    model_config = ConfigDict(extra="forbid")

    values: dict[str, Any] = Field(default_factory=dict)
    reset: list[str] = Field(default_factory=list)


class ProviderProbe(BaseModel):
    """Body for `POST /api/config/providers/probe`."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1, max_length=400)
    api_key: str | None = Field(default=None, max_length=1024)
    capability: str = "rewrite"


def _setting_report(
    spec: ConfigKey,
    settings: Settings,
    overlay_values: dict[str, Any],
    env_vars: frozenset[str],
) -> dict[str, Any]:
    """One setting, with its source -- and **never with a secret in it**."""
    source = runtime_config.source_of(spec, overlay_values, env_vars)
    report: dict[str, Any] = {
        "key": spec.key,
        "env_var": spec.env_var,
        "capability": spec.capability,
        "label": spec.label,
        "kind": spec.kind,
        "help": spec.help,
        "source": source,
        "overridden": source == runtime_config.SOURCE_OVERLAY,
        "secret": spec.secret,
    }
    if spec.choices:
        report["choices"] = list(spec.choices)
    if spec.minimum is not None:
        report["minimum"] = spec.minimum
    if spec.maximum is not None:
        report["maximum"] = spec.maximum
    report["max_chars"] = spec.max_chars

    raw = getattr(settings, spec.key)
    if spec.secret:
        report["value"] = None
        report["api_key_set"] = bool(_secret_text(raw))
    else:
        report["value"] = raw
        default = Settings.model_fields[spec.key].default
        report["default"] = default
    return report


def _secret_text(value: Any) -> str:
    """The plaintext behind a `SecretStr`, stripped. Never leaves this module."""
    getter = getattr(value, "get_secret_value", None)
    text = getter() if callable(getter) else value
    return text.strip() if isinstance(text, str) else ""


def _probe_secrets(settings: Settings, used_key: str) -> list[str]:
    """The strings `_scrub_secrets()` must remove from a probe's own words."""
    found: list[str] = []
    if used_key:
        found.append(used_key)
    for name in sorted(runtime_config.SECRET_CONFIG_KEYS):
        text = _secret_text(getattr(settings, name, None))
        if text and text not in found:
            found.append(text)
    return found


def _scrub_secrets(text: str, secrets: list[str]) -> str:
    """Replace any of `secrets` in `text` with `***`."""
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "***")
    return text


def provider_config_report() -> dict[str, Any]:
    """The whole effective configuration, grouped by capability."""
    settings = get_settings()
    overlay = runtime_config.read_overlay(settings.research_gateway_runtime_config_path)
    env_vars = runtime_config.env_provided_vars(ENV_FILE)

    capabilities: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        specs = keys_for_capability(capability.name)
        capabilities.append(
            {
                "capability": capability.name,
                "label": capability.label,
                "summary": capability.summary,
                "writable": capability.writable,
                "note": capability.note,
                "settings": [
                    _setting_report(spec, settings, overlay.values, env_vars) for spec in specs
                ],
            }
        )
    return {
        "capabilities": capabilities,
        "overlay": {
            "path": settings.research_gateway_runtime_config_path,
            "present": overlay.present,
            "override_count": len(overlay.values),
            "problems": list(overlay.problems),
        },
        "env_file": str(ENV_FILE),
    }


@config_router.get("/config/providers")
async def read_provider_config() -> dict[str, Any]:
    """What model/voice configuration is actually in force, and why."""
    return provider_config_report()


_MISSING = object()


@config_router.put("/config/providers")
async def update_provider_config(body: ProviderConfigUpdate) -> dict[str, Any]:
    """Change one or more settings. Validated first, persisted second."""
    if not body.values and not body.reset:
        raise HTTPException(
            status_code=422,
            detail=("nothing to do: send at least one entry in `values` or one name in `reset`"),
        )

    unknown = sorted(
        {name for name in body.values if name not in CONFIG_KEYS_BY_NAME}
        | {name for name in body.reset if name not in CONFIG_KEYS_BY_NAME}
    )
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown setting(s) {', '.join(repr(n) for n in unknown)}; "
                f"the runtime-configurable settings are "
                f"{', '.join(sorted(CONFIG_KEYS_BY_NAME))}"
            ),
        )
    conflicting = sorted(set(body.values) & set(body.reset))
    if conflicting:
        raise HTTPException(
            status_code=422,
            detail=(
                f"setting(s) {', '.join(repr(n) for n in conflicting)} appear in "
                "both `values` and `reset`; send one or the other, so the result "
                "does not depend on which is applied first"
            ),
        )

    validated: dict[str, Any] = {}
    for name, raw in body.values.items():
        spec = CONFIG_KEYS_BY_NAME[name]
        try:
            validated[name] = runtime_config.validate_value(spec, raw)
        except ConfigValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    settings = get_settings()
    path = settings.research_gateway_runtime_config_path
    current = dict(runtime_config.read_overlay(path).values)
    current.update(validated)
    dropped = [name for name in body.reset if current.pop(name, _MISSING) is not _MISSING]

    try:
        runtime_config.write_overlay(path, current)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"the runtime config overlay at {path} could not be written "
                f"({exc.__class__.__name__}); nothing was changed and the "
                "gateway is still running on .env"
            ),
        ) from exc

    logger.info(
        "runtime provider config updated: set %s, reset %s",
        sorted(validated) or "nothing",
        sorted(dropped) or "nothing",
    )
    return {
        "updated": sorted(validated),
        "reset": sorted(dropped),
        **provider_config_report(),
    }


@config_router.delete("/config/providers/{key}")
async def reset_provider_setting(key: str) -> dict[str, Any]:
    """Drop one override, falling back to `.env` -- the way back to known state."""
    spec = CONFIG_KEYS_BY_NAME.get(key)
    if spec is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown setting {key!r}; the runtime-configurable settings are "
                f"{', '.join(sorted(CONFIG_KEYS_BY_NAME))}"
            ),
        )
    settings = get_settings()
    path = settings.research_gateway_runtime_config_path
    current = dict(runtime_config.read_overlay(path).values)
    was_overridden = key in current
    if was_overridden:
        current.pop(key)
        try:
            runtime_config.write_overlay(path, current)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"the runtime config overlay at {path} could not be written "
                    f"({exc.__class__.__name__}); {key} was not reset"
                ),
            ) from exc
        logger.info("runtime provider config reset: %s now falls back to .env", key)
    return {"reset": key, "was_overridden": was_overridden, **provider_config_report()}


def models_url(base_url: str) -> str:
    """`{base}/models`, tolerating a trailing slash -- `endpoint_url()`'s sibling."""
    return f"{base_url.rstrip('/')}/models"


def models_from_payload(payload: Any) -> list[dict[str, Any]] | None:
    """The model list out of an OpenAI-compatible `/models` body, or None."""
    if isinstance(payload, dict):
        data = payload.get("data")
    elif isinstance(payload, list):
        data = payload
    else:
        return None
    if not isinstance(data, list):
        return None
    models: list[dict[str, Any]] = []
    for entry in data[:MAX_PROBE_MODELS]:
        if isinstance(entry, dict):
            identifier = entry.get("id") or entry.get("name")
            owned_by = entry.get("owned_by")
        elif isinstance(entry, str):
            identifier, owned_by = entry, None
        else:
            continue
        if not isinstance(identifier, str) or not identifier.strip():
            continue
        models.append(
            {
                "id": identifier.strip()[:MAX_MODEL_ID_CHARS],
                "owned_by": owned_by if isinstance(owned_by, str) else None,
            }
        )
    return models


async def probe_models_endpoint(
    base_url: str, api_key: str, *, timeout_s: float = PROBE_TIMEOUT_S
) -> dict[str, Any]:
    """GET `{base}/models` and report, honestly, what happened."""
    url = models_url(base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    result: dict[str, Any] = {
        "base_url": base_url,
        "url": url,
        "authenticated": bool(api_key),
        "reachable": False,
        "status_code": None,
        "outcome": PROBE_UNREACHABLE,
        "models": [],
        "model_count": 0,
        "detail": "",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        result["detail"] = f"no HTTP response from {url}: {exc.__class__.__name__}: {exc}"
        return result

    result["reachable"] = True
    result["status_code"] = response.status_code

    if response.status_code in (401, 403):
        result["outcome"] = PROBE_UNAUTHORIZED
        result["detail"] = (
            f"the endpoint answered HTTP {response.status_code}: "
            f"{response.text[:_PROBE_DETAIL_CHARS]}"
        )
        return result
    if not 200 <= response.status_code < 300:
        result["outcome"] = PROBE_HTTP_ERROR
        result["detail"] = (
            f"the endpoint answered HTTP {response.status_code}: "
            f"{response.text[:_PROBE_DETAIL_CHARS]}"
        )
        return result
    try:
        payload = response.json()
    except ValueError:
        result["outcome"] = PROBE_NOT_JSON
        result["detail"] = (
            "the endpoint answered 200 with something that is not JSON, so it "
            "is probably not an OpenAI-compatible base URL: "
            f"{response.text[:_PROBE_DETAIL_CHARS]}"
        )
        return result

    models = models_from_payload(payload)
    if models is None:
        result["outcome"] = PROBE_NO_MODELS
        result["detail"] = (
            "the endpoint answered 200 with JSON that carries no `data` list of "
            "models, so it is probably not an OpenAI-compatible base URL"
        )
        return result
    result["outcome"] = PROBE_OK
    result["models"] = models
    result["model_count"] = len(models)
    result["detail"] = f"{len(models)} model(s) available"
    return result


@config_router.post("/config/providers/probe")
async def probe_provider(body: ProviderProbe) -> dict[str, Any]:
    """Ask an endpoint which models it serves, so the app can offer a picker."""
    try:
        base_url = runtime_config.validate_value(_PROBE_URL_SPEC, body.base_url)
    except ConfigValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if not base_url:
        raise HTTPException(
            status_code=422,
            detail=(
                "base_url is empty; give the OpenAI-compatible base URL to "
                "probe, including its version segment (e.g. "
                "http://127.0.0.1:8001/v1)"
            ),
        )

    settings = get_settings()
    if body.api_key is None:
        api_key = _stored_key_for(settings, body.capability)
    else:
        api_key = body.api_key.strip()

    logger.info("probing %s for models", models_url(base_url))
    result = await probe_models_endpoint(base_url, api_key)
    secrets = _probe_secrets(settings, api_key)
    detail = result.get("detail")
    if isinstance(detail, str) and detail:
        result["detail"] = _scrub_secrets(detail, secrets)
    for model in result.get("models", []):
        if isinstance(model.get("id"), str):
            model["id"] = _scrub_secrets(model["id"], secrets)
    return {"probe": result}


def _stored_key_for(settings: Settings, capability: str) -> str:
    """The key currently in force for `capability`, for an unkeyed probe body."""
    for spec in keys_for_capability(capability):
        if spec.secret:
            text = _secret_text(getattr(settings, spec.key, None))
            if text:
                return text
    return ""


__all__ = [
    "MAX_MODEL_ID_CHARS",
    "MAX_PROBE_MODELS",
    "PROBE_HTTP_ERROR",
    "PROBE_NOT_JSON",
    "PROBE_NO_MODELS",
    "PROBE_OK",
    "PROBE_OUTCOMES",
    "PROBE_TIMEOUT_S",
    "PROBE_UNAUTHORIZED",
    "PROBE_UNREACHABLE",
    "ProviderConfigUpdate",
    "ProviderProbe",
    "config_router",
    "models_from_payload",
    "models_url",
    "probe_models_endpoint",
    "probe_provider",
    "provider_config_report",
    "read_provider_config",
    "reset_provider_setting",
    "update_provider_config",
]
