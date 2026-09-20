"""The merged model catalog behind `GET /api/models/catalog` (P6, agent model management)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

CATALOG_TTL_S = 3600.0

FAILURE_TTL_S = 60.0

FETCH_TIMEOUT_S = 10.0

FEATURED_ORDER: tuple[str, ...] = ("openrouter", "anthropic", "openai-api")
MAX_FEATURED = 4

KIND_LOCAL = "local"
KIND_OPENROUTER = "openrouter"
KIND_OTHER = "other"

SOURCE_OPENROUTER = "openrouter"
SOURCE_LOCAL = "local-endpoint"
SOURCE_HERMES = "hermes"

_OPENROUTER_TOOLS_PARAMS = frozenset({"tools"})
_OPENROUTER_REASONING_PARAMS = frozenset({"reasoning", "reasoning_effort"})

_LLAMA_FTYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    32: "BF16",
}


# No Authorization header is ever sent: the list is public and this module holds no key.
async def fetch_openrouter_models(
    *, timeout_s: float = FETCH_TIMEOUT_S
) -> dict[str, dict[str, Any]] | None:
    """OpenRouter's public list as `{model_id: entry}`, or None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(
                OPENROUTER_MODELS_URL, headers={"Accept": "application/json"}
            )
        response.raise_for_status()
        payload = response.json()
    # None, not {}: "no answer" retries on the short failure TTL; "none" does not.
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("OpenRouter model list unavailable: %s: %s", exc.__class__.__name__, exc)
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        logger.info("OpenRouter model list had no `data` list; ignoring it")
        return None
    by_id: dict[str, dict[str, Any]] = {}
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            by_id[entry["id"]] = entry
    return by_id


async def probe_local_models(
    api_url: str, *, timeout_s: float = FETCH_TIMEOUT_S
) -> dict[str, dict[str, Any]] | None:
    """`GET {api_url}/models` on a local OpenAI-compatible endpoint, as `{id: entry}`."""
    url = f"{api_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("local model probe %s unavailable: %s: %s", url, exc.__class__.__name__, exc)
        return None
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return None
    by_id: dict[str, dict[str, Any]] = {}
    for entry in data:
        if isinstance(entry, dict):
            identifier = entry.get("id") or entry.get("name")
            if isinstance(identifier, str) and identifier:
                by_id[identifier] = entry
    return by_id


class ModelCatalogCache:
    """One-hour memo of the two external fetches, on the monotonic clock."""

    def __init__(
        self, *, ttl_s: float = CATALOG_TTL_S, failure_ttl_s: float = FAILURE_TTL_S
    ) -> None:
        self.ttl_s = ttl_s
        self.failure_ttl_s = failure_ttl_s
        self._openrouter: tuple[float, dict[str, dict[str, Any]] | None] | None = None
        self._local: dict[str, tuple[float, dict[str, dict[str, Any]] | None]] = {}

    def _fresh(self, entry: tuple[float, Any] | None, now: float) -> bool:
        if entry is None:
            return False
        fetched_at, value = entry
        ttl = self.ttl_s if value is not None else self.failure_ttl_s
        return (now - fetched_at) < ttl

    async def openrouter(self, *, refresh: bool = False) -> dict[str, dict[str, Any]] | None:
        now = time.monotonic()
        if not refresh and self._fresh(self._openrouter, now):
            return self._openrouter[1]  # type: ignore[index]
        value = await fetch_openrouter_models()
        self._openrouter = (time.monotonic(), value)
        return value

    async def local(
        self, api_url: str, *, refresh: bool = False
    ) -> dict[str, dict[str, Any]] | None:
        now = time.monotonic()
        entry = self._local.get(api_url)
        if not refresh and self._fresh(entry, now):
            return entry[1]  # type: ignore[index]
        value = await probe_local_models(api_url)
        self._local[api_url] = (time.monotonic(), value)
        return value

    def cached_openrouter(self) -> dict[str, dict[str, Any]] | None:
        """The last fetched OpenRouter index, **without** fetching -- or None."""
        return self._openrouter[1] if self._openrouter is not None else None

    def clear(self) -> None:
        self._openrouter = None
        self._local.clear()


def _money_string_to_float(value: Any) -> float | None:
    """`"$2.00"` -> `2.0`; `"$0.20"` -> `0.2`; anything unparseable -> None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _per_token_to_per_million(value: Any) -> float | None:
    """OpenRouter's `pricing.*` per-token strings (`"0.000002"`) -> per-1M float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        per_token = float(value)
    elif isinstance(value, str):
        try:
            per_token = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return round(per_token * 1_000_000, 6)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def provider_kind(entry: dict[str, Any]) -> str:
    """`local` when it has an `api_url` or the slug starts with `custom`,
    `openrouter` for OpenRouter, else `other` (§8)."""
    slug = entry.get("slug") if isinstance(entry.get("slug"), str) else ""
    if entry.get("api_url") or slug.startswith("custom"):
        return KIND_LOCAL
    if slug == "openrouter":
        return KIND_OPENROUTER
    return KIND_OTHER


def hermes_price(pricing: Any, model_id: str) -> tuple[dict[str, float | None] | None, bool | None]:
    """`(price_per_million, free)` from Hermes's `pricing` map for one model, or `(None, None)`."""
    if not isinstance(pricing, dict):
        return None, None
    row = pricing.get(model_id)
    if not isinstance(row, dict):
        return None, None
    price = {
        "input": _money_string_to_float(row.get("input")),
        "output": _money_string_to_float(row.get("output")),
        "cache_read": _money_string_to_float(row.get("cache")),
    }
    free = row.get("free") if isinstance(row.get("free"), bool) else None
    if price["input"] is None and price["output"] is None:
        return None, free
    return price, free


def openrouter_facts(entry: dict[str, Any]) -> dict[str, Any]:
    """The §8 facts derivable from one OpenRouter list entry."""
    pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
    price = {
        "input": _per_token_to_per_million(pricing.get("prompt")),
        "output": _per_token_to_per_million(pricing.get("completion")),
        "cache_read": _per_token_to_per_million(pricing.get("input_cache_read")),
    }
    price_known = price["input"] is not None or price["output"] is not None
    architecture = entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
    modalities = architecture.get("input_modalities")
    vision: bool | None = None
    if isinstance(modalities, list):
        vision = "image" in modalities
    params = entry.get("supported_parameters")
    tools: bool | None = None
    reasoning: bool | None = None
    if isinstance(params, list):
        param_set = {p for p in params if isinstance(p, str)}
        tools = bool(param_set & _OPENROUTER_TOOLS_PARAMS)
        reasoning = bool(param_set & _OPENROUTER_REASONING_PARAMS)
    top = entry.get("top_provider") if isinstance(entry.get("top_provider"), dict) else {}
    return {
        "name": entry.get("name") if isinstance(entry.get("name"), str) else None,
        "context_length": _as_int(entry.get("context_length")),
        "max_completion_tokens": _as_int(top.get("max_completion_tokens")),
        "price_per_million": price if price_known else None,
        "free": (price["input"] == 0 and price["output"] == 0) if price_known else None,
        "capabilities": {"reasoning": reasoning, "tools": tools, "vision": vision},
    }


def _quantization_label(ftype: Any) -> str | None:
    if isinstance(ftype, str):
        return ftype or None
    if isinstance(ftype, bool):
        return None
    if isinstance(ftype, int):
        return _LLAMA_FTYPE_NAMES.get(ftype, f"ftype {ftype}")
    return None


def local_facts(entry: dict[str, Any]) -> dict[str, Any] | None:
    """`{parameters, quantization, context_length}` from a llama.cpp `/models` entry, or None."""
    meta = entry.get("meta")
    if not isinstance(meta, dict):
        return None
    facts = {
        "parameters": _as_int(meta.get("n_params")),
        "quantization": _quantization_label(meta.get("ftype")),
        "context_length": _as_int(meta.get("n_ctx")),
    }
    if all(v is None for v in facts.values()):
        return None
    return facts


def _local_entry_for(
    probe: dict[str, dict[str, Any]] | None, model_id: str
) -> dict[str, Any] | None:
    """The probe entry for `model_id`: exact id, else the single loaded model."""
    if not probe:
        return None
    if model_id in probe:
        return probe[model_id]
    # A llama.cpp server serves only the model it was launched with, whatever id it uses.
    if len(probe) == 1:
        return next(iter(probe.values()))
    return None


def _hermes_reasoning(capabilities: Any, model_id: str) -> bool | None:
    if not isinstance(capabilities, dict):
        return None
    row = capabilities.get(model_id)
    if not isinstance(row, dict):
        return None
    value = row.get("reasoning")
    return value if isinstance(value, bool) else None


def _provider_display_name(entry: dict[str, Any], slug: str, kind: str) -> str:
    """Hermes's own `name` when it is a real label; `Local endpoint` for a
    local provider Hermes only calls `custom` (the §8 example)."""
    name = entry.get("name")
    if (
        isinstance(name, str)
        and name.strip()
        and name.strip().lower() not in {slug.lower(), "custom"}
    ):
        return name.strip()
    if kind == KIND_LOCAL:
        return "Local endpoint"
    return slug


def _build_model(
    model_id: str,
    *,
    kind: str,
    entry: dict[str, Any],
    openrouter: dict[str, dict[str, Any]] | None,
    local_probe: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """One §8 model row, merging the three sources by precedence."""
    row: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "context_length": None,
        "max_completion_tokens": None,
        "price_per_million": None,
        "free": False,
        "capabilities": {
            "reasoning": _hermes_reasoning(entry.get("capabilities"), model_id),
            "tools": None,
            "vision": None,
        },
        "local": None,
    }
    hermes_price_row, hermes_free = hermes_price(entry.get("pricing"), model_id)
    if hermes_price_row is not None:
        row["price_per_million"] = hermes_price_row
        row["free"] = hermes_price_row["input"] == 0 and hermes_price_row["output"] == 0
    if hermes_free is True:
        row["free"] = True

    # OpenRouter's live list overrides Hermes's on-disk price; it needs no key to read.
    if kind == KIND_OPENROUTER and openrouter:
        facts = openrouter.get(model_id)
        if isinstance(facts, dict):
            merged = openrouter_facts(facts)
            if merged["name"]:
                row["name"] = merged["name"]
            row["context_length"] = merged["context_length"]
            row["max_completion_tokens"] = merged["max_completion_tokens"]
            if merged["price_per_million"] is not None:
                row["price_per_million"] = merged["price_per_million"]
                row["free"] = bool(merged["free"])
            for key, value in merged["capabilities"].items():
                if value is not None:
                    row["capabilities"][key] = value

    if kind == KIND_LOCAL:
        row["price_per_million"] = {"input": 0, "output": 0, "cache_read": None}
        row["free"] = True
        probe_entry = _local_entry_for(local_probe, model_id)
        if probe_entry is not None:
            facts = local_facts(probe_entry)
            row["local"] = facts
            if facts is not None and facts["context_length"] is not None:
                row["context_length"] = facts["context_length"]
    return row


def _featured_slugs(providers: list[dict[str, Any]]) -> list[str]:
    """The §8 featured order, over authenticated providers only, capped at four."""
    authenticated = [p for p in providers if p.get("authenticated") is True]
    ordered: list[str] = [
        entry["slug"] for entry in authenticated if provider_kind(entry) == KIND_LOCAL
    ]
    by_slug = {p["slug"]: p for p in authenticated}
    for slug in FEATURED_ORDER:
        if slug in by_slug and slug not in ordered:
            ordered.append(slug)
    return ordered[:MAX_FEATURED]


def _current_from_options(
    options: dict[str, Any], providers: list[dict[str, Any]]
) -> dict[str, Any]:
    """`{provider, model}` for the default profile, as far as `model.options` says."""
    current_provider = next((p["slug"] for p in providers if p.get("is_current") is True), None)
    model: str | None = None
    for key in ("current_model", "model", "default_model"):
        value = options.get(key)
        if isinstance(value, str) and value:
            model = value
            break
    return {"provider": current_provider, "model": model}


def _provider_entries(options: Any) -> list[dict[str, Any]]:
    """`model.options`'s provider list, entries with a string `slug` only."""
    providers = options.get("providers") if isinstance(options, dict) else None
    if not isinstance(providers, list):
        return []
    return [
        p for p in providers if isinstance(p, dict) and isinstance(p.get("slug"), str) and p["slug"]
    ]


def model_ids_for(options: Any, provider: str) -> list[str] | None:
    """The id strings `model.options` lists for `provider`, or None if the provider is not configured."""
    for entry in _provider_entries(options):
        if entry["slug"] == provider:
            models = entry.get("models")
            if not isinstance(models, list):
                return []
            return [
                m if isinstance(m, str) else m.get("id")
                for m in models
                if isinstance(m, (str, dict))
            ]
    return None


async def selectable_model_ids(
    options: dict[str, Any], provider: str, *, cache: ModelCatalogCache
) -> list[str] | None:
    """The ids a profile may be pointed at for `provider`: Hermes's list, or --
    for a local provider Hermes lists nothing for -- what its endpoint's
    `/models` probe answers. None when the provider is not configured at all.
    """
    ids = model_ids_for(options, provider)
    if ids:
        return ids
    if ids is None:
        return None
    entry = next((e for e in _provider_entries(options) if e["slug"] == provider), None)
    api_url = entry.get("api_url") if entry else None
    if entry is None or provider_kind(entry) != KIND_LOCAL or not isinstance(api_url, str):
        return ids
    probe = await cache.local(api_url)
    return [mid for mid in (probe or {}) if isinstance(mid, str) and mid]


def is_openrouter(options: Any, provider: str) -> bool:
    """Whether `model.options` lists `provider` as OpenRouter (§8 `kind`)."""
    entry = next((e for e in _provider_entries(options) if e["slug"] == provider), None)
    return entry is not None and provider_kind(entry) == KIND_OPENROUTER


async def build_catalog(
    options: dict[str, Any],
    *,
    cache: ModelCatalogCache,
    refresh: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The `GET /api/models/catalog` payload (§8) from a `model.options` result."""
    providers = _provider_entries(options)
    featured = _featured_slugs(providers)
    needs_openrouter = any(provider_kind(p) == KIND_OPENROUTER for p in providers)
    openrouter = await cache.openrouter(refresh=refresh) if needs_openrouter else None

    built_by_slug: dict[str, dict[str, Any]] = {}
    for entry in providers:
        slug = entry["slug"]
        kind = provider_kind(entry)
        api_url = (
            entry.get("api_url")
            if isinstance(entry.get("api_url"), str) and entry.get("api_url")
            else None
        )
        local_probe = (
            await cache.local(api_url, refresh=refresh)
            if (kind == KIND_LOCAL and api_url)
            else None
        )
        model_ids = model_ids_for(options, slug) or []
        # Hermes lists no ids for a local provider it has not probed; our own probe does.
        if not model_ids and local_probe:
            model_ids = [mid for mid in local_probe if isinstance(mid, str) and mid]
        built_by_slug[slug] = {
            "slug": slug,
            "name": _provider_display_name(entry, slug, kind),
            "kind": kind,
            "api_url": api_url,
            "authenticated": entry.get("authenticated") is True,
            "featured": slug in featured,
            "models": [
                _build_model(
                    mid, kind=kind, entry=entry, openrouter=openrouter, local_probe=local_probe
                )
                for mid in model_ids
                if isinstance(mid, str) and mid
            ],
        }

    ordered = [built_by_slug[slug] for slug in featured]
    ordered += [built_by_slug[p["slug"]] for p in providers if p["slug"] not in featured]
    stamp = now if now is not None else datetime.now(UTC)
    return {
        "fetched_at": stamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "current": _current_from_options(options, providers),
        "providers": ordered,
    }


def find_provider(catalog: dict[str, Any], provider: str) -> dict[str, Any] | None:
    for entry in catalog.get("providers", []):
        if isinstance(entry, dict) and entry.get("slug") == provider:
            return entry
    return None


def _is_openrouter_provider(catalog: dict[str, Any], provider: str) -> bool:
    """Whether `provider` is OpenRouter -- by catalog kind, or by slug when the
    catalog does not list it at all (a failed `model.options` still lets the
    public index answer)."""
    entry = find_provider(catalog, provider)
    if entry is not None:
        return entry.get("kind") == KIND_OPENROUTER
    return provider == "openrouter"


def _facts_from_openrouter_index(
    openrouter: dict[str, dict[str, Any]] | None, model_id: str
) -> dict[str, Any] | None:
    """The §8 `model_facts` for an OpenRouter id known only to the public list."""
    entry = openrouter.get(model_id) if openrouter else None
    if not isinstance(entry, dict):
        return None
    merged = openrouter_facts(entry)
    return {
        "name": merged["name"] or model_id,
        "kind": KIND_OPENROUTER,
        "context_length": merged["context_length"],
        "max_completion_tokens": merged["max_completion_tokens"],
        "price_per_million": merged["price_per_million"],
        "free": bool(merged["free"]),
        "capabilities": merged["capabilities"],
        "local": None,
        "source": SOURCE_OPENROUTER,
    }


def model_facts_for(
    catalog: dict[str, Any],
    provider: str | None,
    model_id: str | None,
    *,
    openrouter: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The §8 `model_facts` for one `(provider, model_id)`, or None if nothing lists it."""
    if not provider or not model_id:
        return None
    entry = find_provider(catalog, provider)
    row = (
        next(
            (m for m in entry.get("models", []) if isinstance(m, dict) and m.get("id") == model_id),
            None,
        )
        if entry is not None
        else None
    )
    if row is None:
        if _is_openrouter_provider(catalog, provider):
            return _facts_from_openrouter_index(openrouter, model_id)
        return None
    kind = entry.get("kind", KIND_OTHER)
    if kind == KIND_LOCAL and row.get("local") is not None:
        source = SOURCE_LOCAL
    # A name differing from the id means the public list answered for this row.
    elif kind == KIND_OPENROUTER and (
        row.get("context_length") is not None or row.get("name") != model_id
    ):
        source = SOURCE_OPENROUTER
    else:
        source = SOURCE_HERMES
    return {
        "name": row.get("name") or model_id,
        "kind": kind,
        "context_length": row.get("context_length"),
        "max_completion_tokens": row.get("max_completion_tokens"),
        "price_per_million": row.get("price_per_million"),
        "free": bool(row.get("free")),
        "capabilities": dict(
            row.get("capabilities") or {"reasoning": None, "tools": None, "vision": None}
        ),
        "local": row.get("local"),
        "source": source,
    }


def price_lookup_from_catalog(
    catalog: dict[str, Any] | None,
    *,
    openrouter: dict[str, dict[str, Any]] | None = None,
) -> Callable[[str], tuple[float, float] | None]:
    """A `price_lookup(model_id) -> (input, output) | None` over every provider in the catalog."""
    table: dict[str, tuple[float, float]] = {}
    for entry in (catalog or {}).get("providers", []):
        if not isinstance(entry, dict):
            continue
        for row in entry.get("models", []):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            model_id = row["id"]
            # First in catalog order wins, so a local model shadows a same-named remote.
            if model_id in table:
                continue
            price = row.get("price_per_million")
            if (
                isinstance(price, dict)
                and price.get("input") is not None
                and price.get("output") is not None
            ):
                table[model_id] = (float(price["input"]), float(price["output"]))
            elif row.get("free") is True:
                table[model_id] = (0.0, 0.0)

    def lookup(model_id: str) -> tuple[float, float] | None:
        price = table.get(model_id)
        if price is not None:
            return price
        entry = openrouter.get(model_id) if openrouter else None
        if not isinstance(entry, dict):
            return None
        fallback = openrouter_facts(entry)["price_per_million"]
        if fallback is None or fallback["input"] is None or fallback["output"] is None:
            return None
        return (float(fallback["input"]), float(fallback["output"]))

    return lookup


__all__ = [
    "CATALOG_TTL_S",
    "FAILURE_TTL_S",
    "FEATURED_ORDER",
    "FETCH_TIMEOUT_S",
    "KIND_LOCAL",
    "KIND_OPENROUTER",
    "KIND_OTHER",
    "MAX_FEATURED",
    "OPENROUTER_MODELS_URL",
    "SOURCE_HERMES",
    "SOURCE_LOCAL",
    "SOURCE_OPENROUTER",
    "ModelCatalogCache",
    "build_catalog",
    "fetch_openrouter_models",
    "find_provider",
    "hermes_price",
    "local_facts",
    "model_facts_for",
    "model_ids_for",
    "openrouter_facts",
    "price_lookup_from_catalog",
    "probe_local_models",
    "provider_kind",
]
