"""Resolve a Hermes profile into the OpenAI-compatible endpoint that serves it.

Owner ask 2026-09-07: *"The settings rewrite for listening accepts ip and port
but I'd prefer if I can select one of my profiles instead."* The profiles are
Hermes's own (`profiles.list`: the agents in the Agents screen), and each names
a `provider` slug and a `model` id. This module turns that pair into what
`api/rewrite.py::rewrite_via_chat_completions` needs -- a base URL and a model
-- so `REWRITE_PROFILE=gemma` means "whatever gemma is running on", and
re-pointing the profile in the Agents screen re-points the rewrite with it.

Two ways a provider yields a base URL, and one way it cannot:

* **A provider with an `api_url`** in `model.options` -- a user-defined local
  endpoint (`custom`, `local-3080ti`; measured on the owner's instance) -- is
  used as-is. That URL is already the OpenAI-compatible base including its
  version segment; it is what Hermes itself posts to.
* **A built-in hosted provider that also speaks `/chat/completions`** --
  OpenRouter, OpenAI, Anthropic -- resolves through `HOSTED_BASE_URLS`. The
  base URL is public knowledge; the *key* is not: Hermes never exposes its
  own (`key_env` is not in `model.options`), so the gateway sends the one it
  holds, `REWRITE_API_KEY`. A hosted profile with no key set there is refused
  at the route with a message that says exactly that.
* **Anything else** (`bedrock`, `moa`, `openai-codex`, `opencode-free` -- the
  owner's instance lists all four) has no endpoint this gateway can call with
  a key of its own. Naming that is the honest answer; guessing a URL is not.

The resolution is cached per profile name for `CACHE_TTL_S` on the app state,
because a document read aloud is dozens of rewrites in a row and each would
otherwise cost two Hermes round trips. Errors are never cached: a profile the
owner is about to create must be found on the next try.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from adapters.hermes import HermesAdapter, HermesError
from domain.hermes_runtime import _with_reconnect

logger = logging.getLogger(__name__)

#: Hermes's built-in hosted providers that also speak OpenAI's
#: `/chat/completions` with a Bearer key, and the base URL each does it at.
#: A slug absent here has no endpoint this gateway can call on its own.
HOSTED_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai-api": "https://api.openai.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

#: How long a resolved endpoint is trusted before `profiles.list` and
#: `model.options` are asked again. One minute: a model change in the Agents
#: screen shows up in the next document, not the next chunk.
CACHE_TTL_S = 60.0

_CACHE_ATTR = "profile_endpoint_cache"


class ProfileEndpointError(Exception):
    """Why a profile could not be turned into an endpoint, in the owner's terms.

    `status_code` is what the route should answer: 503 for a configuration
    the gateway cannot act on (no such profile, a provider with no endpoint),
    502 when Hermes itself could not be asked.
    """

    def __init__(self, detail: str, *, status_code: int = 503) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class ProfileEndpoint:
    profile: str
    provider: str
    model: str
    base_url: str
    #: True when the base URL came from `HOSTED_BASE_URLS`, i.e. the provider
    #: is a keyed service and the key has to be the gateway's own.
    hosted: bool


def _provider_entry(options: Any, slug: str) -> dict[str, Any] | None:
    providers = options.get("providers") if isinstance(options, dict) else None
    if not isinstance(providers, list):
        return None
    for entry in providers:
        if isinstance(entry, dict) and entry.get("slug") == slug:
            return entry
    return None


def endpoint_from(profiles: Any, options: Any, name: str) -> ProfileEndpoint:
    """The pure half: a `profiles.list` result + a `model.options` result -> endpoint.

    Raises `ProfileEndpointError` (503) with a message that names the profile,
    the provider and what to do about it.
    """
    rows = profiles.get("profiles") if isinstance(profiles, dict) else None
    rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    row = next((r for r in rows if r.get("name") == name), None)
    if row is None:
        known = ", ".join(sorted(str(r.get("name")) for r in rows if r.get("name"))) or "none"
        raise ProfileEndpointError(
            f"prose rewrite is configured to use Hermes profile {name!r}, but there is no "
            f"such profile (Hermes lists: {known}). Pick another profile in Settings > "
            "Models & voice, or clear the profile to use the endpoint and model fields."
        )
    provider = row.get("provider")
    model = row.get("model")
    if not isinstance(provider, str) or not provider.strip():
        raise ProfileEndpointError(
            f"Hermes profile {name!r} has no provider set, so there is nothing to send the "
            "rewrite to. Pick a model for it in the Agents screen first."
        )
    if not isinstance(model, str) or not model.strip():
        raise ProfileEndpointError(
            f"Hermes profile {name!r} has no model set, so there is nothing to send the "
            "rewrite to. Pick a model for it in the Agents screen first."
        )
    provider = provider.strip()
    model = model.strip()

    entry = _provider_entry(options, provider)
    api_url = entry.get("api_url") if entry else None
    if isinstance(api_url, str) and api_url.strip():
        return ProfileEndpoint(
            profile=name, provider=provider, model=model, base_url=api_url.strip(), hosted=False
        )
    hosted = HOSTED_BASE_URLS.get(provider)
    if hosted:
        return ProfileEndpoint(
            profile=name, provider=provider, model=model, base_url=hosted, hosted=True
        )
    raise ProfileEndpointError(
        f"Hermes profile {name!r} runs on provider {provider!r}, which has no "
        "OpenAI-compatible endpoint this gateway can call with a key of its own "
        f"(it can call {', '.join(sorted(HOSTED_BASE_URLS))}, or any provider Hermes "
        "lists with a base URL). Pick a profile on one of those, or clear the profile "
        "and set the endpoint and model by hand."
    )


def invalidate_profile_endpoint_cache(app_state: Any) -> None:
    cache = getattr(app_state, _CACHE_ATTR, None)
    if isinstance(cache, dict):
        cache.clear()


async def resolve_profile_endpoint(
    app_state: Any, name: str, *, cache_ttl_s: float = CACHE_TTL_S
) -> ProfileEndpoint:
    """`endpoint_from` over the live instance, memoised on `app_state`.

    Raises `ProfileEndpointError`: 502 when Hermes could not be asked, 503 for
    every answer that is not an endpoint.
    """
    cache = getattr(app_state, _CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(app_state, _CACHE_ATTR, cache)
    hit = cache.get(name)
    now = time.monotonic()
    if hit is not None and now - hit[0] < cache_ttl_s:
        return hit[1]

    adapter: HermesAdapter = app_state.hermes_adapter
    try:
        profiles = await _with_reconnect(app_state, adapter, adapter.profiles_list)
        options = await _with_reconnect(app_state, adapter, adapter.model_options)
    except HermesError as exc:
        raise ProfileEndpointError(
            f"prose rewrite is configured to use Hermes profile {name!r}, but Hermes could "
            f"not be asked which model it runs: {exc}",
            status_code=502,
        ) from exc

    endpoint = endpoint_from(profiles, options, name)
    cache[name] = (now, endpoint)
    logger.info(
        "rewrite profile %r resolved to %s on %s (%s)",
        name,
        endpoint.model,
        endpoint.provider,
        endpoint.base_url,
    )
    return endpoint


__all__ = [
    "CACHE_TTL_S",
    "HOSTED_BASE_URLS",
    "ProfileEndpoint",
    "ProfileEndpointError",
    "endpoint_from",
    "invalidate_profile_endpoint_cache",
    "resolve_profile_endpoint",
]
