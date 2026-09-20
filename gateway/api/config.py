"""Runtime provider configuration: `/api/config/providers` (P5-9).

The owner's principle, verbatim: *"It shouldn't matter the logistics I use --
what should matter is having control over the configs."* Every provider choice
in this service used to be a hand edit of the repo-root `.env` on one specific
Mac, which meant the owner could not change their own setup without a
developer. Which machine or engine is in use is theirs to decide and change at
will, so these four routes make it changeable from the app:

    GET    /api/config/providers          what is in force, and where it came from
    PUT    /api/config/providers          change one or more values
    DELETE /api/config/providers/{key}    drop the override, fall back to .env
    POST   /api/config/providers/probe    ask an endpoint which models it serves

`config/runtime_config.py` owns the store, the registry and the validation; this module
owns the HTTP shape. Nothing here changes what a setting *does* -- `api/rewrite.py`
and `api/transcribe.py` read the same `Settings` fields they always did. What
changes is **who can change them and from where**.

## The two rules that shape every response

**1. No secret ever leaves this process.** `GET` reports a key as
`{"api_key_set": true}` and `"value": null`; there is no route, parameter or
error path that returns one. Secrets are write-only: `PUT` accepts them,
nothing reads them back. `tests/test_config.py::test_no_route_can_return_a_secret`
drives every route with a distinctive fake key configured and asserts that
string appears in no response body, and `_scrub_secrets()` additionally strips
any configured key out of an upstream error detail before it is echoed -- a
belt-and-braces guard for the one place a foreign server's words reach the app.

The `.env` field validation is the other half of it: a base URL carrying
`user:password@host` is a **422**, because a credential typed into a URL would
be reported back by every one of these routes, by `provider_label()` and by the
log line `api/rewrite.py` writes on every rewrite.

**2. A wrong answer is worse than an honest failure.** `PUT` validates before
persisting (422, naming the key and the bound), an unknown key is a 422 rather
than a silently-dropped instruction (the `extra="forbid"` discipline every body
in this service follows), and `probe` reports *unreachable*, *unauthorized*,
*http_error*, *not_json* and *no_models* as five distinct outcomes rather than
one "failed" -- because "the box is off", "the key is wrong" and "that is not an
OpenAI-compatible endpoint" need three different fixes.

## Why probe exists

A text field for the model id is configurable but not usable: model ids are
opaque, namespaced, and differ per endpoint, so the owner would be typing a
string they can only verify by getting a rewrite back wrong. `probe` GETs
`{base}/models` -- the OpenAI-compatible discovery route every server in the
measured list speaks (MTPLX on 127.0.0.1:8001, Ollama on 11434, OpenRouter) --
and hands back the list, so the app offers a picker. That is the difference
between configurable and usable.

## Freshness

`get_settings()` builds a fresh `Settings` per call and applies the overlay each
time, so a `PUT` here takes effect on the **next request** -- no restart, and no
process state to keep in step. The overlay parse is cached against the file's
`(st_mtime_ns, st_size)` and `write_overlay()` invalidates explicitly, so the
per-request cost is a `stat()` and correctness does not depend on the cache.
"""

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

#: Authenticated routes (mounted under `/api` in `api.main`).
config_router = APIRouter(tags=["config"])

#: How long a discovery probe waits. Short on purpose: the operator is standing in
#: a settings screen watching a spinner, and "that box is not answering" is a
#: useful answer delivered in seconds rather than a useful answer delivered in
#: two minutes.
PROBE_TIMEOUT_S = 10.0

#: How much of an upstream body is echoed into a probe detail. Same bound as
#: `api/rewrite.py`'s `_UPSTREAM_DETAIL_CHARS`, and scrubbed of every
#: configured secret before it goes out.
_PROBE_DETAIL_CHARS = 300

#: Cap on what a probe reports back. A hosted router lists hundreds of models;
#: a picker does not need more than this and the app should never be handed an
#: unbounded list off a foreign server.
MAX_PROBE_MODELS = 500
MAX_MODEL_ID_CHARS = 200

# --- probe outcomes -------------------------------------------------------
#
# Five, not one. Each names a different fix, and collapsing them into "failed"
# is how a settings screen tells the operator to check the network when the real
# problem is a rejected key.
PROBE_OK = "ok"  # answered, and served a model list
PROBE_UNREACHABLE = "unreachable"  # no HTTP response at all: DNS, refused, timeout
PROBE_UNAUTHORIZED = "unauthorized"  # answered 401/403: the key is missing or wrong
PROBE_HTTP_ERROR = "http_error"  # answered, but not 2xx and not an auth refusal
PROBE_NOT_JSON = "not_json"  # answered 2xx with something that is not JSON
PROBE_NO_MODELS = "no_models"  # answered 2xx JSON with no recognisable model list

#: The validator `POST .../probe` reuses for its `base_url`, so "that is not a
#: URL" -- including the refusal of `user:key@host`, which would otherwise put a
#: secret into every later report -- is answered identically wherever the operator
#: types one. Resolved at import so a registry that ever lost the key fails
#: loudly at startup rather than as a 500 on the probe route.
_PROBE_URL_SPEC: ConfigKey = CONFIG_KEYS_BY_NAME["rewrite_base_url"]

PROBE_OUTCOMES: tuple[str, ...] = (
    PROBE_OK,
    PROBE_UNREACHABLE,
    PROBE_UNAUTHORIZED,
    PROBE_HTTP_ERROR,
    PROBE_NOT_JSON,
    PROBE_NO_MODELS,
)


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------


class ProviderConfigUpdate(BaseModel):
    """Body for `PUT /api/config/providers`.

    Closed schema like every other body here, and closed *twice*: the model
    refuses an unknown top-level field, and the route refuses an unknown name
    inside `values`/`reset`. A typo'd setting name that was quietly dropped
    would leave the owner believing they had changed something they had not --
    the exact failure the no-silent-fallback discipline exists to prevent.
    """

    model_config = ConfigDict(extra="forbid")

    #: Setting name -> new value. Secrets are accepted here and never read back.
    values: dict[str, Any] = Field(default_factory=dict)
    #: Setting names whose override should be dropped, falling back to `.env`.
    #: Applied after `values`, so sending the same key in both is a 422 rather
    #: than an order-dependent surprise.
    reset: list[str] = Field(default_factory=list)


class ProviderProbe(BaseModel):
    """Body for `POST /api/config/providers/probe`.

    `api_key` is a **tri-state** and the distinction is the point:

    * **absent / null** -- use whatever key is currently in force for
      `capability`, so the owner can probe an already-configured endpoint
      without re-typing a secret they cannot read back.
    * **`""`** -- probe with no `Authorization` header at all. This is the
      normal case for a local llama.cpp / vLLM / MTPLX server and must be
      expressible, or a keyless endpoint could never be probed once a key had
      been configured for something else.
    * **a string** -- probe with exactly that key, *before* saving it, so a bad
      key is discovered in the settings screen and not by a failed rewrite.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1, max_length=400)
    api_key: str | None = Field(default=None, max_length=1024)
    #: Which capability's stored key to borrow when `api_key` is absent.
    capability: str = "rewrite"


# ---------------------------------------------------------------------------
# Reporting what is in force
# ---------------------------------------------------------------------------


def _setting_report(
    spec: ConfigKey,
    settings: Settings,
    overlay_values: dict[str, Any],
    env_vars: frozenset[str],
) -> dict[str, Any]:
    """One setting, with its source -- and **never with a secret in it**.

    A secret reports `value: null` and `api_key_set`, which is all the app
    needs to render "Set"/"Not set" and a way to clear it. Everything else
    reports its effective value plus the field default, so the app can show
    what resetting would fall back towards.
    """
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
        # The whole contract in two lines: no value, ever, and a boolean that
        # says whether one is configured.
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
    """The strings `_scrub_secrets()` must remove from a probe's own words.

    **Scoped to what could actually be in that text**, and no wider. Two
    sources: the key this probe presented (the only string an endpoint could
    plausibly echo back at all), and the other configured *provider* keys,
    because the realistic way one of those reaches a foreign server is the
    owner pasting the wrong one into the field.

    Deliberately NOT the gateway's own inbound password or Hermes's password.
    Neither is ever transmitted to a probed endpoint,
    so scrubbing them removes no risk -- and it does real damage: measured on
    2026-09-02 against the live services with a one-character
    `RESEARCH_GATEWAY_PASSWORD` set, a blanket scrub rewrote every letter "p"
    in the error text ("no HTTP res***onse from htt***://..."), turning an
    honest diagnostic into noise. A scrubber that mangles the message is a
    scrubber the owner learns to ignore.

    Built fresh per call, never stored, logged or returned.
    """
    found: list[str] = []
    if used_key:
        found.append(used_key)
    for name in sorted(runtime_config.SECRET_CONFIG_KEYS):
        text = _secret_text(getattr(settings, name, None))
        if text and text not in found:
            found.append(text)
    return found


def _scrub_secrets(text: str, secrets: list[str]) -> str:
    """Replace any of `secrets` in `text` with `***`.

    Belt and braces. A probe echoes a *foreign* server's error body, which this
    gateway does not control; nothing observed ever echoes the bearer token
    back, but "has never been observed to" is not a guarantee and a settings
    screen is exactly where a leaked key would be read aloud. No length floor:
    if a configured key is short enough that scrubbing it garbles the message,
    the garbled message is still the right trade.
    """
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "***")
    return text


def provider_config_report() -> dict[str, Any]:
    """The whole effective configuration, grouped by capability.

    Read straight off a fresh `get_settings()`, so what this reports is exactly
    what the next `/api/rewrite` or `/api/transcribe` will use -- there is no
    second copy of the resolution to drift.
    """
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
        # Where the overlay lives and whether it is currently in play. The path
        # is a filesystem location the operator configured, not a secret.
        "overlay": {
            "path": settings.research_gateway_runtime_config_path,
            "present": overlay.present,
            "override_count": len(overlay.values),
            # Never a value -- `read_overlay()` builds these from validator
            # messages precisely so a rejected secret is not echoed.
            "problems": list(overlay.problems),
        },
        "env_file": str(ENV_FILE),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@config_router.get("/config/providers")
async def read_provider_config() -> dict[str, Any]:
    """What model/voice configuration is actually in force, and why.

    Response: `{"capabilities": [...], "overlay": {...}, "env_file": "..."}`.
    Each capability carries its settings; each setting carries `value`,
    `source` (`default` | `env` | `overlay`), `env_var`, and the bounds the app
    needs to build a control for it.

    **A secret's `value` is always `null`** and its `api_key_set` boolean is the
    only thing reported about it. There is no parameter that changes that.

    Two capabilities are listed with `writable: false` and a `note` saying why:
    speech synthesis runs on the device, and the conversation model is the
    Hermes instance's own (measured: `session.create` accepts a model and
    ignores it). Saying so beats a screen that is silent about them.
    """
    return provider_config_report()


#: Sentinel for "the overlay had no entry for this key", so `reset` can
#: distinguish dropping something from dropping nothing without a second lookup.
_MISSING = object()


@config_router.put("/config/providers")
async def update_provider_config(body: ProviderConfigUpdate) -> dict[str, Any]:
    """Change one or more settings. Validated first, persisted second.

    Body: `{"values": {"rewrite_model": "..."}, "reset": ["rewrite_api_key"]}`
    -- both optional, but a body that does neither is a 422 rather than a
    round trip that changes nothing.

    **Every value is validated before anything is written**, so a request that
    is rejected leaves the previous configuration exactly as it was: there is
    no partially-applied state. An unknown key -- in `values` or in `reset` --
    is a 422 naming it, never a silent drop.

    `reset` drops the overlay entry so the value falls back to `.env`, which is
    how the owner gets back to a known state from the app. `.env` itself is
    never written by this route or any other.

    Response: `{"updated": [...], "reset": [...]}` plus the full
    `GET /api/config/providers` body, so the app renders the new effective
    state -- including its `source` -- without a second round trip.
    """
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

    # Validate the whole batch before touching the file. A half-applied config
    # change is worse than a rejected one -- the operator would be looking at a
    # screen that is partly what they asked for and partly not.
    validated: dict[str, Any] = {}
    for name, raw in body.values.items():
        spec = CONFIG_KEYS_BY_NAME[name]
        try:
            validated[name] = runtime_config.validate_value(spec, raw)
        except ConfigValueError as exc:
            # `str(exc)` is built by the validator from the key name and the
            # bound, never from the value, so a rejected secret is not echoed.
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

    # Names only. Values -- one of which may be a key -- are never logged.
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
    """Drop one override, falling back to `.env` -- the way back to known state.

    **Idempotent.** Resetting a key that was never overridden is a 200 with
    `"was_overridden": false`, not a 404: the caller asked for "this key should
    come from `.env`", and it already does. A 404 would make a settings screen
    show an error for an outcome that is exactly what was wanted.

    An unknown key is a 422, the same as on `PUT` -- a name this gateway does
    not know is a client bug either way, and answering 200 to it would let a
    typo read as a successful reset.

    Response: `{"reset": key, "was_overridden": bool}` plus the full
    `GET /api/config/providers` body.
    """
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
    """`{base}/models`, tolerating a trailing slash -- `endpoint_url()`'s sibling.

    The base URL already includes the version segment the server expects
    (`.../v1`), exactly as `REWRITE_BASE_URL` does and as every
    OpenAI-compatible client takes it. This function does not invent one, for
    the same reason `api/rewrite.py::endpoint_url` does not.
    """
    return f"{base_url.rstrip('/')}/models"


def models_from_payload(payload: Any) -> list[dict[str, Any]] | None:
    """The model list out of an OpenAI-compatible `/models` body, or None.

    Accepts the documented `{"object": "list", "data": [...]}` and the bare
    list some servers answer with. `None` -- not `[]` -- when the body is
    neither, because "this endpoint serves no models" and "this is not a models
    endpoint" are different answers and only the second means the owner has
    typed the wrong URL.

    Entries are bounded (`MAX_PROBE_MODELS`, `MAX_MODEL_ID_CHARS`): this list
    comes off a foreign server and is rendered in a picker.
    """
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
    """GET `{base}/models` and report, honestly, what happened.

    Module-level so the tests can drive it against a mock OpenAI-compatible
    server on loopback and the route tests can replace it -- the same injection
    point `api/rewrite.py` exposes as `rewrite_via_chat_completions`.

    **Never raises for an upstream problem.** Every outcome is data: the caller
    is a settings screen and "the box refused the key" is an answer, not an
    error. Only a malformed *request* is an HTTP error, and that is decided by
    the route before this is called.

    The key is used and never stored, logged or returned; `build_headers`-style,
    an empty key sends no `Authorization` header at all, because `Bearer ` with
    nothing after it is a 401 on some servers and an odd log line on the rest.
    """
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

    # An HTTP answer of any kind means the endpoint is there. A 401 is
    # reachable-and-refusing, which is a different fix from "not answering".
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
    """Ask an endpoint which models it serves, so the app can offer a picker.

    Body: `{"base_url": "...", "api_key": null | "" | "...", "capability":
    "rewrite"}`. See `ProviderProbe` for why `api_key` is a tri-state.

    Response: `{"probe": {"base_url", "url", "reachable", "status_code",
    "outcome", "models", "model_count", "detail", "authenticated"}}`.

    **Always HTTP 200 when the request itself was well formed**, whatever the
    endpoint did. The probe's job is to find out, and it succeeded at that; the
    app renders `outcome`. A malformed `base_url` is a 422 from the same
    validator `PUT` uses, so "that is not a URL" is answered identically
    wherever the owner types it -- including the refusal of `user:key@host`,
    which would otherwise put a secret into every later report.

    `outcome` is one of `ok`, `unreachable`, `unauthorized`, `http_error`,
    `not_json`, `no_models`. `detail` is bounded and scrubbed of every
    configured secret before it goes out.
    """
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

    # The URL is logged, the key never is -- and the URL cannot carry one,
    # because the validator above refuses userinfo.
    logger.info("probing %s for models", models_url(base_url))
    result = await probe_models_endpoint(base_url, api_key)
    secrets = _probe_secrets(settings, api_key)
    detail = result.get("detail")
    if isinstance(detail, str) and detail:
        result["detail"] = _scrub_secrets(detail, secrets)
    # The model ids come off a foreign server too. Nothing sane puts a key in
    # one, but they are rendered in a picker and scrubbing them costs a pass
    # over a bounded list.
    for model in result.get("models", []):
        if isinstance(model.get("id"), str):
            model["id"] = _scrub_secrets(model["id"], secrets)
    return {"probe": result}


def _stored_key_for(settings: Settings, capability: str) -> str:
    """The key currently in force for `capability`, for an unkeyed probe body.

    Used and immediately discarded. An unknown capability, or one with no
    secret of its own, yields `""` -- probe keyless rather than reach for
    someone else's credential.
    """
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
