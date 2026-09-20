"""Prose rewrite for speech: `POST /api/rewrite` (P5-4a)."""

from __future__ import annotations

import logging
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.validators import reject_blank_text
from config.settings import Settings, get_settings
from domain.profile_endpoint import (
    ProfileEndpoint,
    ProfileEndpointError,
    resolve_profile_endpoint,
)
from domain.prompt_files import PROMPT_FILE_FOR_DEPTH, PROMPT_FILE_FOR_STYLE

logger = logging.getLogger(__name__)

rewrite_router = APIRouter(tags=["rewrite"])

_UPSTREAM_DETAIL_CHARS = 300

DEFAULT_REWRITE_STYLE = "listen"

DEFAULT_REWRITE_DEPTH = "full"

REWRITE_DEPTH_SUFFIXES: dict[str, str] = {
    "brief": (
        " Keep this to the SHORTEST version that is still true: lead with "
        "the outcome or bottom line, then only the handful of facts that "
        "change what the listener would do next. Everything you keep must "
        "still be a fact from the input, stated correctly -- brevity comes "
        "from choosing what to omit, never from rounding or guessing a "
        "number you dropped."
    ),
    "medium": (
        " Aim for a middle length: the outcome, the reasoning that supports "
        "it, and the numbers and decisions that matter, but skip minor "
        "detail, secondary caveats and anything a listener would not act on. "
        "Every fact you keep must be stated exactly as the input states it."
    ),
}

# Fractions of the base ceiling; reasoning models spend it before content, so short depths cap low.
REWRITE_DEPTH_MAX_TOKENS_FRACTION: dict[str, float] = {
    "brief": 0.35,
    "medium": 0.65,
}

REWRITE_DEPTHS = ("brief", "medium", "full", "raw")

RewriteDepth = Literal["brief", "medium", "full", "raw"]

REWRITE_STYLE_PROMPT_SETTINGS: dict[str, str] = {
    "listen": "rewrite_prompt",
    "explain": "rewrite_code_prompt",
    "document": "rewrite_doc_prompt",
}

REWRITE_STYLES = tuple(REWRITE_STYLE_PROMPT_SETTINGS)

# Spelled out rather than derived from the tuple: Literal needs literal members at type-check time.
RewriteStyle = Literal["listen", "explain", "document"]


class RewriteRequest(BaseModel):
    """Body for `POST /api/rewrite`."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    style: RewriteStyle = DEFAULT_REWRITE_STYLE
    depth: RewriteDepth = DEFAULT_REWRITE_DEPTH

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        return reject_blank_text(value)


def prompt_for_style(settings: Settings, style: str) -> str:
    """The configured system prompt for `style`."""
    try:
        attribute = REWRITE_STYLE_PROMPT_SETTINGS[style]
    except KeyError:  # pragma: no cover - unreachable behind the schema
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown rewrite style {style!r}; known styles are {', '.join(REWRITE_STYLES)}"
            ),
        ) from None
    return getattr(settings, attribute)


def prompt_for_depth(base_prompt: str, depth: str) -> str:
    """`base_prompt` (already resolved for `style`) plus `depth`'s suffix."""
    return compose_prompt(base_prompt, REWRITE_DEPTH_SUFFIXES.get(depth, ""))


def compose_prompt(base_prompt: str, suffix: str) -> str:
    """`base_prompt` plus `suffix`, which may be empty. The one place the two
    halves of a system prompt are joined, so the file-backed path and the
    settings path cannot disagree about the join."""
    return base_prompt + suffix if suffix else base_prompt


async def resolve_system_prompt(app_state: Any, settings: Settings, style: str, depth: str) -> str:
    """The system prompt for `style` at `depth`, files first."""
    store = getattr(app_state, "prompt_files", None)
    base = prompt_for_style(settings, style)
    suffix = REWRITE_DEPTH_SUFFIXES.get(depth, "")
    if store is None:
        return compose_prompt(base, suffix)
    base = await store.text(PROMPT_FILE_FOR_STYLE.get(style)) or base
    file_suffix = await store.text(PROMPT_FILE_FOR_DEPTH.get(depth))
    if file_suffix is not None:
        suffix = file_suffix if file_suffix.startswith(" ") else " " + file_suffix
    return compose_prompt(base, suffix)


def max_tokens_for_depth(base_max_tokens: int, depth: str) -> int:
    """`base_max_tokens` scaled by `depth`'s fraction, floored at 1."""
    fraction = REWRITE_DEPTH_MAX_TOKENS_FRACTION.get(depth)
    if fraction is None:
        return base_max_tokens
    return max(1, int(base_max_tokens * fraction))


def endpoint_url(base_url: str) -> str:
    """`{base}/chat/completions`, tolerating a trailing slash on the base."""
    return f"{base_url.rstrip('/')}/chat/completions"


def provider_label(base_url: str) -> str:
    """The configured endpoint's `host[:port]`, for the response's `provider`."""
    netloc = urlsplit(base_url).netloc
    return netloc or base_url


def build_headers(api_key: str) -> dict[str, str]:
    """Request headers. **An empty key omits `Authorization` entirely.**"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_payload(
    text: str,
    *,
    model: str,
    system_prompt: str,
    max_tokens: int,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """The chat-completions body: system prompt + the user's prose."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        # Sent explicitly: unset, a reasoning model can spend the whole budget and return empty content.
        "max_tokens": max_tokens,
        # Omitted entirely when off, so a provider that rejects unknown body fields never sees the key.
        **({"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}),
    }


def content_from_completion(payload: Any, *, label: str = "rewrite") -> str:
    """`choices[0].message.content`, asserted to be a non-empty string."""
    if not isinstance(payload, dict):
        raise _no_content_error("the endpoint's JSON body was not an object", payload, label=label)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        detail = payload.get("error", payload)
        raise _no_content_error(
            "the endpoint answered HTTP 200 with no `choices`", detail, label=label
        )
    first = choices[0]
    # finish_reason "length" is a truncated 200; refusing it lets the caller use the full original.
    finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
    if finish_reason == "length":
        raise _no_content_error(
            "the endpoint truncated the answer at the token ceiling "
            f"(finish_reason: length) -- raise {label.upper()}_MAX_TOKENS or "
            "shorten the input; speaking a half-finished answer would hide "
            "the rest",
            first.get("message", {}).get("content", "")[:200]
            if isinstance(first.get("message"), dict)
            else first,
            label=label,
        )
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise _no_content_error(
            "the endpoint's first choice carried no usable message content",
            first,
            label=label,
        )
    return content.strip()


def _no_content_error(reason: str, detail: Any, *, label: str = "rewrite") -> HTTPException:
    return HTTPException(
        status_code=502,
        detail=(
            f"the {label} endpoint returned no usable text: {reason} "
            f"({str(detail)[:_UPSTREAM_DETAIL_CHARS]})"
        ),
    )


async def rewrite_via_chat_completions(
    text: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    timeout_s: float,
    max_tokens: int,
    disable_thinking: bool = False,
) -> str:
    """One POST to `{base}/chat/completions`; the rewritten text out."""
    url = endpoint_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                url,
                headers=build_headers(api_key),
                json=build_payload(
                    text,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"could not reach the rewrite endpoint at {url}: {exc}",
        ) from exc
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"the rewrite endpoint answered HTTP {response.status_code}: "
                f"{response.text[:_UPSTREAM_DETAIL_CHARS]}"
            ),
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"the rewrite endpoint answered non-JSON: {response.text[:_UPSTREAM_DETAIL_CHARS]}"
            ),
        ) from exc
    return content_from_completion(payload)


def configured_endpoint(settings: Settings) -> tuple[str, str]:
    """`(base_url, model)` from `REWRITE_BASE_URL` / `REWRITE_MODEL`, or a 503 naming the gap."""
    base_url = settings.rewrite_base_url.strip()
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail=(
                "prose rewrite is not configured: REWRITE_BASE_URL is empty. "
                "Point it at any OpenAI-compatible /chat/completions endpoint "
                "(e.g. https://openrouter.ai/api/v1, or a local server such as "
                "http://<llm-host>:8080/v1) in the gateway's .env, or pick a "
                "Hermes profile in Settings > Models & voice. This gateway "
                "never falls back to a provider you did not configure."
            ),
        )
    model = settings.rewrite_model.strip()
    if not model:
        raise HTTPException(
            status_code=503,
            detail=(
                "prose rewrite is not configured: REWRITE_MODEL is empty. Set "
                "it to a model the configured endpoint serves (ids are often "
                "namespaced, e.g. 'anthropic/claude-3.5-haiku')."
            ),
        )
    return base_url, model


async def rewrite_via_host_llm(
    app_state: Any,
    text: str,
    *,
    profile: str,
    system_prompt: str,
    max_tokens: int,
) -> str | None:
    """Ask the HOST to run the completion, on the profile's own model."""
    host_llm = getattr(app_state, "profile_llm", None)
    if host_llm is None:
        return None
    return await host_llm(
        profile=profile,
        system_prompt=system_prompt,
        text=text,
        max_tokens=max_tokens,
    )


async def resolve_rewrite_endpoint(
    app_state: Any, profile: str, *, api_key: str
) -> ProfileEndpoint:
    """The endpoint `REWRITE_PROFILE` names, or the HTTP error that says why not."""
    try:
        endpoint = await resolve_profile_endpoint(app_state, profile)
    except ProfileEndpointError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if endpoint.hosted and not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Hermes profile {profile!r} runs on {endpoint.provider}, which needs an API "
                "key, and the rewrite has none set. This gateway cannot read Hermes's own "
                f"key; set one for {endpoint.provider} in Settings > Models & voice > Rewrite "
                "for listening (REWRITE_API_KEY)."
            ),
        )
    return endpoint


@rewrite_router.post("/rewrite")
async def rewrite_prose(body: RewriteRequest, request: Request) -> dict[str, Any]:
    """Rewrite text so a TTS voice can read it aloud (P5-4a)."""
    settings: Settings = get_settings()
    text = body.text
    max_chars = settings.rewrite_max_input_chars
    if len(text) > max_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"text is {len(text)} characters; the rewrite cap is "
                f"{max_chars} (REWRITE_MAX_INPUT_CHARS)"
            ),
        )

    # Stays ahead of the config checks: the raw escape hatch must not depend on the provider.
    if body.depth == "raw":
        logger.info("rewrite raw passthrough: %d chars, style %s", len(text), body.style)
        return {
            "rewrite": {
                "text": text,
                "provider": "raw",
                "model": None,
                "profile": None,
                "style": body.style,
                "depth": "raw",
                "input_chars": len(text),
                "output_chars": len(text),
            }
        }

    api_key = settings.rewrite_api_key.get_secret_value().strip()
    profile = settings.rewrite_profile.strip()
    system_prompt = await resolve_system_prompt(request.app.state, settings, body.style, body.depth)
    max_tokens = max_tokens_for_depth(settings.rewrite_max_tokens, body.depth)

    if profile:
        hosted = await rewrite_via_host_llm(
            request.app.state,
            text,
            profile=profile,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        if hosted is not None:
            logger.info(
                "rewrote %d chars -> %d chars in style %s depth %s via the host LLM (profile %s)",
                len(text),
                len(hosted),
                body.style,
                body.depth,
                profile,
            )
            return {
                "rewrite": {
                    "text": hosted,
                    "provider": f"hermes ({profile})",
                    "model": None,
                    "profile": profile,
                    "style": body.style,
                    "depth": body.depth,
                    "input_chars": len(text),
                    "output_chars": len(hosted),
                }
            }
        endpoint = await resolve_rewrite_endpoint(request.app.state, profile, api_key=api_key)
        base_url, model = endpoint.base_url, endpoint.model
    else:
        base_url, model = configured_endpoint(settings)

    rewritten = await rewrite_via_chat_completions(
        text,
        base_url=base_url,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        timeout_s=settings.rewrite_timeout_s,
        max_tokens=max_tokens,
        disable_thinking=settings.rewrite_disable_thinking,
    )
    logger.info(
        "rewrote %d chars -> %d chars in style %s depth %s via %s (%s)%s",
        len(text),
        len(rewritten),
        body.style,
        body.depth,
        provider_label(base_url),
        model,
        f" for profile {profile!r}" if profile else "",
    )
    return {
        "rewrite": {
            "text": rewritten,
            "provider": provider_label(base_url),
            "model": model,
            "profile": profile or None,
            "style": body.style,
            "depth": body.depth,
            "input_chars": len(text),
            "output_chars": len(rewritten),
        }
    }


__all__ = [
    "DEFAULT_REWRITE_DEPTH",
    "DEFAULT_REWRITE_STYLE",
    "REWRITE_DEPTHS",
    "REWRITE_DEPTH_MAX_TOKENS_FRACTION",
    "REWRITE_DEPTH_SUFFIXES",
    "REWRITE_STYLES",
    "REWRITE_STYLE_PROMPT_SETTINGS",
    "RewriteDepth",
    "RewriteRequest",
    "RewriteStyle",
    "build_headers",
    "build_payload",
    "compose_prompt",
    "content_from_completion",
    "endpoint_url",
    "max_tokens_for_depth",
    "prompt_for_depth",
    "prompt_for_style",
    "provider_label",
    "resolve_system_prompt",
    "rewrite_prose",
    "rewrite_router",
    "rewrite_via_chat_completions",
    "rewrite_via_host_llm",
]
