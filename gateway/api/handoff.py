"""`POST /api/sessions/{stored}/handoff` -- the opening prompt for a session that continues this one."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

import api.rewrite as rewrite_mod
from adapters.hermes import HermesAdapter, HermesError
from config.settings import Settings, get_settings
from domain.handoff import HandoffExcerpt, build_excerpt, suggested_title
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)
from domain.live_handles import LiveHandleCache
from domain.prompt_files import HANDOFF_PROMPT_FILE
from domain.transcript import _transcript

logger = logging.getLogger(__name__)

handoff_router = APIRouter(tags=["handoff"])


class HandoffRequest(BaseModel):
    """Both fields optional; an empty body `{}` is the common case."""

    model_config = ConfigDict(extra="forbid")

    last_n: int | None = Field(default=None, ge=1, le=200)
    source_title: str | None = Field(default=None, max_length=500)


async def resolve_handoff_prompt(app_state: Any, settings: Settings) -> str:
    """`handoff.md` if the operator wrote one, else `HANDOFF_PROMPT`."""
    store = getattr(app_state, "prompt_files", None)
    base = settings.handoff_prompt
    if store is None:
        return base
    return await store.text(HANDOFF_PROMPT_FILE) or base


async def _session_messages(app_state: Any, profile: str, stored_id: str) -> Any:
    """The session's transcript rows, by stored id -> live handle, exactly as
    `GET /sessions/{id}/messages` resolves them."""
    adapter: HermesAdapter = resolve_profile_adapter(app_state, profile)
    cache: LiveHandleCache = resolve_live_handle_cache(app_state, profile)
    try:
        _live_id, history = await _with_reconnect(
            app_state,
            adapter,
            lambda: _with_live_handle(
                adapter, cache, stored_id, adapter.session_history, profile=profile
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc
    _count, messages = _transcript(history, "count")
    return messages


def excerpt_or_422(messages: Any, *, last_n: int, max_chars: int) -> HandoffExcerpt:
    """`build_excerpt`, refusing a session with nothing to continue from."""
    excerpt = build_excerpt(messages, last_n=last_n, max_chars=max_chars)
    if excerpt.messages_used == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "nothing to continue from: this conversation has no messages "
                "yet (tool calls and notices do not count). Send a message first."
            ),
        )
    return excerpt


@handoff_router.post("/sessions/{stored_session_id}/handoff")
async def session_handoff(
    stored_session_id: str,
    body: HandoffRequest,
    request: Request,
    profile: str = Query(default="default"),
) -> dict[str, Any]:
    """Distil the last several messages into the opening prompt of a new session."""
    settings: Settings = get_settings()
    stored_id = _validate_stored_session_id(stored_session_id)
    messages = await _session_messages(request.app.state, profile, stored_id)
    excerpt = excerpt_or_422(
        messages,
        last_n=body.last_n or settings.handoff_last_messages,
        max_chars=settings.rewrite_max_input_chars,
    )

    api_key = settings.rewrite_api_key.get_secret_value().strip()
    rewrite_profile = settings.rewrite_profile.strip()
    system_prompt = await resolve_handoff_prompt(request.app.state, settings)

    base_url = ""
    model: str | None = None
    provider_label = ""

    prompt: str | None = None
    if rewrite_profile:
        prompt = await rewrite_mod.rewrite_via_host_llm(
            request.app.state,
            excerpt.text,
            profile=rewrite_profile,
            system_prompt=system_prompt,
            max_tokens=settings.rewrite_max_tokens,
        )
        if prompt is not None:
            provider_label = f"hermes ({rewrite_profile})"

    if prompt is None:
        if rewrite_profile:
            endpoint = await rewrite_mod.resolve_rewrite_endpoint(
                request.app.state, rewrite_profile, api_key=api_key
            )
            base_url, model = endpoint.base_url, endpoint.model
        else:
            base_url, model = rewrite_mod.configured_endpoint(settings)
        provider_label = rewrite_mod.provider_label(base_url)
        prompt = await rewrite_mod.rewrite_via_chat_completions(
            excerpt.text,
            base_url=base_url,
            model=model,
            api_key=api_key,
            system_prompt=system_prompt,
            timeout_s=settings.rewrite_timeout_s,
            max_tokens=settings.rewrite_max_tokens,
            disable_thinking=settings.rewrite_disable_thinking,
        )
    logger.info(
        "handoff for %s: %d of %d messages, %d chars -> %d chars via %s (%s)%s%s",
        stored_id,
        excerpt.messages_used,
        excerpt.messages_available,
        len(excerpt.text),
        len(prompt),
        provider_label,
        model,
        f" for profile {rewrite_profile!r}" if rewrite_profile else "",
        " (truncated)" if excerpt.truncated else "",
    )
    return {
        "handoff": {
            "prompt": prompt,
            "title": suggested_title(body.source_title),
            "messages_used": excerpt.messages_used,
            "messages_available": excerpt.messages_available,
            "truncated": excerpt.truncated,
            "input_chars": len(excerpt.text),
            "output_chars": len(prompt),
            "provider": provider_label,
            "model": model,
            "profile": rewrite_profile or None,
        }
    }


__all__ = ["HandoffRequest", "handoff_router", "resolve_handoff_prompt", "session_handoff"]
