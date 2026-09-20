"""`POST /api/sessions/{stored}/handoff` -- the opening prompt for a session that continues this one.

Owner ask, 2026-09-07: *"When in a chat session and I want to start a new
session continuing from where I left off ... analyze the last several
messages and come up with a good prompt to start a new session ... allows me
to select a new profile and will default to current project and current
profile."* And on the model: *"Maybe it could use the same AI as rewrite for
listening."*

This route is the analysis half. It reads the session's transcript, keeps
the last N things the user and the assistant said (`domain/handoff.py`), and
asks the rewrite endpoint to distil them into the message the owner will
send to a fresh session. It returns that text for the owner to READ AND EDIT
in the new-session sheet; it creates nothing and sends nothing to Hermes.
Creating the session is the existing `POST /api/projects/{id}/sessions/new`,
untouched, so a continuation is a normal new session that happens to open
with a good first message.

    POST /api/sessions/{stored}/handoff?profile=<name>
    {"last_n": 12, "source_title": "..."}          # both optional
    -> {"handoff": {"prompt", "title", "messages_used", "messages_available",
                    "truncated", "input_chars", "output_chars",
                    "provider", "model", "profile"}}

## Why the rewrite path and not a Hermes turn

The same three measurements that keep `api/rewrite.py` off Hermes: a turn on
the live instance is 2.5-15 minutes, completion signals arrive late (B-62),
and `prompt.submit` would write the request into the transcript being
continued. A sheet the owner just opened has to fill in seconds and must
leave the old session exactly as it was.

## What is shared, deliberately

Endpoint resolution (`REWRITE_PROFILE` first, else `REWRITE_BASE_URL` /
`REWRITE_MODEL`), the key, the timeout, the token ceiling, the input cap and
the no-content guard are all `api/rewrite.py`'s, called rather than copied:
one wire shape, one set of settings, one place to keep honest. Only the
prompt (`HANDOFF_PROMPT`, or a `handoff.md` file the app's editor can save)
and the window (`HANDOFF_LAST_MESSAGES`, overridable per request) are this
route's own. The completion call is reached through the `api.rewrite` module
attribute at call time, so the same monkeypatch that fakes the rewrite in
tests fakes this too.

## Error contract

| Status | When |
|---|---|
| 404 | unknown stored id (Hermes `[4001]`/`[4007]`, mapped by `_http_error_from_hermes`) |
| 422 | the session has no user/assistant text to continue from; or a bad `last_n` |
| 502 | the endpoint is unreachable, non-2xx, non-JSON, or 200-with-no-content; or Hermes failed |
| 503 | no rewrite endpoint configured, or the named profile cannot answer |

A 422 for an empty session mirrors Fork's rule (`nothing to branch -- send a
message first`): the app disables the action on a known-empty session and
reports this if it guessed wrong.
"""

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
    """Both fields optional; an empty body `{}` is the common case.

    `last_n` overrides `HANDOFF_LAST_MESSAGES` for this one request. Bounded
    the same way the setting is. `source_title` is the session's title as
    the app knows it -- the transcript does not carry it, and a second Hermes
    round trip (`session.list`) to learn what the caller already has would be
    waste -- and it only feeds the suggested title.
    """

    model_config = ConfigDict(extra="forbid")

    last_n: int | None = Field(default=None, ge=1, le=200)
    source_title: str | None = Field(default=None, max_length=500)


async def resolve_handoff_prompt(app_state: Any, settings: Settings) -> str:
    """`handoff.md` if the owner wrote one, else `HANDOFF_PROMPT`.

    The same resolution the speech prompts use (`domain/prompt_files.py`): the
    file is an override, never a requirement, and a missing, empty or
    unreadable file means the setting.
    """
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
            lambda: _with_live_handle(adapter, cache, stored_id, adapter.session_history, profile=profile),
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
    """Distil the last several messages into the opening prompt of a new session.

    `profile` is the profile the SOURCE session lives on (a stored id is
    unique only within a profile, B-136) -- not the profile the new session
    will use; that is the sheet's choice and this route never sees it.

    Reads only. Nothing here calls `prompt.submit`, `session.create` or
    anything else that writes, so a failed or abandoned handoff leaves no
    trace on Hermes.
    """
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

    # Assigned up front, NOT inside the fallback branch below: when the host
    # LLM answers, that branch never runs, and the logging line and the
    # response body both read these. Leaving them to the branch raised
    # UnboundLocalError and turned a working handoff into a 500.
    base_url = ""
    model: str | None = None
    provider_label = ""

    prompt: str | None = None
    if rewrite_profile:
        # Same route as the rewrite: inside Hermes the HOST runs it on the
        # profile's own model, which is the only thing that works for a
        # provider with no endpoint this process could call. This
        # screen shares `resolve_rewrite_endpoint`, so without it "Continue in
        # new session" hits the identical 503.
        prompt = await rewrite_mod.rewrite_via_host_llm(
            request.app.state,
            excerpt.text,
            profile=rewrite_profile,
            system_prompt=system_prompt,
            max_tokens=settings.rewrite_max_tokens,
        )
        if prompt is not None:
            # Name where it actually ran. There is no endpoint to label: the
            # host chose the transport.
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
