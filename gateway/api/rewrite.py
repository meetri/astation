"""Prose rewrite for speech: `POST /api/rewrite` (P5-4a).

One route: text in, prose out, shaped so a text-to-speech voice can read it
and a listener can follow it by ear:

    {"rewrite": {"text", "provider", "model", "style",
                 "input_chars", "output_chars"}}

JSON in, not multipart -- there is no upload here, only text:

    {"text": "...", "style": "listen"}   # `style` optional, default "listen"

## Three styles, three prompts

What the text IS changes what "read this aloud" should mean, so `style`
picks the system prompt and nothing else:

| `style` | Input | Setting |
|---|---|---|
| `listen` (default) | a chat reply, prose | `REWRITE_PROMPT` |
| `explain` | source code | `REWRITE_CODE_PROMPT` |
| `document` | one chunk of a long markdown/txt/PDF document | `REWRITE_DOC_PROMPT` |

`listen` and `document` share the faithful-conversion discipline B-91 was
corrected into: this is CONVERSION, not summarisation, and every finding,
number, name, decision and caveat survives (measured: 34/34 numeric facts
of a 931-word reply kept, where an earlier terse prompt kept 2/34 and the
owner noticed at once). `document` adds only what a chunk needs -- no
opening preamble, no closing summary, and no page furniture (page numbers,
running headers, hyphenation split across a line break), because the chunk
is spoken directly between its neighbours.

**`explain` is deliberately NOT faithful to every token, and that is the
point.** Read aloud literally, code is unlistenable -- brackets,
punctuation, import lists, identifiers spelled out letter by letter. So
that prompt describes instead: what the file is for, its functions,
classes and endpoints and what each does, the notable logic, and the
surprising parts (guards, error handling, security-relevant checks), while
keeping the concrete values that carry meaning (limits, timeouts, status
codes). The capability is measured, not assumed: on 2026-09-01 this very
module was fed through `listen` and came back a genuinely useful spoken
explanation -- `explain` exists to give that its own prompt rather than
borrow one tuned for prose.

An unknown `style` is a **422**, never a silent fallback to `listen`: a new
style means a new prompt, and quietly rewriting code with the prose prompt
is exactly the failure the enum prevents.

## Why this does not go through Hermes

Three independent measurements kill the Hermes-routed version: real turn
latency on the live instance is 2.5-15 minutes; B-62 completion signals
arrive 8-10 minutes late; and `prompt.submit` would write the rewrite into
the user's transcript as if they had asked for it. A speak button has to
answer in seconds and must not touch the conversation. So the gateway calls
a chat-completions endpoint itself.

## One implementation, a configurable base URL -- deliberately

`REWRITE_BASE_URL` points at any **OpenAI-compatible** `/chat/completions`
endpoint. That is the whole provider abstraction: OpenRouter today, and the
a self-hosted server on your own LAN (llama.cpp, vLLM, Ollama,
LM Studio — anything speaking that wire shape, typically with no key required)
is the same setting. Moving the rewrite between hosted and local hardware is a
`.env` edit with **no code change**. A second provider
branch would have to be kept honest twice for no gain; there is exactly one
wire shape here, and it is the one every server in that list speaks.

Consequences of that choice, made explicit:

* **An empty `REWRITE_API_KEY` is normal, not an error.** A keyless local
  server gets no `Authorization` header at all -- sending `Bearer ` would be
  a 401 from some servers and an odd log line on the rest. Only an empty
  `REWRITE_BASE_URL` (or `REWRITE_MODEL`) disables the feature, and it does
  so with a 503 naming the variable -- never a silent fallback to a provider
  the operator did not configure (the `api/transcribe.py` rule).
* **Model ids are namespaced with a slash** (`anthropic/claude-3.5-haiku`,
  `qwen/qwen3-27b`). Nothing here validates the model against a slash-free
  pattern; it is an opaque string forwarded to the endpoint, which is the
  only authority on whether it exists.
* `provider` in the response is the **host (and port) of the configured base
  URL** -- `openrouter.ai`, or a LAN `host:port` -- so the app can show which
  box actually answered. Never the path, never anything derived from the key.

## The guard that matters most (review §2, wave 1)

**OpenRouter can answer HTTP 200 with an `error` key and no `choices`.** A
naive `payload["choices"][0]["message"]["content"]` reader would then hand
the app an empty string and the speak button would silently say nothing --
indistinguishable, to the operator, from a broken feature. So
`content_from_completion()` asserts a non-empty string is really there and
raises **502 carrying the upstream detail** otherwise. Every no-content
shape (an `error` body, an empty `choices` list, a null/blank `content`)
lands on that same honest 502.

## Error contract

| Status | When |
|---|---|
| 413 | `text` longer than `REWRITE_MAX_INPUT_CHARS` (default 24000) |
| 422 | `text` empty/whitespace, or an unknown `style` |
| 502 | endpoint unreachable, non-2xx, non-JSON, or 200-with-no-content |
| 503 | `REWRITE_BASE_URL` / `REWRITE_MODEL` unset (feature disabled) |

Tests inject a fake by monkeypatching the module-level
`rewrite_via_chat_completions`, and exercise the real one against a mock
OpenAI-compatible server on loopback (`test_rewrite.py`) -- no test reaches
the internet or spends a token.
"""

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

#: Authenticated routes (mounted under `/api` in `api.main`).
rewrite_router = APIRouter(tags=["rewrite"])

#: How much of an upstream error body is echoed into a 502 detail. The
#: bodies are short structured JSON in practice and never contain the API key
#: (it travels in a request header); bounded anyway, same as
#: `api/transcribe.py::transcribe_cloud`.
_UPSTREAM_DETAIL_CHARS = 300

#: The style whose prompt is the historical default, used when the body omits
#: `style` entirely -- so an older client that never learned about styles keeps
#: getting exactly the rewrite it got before.
DEFAULT_REWRITE_STYLE = "listen"

#: P6-2: "the default is to give a very clear pros
#: reconstruction ... if I tap and hold ... brief ... medium ... full ... or
#: raw." `depth` is ORTHOGONAL to `style` -- style picks WHICH prompt (chat
#: reply / source / document chunk), depth picks HOW MUCH of it survives. The
#: two compose: "explain this source file, briefly" is a real request.
#:
#: `full` is the historical, and default, behaviour -- the faithful
#: conversion B-91 was corrected into (34/34 facts kept). It is the operator's
#: "very clear pros reconstruction" and what every older client already gets,
#: so making it the default costs an old client nothing.
#:
#: `raw` is NOT "a very short rewrite" -- it skips the rewrite call ENTIRELY
#: (see `rewrite_prose` below) and returns the source text verbatim, because
#: it is the escape hatch for when the rewrite is wrong or the configured
#: model is down, and an escape hatch that depends on the thing it is
#: escaping from is not one.
DEFAULT_REWRITE_DEPTH = "full"

#: `depth` -> the extra instruction appended to the style's system prompt.
#: A suffix, not a separate prompt per depth: `brief`/`medium` must keep the
#: SAME faithfulness discipline (never invent, never drop a fact silently)
#: and only the SELECTION of what survives changes with length -- writing
#: three separate prompts risks each depth quietly drifting its own rules.
#: `full` and `raw` need no suffix: `full` IS the base prompt's own
#: instruction, and `raw` never reaches a prompt at all.
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

#: `depth` -> the `REWRITE_MAX_TOKENS`-relative ceiling for that depth, as a
#: fraction. `full` keeps the server's own configured ceiling (whatever the
#: owner tuned `REWRITE_MAX_TOKENS` to); `brief`/`medium` are capped tighter
#: so a REASONING model cannot spend the whole budget on `reasoning_content`
#: and get truncated mid-sentence on what was supposed to be the SHORT
#: option -- see `REWRITE_MAX_TOKENS`'s own comment on why that budget is
#: generous. Fractions rather than fixed token counts so raising the base
#: ceiling raises every depth with it.
REWRITE_DEPTH_MAX_TOKENS_FRACTION: dict[str, float] = {
    "brief": 0.35,
    "medium": 0.65,
}

#: Every depth this gateway knows how to ask for. `raw` is included so the
#: schema accepts it even though the route never builds a prompt for it.
REWRITE_DEPTHS = ("brief", "medium", "full", "raw")

RewriteDepth = Literal["brief", "medium", "full", "raw"]

#: `style` -> the `Settings` attribute holding that style's system prompt. The
#: single source of truth for both the request Literal and the route, so a
#: style can never exist in the schema without a prompt behind it (a test
#: asserts the two agree).
REWRITE_STYLE_PROMPT_SETTINGS: dict[str, str] = {
    "listen": "rewrite_prompt",  # REWRITE_PROMPT -- prose, faithful conversion
    "explain": "rewrite_code_prompt",  # REWRITE_CODE_PROMPT -- source code
    "document": "rewrite_doc_prompt",  # REWRITE_DOC_PROMPT -- one doc chunk
}

#: Every style this gateway knows how to ask for. A new style means a new
#: prompt, not a silently-ignored field -- so an unknown value is a 422.
REWRITE_STYLES = tuple(REWRITE_STYLE_PROMPT_SETTINGS)

#: The request schema's enum. Spelled out rather than built from the tuple
#: because `Literal[...]` needs literal members at type-check time; the test
#: `test_every_style_has_a_prompt_setting` keeps the two in lockstep.
RewriteStyle = Literal["listen", "explain", "document"]


class RewriteRequest(BaseModel):
    """Body for `POST /api/rewrite`.

    Closed schema like every other body in this service: a typo'd field is a
    422, never a silently-dropped instruction.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    #: Optional. `listen` is read-aloud prose (the default), `explain`
    #: describes SOURCE CODE, `document` speaks one chunk of a longer
    #: document. Typed as an enum so an unknown style is refused rather than
    #: quietly rewritten with the wrong prompt -- speaking code through the
    #: prose prompt is a bad answer, not a near-miss.
    style: RewriteStyle = DEFAULT_REWRITE_STYLE
    #: Optional, P6-2. How much of the source survives: `brief` / `medium` /
    #: `full` (the default -- today's faithful conversion, unchanged) /
    #: `raw` (no rewrite at all; the route returns the source verbatim
    #: without calling a model). Orthogonal to `style`.
    depth: RewriteDepth = DEFAULT_REWRITE_DEPTH

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        return reject_blank_text(value)


def prompt_for_style(settings: Settings, style: str) -> str:
    """The configured system prompt for `style`.

    Each style reads its OWN `REWRITE_*` setting, so the operator can tune any
    one of them from `.env` without a rebuild and without disturbing the other
    two. An unknown style cannot reach here -- the request schema refuses it
    with a 422 first -- but it raises rather than falling back, because a
    silent fallback is the one behaviour this route must never have.
    """
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
    """`base_prompt` (already resolved for `style`) plus `depth`'s suffix.

    `full` returns `base_prompt` unchanged -- it is not "no suffix", it is
    the base prompt's OWN instruction, which already asks for the complete
    faithful conversion. `raw` never reaches here; the route short-circuits
    before building a prompt at all.
    """
    return compose_prompt(base_prompt, REWRITE_DEPTH_SUFFIXES.get(depth, ""))


def compose_prompt(base_prompt: str, suffix: str) -> str:
    """`base_prompt` plus `suffix`, which may be empty. The one place the two
    halves of a system prompt are joined, so the file-backed path and the
    settings path cannot disagree about the join."""
    return base_prompt + suffix if suffix else base_prompt


async def resolve_system_prompt(app_state: Any, settings: Settings, style: str, depth: str) -> str:
    """The system prompt for `style` at `depth`, files first.

    Each half is independently overridable by a file the app's editor can
    save (`domain/prompt_files.py`): the style prompt by `listen.md` /
    `document.md` / `code.md`, the depth suffix by `depth-brief.md` /
    `depth-medium.md`. A missing, empty or unreadable file means "use the
    setting", so this is exactly `prompt_for_depth(prompt_for_style(...))`
    on a gateway with no prompt files -- which is every gateway until the
    owner writes one.

    `full` and `raw` have no depth file by construction: `full` IS the style
    prompt's own instruction, and `raw` never reaches a prompt at all.
    """
    store = getattr(app_state, "prompt_files", None)
    base = prompt_for_style(settings, style)
    suffix = REWRITE_DEPTH_SUFFIXES.get(depth, "")
    if store is None:
        return compose_prompt(base, suffix)
    base = await store.text(PROMPT_FILE_FOR_STYLE.get(style)) or base
    file_suffix = await store.text(PROMPT_FILE_FOR_DEPTH.get(depth))
    if file_suffix is not None:
        # A depth file replaces the built-in suffix outright. It is prefixed
        # with a space for the same reason the built-ins are: it is appended
        # to a prompt that ends in a full stop.
        suffix = file_suffix if file_suffix.startswith(" ") else " " + file_suffix
    return compose_prompt(base, suffix)


def max_tokens_for_depth(base_max_tokens: int, depth: str) -> int:
    """`base_max_tokens` scaled by `depth`'s fraction, floored at 1.

    Never zero: a `max_tokens: 0` request is a malformed request to most
    OpenAI-compatible servers, not a valid "give me nothing back".
    """
    fraction = REWRITE_DEPTH_MAX_TOKENS_FRACTION.get(depth)
    if fraction is None:
        return base_max_tokens
    return max(1, int(base_max_tokens * fraction))


def endpoint_url(base_url: str) -> str:
    """`{base}/chat/completions`, tolerating a trailing slash on the base.

    The base URL is expected to already include the API version segment the
    server expects (`.../v1`), exactly as every OpenAI-compatible client
    takes it -- this function does not invent one.
    """
    return f"{base_url.rstrip('/')}/chat/completions"


def provider_label(base_url: str) -> str:
    """The configured endpoint's `host[:port]`, for the response's `provider`.

    Honest and non-branded: whatever is actually serving. Falls back to the
    raw configured string when it has no parseable netloc, so a misconfigured
    URL is visible rather than blank.
    """
    netloc = urlsplit(base_url).netloc
    return netloc or base_url


def build_headers(api_key: str) -> dict[str, str]:
    """Request headers. **An empty key omits `Authorization` entirely.**

    Local llama.cpp / vLLM servers are keyless; `Bearer ` with nothing after
    it is worse than no header at all (some servers 401 on it). The key is
    never logged here or anywhere else.
    """
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
    """The chat-completions body: system prompt + the user's prose.

    `stream` is deliberately absent (the default is false): the app wants one
    finished string to hand to AVSpeechSynthesizer, and a streamed answer
    would buy nothing but partial-frame handling.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        # Sent explicitly rather than left to the server's default: measured
        # against a local reasoning model, a small budget is spent entirely on
        # `reasoning_content` and `content` comes back EMPTY (see
        # `Settings.rewrite_max_tokens`). Leaving it unset makes the result
        # depend on whichever server the operator points at that week.
        "max_tokens": max_tokens,
        # Qwen-family template switch; see `Settings.rewrite_disable_thinking`.
        # Omitted entirely when off, so a provider that rejects unknown body
        # fields never sees it.
        **({"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}),
    }


def content_from_completion(payload: Any, *, label: str = "rewrite") -> str:
    """`choices[0].message.content`, asserted to be a non-empty string.

    `label` names the caller in the 502 detail, so `api/converse.py` -- which
    speaks the identical wire shape to the identical class of endpoint and
    reuses this guard rather than growing a second copy of it to keep honest --
    reports "the converse endpoint" and not "the rewrite endpoint". It changes
    nothing else, and the default keeps every existing message byte-identical.

    **This is the guard the whole route exists around** (module docstring):
    an OpenRouter 200 carrying `{"error": {...}}` and no `choices` must
    become a 502 with the upstream detail, not an empty rewrite that makes
    the speak button silently say nothing.

    Raises `HTTPException(502)` for every shape that is not usable text: a
    non-dict body, an `error` body, missing/empty `choices`, a non-dict
    `message`, or `content` that is absent, not a string, or blank. A
    content that arrives as a list of parts (an Anthropic-native shape, not
    the OpenAI-compatible one this route speaks) is also refused rather than
    guessed at -- if a server ever answers that way, it gets handled with a
    measurement, not an assumption.
    """
    if not isinstance(payload, dict):
        raise _no_content_error("the endpoint's JSON body was not an object", payload, label=label)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        # The measured OpenRouter failure mode: HTTP 200, an `error` key, no
        # `choices`. Surface the upstream's own words -- they name the real
        # cause (no credits, bad model id, rate limit).
        detail = payload.get("error", payload)
        raise _no_content_error(
            "the endpoint answered HTTP 200 with no `choices`", detail, label=label
        )
    first = choices[0]
    # a response cut off at the token ceiling arrives as a normal 200
    # with `finish_reason: "length"` and a half-finished final sentence. The
    # route used to hand that straight to the synthesizer, so the operator heard
    # "less than 25% of the message" and had no way to know it was truncated.
    # Refusing it is the honest move AND the better outcome: the app's
    # fallback then speaks the FULL original verbatim, so nothing is lost --
    # where speaking the fragment would have silently swallowed the rest.
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
    """One POST to `{base}/chat/completions`; the rewritten text out.

    Module-level on purpose: this is the injection point the route tests
    replace with a fake, the same way `api/transcribe.py` exposes
    `transcribe_local`. The real function is itself covered against a mock
    OpenAI-compatible server on loopback.

    Every failure is a 502 that names what happened: transport failure,
    non-2xx (with the bounded upstream body), non-JSON, or the
    200-with-no-content case `content_from_completion` catches.
    """
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
    """Ask the HOST to run the completion, on the profile's own model.

    Returns the rewritten text, or `None` when no host LLM is available — in
    which case the caller falls back to the HTTP path below.

    This exists because resolving a profile to an OpenAI-compatible base URL
    plus a key of the gateway's own cannot work for every provider: a
    profile on a local llama.cpp server, on `bedrock`, `moa` or `openai-codex`
    has no such endpoint the gateway can call, and the honest answer was a 503.
    Hermes already knows how to talk to every provider it is configured with,
    including those, so when the gateway runs INSIDE Hermes the right move is
    to hand it the messages and let it choose the transport.

    Installed by the plugin (`plugin/dashboard/plugin_api.py`); absent in the
    standalone gateway, where this returns None and nothing changes.
    """
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
    """The endpoint `REWRITE_PROFILE` names, or the HTTP error that says why not.

    A hosted provider (OpenRouter, OpenAI, Anthropic) with no `REWRITE_API_KEY`
    is refused HERE, before any request goes out: the upstream would answer
    401 and the 502 that turned into would read as "the endpoint is broken"
    when the fix is one field in the settings screen.
    """
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
    """Rewrite text so a TTS voice can read it aloud (P5-4a).

    Body: `{"text": str, "style": "listen"|"explain"|"document"}` (`style`
    optional, default `"listen"`; an unknown value is a 422, never a silent
    fallback). The style selects the system prompt and nothing else --
    `REWRITE_PROMPT` for prose, `REWRITE_CODE_PROMPT` for source code,
    `REWRITE_DOC_PROMPT` for one chunk of a long document.

    Response: `{"rewrite": {"text", "provider", "model", "style",
    "input_chars", "output_chars"}}` -- `provider` is the configured
    endpoint's host, so the app can show whether OpenRouter or the LAN box
    answered, and `style` is the style actually used, so it can show that too.

    Endpoint, model, system prompt, size cap and timeout all come from server
    settings (`REWRITE_*`); a request cannot choose them, the same rule
    `POST /api/transcribe` follows. The full error contract is in the module
    docstring -- and note the two that are easy to get wrong: an **empty
    `REWRITE_API_KEY` is allowed** (keyless local servers), while an empty
    `REWRITE_BASE_URL` disables the feature with a 503 naming it.
    """
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

    # P6-2: `raw` is the escape hatch, so it must not depend on the thing it
    # escapes from. Deliberately BEFORE the REWRITE_BASE_URL/REWRITE_MODEL
    # checks below, not just before the model call -- an earlier version of
    # this route put the raw branch after those checks, so `raw` still 503'd
    # whenever the rewrite provider was unconfigured or down, which defeats
    # the entire point of an escape hatch. It always succeeds if the request
    # itself was valid, regardless of REWRITE_* configuration.
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

    # May be empty -- `build_headers` then sends no Authorization header.
    api_key = settings.rewrite_api_key.get_secret_value().strip()
    profile = settings.rewrite_profile.strip()
    system_prompt = await resolve_system_prompt(
        request.app.state, settings, body.style, body.depth
    )
    max_tokens = max_tokens_for_depth(settings.rewrite_max_tokens, body.depth)

    if profile:
        # Inside Hermes, let the HOST run it on the profile's own model. That
        # works for providers the gateway could never call itself -- a local
        # server, bedrock, moa, openai-codex -- which is the whole of B-195.
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
                len(text), len(hosted), body.style, body.depth, profile,
            )
            # EXACTLY the shape every other branch of this route returns:
            # `{"rewrite": {...}}` with those eight keys. The app decodes a
            # `{"rewrite": …}` wrapper (`RewriteClient.swift`), so inventing a
            # different envelope here 200s and then fails in the decoder with
            # "Key 'rewrite' not found" -- which reads to the user as a broken
            # gateway rather than a shape mismatch.
            return {
                "rewrite": {
                    "text": hosted,
                    # Named so the Test row can say where it actually ran,
                    # rather than an endpoint host that was never dialled.
                    "provider": f"hermes ({profile})",
                    "model": None,
                    "profile": profile,
                    "style": body.style,
                    "depth": body.depth,
                    "input_chars": len(text),
                    "output_chars": len(hosted),
                }
            }
        # A standalone gateway: resolve an endpoint and call it directly.
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
            # The profile that chose the endpoint and model, or null when the
            # fields did -- so the app's Test can say "gemma (gemma-4-12b-qat)".
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
