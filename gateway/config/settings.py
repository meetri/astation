"""Research Gateway settings.

Loaded from the repo-root `.env` (four directories above this file), never from
a copy inside `services/research-gateway/`.

`HERMES_PASSWORD` must never be printed or logged. It is typed as `SecretStr`
(so `str()`/default `repr()` never leak the raw value) and `Settings.__repr__`
is additionally overridden to omit it entirely, so debugging tools that call
`repr(settings)` cannot leak it even indirectly via a masked placeholder.

`RESEARCH_GATEWAY_PASSWORD` gets exactly the same treatment. Note the two
credential pairs are different things and must not be conflated:
`HERMES_USERNAME`/`HERMES_PASSWORD` are what this service presents *upstream*
to Hermes; `RESEARCH_GATEWAY_USERNAME`/`RESEARCH_GATEWAY_PASSWORD` are this
service's *own* inbound credential, which clients (the phone app) present to
it -- see `api/auth.py`.

## Three layers, and `.env` is the middle one

    defaults  ->  .env  ->  runtime overlay

`get_settings()` applies a **runtime configuration overlay** on top of what is
loaded here (`config/runtime_config.py`, `api/config.py`): a small JSON file the
gateway owns and writes, so the owner can change which endpoint, which model
and which key are in force **from the app**, with no file edit and no restart.
`.env` keeps its job unchanged -- it is the floor, it is hand-maintained, and
nothing in this service ever writes to it.

Precedence works because pydantic-settings ranks **init kwargs above every
other source**: `Settings(**overlay)` therefore beats both the process
environment and `.env`, and removing a key from the overlay falls straight back
to whichever of those provided it. Constructing `Settings()` with no kwargs is
still exactly the old two-layer behaviour, which is what every test that wants
`.env`-only semantics uses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import runtime_config

logger = logging.getLogger(__name__)

# Repo root is four levels up from this file:
#   services/research-gateway/config/settings.py -> config -> services/research-gateway
#   -> services -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _REPO_ROOT / ".env"

#: The `.env` this service reads, exported so `api/config.py` can report which
#: values the owner has written down there (presence only, never a value).
ENV_FILE = _ENV_FILE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Hermes connection
    hermes_scheme: str = Field(alias="HERMES_SCHEME", default="http")
    hermes_host: str = Field(alias="HERMES_HOST", default="127.0.0.1")
    hermes_port: int = Field(alias="HERMES_PORT", default=9119)
    hermes_username: str = Field(alias="HERMES_USERNAME", default="")
    hermes_password: SecretStr = Field(alias="HERMES_PASSWORD", default=SecretStr(""))

    # Research Gateway's own inbound credential (HTTP Basic, see api/auth.py).
    # Distinct from the Hermes credentials above. Empty by default so a
    # misconfigured deployment fails *closed* (auth.py rejects every request)
    # rather than serving the LAN unauthenticated.
    research_gateway_username: str = Field(alias="RESEARCH_GATEWAY_USERNAME", default="")
    research_gateway_password: SecretStr = Field(
        alias="RESEARCH_GATEWAY_PASSWORD", default=SecretStr("")
    )

    # The Hermes sandbox root the gateway's own path validation confines
    # `/api/sandbox/*` requests to (P3-1a). Hermes enforces its own
    # `locked_root` upstream (measured: 403 outside `/opt/data`, PV "Phase 3
    # probe"), but the gateway re-validates against this value regardless --
    # defense in depth, and because the upstream routes are undocumented and a
    # Hermes upgrade could change that confinement silently.
    hermes_sandbox_root: str = Field(alias="HERMES_SANDBOX_ROOT", default="/opt/data")

    # B-61: Hermes-internal churn the sandbox-diff auto-promoter must never
    # ingest. Empty (the default) means "use the built-in defaults" in
    # api/artifacts.py (SANDBOX_DIFF_DENYLIST_DIRS /
    # SANDBOX_DIFF_DENYLIST_FILENAME_GLOBS -- the rationale lives there);
    # a non-empty value is comma-separated and REPLACES the corresponding
    # default wholesale, so a future Hermes layout change is a config edit,
    # not a code edit. Dirs are top-level names under the sandbox root
    # (trailing slash tolerated: "logs/" == "logs"); globs are fnmatch
    # patterns applied to basenames.
    hermes_sandbox_denylist_dirs: str = Field(alias="HERMES_SANDBOX_DENYLIST_DIRS", default="")
    hermes_sandbox_denylist_globs: str = Field(alias="HERMES_SANDBOX_DENYLIST_GLOBS", default="")

    # The sandbox directory the P3-3 priming turn curls composer attachments
    # into. Must sit inside `hermes_sandbox_root` (validated at use); kept
    # separate from `probe_scratch` so user attachments are never mixed into
    # probe cleanup sweeps.
    hermes_attachment_dir: str = Field(
        alias="HERMES_ATTACHMENT_DIR", default="/opt/data/attachments"
    )

    # Extra roots `GET /api/sandbox/text` may open READ-ONLY, beyond
    # `hermes_sandbox_root`. Comma-separated absolute paths; `/` means the
    # whole filesystem the Hermes container can see. Empty (the default)
    # leaves the viewer confined exactly like every other route.
    #
    # **Why this is not just a wider `HERMES_SANDBOX_ROOT`.** That value is not
    # only a permission: `domain/artifact_ingest.py` WALKS it on every sandbox
    # diff, so widening it to read one file would point the auto-ingest sweep
    # at the whole filesystem. This setting is read by nothing but the viewer.
    #
    # **Read-only is enforced, not advisory.** A path admitted only by this
    # setting comes back `editable: false`, and `PUT /api/sandbox/text` keeps
    # running the strict `hermes_sandbox_root` rule, so no value here can widen
    # what the app is able to write. Upstream reality is why the split matters:
    # Hermes's `fs/read-text` confines nothing (measured: it reads
    # `/opt/hermes/...` and `/etc/hostname`), while its `/api/files*` browse and
    # download routes enforce their own `locked_root` and 403 outside it -- so
    # this widens the text viewer only, never the file browser.
    #
    # Anyone holding `RESEARCH_GATEWAY_PASSWORD` can read anything this admits.
    research_gateway_view_roots: str = Field(alias="RESEARCH_GATEWAY_VIEW_ROOTS", default="")

    # Base URL the HERMES SANDBOX HOST can reach this gateway at, for the
    # P3-3 capability serve URL the priming turn curls (e.g.
    # "http://192.168.1.10:8000"). Empty (the default) derives it from the
    # inbound request's Host header, which is correct whenever the phone and
    # the Hermes host both reach the gateway at the same address.
    # Set it explicitly if the gateway ever sits behind a proxy whose Host
    # the sandbox cannot resolve.
    research_gateway_public_base_url: str = Field(
        alias="RESEARCH_GATEWAY_PUBLIC_BASE_URL", default=""
    )

    # Speech-to-text (P5-2a, `api/transcribe.py`). Aliases mirror Hermes's own
    # transcription-tool conventions (`STT_*`, provider names
    # `local`/`openai`/`groq`), so config knowledge transfers if Hermes's
    # voice stack is ever enabled. The two API keys get the same SecretStr
    # treatment as the passwords above and are masked in `__repr__` below.
    stt_provider: str = Field(alias="STT_PROVIDER", default="local")
    stt_local_model: str = Field(alias="STT_LOCAL_MODEL", default="small")
    stt_language: str = Field(alias="STT_LANGUAGE", default="en")
    # Server-configured jargon hint, prepended to any per-request vocab_hint
    # and passed to Whisper as `initial_prompt` (local) / `prompt` (cloud).
    stt_vocab_hint: str = Field(alias="STT_VOCAB_HINT", default="")
    stt_openai_key: SecretStr = Field(alias="STT_OPENAI_KEY", default=SecretStr(""))
    stt_groq_key: SecretStr = Field(alias="STT_GROQ_KEY", default=SecretStr(""))
    stt_openai_model: str = Field(alias="STT_OPENAI_MODEL", default="whisper-1")
    stt_groq_model: str = Field(alias="STT_GROQ_MODEL", default="whisper-large-v3-turbo")

    # Prose rewrite for speech (P5-4a, `api/rewrite.py`). Mirrors the `STT_*`
    # block above. ONE implementation -- an OpenAI-compatible
    # `/chat/completions` endpoint -- behind a configurable base URL, so
    # moving from a hosted router to a model on your own LAN is a
    # config edit and not a code change (measured: a keyless llama.cpp
    # server on a LAN serving a local model).
    #
    # `REWRITE_BASE_URL` empty (the default) cleanly DISABLES the feature:
    # the route answers an honest 503 naming the variable, and never falls
    # back to some other provider the owner did not configure.
    rewrite_base_url: str = Field(alias="REWRITE_BASE_URL", default="")
    rewrite_model: str = Field(alias="REWRITE_MODEL", default="anthropic/claude-3.5-haiku")
    # May legitimately be EMPTY: a local llama.cpp/vLLM server needs no key,
    # and an empty key means "send no Authorization header" -- not a 503.
    # SecretStr + the `__repr__` override below for the same reason as every
    # other credential here.
    rewrite_api_key: SecretStr = Field(alias="REWRITE_API_KEY", default=SecretStr(""))
    # The name of a Hermes profile whose provider and model answer the rewrite
    # INSTEAD of REWRITE_BASE_URL / REWRITE_MODEL (owner ask 2026-09-07: "I'd
    # prefer if I can select one of my profiles"). Resolved on every request
    # from `profiles.list` + `model.options` (`domain/profile_endpoint.py`),
    # so re-pointing the profile in the Agents screen re-points the rewrite
    # too. The key is still REWRITE_API_KEY: Hermes never exposes its own.
    rewrite_profile: str = Field(alias="REWRITE_PROFILE", default="")
    # The system prompt, overridable so the owner can tune the rewrite by ear
    # from `.env` without a rebuild.
    rewrite_prompt: str = Field(
        alias="REWRITE_PROMPT",
        default=(
            "Rewrite the user's text so it can be READ ALOUD and "
            "understood by ear alone. This is CONVERSION TO SPEAKABLE "
            "PROSE, not summarisation. Keep EVERY finding, measurement, "
            "number, name, decision, caveat and recommendation that appears "
            "in the input -- if the input states a value, your rewrite "
            "states it too. Never drop a fact because it seems minor. What "
            "you remove is only what cannot be heard or does not carry "
            "meaning: markup, repetition, hedging, throat-clearing, and "
            "narration of the process. Write densely -- pack related facts "
            "into the same sentence rather than giving each its own. Lead "
            "with the outcome, then the evidence that supports it. Turn code "
            "fences, tables and lists into flowing spoken sentences, and "
            "describe code by what it does. Say paths, symbols and "
            "abbreviations the way a person would say them aloud, and never "
            "spell identifiers letter by letter. No preamble and no "
            "commentary about the rewrite itself. Reply with the spoken text "
            "only, and always finish your final sentence."
        ),
    )
    # `style: "explain"` -- SOURCE CODE. A separate prompt because the
    # faithful-conversion discipline above is exactly wrong here: read aloud,
    # code is unlistenable (punctuation, brackets, import lists), so this one
    # DESCRIBES what the code does instead of converting it token for token.
    # Overridable from `.env` like every other prompt here.
    rewrite_code_prompt: str = Field(
        alias="REWRITE_CODE_PROMPT",
        default=(
            "The user's text is SOURCE CODE. Produce a spoken explanation of "
            "it that a listener can follow with nothing in front of them. "
            "This is DESCRIPTION, not transcription -- reading code aloud "
            "token by token is useless. Say what the file or section is for, "
            "then walk its main structures -- functions, classes, endpoints, "
            "data types, configuration -- and what each one does, in the "
            "order a reader meets them. Describe the notable logic, and call "
            "out anything surprising or important: guards, error handling, "
            "fallbacks, retries, limits, and security-relevant checks. Keep "
            "the concrete values that carry meaning -- limits, timeouts, "
            "sizes, status codes, defaults -- and say them as a person would. "
            "Never read syntax aloud: no brackets, no punctuation, no "
            "indentation, no import lists, no decorator noise. Never spell an "
            "identifier letter by letter -- say rewrite_via_chat_completions "
            'as "rewrite via chat completions", and say a file path the way '
            "a person would read it out. Do not invent behaviour the code "
            "does not have, and say plainly when something cannot be told "
            "from the code shown. No preamble and no commentary about the "
            "explanation itself. Reply with the spoken text only, and always "
            "finish your final sentence."
        ),
    )
    # `style: "document"` -- one CHUNK of a long markdown/txt/PDF document.
    # Same faithfulness discipline as `REWRITE_PROMPT` (this is conversion,
    # not summarisation) plus the two things a chunk needs: no preamble and no
    # closing summary, because it is spoken directly between its neighbours,
    # and no page furniture, because page numbers and running headers cannot
    # be heard.
    rewrite_doc_prompt: str = Field(
        alias="REWRITE_DOC_PROMPT",
        default=(
            "Rewrite the user's text so it can be READ ALOUD and understood "
            "by ear alone. This is CONVERSION TO SPEAKABLE PROSE, not "
            "summarisation. Keep EVERY finding, measurement, number, name, "
            "decision, caveat and recommendation that appears in the input -- "
            "if the input states a value, your rewrite states it too. Never "
            "drop a fact because it seems minor. What you remove is only what "
            "cannot be heard or does not carry meaning: markup, repetition, "
            "hedging, and throat-clearing. Keep the text's own order. This "
            "text is ONE CHUNK OF A LONGER DOCUMENT: it will be spoken "
            "directly after the previous chunk and directly before the next, "
            'so do NOT open with a preamble such as "This document '
            'describes" and do NOT close with a summary or a conclusion -- '
            "begin where the text begins and end where it ends, mid-argument "
            "if that is where it ends. Drop the page furniture that cannot be "
            "heard: page numbers, running headers and footers, figure and "
            "table numbering artifacts, and footnote markers; rejoin any word "
            "hyphenated across a line break. Turn code fences, tables and "
            "lists into flowing spoken sentences. Say paths, symbols and "
            "abbreviations the way a person would say them aloud, and never "
            "spell identifiers letter by letter. Reply with the spoken text "
            "only, and always complete the sentence you are writing."
        ),
    )
    rewrite_max_input_chars: int = Field(alias="REWRITE_MAX_INPUT_CHARS", default=24000)
    # Measured 2026-09-01 against a self-hosted 27B model on a consumer GPU
    # over the LAN: a rewrite of
    # a short technical paragraph took **132 s** end to end and produced a
    # genuinely good spoken version. 20 s was the original default and 502'd
    # every local call before the model had said anything. A cloud model
    # answers in seconds, so this ceiling costs nothing there -- it only stops
    # the local path from being dead on arrival. The app's own client uses the
    # long-call URLSession (120 s idle) for the same reason.
    rewrite_timeout_s: float = Field(alias="REWRITE_TIMEOUT_S", default=90.0)
    # Sent as `max_tokens`. Generous ON PURPOSE: a REASONING model spends this
    # budget on `reasoning_content` FIRST and only then writes `content`.
    # Measured on the same endpoint: at `max_tokens: 16` the reply came back
    # with 0 characters of `content` and a full `reasoning_content` -- which
    # `content_from_completion` correctly refuses as a 502, so a stingy value
    # reads to the user as "the rewrite endpoint is broken". At 600 the same
    # prompt returned 517 characters of clean spoken prose.
    rewrite_max_tokens: int = Field(alias="REWRITE_MAX_TOKENS", default=2400)
    # Reasoning models spend most of their output budget THINKING before they
    # write, and for this task that thinking is pure waste -- rewriting text
    # for the ear needs no deliberation. Measured 2026-09-01 against the
    # owner's `qwen3.8-27b`: a baseline call generated 2,193 characters of
    # `reasoning_content` to produce 1,247 characters of speech, i.e. **64% of
    # everything generated was discarded**, and total completion tokens fell
    # 602 -> 252 (-58%) with thinking switched off.
    #
    # Sent as `chat_template_kwargs: {"enable_thinking": false}`, which
    # llama.cpp/vLLM honour for Qwen-family templates. Default FALSE because
    # it is a template-level extension, not part of the OpenAI schema, and a
    # stricter hosted router could reject an unknown body field -- turn it on
    # for a local reasoning model, leave it off for a hosted one that is
    # already fast.
    rewrite_disable_thinking: bool = Field(alias="REWRITE_DISABLE_THINKING", default=False)

    # File-backed rewrite prompts (owner, 2026-09-07): the speech prompts as
    # files the app's own editor can open and save, rather than a redeploy or
    # a Settings text field. Empty (the default) means
    # `<HERMES_SANDBOX_ROOT>/trg-researcher/prompts`. The directory MUST be
    # inside the sandbox root -- that is the only place the app's editor can
    # write (`PUT /api/sandbox/text`), and the gateway validates against the
    # same root before reading. A file that is missing, empty, oversized or
    # unreadable simply falls back to the settings below; see
    # `domain/prompt_files.py` for the full resolution order.
    rewrite_prompt_dir: str = Field(alias="REWRITE_PROMPT_DIR", default="")
    # How long a prompt file (or the knowledge that one is absent) is held
    # before it is read again. A document plays as dozens of chunks and each
    # resolves a style prompt and a depth suffix, so this is what keeps that
    # from being a Hermes round trip per chunk. An edit saved from the phone
    # takes effect within this window, with no restart. 0 disables caching.
    rewrite_prompt_file_ttl_s: float = Field(alias="REWRITE_PROMPT_FILE_TTL_S", default=15.0)

    # Continue in a new session (owner ask 2026-09-07, `api/handoff.py`): the
    # last several messages of a conversation distilled into the opening
    # prompt of a fresh session. Goes down the SAME path as the rewrite above
    # -- REWRITE_PROFILE (or REWRITE_BASE_URL / REWRITE_MODEL), the same key,
    # timeout, token ceiling and input cap -- because it is the same class of
    # call (seconds, not a Hermes turn; must not touch the transcript) and a
    # second set of endpoint settings would be a second thing to keep honest.
    # Only the prompt and the window are its own. The prompt is overridable
    # here and, like the speech prompts, by a `handoff.md` file in
    # REWRITE_PROMPT_DIR that the app's editor can save.
    handoff_prompt: str = Field(
        alias="HANDOFF_PROMPT",
        default=(
            "You are preparing the OPENING MESSAGE for a fresh assistant "
            "session that will continue the conversation in the excerpt "
            "below, but has none of its history. Write that message in the "
            "user's own voice, addressed to the new assistant, in plain prose "
            "under four short headings: Goal (what the user is trying to "
            "achieve), Established (the facts, decisions, numbers, names, "
            "commands and file paths that were settled, stated exactly as the "
            "excerpt states them), Open (questions still unanswered, and "
            "things that were tried and did not work), Next (the one concrete "
            "step to take first). Keep every specific value a continuation "
            "would need; drop greetings, process narration and anything that "
            "was resolved and no longer matters. Never invent a fact that is "
            "not in the excerpt. Do not describe this as a summary or a "
            "handoff, and do not address the previous assistant. Reply with "
            "the message only."
        ),
    )
    # How many conversational messages (user and assistant text, never tool
    # calls or display-only rows) the excerpt holds by default. A request may
    # ask for a different window. Twelve is roughly the last six exchanges:
    # enough to know where the work stands, short enough that a small local
    # model reads it whole.
    handoff_last_messages: int = Field(alias="HANDOFF_LAST_MESSAGES", default=12, ge=1, le=200)

    # Per-project workspace directories (owner, 2026-09-07): a project's
    # instructions are a `HERMES.md` file the gateway keeps under
    # `<HERMES_SANDBOX_ROOT>/<this>/<project_id>/`, edited with the app's file
    # editor and baked into each new session's system prompt via its `cwd`
    # (`domain/project_workspace.py`, `docs/PROJECT_INSTRUCTIONS_DESIGN.md`).
    # Empty (the default) means `trg-researcher/projects`, alongside the
    # speech prompts. MUST resolve inside the sandbox root -- that is the only
    # place the app's editor can write and the gateway validates before it
    # `mkdir`s or writes; a subdir that escapes the root disables the feature
    # (the workspace helper no-ops) rather than writing outside it.
    project_workspace_subdir: str = Field(alias="PROJECT_WORKSPACE_SUBDIR", default="")

    # Conversation about the owner's own research (`api/converse.py`).
    # Mirrors the `REWRITE_*` block above -- one implementation, an
    # OpenAI-compatible `/chat/completions` endpoint behind a configurable base
    # URL -- and exists for the same measured reason: a Hermes agent turn takes
    # 2.5-15 minutes and B-62 completion signals arrive 8-10 minutes late, so
    # a spoken back-and-forth about a session can only be served by a fast
    # model sitting BESIDE the agent, reading the gateway's own durable state.
    #
    # Like `REWRITE_BASE_URL`, EMPTY (the default) cleanly DISABLES the route
    # with an honest 503 naming the variable. It used to default to Ollama on
    # 127.0.0.1:11434 (measured 2026-09-02: `llama3.2:3b` answers in ~2.8 s),
    # but inside Docker that address is the gateway container itself, so the
    # "works with no config edit" intent never held for the deployed form and
    # only produced a confusing 502. Set it to Ollama's OpenAI-compatible base
    # (`http://<ollama-host>:11434/v1`) or any other endpoint to turn it on. An
    # endpoint that is configured but not running is a 502, never a silent
    # fallback to somewhere the owner did not configure.
    converse_base_url: str = Field(alias="CONVERSE_BASE_URL", default="")
    converse_model: str = Field(alias="CONVERSE_MODEL", default="llama3.2:3b")
    # May legitimately be EMPTY: Ollama and every local llama.cpp/vLLM server
    # needs no key, and an empty key means "send no Authorization header".
    converse_api_key: SecretStr = Field(alias="CONVERSE_API_KEY", default=SecretStr(""))
    # The system prompt. Four things it must do, in this order of importance:
    # answer only from the provided context; REFUSE with a machine-readable
    # sentinel when the context does not support an answer; speak two or three
    # plain sentences (this is going to a voice, so `REWRITE_PROMPT`'s spoken
    # conventions apply); and declare which numbered context entries it used,
    # so the citations the route returns can be narrowed to the ones the answer
    # actually rests on.
    #
    # The refusal is a PROTOCOL, not a hint: an earlier automated "did it
    # refuse?" classifier on this task keyword-matched the prose and was wrong
    # twice, so the route parses an exact leading token instead of guessing
    # from the wording.
    #
    # **Rule 2 is a FORCED BINARY, and that is a measurement, not a style.**
    # Measured 2026-09-02 against `llama3.2:3b` over 29 real questions on the
    # owner's own sessions: an earlier wording that asked for `NOANSWER` only
    # when refusing was obeyed **1 time in 29** -- the model refused in
    # perfectly good prose ("there is no mention of that in the context") and
    # the route, correctly refusing to guess from wording, reported those as
    # answers. Requiring one of exactly two opening words on EVERY reply
    # turned protocol compliance into 29 of 29. A refusal the client cannot
    # detect is a refusal presented as an answer, which is the failure this
    # whole feature is built to avoid.
    #
    # Rule 3 exists because a false-premise question is where a small model
    # fails worst: asked "how many skills did you enable?" about records
    # saying thirteen were DISABLED, it inverted the record rather than
    # correcting the question.
    converse_prompt: str = Field(
        alias="CONVERSE_PROMPT",
        default=(
            "You answer the user's questions about their OWN research, using "
            "only the numbered CONTEXT below it. The context holds real "
            "records from their research gateway: transcript rows, run events "
            "and artifacts. Your answer will be READ ALOUD by a voice.\n"
            "\n"
            "Rules, in order of importance:\n"
            "1. Answer ONLY from the CONTEXT. Never use outside knowledge, "
            "never guess, and never fill a gap with something that merely "
            "sounds likely. If the context states a number, a name or a path, "
            "use exactly that one.\n"
            "2. Begin EVERY reply with one of exactly two words. Use "
            "ANSWER: when the context does answer the question. Use "
            "NOANSWER: when it does not. There is no third option and no "
            "reply without one of them. Refusing is always better than "
            "guessing and costs nothing. A context that is merely about the "
            "same general subject is NOT an answer.\n"
            "3. If the question takes something for granted that the context "
            "does not support, CORRECT IT: begin with ANSWER: and say what "
            "the context actually says instead. Never repeat the question's "
            "assumption back as if it were a fact, and never invent a reason "
            "for something the context does not say happened.\n"
            "4. Answer in two or three sentences of plain spoken prose. No "
            "markdown, no bullet points, no headings, no code fences, no "
            "asterisks, no numbered lists.\n"
            "5. Say numbers, paths, file names and abbreviations the way a "
            "person would say them aloud, and never spell an identifier letter "
            "by letter.\n"
            "6. Finish with one final line naming the context entries you "
            "used, in the form SOURCES: 2, 5 -- numbers only. If you refused, "
            "write SOURCES: none."
        ),
    )
    converse_max_question_chars: int = Field(alias="CONVERSE_MAX_QUESTION_CHARS", default=600)
    # The character budget for the assembled context. Deliberately small: a 3B
    # model is reliable over a few thousand characters and degrades over a
    # window stuffed to its nominal 128k, and a bigger window is precisely how
    # the measured confabulation happened. The route reports the chars it
    # actually used, so this can be tuned against evidence rather than feel.
    converse_max_context_chars: int = Field(alias="CONVERSE_MAX_CONTEXT_CHARS", default=6000)
    # How many of the newest records from the scope's primary source are
    # included whatever the question -- so "what did we just do" works without
    # a lexical hook. Kept small: this is the ONLY part of the window that is
    # not justified by the question, and it is therefore the only part that can
    # tempt a model into answering from unrelated recent material.
    converse_anchor_units: int = Field(alias="CONVERSE_ANCHOR_UNITS", default=3)
    converse_timeout_s: float = Field(alias="CONVERSE_TIMEOUT_S", default=60.0)
    # Two or three spoken sentences plus a SOURCES line. Generous enough that a
    # reasoning model can think first (the `REWRITE_MAX_TOKENS` lesson) without
    # being large enough to let a model ramble into a wall of prose.
    converse_max_tokens: int = Field(alias="CONVERSE_MAX_TOKENS", default=800)
    # **Zero on purpose, and this is a correctness setting, not a style one.**
    # Ollama's default sampling temperature is 0.8; for an extraction task
    # whose whole value is that it does not invent, sampling creativity is
    # exactly the wrong knob to leave at a chat default.
    converse_temperature: float = Field(alias="CONVERSE_TEMPERATURE", default=0.0)
    # Same Qwen-template switch as `REWRITE_DISABLE_THINKING`, same default and
    # same rationale.
    converse_disable_thinking: bool = Field(alias="CONVERSE_DISABLE_THINKING", default=False)

    # Text-to-speech (P5-10, `api/speak.py`). Same shape as the `STT_*` block:
    # a provider name, and the settings each shipped provider needs.
    #
    # `piper` is the default because it is the only option that is BOTH local
    # and keyless -- the reply never leaves the machine and there is no account
    # to hold. `edge` is the keyless cloud alternative. The other six names
    # `api/speak.py` recognises are not implemented here and answer an honest
    # 503 naming what is missing; none of them is ever fallen back to.
    tts_provider: str = Field(alias="TTS_PROVIDER", default="piper")
    # Empty (the default) means "the first voice the provider actually has",
    # which is resolved per request and reported back in `X-TTS-Voice`. A voice
    # id set here that the provider does NOT have is a 503 naming it, never a
    # quiet substitution.
    tts_voice: str = Field(alias="TTS_VOICE", default="")
    # Where the piper `.onnx` + `.onnx.json` voice files live -- beside the DB,
    # the artifact store and the runtime overlay, inside the already-gitignored
    # `data/`. Regenerable on demand:
    #   uv run python -m piper.download_voices \
    #       --download-dir data/piper-voices en_US-lessac-medium
    tts_piper_voice_dir: str = Field(alias="TTS_PIPER_VOICE_DIR", default="./data/piper-voices")
    # Piper's phoneme length scale: < 1 speaks faster, > 1 slower. 1.0 is the
    # voice's own trained pace and is what the model config asks for, so the
    # default changes nothing about how the voice sounds.
    tts_piper_length_scale: float = Field(alias="TTS_PIPER_LENGTH_SCALE", default=1.0)
    # Edge's own rate control, as its API takes it: a signed percentage string.
    # "+0%" is the voice's natural pace.
    tts_edge_rate: str = Field(alias="TTS_EDGE_RATE", default="+0%")
    # Refused with a 413 past this, rather than synthesised into minutes of
    # audio nobody waits for. The app speaks a sentence at a time, so this is a
    # ceiling on a *chunk*, not on a reply.
    tts_max_input_chars: int = Field(alias="TTS_MAX_INPUT_CHARS", default=6000)

    # Research Gateway's own storage
    research_gateway_db_path: str = Field(
        alias="RESEARCH_GATEWAY_DB_PATH", default="./data/research-gateway.db"
    )
    research_gateway_artifact_root: str = Field(
        alias="RESEARCH_GATEWAY_ARTIFACT_ROOT", default="./data/artifacts"
    )
    # Where the runtime configuration overlay lives -- beside the DB and the
    # artifact store, inside the already-gitignored `data/` directory. It holds
    # API keys in plaintext (like `.env`) and is written 0600.
    #
    # **Deliberately NOT itself overlayable.** It is read from `.env` before the
    # overlay is applied, so a bad overlay can never move the file the gateway
    # would have to read to fix itself.
    research_gateway_runtime_config_path: str = Field(
        alias="RESEARCH_GATEWAY_RUNTIME_CONFIG_PATH",
        default="./data/runtime-config.json",
    )

    # The snapshot sweep (P6-3, `api/snapshot_sweep.py`): the timer that
    # re-snapshots filed sessions whose transcript moved, so they are durable
    # without a tap. `docs/SESSION_ARCHIVE_DESIGN.md` §6.5.
    #
    # `INTERVAL_S` is the period between passes; `0` disables the timer and
    # `POST /api/snapshot-sweeps` still runs one by hand (the test suite sets
    # `0` so no timer runs under `TestClient`). `MAX_PER_PASS` caps how many
    # sessions one pass resumes -- a cold resume of the 3.96 MB session took
    # 1.84 s and leaves a live handle behind, so a pass over 130 sessions is
    # bounded on purpose. `STARTUP_DELAY_S` is how long after the gateway
    # starts the timer's first pass may run; the first pass additionally waits
    # for the adapter to be connected by a real request rather than connecting
    # on its own. `SCOPE` is `filed` (filed sessions plus anything that already
    # has a snapshot) or `all` (every session Hermes lists).
    research_gateway_snapshot_sweep_interval_s: int = Field(
        alias="RESEARCH_GATEWAY_SNAPSHOT_SWEEP_INTERVAL_S", default=3600, ge=0
    )
    research_gateway_snapshot_sweep_max_per_pass: int = Field(
        alias="RESEARCH_GATEWAY_SNAPSHOT_SWEEP_MAX_PER_PASS", default=10, ge=1
    )
    research_gateway_snapshot_sweep_startup_delay_s: int = Field(
        alias="RESEARCH_GATEWAY_SNAPSHOT_SWEEP_STARTUP_DELAY_S", default=120, ge=0
    )
    research_gateway_snapshot_sweep_scope: Literal["filed", "all"] = Field(
        alias="RESEARCH_GATEWAY_SNAPSHOT_SWEEP_SCOPE", default="filed"
    )

    # The `ProfileConnectionManager` reconciliation timer (B-136,
    # `docs/CHAT_HISTORY_DESIGN.md` §4, `domain/profile_connection.py`): polls
    # `profiles.list` and auto-provisions a per-profile isolated dashboard
    # connection for anything beyond `default`, via `SubprocessProfileLauncher`
    # (a real `hermes -p <profile> dashboard --isolated --port 0` subprocess).
    #
    # **`0` (the default) disables the timer entirely.** The manager is still
    # built at startup (`api/main.py`'s `lifespan`) and reachable on
    # `app.state.profile_connection_manager` -- constructing it and resolving
    # its default-profile connection through `app.state` at the point of use
    # costs nothing and fixes a real staleness hazard either way (see the
    # module docstring) -- but nothing calls out to spawn a real OS process
    # until this is set to a positive value. Actually launching per-profile
    # dashboards against the deployed Hermes host is a deliberate, owner-timed
    # step (`docs/CHAT_HISTORY_DESIGN.md` §9's "Step 2: Go before gateway work
    # starts" was for building this; turning the timer on is a separate,
    # later decision), not something this change turns on by shipping.
    #
    # B-137 (`docs/BUGS.md`, fixed `63a6fce`) is why it was unsafe to set this
    # above `0` before: the default profile's connection reused
    # `app.state.hermes_adapter` -- the SAME object `EventBroadcaster` already
    # drains for `/ws/events` -- and two concurrent consumers of
    # `HermesAdapter.events()`'s one shared `asyncio.Queue` split the stream
    # instead of each seeing it all (B-08's own failure class). The default
    # connection now subscribes through `EventBroadcaster.subscribe()`
    # instead, so that hazard no longer applies.
    research_gateway_profile_reconcile_interval_s: int = Field(
        alias="RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S", default=0, ge=0
    )

    # Non-default profile dashboards are launched via `docker exec <target>
    # hermes -p <profile> dashboard ...` rather than a bare `hermes` subprocess
    # (`SubprocessProfileLauncher`'s `command_prefix`) -- the gateway and
    # Hermes run as separate containers (2026-09-05 Docker deploy), so the
    # `hermes` CLI isn't present in the gateway's own image. Empty (the
    # default) means "run `hermes` as a bare subprocess", which only works if
    # the gateway process and Hermes's CLI share a host/filesystem.
    research_gateway_profile_docker_exec_target: str = Field(
        alias="RESEARCH_GATEWAY_PROFILE_DOCKER_EXEC_TARGET", default=""
    )

    # The host the gateway dials to *reach* a launched per-profile dashboard
    # -- distinct from `hermes_host` (the unified/default connection's LAN
    # address) because a `docker exec`-launched dashboard's port is only
    # reachable from inside the Docker network the Hermes container is on,
    # via its container name, not the host's LAN IP (an ephemeral `--port 0`
    # port is never published to the host). Empty (the default) falls back to
    # `hermes_host`, correct only when profile dashboards and the unified
    # connection really are reachable the same way (e.g. no Docker split).
    research_gateway_profile_dashboard_host: str = Field(
        alias="RESEARCH_GATEWAY_PROFILE_DASHBOARD_HOST", default=""
    )

    @field_validator("research_gateway_snapshot_sweep_scope", mode="before")
    @classmethod
    def _normalize_sweep_scope(cls, value: object) -> object:
        """`" All "` in `.env` means `all`; anything outside the two values still fails."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @property
    def hermes_base_url(self) -> str:
        return f"{self.hermes_scheme}://{self.hermes_host}:{self.hermes_port}"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Settings(hermes_scheme={self.hermes_scheme!r}, "
            f"hermes_host={self.hermes_host!r}, "
            f"hermes_port={self.hermes_port!r}, "
            f"hermes_username={self.hermes_username!r}, "
            f"hermes_password=SecretStr('**********'), "
            f"research_gateway_username={self.research_gateway_username!r}, "
            f"research_gateway_password=SecretStr('**********'), "
            f"hermes_sandbox_root={self.hermes_sandbox_root!r}, "
            f"hermes_sandbox_denylist_dirs={self.hermes_sandbox_denylist_dirs!r}, "
            f"hermes_sandbox_denylist_globs={self.hermes_sandbox_denylist_globs!r}, "
            f"hermes_attachment_dir={self.hermes_attachment_dir!r}, "
            f"research_gateway_view_roots={self.research_gateway_view_roots!r}, "
            f"stt_provider={self.stt_provider!r}, "
            f"stt_local_model={self.stt_local_model!r}, "
            f"stt_language={self.stt_language!r}, "
            f"stt_vocab_hint={self.stt_vocab_hint!r}, "
            f"stt_openai_key=SecretStr('**********'), "
            f"stt_groq_key=SecretStr('**********'), "
            f"stt_openai_model={self.stt_openai_model!r}, "
            f"stt_groq_model={self.stt_groq_model!r}, "
            f"rewrite_base_url={self.rewrite_base_url!r}, "
            f"rewrite_model={self.rewrite_model!r}, "
            f"rewrite_api_key=SecretStr('**********'), "
            f"rewrite_profile={self.rewrite_profile!r}, "
            f"rewrite_prompt={self.rewrite_prompt!r}, "
            f"rewrite_code_prompt={self.rewrite_code_prompt!r}, "
            f"rewrite_doc_prompt={self.rewrite_doc_prompt!r}, "
            f"rewrite_max_input_chars={self.rewrite_max_input_chars!r}, "
            f"rewrite_timeout_s={self.rewrite_timeout_s!r}, "
            f"rewrite_max_tokens={self.rewrite_max_tokens!r}, "
            f"rewrite_disable_thinking={self.rewrite_disable_thinking!r}, "
            f"rewrite_prompt_dir={self.rewrite_prompt_dir!r}, "
            f"rewrite_prompt_file_ttl_s={self.rewrite_prompt_file_ttl_s!r}, "
            f"handoff_prompt={self.handoff_prompt!r}, "
            f"handoff_last_messages={self.handoff_last_messages!r}, "
            f"project_workspace_subdir={self.project_workspace_subdir!r}, "
            f"converse_base_url={self.converse_base_url!r}, "
            f"converse_model={self.converse_model!r}, "
            f"converse_api_key=SecretStr('**********'), "
            f"converse_prompt={self.converse_prompt!r}, "
            f"converse_max_question_chars={self.converse_max_question_chars!r}, "
            f"converse_max_context_chars={self.converse_max_context_chars!r}, "
            f"converse_anchor_units={self.converse_anchor_units!r}, "
            f"converse_timeout_s={self.converse_timeout_s!r}, "
            f"converse_max_tokens={self.converse_max_tokens!r}, "
            f"converse_temperature={self.converse_temperature!r}, "
            f"converse_disable_thinking={self.converse_disable_thinking!r}, "
            f"tts_provider={self.tts_provider!r}, "
            f"tts_voice={self.tts_voice!r}, "
            f"tts_piper_voice_dir={self.tts_piper_voice_dir!r}, "
            f"tts_piper_length_scale={self.tts_piper_length_scale!r}, "
            f"tts_edge_rate={self.tts_edge_rate!r}, "
            f"tts_max_input_chars={self.tts_max_input_chars!r}, "
            f"research_gateway_snapshot_sweep_interval_s="
            f"{self.research_gateway_snapshot_sweep_interval_s!r}, "
            f"research_gateway_snapshot_sweep_max_per_pass="
            f"{self.research_gateway_snapshot_sweep_max_per_pass!r}, "
            f"research_gateway_snapshot_sweep_startup_delay_s="
            f"{self.research_gateway_snapshot_sweep_startup_delay_s!r}, "
            f"research_gateway_snapshot_sweep_scope="
            f"{self.research_gateway_snapshot_sweep_scope!r}, "
            f"research_gateway_profile_reconcile_interval_s="
            f"{self.research_gateway_profile_reconcile_interval_s!r}, "
            f"research_gateway_profile_docker_exec_target="
            f"{self.research_gateway_profile_docker_exec_target!r}, "
            f"research_gateway_profile_dashboard_host="
            f"{self.research_gateway_profile_dashboard_host!r}, "
            f"research_gateway_public_base_url={self.research_gateway_public_base_url!r}, "
            f"research_gateway_db_path={self.research_gateway_db_path!r}, "
            f"research_gateway_artifact_root={self.research_gateway_artifact_root!r}, "
            f"research_gateway_runtime_config_path="
            f"{self.research_gateway_runtime_config_path!r})"
        )

    __str__ = __repr__


def get_settings() -> Settings:
    """Return a freshly-loaded Settings instance, overlay applied.

    Not cached at module import time so tests can point `env_file` elsewhere by
    constructing `Settings(_env_file=...)` directly if needed -- and, since the
    runtime overlay arrived, so that **a configuration change takes effect on
    the next request with no restart**. Every endpoint calls this per request;
    that property is load-bearing and nothing here may quietly cache a
    `Settings` across calls.

    Three layers, in order: field defaults, then `.env` / the process
    environment, then the overlay `api/config.py` writes. The overlay wins
    because pydantic-settings ranks init kwargs above every other source.

    **What that costs, measured** (2026-09-02, this machine, 2,000 calls):
    a bare `Settings()` is ~1,000 us, dominated by re-parsing `.env` -- which
    is what it already cost before the overlay existed. With **no overlay
    file** this function adds **~17 us** (a `stat()` and a dict check). With an
    overlay present it constructs `Settings` a second time, so ~2,080 us. That
    second construction is the price of resolving the overlay path through
    exactly the same rules as every other setting rather than through a
    private second lookup that could drift from them, and it is paid at most
    twice per HTTP request (auth plus the route). Nothing per event frame
    calls this -- the WebSocket fan-out does not touch settings at all -- so
    the busiest path in the service is unaffected.

    **Never raises because of the overlay.** `read_overlay()` already drops
    anything unreadable, unknown or invalid, and the construction is guarded
    anyway: if an overlay value still cannot build a `Settings`, the
    `.env`-only instance is returned and the gateway behaves exactly as it did
    before the overlay existed. Configurability must not create a new way to
    break speech.
    """
    base = Settings()
    overlay = runtime_config.read_overlay(base.research_gateway_runtime_config_path)
    if not overlay.values:
        return base
    try:
        return Settings(**overlay.values)
    except Exception as exc:
        # **The exception CLASS and the keys, never the message.** A pydantic
        # `ValidationError` can carry the offending `input` value, and one of
        # these keys may be an API key -- so this is deliberately not
        # `logger.exception`. `read_overlay()` has already validated every
        # value, so reaching here at all means a shape `runtime_config` and
        # `Settings` disagree about, which the key names are enough to find.
        logger.error(
            "the runtime config overlay at %s could not be applied (%s); "
            "falling back to .env for every value. Keys in the overlay: %s",
            base.research_gateway_runtime_config_path,
            exc.__class__.__name__,
            sorted(overlay.values),
        )
        return base
