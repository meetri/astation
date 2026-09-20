"""The runtime configuration overlay: the operator's own provider settings, writable.

The operator's principle, verbatim: *"It shouldn't matter the logistics I use --
what should matter is having control over the configs."* Until this module
existed, every provider choice (which endpoint answers a rewrite, which Whisper
runs, which key is presented) was a hand edit of the repo-root `.env` on one
specific Mac. The operator could not change their own setup without a developer,
and the machine or engine in use is theirs to decide and change at will.

## The load order, and what "overlay" means

    defaults  ->  .env  ->  overlay

`.env` keeps doing exactly what it did: it is the **floor**, the hand-maintained
gitignored file that a fresh checkout or a Docker deploy is configured from.
This module adds a layer **above** it -- a small JSON file the gateway itself
owns and writes -- so a value the operator sets from their phone wins over the one
in `.env`, and dropping the overlay value (`reset`) falls straight back to it.

**`.env` is never written.** It is hand-maintained, it carries comments the
owner wrote, and it is the recovery path when the overlay is wrong; a program
that rewrites it destroys both.

## Where it lives

`RESEARCH_GATEWAY_RUNTIME_CONFIG_PATH`, default `./data/runtime-config.json` --
beside `research-gateway.db` and the artifact store, inside the
already-gitignored `services/research-gateway/data/`. It holds secrets in
plaintext, exactly as `.env` does, so it is written `0600` and it must never be
committed. The file itself is written atomically (temp file in the same
directory, then `os.replace`), so a crash mid-write leaves the previous
configuration intact rather than a half-file.

Format::

    {"version": 1, "values": {"rewrite_model": "mtplx-27b", ...}}

Keys are `Settings` **field names**, not env-var aliases, because that is the
vocabulary the HTTP API and the app speak and one vocabulary is better than two.
A hand-written file that omits the `{"version", "values"}` wrapper and is just a
flat `{key: value}` map is read as the values map -- forgiving, and unambiguous
because every recognised key is in `CONFIG_KEYS`.

## Every failure degrades to ".env only"

The same rule `SpokenRewriteCacheStore` and `OfflineCacheStores` already follow
on the app side: a store that cannot be read is not an outage. `load_overlay()`
**never raises**. A missing file, unreadable bytes, malformed JSON, a non-object
payload, a key no longer in `CONFIG_KEYS`, a value that no longer validates --
each is dropped (with a warning, once per file version, so a broken overlay is
visible in the log without spamming it) and everything still readable is kept.
Worst case the whole overlay is ignored and the gateway runs on `.env` exactly
as it did before this module existed.

That is the load-bearing property behind the promise that configurability must
not create a new way to break speech: there is no overlay state that can stop
`get_settings()` from returning a usable `Settings`.

## Freshness

`get_settings()` builds a fresh `Settings()` per call and every endpoint calls
it per request, so a configuration change takes effect on the **next request**
with no restart. This module keeps that property and does not pay for it twice:
the parsed overlay is cached against the file's `(st_mtime_ns, st_size)`, so the
per-request cost is one `stat()`. Any write -- ours through `write_overlay()`,
which also invalidates explicitly, or a hand edit -- moves the stamp and the
next call re-reads. The stamp is taken *after* the read and compared with the
one taken before it; a file that changed underneath the read is used but not
cached, so a torn read can never become the cached answer.

## Validation happens here, not at the edge

`validate_value()` is the single authority on what a key may hold, and the HTTP
layer turns its `ConfigValueError` into a 422. It runs **before** anything is
persisted (so an invalid value never reaches the file) and **again** on load (so
an overlay written by an older build, or edited by hand, cannot feed `Settings`
something that would raise during construction).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: Overlay file format version. Bumped only if the on-disk shape changes in a
#: way a reader has to branch on; the loader tolerates its absence.
OVERLAY_VERSION = 1

#: File mode for the overlay. It carries API keys in plaintext, like `.env`.
OVERLAY_FILE_MODE = 0o600


class ConfigValueError(ValueError):
    """A value that is not allowed for this key. Refused before it is stored."""


# ---------------------------------------------------------------------------
# The registry: every key the operator may change at runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigKey:
    """One runtime-changeable setting.

    `key` is the `Settings` field name and is simultaneously the API key, the
    overlay JSON key and the identifier the app sends -- deliberately one
    vocabulary. `env_var` is the `.env` alias it shadows, reported so the operator
    can see which line of their `.env` an overlay value is overriding.
    """

    key: str
    env_var: str
    capability: str
    label: str
    #: `url` | `text` | `secret` | `choice` | `int` | `float` | `bool`
    kind: str
    help: str = ""
    choices: tuple[str, ...] = ()
    max_chars: int = 512
    minimum: float | None = None
    maximum: float | None = None
    #: Whether an empty string is a meaningful value rather than a mistake.
    #: True for `rewrite_base_url` (empty disables the feature, honestly) and
    #: for every key whose emptiness the existing code already handles.
    allow_empty: bool = True
    #: Whether a newline is legitimate content. True for exactly the three
    #: rewrite prompts, which are multi-line prose. **False everywhere else,
    #: and that is a security property, not formatting**: a newline in an API
    #: key or a model id is a header-injection shape (`Authorization: Bearer
    #: <key>`) and a log-forging shape. Per key rather than per kind, because
    #: `rewrite_model` and `rewrite_prompt` are both free text and only one of
    #: them may ever contain a line break.
    multiline: bool = False
    #: A regex a non-empty value must fullmatch. Set for `rewrite_profile`,
    #: whose value names a Hermes profile: the same shape
    #: `api/profile_admin.py` enforces on a name that becomes a CLI argument.
    pattern: str | None = None

    @property
    def secret(self) -> bool:
        return self.kind == "secret"


@dataclass(frozen=True)
class Capability:
    """A family of settings the operator thinks about as one thing.

    `writable` is False for the two capabilities this gateway does **not**
    configure. They are listed anyway, with the reason, because "the app shows
    nothing about TTS" and "the gateway has no TTS configuration" are different
    facts and only the second is true -- and because leaving room for them here
    is what makes adding one later a registry entry rather than a new surface.
    """

    name: str
    label: str
    summary: str
    writable: bool = True
    note: str = ""


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="rewrite",
        label="Rewrite for listening",
        summary=(
            "The OpenAI-compatible /chat/completions endpoint that turns a "
            "reply into prose a voice can read aloud."
        ),
    ),
    Capability(
        name="converse",
        label="Ask about your research",
        summary=(
            "The fast OpenAI-compatible endpoint that answers spoken questions "
            "about a session, a project or a document from this gateway's own "
            "records. Separate from the conversation model below: that one is "
            "the agent doing the work, this one only reads what it recorded."
        ),
    ),
    Capability(
        name="stt",
        label="Transcription",
        summary="What turns a recording from the mic button into text.",
    ),
    Capability(
        name="tts",
        label="Speech synthesis",
        summary=(
            "The engine that turns a reply into audio. Piper runs on this "
            "gateway's own CPU with no key; Edge is Microsoft's, also with no "
            "key. The phone's own voice is what speaks when neither can."
        ),
        note=(
            "This picks the SERVER voice. The device's own AVSpeechSynthesizer "
            "voice is still chosen in Settings > Voice, and it is what speaks "
            "whenever this engine cannot -- so a 503 here is a downgrade, "
            "never silence."
        ),
    ),
    Capability(
        name="artifacts",
        label="Artifact ingest",
        summary=(
            "Which files the agent writes get filed into the artifact "
            "library, and which are skipped as operational noise."
        ),
        note=(
            "These ADD to the built-in rules, they never replace them, so "
            "Hermes's own state files stay out whatever is set here. They "
            "gate NEW ingests only -- use Archive ignored files to clear what "
            "is already filed. Promotion from the sandbox browser is never "
            "blocked: an explicit ask to keep a file wins over a rule."
        ),
    ),
    Capability(
        name="conversation",
        label="Conversation model",
        summary="The model that answers in a session.",
        writable=False,
        note=(
            "Chosen by the Hermes instance, not by this gateway: session.create "
            "accepts a model and silently ignores it (measured -- see "
            "GET /api/profiles), so a picker here would only appear to work. "
            "Which Hermes instance is used is a HERMES_* value in .env and "
            "needs a restart, because the gateway holds one long-lived socket "
            "to it."
        ),
    ),
)

CAPABILITIES_BY_NAME: dict[str, Capability] = {c.name: c for c in CAPABILITIES}


#: The shape of a Hermes profile name -- `api/profile_admin.py`'s
#: `PROFILE_NAME_PATTERN`, duplicated here rather than imported because
#: `config/` must not depend on `api/`. `tests/test_config.py` asserts the two
#: are the same string.
PROFILE_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,39}$"

CONFIG_KEYS: tuple[ConfigKey, ...] = (
    # --- artifact ingest (domain/artifact_ingest.py) -----------------------
    ConfigKey(
        key="artifact_ignore_dirs",
        env_var="ARTIFACT_IGNORE_DIRS",
        capability="artifacts",
        label="Ignore folders",
        kind="text",
        help=(
            "Comma-separated folder names to keep out of the library, matched "
            "wherever they appear in a path -- node_modules, .git, build. "
            "Added to the built-in list, which already covers .npm, .cache, "
            "__pycache__, .venv and the rest."
        ),
        max_chars=400,
    ),
    ConfigKey(
        key="artifact_ignore_globs",
        env_var="ARTIFACT_IGNORE_GLOBS",
        capability="artifacts",
        label="Ignore filenames",
        kind="text",
        help=(
            "Comma-separated filename patterns to keep out of the library, "
            "such as *.log or *.tmp. Added to the built-in list."
        ),
        max_chars=400,
    ),
    # --- rewrite (api/rewrite.py) ------------------------------------------
    ConfigKey(
        key="rewrite_profile",
        env_var="REWRITE_PROFILE",
        capability="rewrite",
        label="Agent profile",
        kind="text",
        help=(
            "The name of a Hermes profile (an agent in the Agents screen). "
            "When set, the rewrite goes to that profile's provider and model, "
            "and the endpoint and model fields below are ignored. Empty uses "
            "the endpoint and model below. The API key is still the one here: "
            "Hermes never exposes its own keys, so a profile on a hosted "
            "provider (OpenRouter, OpenAI, Anthropic) needs a key set below."
        ),
        max_chars=40,
        pattern=PROFILE_NAME_PATTERN,
    ),
    ConfigKey(
        key="rewrite_base_url",
        env_var="REWRITE_BASE_URL",
        capability="rewrite",
        label="Endpoint",
        kind="url",
        help=(
            "Any OpenAI-compatible base URL, including the version segment "
            "(e.g. http://127.0.0.1:8001/v1). Empty disables the rewrite: the "
            "app then reads replies as written, on-device."
        ),
        max_chars=400,
    ),
    ConfigKey(
        key="rewrite_model",
        env_var="REWRITE_MODEL",
        capability="rewrite",
        label="Model",
        kind="text",
        help="A model id the endpoint serves. Probe the endpoint to list them.",
        max_chars=200,
    ),
    ConfigKey(
        key="rewrite_api_key",
        env_var="REWRITE_API_KEY",
        capability="rewrite",
        label="API key",
        kind="secret",
        help=(
            "Sent as a Bearer token. Empty is normal and correct for a keyless "
            "local server -- no Authorization header is sent at all."
        ),
        max_chars=1024,
    ),
    ConfigKey(
        key="rewrite_timeout_s",
        env_var="REWRITE_TIMEOUT_S",
        capability="rewrite",
        label="Timeout (seconds)",
        kind="float",
        help="How long to wait for the endpoint before giving up.",
        minimum=1.0,
        maximum=600.0,
    ),
    ConfigKey(
        key="rewrite_max_tokens",
        env_var="REWRITE_MAX_TOKENS",
        capability="rewrite",
        label="Max output tokens",
        kind="int",
        help=(
            "A reasoning model spends this budget on thinking first, so a "
            "stingy value returns an empty rewrite."
        ),
        minimum=16,
        maximum=200_000,
    ),
    ConfigKey(
        key="rewrite_max_input_chars",
        env_var="REWRITE_MAX_INPUT_CHARS",
        capability="rewrite",
        label="Max input characters",
        kind="int",
        help="Longer text is refused with a 413 rather than truncated.",
        minimum=100,
        maximum=2_000_000,
    ),
    ConfigKey(
        key="rewrite_disable_thinking",
        env_var="REWRITE_DISABLE_THINKING",
        capability="rewrite",
        label="Disable thinking",
        kind="bool",
        help=(
            "Sends the Qwen-template chat_template_kwargs switch. Turn it on "
            "for a local reasoning model; a stricter hosted router may reject "
            "the unknown field."
        ),
    ),
    ConfigKey(
        key="rewrite_prompt",
        env_var="REWRITE_PROMPT",
        capability="rewrite",
        label="Prose prompt",
        kind="text",
        help="System prompt for style 'listen' -- an ordinary reply.",
        max_chars=8000,
        multiline=True,
    ),
    ConfigKey(
        key="rewrite_code_prompt",
        env_var="REWRITE_CODE_PROMPT",
        capability="rewrite",
        label="Source-code prompt",
        kind="text",
        help="System prompt for style 'explain' -- source code.",
        max_chars=8000,
        multiline=True,
    ),
    ConfigKey(
        key="rewrite_doc_prompt",
        env_var="REWRITE_DOC_PROMPT",
        capability="rewrite",
        label="Document prompt",
        kind="text",
        help="System prompt for style 'document' -- one chunk of a document.",
        max_chars=8000,
        multiline=True,
    ),
    # --- converse (api/converse.py) ----------------------------------------
    ConfigKey(
        key="converse_base_url",
        env_var="CONVERSE_BASE_URL",
        capability="converse",
        label="Endpoint",
        kind="url",
        help=(
            "Any OpenAI-compatible base URL, including the version segment "
            "(e.g. http://<ollama-host>:11434/v1 for Ollama). Empty disables "
            "asking questions about your research."
        ),
        max_chars=400,
    ),
    ConfigKey(
        key="converse_model",
        env_var="CONVERSE_MODEL",
        capability="converse",
        label="Model",
        kind="text",
        help=(
            "A model id the endpoint serves. Small and fast beats large and "
            "clever here: the answer has to arrive while you are still holding "
            "the phone. Probe the endpoint to list them."
        ),
        max_chars=200,
    ),
    ConfigKey(
        key="converse_api_key",
        env_var="CONVERSE_API_KEY",
        capability="converse",
        label="API key",
        kind="secret",
        help=(
            "Sent as a Bearer token. Empty is normal and correct for Ollama "
            "and every keyless local server."
        ),
        max_chars=1024,
    ),
    ConfigKey(
        key="converse_prompt",
        env_var="CONVERSE_PROMPT",
        capability="converse",
        label="Prompt",
        kind="text",
        help=(
            "The system prompt. It must keep the answer inside the provided "
            "context, keep the NOANSWER refusal token, and keep the SOURCES "
            "line -- the route parses both."
        ),
        max_chars=8000,
        multiline=True,
    ),
    ConfigKey(
        key="converse_max_context_chars",
        env_var="CONVERSE_MAX_CONTEXT_CHARS",
        capability="converse",
        label="Context budget (characters)",
        kind="int",
        help=(
            "How much of your own records is put in front of the model. "
            "Bigger is not better: a padded window is what makes a small model "
            "invent an answer."
        ),
        minimum=200,
        maximum=200_000,
    ),
    ConfigKey(
        key="converse_anchor_units",
        env_var="CONVERSE_ANCHOR_UNITS",
        capability="converse",
        label="Recent records always included",
        kind="int",
        help=(
            'The newest records included whatever you asked, so "what did we '
            'just do" works. Zero means the question alone decides.'
        ),
        minimum=0,
        maximum=50,
    ),
    ConfigKey(
        key="converse_max_question_chars",
        env_var="CONVERSE_MAX_QUESTION_CHARS",
        capability="converse",
        label="Max question characters",
        kind="int",
        help="A longer question is refused with a 413 rather than truncated.",
        minimum=20,
        maximum=100_000,
    ),
    ConfigKey(
        key="converse_timeout_s",
        env_var="CONVERSE_TIMEOUT_S",
        capability="converse",
        label="Timeout (seconds)",
        kind="float",
        help="How long to wait for the endpoint before giving up.",
        minimum=1.0,
        maximum=600.0,
    ),
    ConfigKey(
        key="converse_max_tokens",
        env_var="CONVERSE_MAX_TOKENS",
        capability="converse",
        label="Max output tokens",
        kind="int",
        help=(
            "A reasoning model spends this budget on thinking first, so a "
            "stingy value returns an empty answer."
        ),
        minimum=16,
        maximum=200_000,
    ),
    ConfigKey(
        key="converse_temperature",
        env_var="CONVERSE_TEMPERATURE",
        capability="converse",
        label="Temperature",
        kind="float",
        help=(
            "Zero is the right value. This job is reading your records back to "
            "you, and sampling creativity is how a model invents a finding you "
            "never made."
        ),
        minimum=0.0,
        maximum=2.0,
    ),
    ConfigKey(
        key="converse_disable_thinking",
        env_var="CONVERSE_DISABLE_THINKING",
        capability="converse",
        label="Disable thinking",
        kind="bool",
        help=(
            "Sends the Qwen-template chat_template_kwargs switch. Turn it on "
            "for a local reasoning model; a stricter hosted router may reject "
            "the unknown field."
        ),
    ),
    # --- speech to text (api/transcribe.py) --------------------------------
    ConfigKey(
        key="stt_provider",
        env_var="STT_PROVIDER",
        capability="stt",
        label="Provider",
        kind="choice",
        choices=("local", "openai", "groq"),
        help=(
            "local runs faster-whisper in this gateway's own process; the "
            "other two are keyed passthroughs."
        ),
        allow_empty=False,
    ),
    ConfigKey(
        key="stt_local_model",
        env_var="STT_LOCAL_MODEL",
        capability="stt",
        label="Local model",
        kind="text",
        help=(
            "A faster-whisper model size (tiny/base/small/medium/large-v3) or "
            "a CTranslate2 model id it can fetch."
        ),
        max_chars=200,
        allow_empty=False,
    ),
    ConfigKey(
        key="stt_language",
        env_var="STT_LANGUAGE",
        capability="stt",
        label="Language",
        kind="text",
        help="An ISO code such as 'en'. Empty lets Whisper detect it.",
        max_chars=16,
    ),
    ConfigKey(
        key="stt_vocab_hint",
        env_var="STT_VOCAB_HINT",
        capability="stt",
        label="Vocabulary hint",
        kind="text",
        help=(
            "Jargon biasing, prepended to any per-request hint. A "
            "comma-separated glossary works best."
        ),
        max_chars=2000,
    ),
    ConfigKey(
        key="stt_openai_key",
        env_var="STT_OPENAI_KEY",
        capability="stt",
        label="OpenAI key",
        kind="secret",
        help="Required only when the provider is 'openai'.",
        max_chars=1024,
    ),
    ConfigKey(
        key="stt_openai_model",
        env_var="STT_OPENAI_MODEL",
        capability="stt",
        label="OpenAI model",
        kind="text",
        max_chars=200,
    ),
    ConfigKey(
        key="stt_groq_key",
        env_var="STT_GROQ_KEY",
        capability="stt",
        label="Groq key",
        kind="secret",
        help="Required only when the provider is 'groq'.",
        max_chars=1024,
    ),
    ConfigKey(
        key="stt_groq_model",
        env_var="STT_GROQ_MODEL",
        capability="stt",
        label="Groq model",
        kind="text",
        max_chars=200,
    ),
    # --- speech synthesis (api/speak.py) -----------------------------------
    #
    # The `tts` capability used to be listed here as `writable: False` with a
    # note saying the gateway held no speech-synthesis configuration. That was
    # true and is not any more: `POST /api/speak` gives the operator a choice of
    # engine, so the choice belongs where every other provider choice does --
    # changeable from the phone, reporting its source, with no `.env` edit.
    ConfigKey(
        key="tts_provider",
        env_var="TTS_PROVIDER",
        capability="tts",
        label="Engine",
        kind="choice",
        choices=(
            "piper",
            "edge",
            "kittentts",
            "neutts",
            "elevenlabs",
            "openai",
            "mistral",
            "xai",
        ),
        help=(
            "piper runs on this gateway's own CPU and needs no key; edge is "
            "Microsoft's service and also needs no key. The rest are "
            "recognised names this gateway does not implement -- they answer "
            "an honest 503 saying what is missing, and the phone reads the "
            "reply on-device instead."
        ),
        allow_empty=False,
    ),
    ConfigKey(
        key="tts_voice",
        env_var="TTS_VOICE",
        capability="tts",
        label="Voice",
        kind="text",
        help=(
            "A voice id the chosen engine actually has. Empty uses the first "
            "one it has. GET /api/speak/voices lists them by asking the "
            "engine rather than from a table."
        ),
        max_chars=200,
    ),
    ConfigKey(
        key="tts_piper_voice_dir",
        env_var="TTS_PIPER_VOICE_DIR",
        capability="tts",
        label="Piper voice folder",
        kind="text",
        help=(
            "Where the .onnx voice files live on the gateway's machine. "
            "Download one with piper.download_voices."
        ),
        max_chars=400,
        allow_empty=False,
    ),
    ConfigKey(
        key="tts_piper_length_scale",
        env_var="TTS_PIPER_LENGTH_SCALE",
        capability="tts",
        label="Piper pace",
        kind="float",
        help="Below 1 speaks faster, above 1 slower. 1 is the voice's own pace.",
        minimum=0.25,
        maximum=4.0,
    ),
    ConfigKey(
        key="tts_edge_rate",
        env_var="TTS_EDGE_RATE",
        capability="tts",
        label="Edge pace",
        kind="text",
        help="Edge's own rate, as a signed percentage: +0%, -10%, +25%.",
        max_chars=16,
        allow_empty=False,
    ),
    ConfigKey(
        key="tts_max_input_chars",
        env_var="TTS_MAX_INPUT_CHARS",
        capability="tts",
        label="Max characters per request",
        kind="int",
        help="Longer text is refused with a 413 rather than synthesised.",
        minimum=100,
        maximum=200_000,
    ),
)

CONFIG_KEYS_BY_NAME: dict[str, ConfigKey] = {spec.key: spec for spec in CONFIG_KEYS}

#: Every key whose value must never leave this process.
SECRET_CONFIG_KEYS: frozenset[str] = frozenset(spec.key for spec in CONFIG_KEYS if spec.secret)


def keys_for_capability(capability: str) -> tuple[ConfigKey, ...]:
    return tuple(spec for spec in CONFIG_KEYS if spec.capability == capability)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

#: Characters that must never appear in a configured value. A newline in a
#: model id or a key would be smuggled into an HTTP header (`Authorization:
#: Bearer <key>`) or a log line; a NUL breaks both. Rejected for every kind.
_FORBIDDEN_CHARS = ("\n", "\r", "\x00")


def validate_value(spec: ConfigKey, raw: Any) -> Any:
    """The normalized value for `spec`, or `ConfigValueError`.

    The single authority on what a key may hold. Run before persisting, so an
    invalid value never reaches the file, and again on load, so an overlay from
    an older build or a hand edit cannot hand `Settings` something that would
    raise while it is being constructed.
    """
    if spec.kind == "bool":
        return _validate_bool(spec, raw)
    if spec.kind == "int":
        return _validate_number(spec, raw, integer=True)
    if spec.kind == "float":
        return _validate_number(spec, raw, integer=False)

    if not isinstance(raw, str):
        raise ConfigValueError(f"{spec.key} must be a string; got {type(raw).__name__}")
    # Only the prompts may be multi-line, and only they keep their internal
    # whitespace. Everything else -- keys, model ids, URLs, language codes --
    # is a single-line token, stripped, with newlines refused outright.
    value = raw if spec.multiline else raw.strip()
    forbidden = ("\x00",) if spec.multiline else _FORBIDDEN_CHARS
    if any(char in value for char in forbidden):
        raise ConfigValueError(
            f"{spec.key} must not contain control characters (NUL always, and "
            "newlines for every key but the prompts: a newline in a key or a "
            "model id is smuggled into a request header or a log line)"
        )
    if len(value) > spec.max_chars:
        raise ConfigValueError(
            f"{spec.key} is {len(value)} characters; the cap is {spec.max_chars}"
        )
    if not value and not spec.allow_empty:
        raise ConfigValueError(
            f"{spec.key} may not be empty. Reset it instead to fall back to {spec.env_var} in .env."
        )
    if spec.kind == "choice":
        if value not in spec.choices:
            raise ConfigValueError(
                f"{spec.key} must be one of {', '.join(spec.choices)}; got {value!r}"
            )
        return value
    if spec.kind == "url" and value:
        _validate_url(spec, value)
    if spec.pattern is not None and value and not re.fullmatch(spec.pattern, value):
        raise ConfigValueError(f"{spec.key} must match {spec.pattern}; got {value!r}")
    return value


def _validate_bool(spec: ConfigKey, raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise ConfigValueError(f"{spec.key} must be true or false; got {raw!r}")


def _validate_number(spec: ConfigKey, raw: Any, *, integer: bool) -> Any:
    # `bool` is an `int` subclass; `True` is not a timeout.
    if isinstance(raw, bool):
        raise ConfigValueError(f"{spec.key} must be a number; got {raw!r}")
    if isinstance(raw, str):
        try:
            raw = int(raw) if integer else float(raw)
        except ValueError:
            raise ConfigValueError(f"{spec.key} must be a number; got {raw!r}") from None
    if integer:
        if not isinstance(raw, int):
            raise ConfigValueError(f"{spec.key} must be a whole number; got {raw!r}")
        value: Any = int(raw)
    else:
        if not isinstance(raw, (int, float)):
            raise ConfigValueError(f"{spec.key} must be a number; got {raw!r}")
        value = float(raw)
    if spec.minimum is not None and value < spec.minimum:
        raise ConfigValueError(
            f"{spec.key} must be at least {_pretty(spec.minimum, integer)}; got "
            f"{_pretty(value, integer)}"
        )
    if spec.maximum is not None and value > spec.maximum:
        raise ConfigValueError(
            f"{spec.key} must be at most {_pretty(spec.maximum, integer)}; got "
            f"{_pretty(value, integer)}"
        )
    return value


def _pretty(value: float, integer: bool) -> str:
    return str(int(value)) if integer else str(value)


def _validate_url(spec: ConfigKey, value: str) -> None:
    """http/https, a real host, and **no credentials in the URL**.

    The userinfo rule is a leak guard, not tidiness: `https://user:key@host/v1`
    puts a secret into a value this gateway reports back over
    `GET /api/config/providers`, logs as `provider_label()`, and shows in the
    app. Refusing the shape is the only way the "no secret is ever returned"
    promise can hold for a field the operator types freely into.
    """
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https"):
        raise ConfigValueError(f"{spec.key} must start with http:// or https://; got {value!r}")
    if not parts.netloc:
        raise ConfigValueError(f"{spec.key} has no host: {value!r}")
    if "@" in parts.netloc:
        raise ConfigValueError(
            f"{spec.key} must not carry credentials in the URL "
            "(user:password@host). Put the key in the API key field, which is "
            "write-only and is never reported back."
        )
    if parts.query or parts.fragment:
        raise ConfigValueError(
            f"{spec.key} must be a plain base URL with no query string or fragment; got {value!r}"
        )


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlayLoad:
    """What `read_overlay()` found. Never an exception."""

    #: Validated, known keys only. Safe to splat into `Settings(**values)`.
    values: dict[str, Any] = field(default_factory=dict)
    #: True when the file exists and parsed into something usable.
    present: bool = False
    #: Why the file (or part of it) was ignored. Never contains a value, so it
    #: is safe to return over the API and to log.
    problems: tuple[str, ...] = ()


_CACHE_LOCK = threading.Lock()
#: path -> ((st_mtime_ns, st_size), OverlayLoad). One entry per path; the
#: gateway has exactly one overlay, and a test pointing elsewhere adds one more.
_CACHE: dict[str, tuple[tuple[int, int], OverlayLoad]] = {}
#: Paths whose problems have already been logged, keyed by stamp, so a broken
#: overlay warns once per version of the file instead of once per request.
_WARNED: dict[str, tuple[int, int]] = {}


def invalidate_cache(path: str | os.PathLike[str] | None = None) -> None:
    """Drop the parsed-overlay cache (all of it, or one path)."""
    with _CACHE_LOCK:
        if path is None:
            _CACHE.clear()
            _WARNED.clear()
        else:
            _CACHE.pop(str(path), None)
            _WARNED.pop(str(path), None)


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def read_overlay(path: str | os.PathLike[str]) -> OverlayLoad:
    """The overlay at `path`, parsed and validated. **Never raises.**

    Cached against the file's `(st_mtime_ns, st_size)`, so the per-request cost
    is one `stat()` and any write -- ours or a hand edit -- is picked up on the
    next call. The stamp is re-read after the parse and the result is cached
    only if it did not move, so a file rewritten mid-read is used for this call
    but never becomes the cached answer.
    """
    key = str(path)
    before = _stamp(Path(path))
    if before is None:
        # No file is the normal state: nothing has been overridden yet.
        with _CACHE_LOCK:
            _CACHE.pop(key, None)
        return OverlayLoad()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None and cached[0] == before:
        return cached[1]

    load = _parse_overlay(Path(path))
    after = _stamp(Path(path))
    if after == before:
        with _CACHE_LOCK:
            _CACHE[key] = (before, load)
    _warn_once(key, before, load)
    return load


def _warn_once(key: str, stamp: tuple[int, int], load: OverlayLoad) -> None:
    if not load.problems:
        return
    with _CACHE_LOCK:
        if _WARNED.get(key) == stamp:
            return
        _WARNED[key] = stamp
    logger.warning(
        "runtime config overlay %s: %s. Those values are being ignored and the "
        "gateway is using .env for them; nothing else changed.",
        key,
        "; ".join(load.problems),
    )


def _parse_overlay(path: Path) -> OverlayLoad:
    """Bytes on disk -> a validated values map, degrading at every step."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return OverlayLoad(problems=(f"could not be read ({exc.__class__.__name__})",))
    try:
        payload = json.loads(raw)
    except ValueError:
        return OverlayLoad(problems=("is not valid JSON",))
    if not isinstance(payload, dict):
        return OverlayLoad(problems=(f"is a JSON {type(payload).__name__}, not an object",))
    body = payload.get("values")
    if not isinstance(body, dict):
        # A hand-written flat `{key: value}` map is accepted too -- forgiving,
        # and unambiguous because only `CONFIG_KEYS` names are recognised.
        body = {k: v for k, v in payload.items() if k != "version"}

    values: dict[str, Any] = {}
    problems: list[str] = []
    for name, raw_value in body.items():
        spec = CONFIG_KEYS_BY_NAME.get(name)
        if spec is None:
            problems.append(f"key {name!r} is not a runtime-configurable setting")
            continue
        try:
            values[name] = validate_value(spec, raw_value)
        except ConfigValueError as exc:
            # Deliberately the message, never the value: a rejected secret must
            # not be echoed into a log line.
            problems.append(str(exc))
    return OverlayLoad(values=values, present=True, problems=tuple(problems))


def write_overlay(path: str | os.PathLike[str], values: dict[str, Any]) -> None:
    """Persist `values` as the whole overlay, atomically and `0600`.

    Callers pass the complete desired map (read-modify-write), so there is one
    place that decides what the overlay contains. The temp file is created in
    the destination directory so `os.replace` is an atomic rename on the same
    filesystem, and the mode is set on the temp file *before* the rename, so
    the secrets it carries are never briefly world-readable.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"version": OVERLAY_VERSION, "values": values}, indent=2, sort_keys=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, OVERLAY_FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, OVERLAY_FILE_MODE)
        os.replace(temporary, destination)
    except BaseException:
        # A failed write must not leave a half-file lying next to the real one.
        with contextlib.suppress(OSError):  # pragma: no cover - the rename already consumed it
            os.unlink(temporary)
        raise
    finally:
        invalidate_cache(destination)


# ---------------------------------------------------------------------------
# Where a value came from
# ---------------------------------------------------------------------------

SOURCE_DEFAULT = "default"
SOURCE_ENV = "env"
SOURCE_OVERLAY = "overlay"


def env_provided_vars(env_file: str | os.PathLike[str] | None) -> frozenset[str]:
    """Env-var names `.env` or the process environment actually provide.

    Presence, not value: a `.env` line that happens to repeat the default is
    still the operator having written it down, and reporting it as `default` would
    send them looking in the wrong place. Upper-cased because pydantic-settings
    matches env vars case-insensitively.

    Never raises -- an unreadable `.env` degrades to "the process environment
    is all we can see", which is exactly what `Settings` would then load from.
    """
    provided = {name.upper() for name in os.environ}
    if env_file is None:
        return frozenset(provided)
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - a pydantic-settings dependency
        return frozenset(provided)
    with contextlib.suppress(OSError):  # pragma: no cover - defensive
        provided.update(
            name.upper() for name in dotenv_values(str(env_file)) if isinstance(name, str)
        )
    return frozenset(provided)


def source_of(spec: ConfigKey, overlay_values: dict[str, Any], env_vars: frozenset[str]) -> str:
    if spec.key in overlay_values:
        return SOURCE_OVERLAY
    if spec.env_var.upper() in env_vars:
        return SOURCE_ENV
    return SOURCE_DEFAULT


__all__ = [
    "CAPABILITIES",
    "CAPABILITIES_BY_NAME",
    "CONFIG_KEYS",
    "CONFIG_KEYS_BY_NAME",
    "OVERLAY_FILE_MODE",
    "OVERLAY_VERSION",
    "SECRET_CONFIG_KEYS",
    "SOURCE_DEFAULT",
    "SOURCE_ENV",
    "SOURCE_OVERLAY",
    "Capability",
    "ConfigKey",
    "ConfigValueError",
    "OverlayLoad",
    "env_provided_vars",
    "invalidate_cache",
    "keys_for_capability",
    "read_overlay",
    "source_of",
    "validate_value",
    "write_overlay",
]
