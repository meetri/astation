"""Text-to-speech: `POST /api/speak` and `GET /api/speak/voices` (P5-10).

Two routes. One turns text into audio bytes the app plays as a file; the
other says which voices the configured engine *actually* has, so the app
offers a dropdown instead of asking the owner to type an opaque id.

    POST /api/speak         {"text": str, "voice": str?}  ->  audio bytes
    GET  /api/speak/voices  ->  {"provider", "voices": [...], "voice", ...}

## Why this exists at all

Speech output was `AVSpeechSynthesizer` on the phone and nothing else. That
is a fine floor -- it needs no network, it is why the speak button can never
do nothing -- but it is the only voice the owner had. This route gives them
a choice of engine without giving up the floor: the app still falls back to
on-device synthesis on **every** failure this module can produce.

## The provider split that matters is credentials, not vendor

Hermes names eight TTS providers and has **none of them installed**
(measured twice), so there was nothing to call into and this gateway runs
its own. Grouped by what they need rather than by who sells them:

| Provider | Where it runs | Credential | Shipped here |
|---|---|---|---|
| `piper` | this machine's CPU | none | **yes** |
| `edge` | Microsoft's endpoint | none | **yes** |
| `kittentts` | this machine | none | no -- package absent |
| `neutts` | this machine | none | no -- package absent |
| `elevenlabs` | cloud | key | no |
| `openai` | cloud | key | no |
| `mistral` | cloud | key | no |
| `xai` | cloud | key | no |

The six that are not shipped answer an honest **503 that names what is
missing** -- a package to install, or a key to set -- and never a fallback
to one of the two that work. Getting audio from a provider the owner did
not choose is exactly the failure `api/transcribe.py` refuses, and it is
worse here: the owner would hear a different voice and have no way to know
why.

## `piper`: local, keyless, and it needs a voice file

`piper-tts` is a dependency of this service; the neural voice itself is a
~60 MB `.onnx` + its `.onnx.json`, downloaded once into
`TTS_PIPER_VOICE_DIR` (default `./data/piper-voices`, beside the DB and the
artifact store, inside the already-gitignored `data/`)::

    uv run python -m piper.download_voices \\
        --download-dir data/piper-voices en_US-lessac-medium

That directory is the **only** authority on which piper voices exist. There
is no hardcoded catalog anywhere in this file, because the owner already hit
the other thing: Hermes reported `tts.provider = edge` while `edge_tts` was
not installed, i.e. a configuration that named a capability nothing could
deliver. `GET /api/speak/voices` scans the directory and asks Microsoft,
respectively -- it reports what is *there*.

The loaded model is a lazy singleton per voice behind a `threading.Lock`
(the `api/transcribe.py` pattern, same reason: two concurrent first requests
must not both load it), and synthesis runs through `asyncio.to_thread` so a
long paragraph never blocks the event loop the rest of the gateway serves
on. Measured on this machine, `en_US-lessac-medium`: 0.39 s one-time load,
then 0.21 s to synthesise a 3.7 s sentence -- about 18x realtime.

## `edge`: cloud, keyless

`edge_tts` streams MP3 from Microsoft's endpoint with no credential at all,
which makes it the useful counterweight to piper: a second voice, a second
failure mode, and nothing to configure. Its voice list is a live query, so
"which voices are available" is answered by the service rather than by a
constant in this file.

## Complete files, not chunked transfer -- and where the latency win is

The response is a **complete** audio file with an honest `Content-Length`.
The client is `AVAudioPlayer`, which needs a whole file before it can play
one, so streaming the body would buy the app nothing and cost it a truthful
length. That is the *transport* decision, and it is separate from the
latency decision.

The latency decision lives in the app: generation outpaces speech roughly
10:1, so the number that matters is **time to first audio**, not total. The
app splits the text it is about to speak on sentence boundaries (it already
has them -- `SpeechScript`), asks for sentence 1, starts playing it, and
fetches the rest while it plays. So `POST /api/speak` is deliberately a
small, sharp primitive: synthesise exactly the text you were given, quickly.

## Error contract

| Status | When |
|---|---|
| 413 | `text` longer than `TTS_MAX_INPUT_CHARS` |
| 422 | `text` empty/whitespace, or a `voice` the configured provider does not have |
| 502 | the provider was reachable and failed (edge transport, a synthesis crash) |
| 503 | engine missing, no voice installed, unknown `TTS_PROVIDER`, or a provider this gateway does not implement -- always naming what is missing |

Tests inject fakes by monkeypatching the module-level `synthesize_piper` /
`synthesize_edge` / `list_edge_voices` functions (and the two availability
flags), so no test downloads a voice model or reaches Microsoft.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import wave
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.validators import reject_blank_text
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

#: Authenticated routes (mounted under `/api` in `api.main`).
speak_router = APIRouter(tags=["speak"])

#: Import guards, flags rather than inline imports in the route, for the two
#: reasons `api/transcribe.py` gives: the import cost is paid once, and a test
#: can monkeypatch the "not installed" state without uninstalling anything.
try:  # pragma: no cover - exercised via the flag, not the import machinery
    from piper import PiperVoice, SynthesisConfig  # type: ignore[import-not-found]

    PIPER_AVAILABLE = True
except ImportError:  # pragma: no cover - dep is installed in this project
    PiperVoice = None  # type: ignore[assignment]
    SynthesisConfig = None  # type: ignore[assignment]
    PIPER_AVAILABLE = False

try:  # pragma: no cover - same
    import edge_tts  # type: ignore[import-not-found]

    EDGE_AVAILABLE = True
except ImportError:  # pragma: no cover - dep is installed in this project
    edge_tts = None  # type: ignore[assignment]
    EDGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# The provider registry
# ---------------------------------------------------------------------------


class TTSProvider:
    """One synthesis backend, and what it needs before it can speak.

    `local` and `keyless` are the split the owner actually cares about --
    "does this leave my machine" and "does this need an account" -- and they
    are what the app's picker groups on. `shipped` is the honest third fact:
    this gateway implements two of the eight, and the other six say so.
    """

    __slots__ = (
        "keyless",
        "label",
        "local",
        "mime",
        "missing",
        "name",
        "preferred_voice",
        "shipped",
    )

    def __init__(
        self,
        name: str,
        label: str,
        *,
        local: bool,
        keyless: bool,
        shipped: bool,
        mime: str,
        missing: str = "",
        preferred_voice: str = "",
    ) -> None:
        self.name = name
        self.label = label
        self.local = local
        self.keyless = keyless
        self.shipped = shipped
        self.mime = mime
        self.missing = missing
        #: The engine's own documented default, used ONLY when the owner has
        #: configured no voice AND this id is in the probed list. It is a hint,
        #: never a claim: see `default_voice_id()` for why the alternative
        #: (the first voice in the list) is measurably wrong for `edge`.
        self.preferred_voice = preferred_voice


#: Every provider name this gateway recognises. The four keyed ones and the
#: two unimplemented local ones are listed rather than omitted, because
#: "TTS_PROVIDER is 'neutts' and nothing happens" and "TTS_PROVIDER is
#: 'neutts', which this gateway knows about but cannot run because the package
#: is not installed" are different facts and only the second is actionable.
TTS_PROVIDERS: dict[str, TTSProvider] = {
    "piper": TTSProvider(
        "piper",
        "Piper (this machine)",
        local=True,
        keyless=True,
        shipped=True,
        mime="audio/wav",
    ),
    "edge": TTSProvider(
        "edge",
        "Edge (Microsoft, no key)",
        local=False,
        keyless=True,
        shipped=True,
        mime="audio/mpeg",
        # `edge_tts.Communicate`'s own default. Measured 2026-09-02: the
        # service lists 322 voices ordered by locale, so "the first one" is
        # `af-ZA-AdriNeural` -- Afrikaans, for an English-speaking owner who
        # configured nothing. A default that is wrong in a way the owner can
        # HEAR is worse than one that is arbitrary.
        preferred_voice="en-US-EmmaMultilingualNeural",
    ),
    "kittentts": TTSProvider(
        "kittentts",
        "KittenTTS (this machine)",
        local=True,
        keyless=True,
        shipped=False,
        mime="audio/wav",
        missing=(
            "the kittentts package is not installed in this gateway's "
            "environment, so there is no engine to run"
        ),
    ),
    "neutts": TTSProvider(
        "neutts",
        "NeuTTS (this machine)",
        local=True,
        keyless=True,
        shipped=False,
        mime="audio/wav",
        missing=(
            "the neutts package is not installed in this gateway's "
            "environment, so there is no engine to run"
        ),
    ),
    "elevenlabs": TTSProvider(
        "elevenlabs",
        "ElevenLabs (key)",
        local=False,
        keyless=False,
        shipped=False,
        mime="audio/mpeg",
        missing="this gateway has no ElevenLabs client; set TTS_PROVIDER to 'piper' or 'edge'",
    ),
    "openai": TTSProvider(
        "openai",
        "OpenAI (key)",
        local=False,
        keyless=False,
        shipped=False,
        mime="audio/mpeg",
        missing="this gateway has no OpenAI speech client; set TTS_PROVIDER to 'piper' or 'edge'",
    ),
    "mistral": TTSProvider(
        "mistral",
        "Mistral (key)",
        local=False,
        keyless=False,
        shipped=False,
        mime="audio/mpeg",
        missing="this gateway has no Mistral speech client; set TTS_PROVIDER to 'piper' or 'edge'",
    ),
    "xai": TTSProvider(
        "xai",
        "xAI (key)",
        local=False,
        keyless=False,
        shipped=False,
        mime="audio/mpeg",
        missing="this gateway has no xAI speech client; set TTS_PROVIDER to 'piper' or 'edge'",
    ),
}

#: The `TTS_PROVIDER` choice list, in the order the app renders it: the two
#: that work first, then the rest.
TTS_PROVIDER_NAMES: tuple[str, ...] = tuple(TTS_PROVIDERS)

#: Providers this gateway can actually produce audio from today.
SHIPPED_PROVIDERS: tuple[str, ...] = tuple(
    name for name, spec in TTS_PROVIDERS.items() if spec.shipped
)

#: How long to wait on the cloud provider. Short compared with the rewrite's
#: 90 s: edge answers a sentence in well under a second, and the app has a
#: working on-device voice to fall through to, so waiting a long time for a
#: silent endpoint is strictly worse than failing fast.
EDGE_TIMEOUT_S = 30.0

#: Cap on what `GET /api/speak/voices` reports. Edge lists >300 voices across
#: every language; the list is rendered in a picker and comes off a foreign
#: service, so it is bounded like `api/config.py`'s model list is.
MAX_VOICES = 500


# ---------------------------------------------------------------------------
# piper: local, keyless
# ---------------------------------------------------------------------------

#: Guards the singleton load, held across it: two concurrent first requests
#: must not both load the same ~60 MB ONNX graph. Keyed by resolved model
#: path, so switching voices does not evict the one still in use.
_VOICE_LOCK = threading.Lock()
_VOICES: dict[str, Any] = {}


def piper_voice_dir(settings: Settings) -> Path:
    """The configured voice directory, resolved against the service root.

    Relative by default (`./data/piper-voices`) for the same reason the DB
    and artifact paths are: the service is started from
    `services/research-gateway/`, and an absolute default would be wrong on
    every machine but one.
    """
    return Path(settings.tts_piper_voice_dir).expanduser()


def piper_installed_voices(directory: Path) -> list[dict[str, Any]]:
    """Every usable piper voice in `directory`. **Scanned, never assumed.**

    A voice is usable only when both halves are present: `X.onnx` is the
    model and `X.onnx.json` carries the sample rate and the phoneme map
    without which the model cannot be run. A lone `.onnx` is a half-finished
    download, and reporting it would put an id in the app's picker that
    fails the moment it is chosen.

    The config JSON also carries the language and the dataset name, which is
    all the app needs to render a readable row -- so the label comes from the
    voice's own file rather than from a table in this module that could
    disagree with it.
    """
    if not directory.is_dir():
        return []
    voices: list[dict[str, Any]] = []
    for model in sorted(directory.glob("*.onnx")):
        config = model.with_suffix(".onnx.json")
        if not config.is_file():
            logger.warning(
                "piper voice %s has no %s beside it; skipping (incomplete download)",
                model.name,
                config.name,
            )
            continue
        details: dict[str, Any] = {}
        try:
            details = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Unreadable config: the voice still cannot be loaded, but say so
            # with the id rather than dropping it silently.
            logger.warning("piper voice config %s could not be read", config)
            continue
        language = details.get("language") or {}
        locale = language.get("code") if isinstance(language, dict) else None
        quality = (details.get("audio") or {}).get("quality")
        voices.append(
            {
                "id": model.stem,
                "name": model.stem,
                "locale": locale if isinstance(locale, str) else None,
                "provider": "piper",
                "description": " ".join(
                    part
                    for part in (
                        language.get("name_english") if isinstance(language, dict) else None,
                        f"{quality} quality" if isinstance(quality, str) else None,
                    )
                    if part
                ),
            }
        )
    return voices[:MAX_VOICES]


def _load_piper_voice(model_path: Path) -> Any:
    """The process-wide `PiperVoice` for `model_path`, loaded at most once."""
    key = str(model_path)
    with _VOICE_LOCK:
        voice = _VOICES.get(key)
        if voice is None:
            logger.info("loading piper voice %s (first use)", model_path.name)
            voice = PiperVoice.load(model_path)
            _VOICES[key] = voice
        return voice


def synthesize_piper(text: str, *, model_path: Path, length_scale: float | None) -> bytes:
    """One complete WAV, synthesised on this machine's CPU.

    Module-level on purpose: this is the injection point the route tests
    replace with a fake, exactly as `api/transcribe.py` exposes
    `transcribe_local`, so the whole HTTP surface is testable without a
    60 MB model download.

    `voice.synthesize()` yields one `AudioChunk` per sentence, which is why
    the frames are concatenated rather than assumed to be one blob -- and it
    is the same property that would let this stream if the client could use
    a partial file. It cannot (`AVAudioPlayer` needs a whole one), so the
    frames are written into a single in-memory WAV with a correct header,
    and the honest `Content-Length` that comes with it.
    """
    voice = _load_piper_voice(model_path)
    config = None
    if length_scale is not None:
        config = SynthesisConfig(length_scale=length_scale)

    frames: list[bytes] = []
    sample_rate = 22050
    sample_width = 2
    channels = 1
    for chunk in voice.synthesize(text, syn_config=config):
        frames.append(chunk.audio_int16_bytes)
        sample_rate = chunk.sample_rate
        sample_width = chunk.sample_width
        channels = chunk.sample_channels

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(frames))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# edge: cloud, keyless
# ---------------------------------------------------------------------------


async def list_edge_voices() -> list[dict[str, Any]]:
    """Ask Microsoft which voices it serves. **A live query, not a table.**

    Module-level for the same injection reason as the synth functions, and a
    live query for the reason in the module docstring: a hardcoded list is
    how a gateway ends up claiming a capability it does not have.
    """
    raw = await edge_tts.list_voices()
    voices: list[dict[str, Any]] = []
    for entry in raw[:MAX_VOICES]:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("ShortName")
        if not isinstance(identifier, str) or not identifier:
            continue
        tags = entry.get("VoiceTag") or {}
        personalities = tags.get("VoicePersonalities") if isinstance(tags, dict) else None
        voices.append(
            {
                "id": identifier,
                "name": entry.get("FriendlyName") or identifier,
                "locale": entry.get("Locale"),
                "provider": "edge",
                "description": ", ".join(str(tag) for tag in personalities if isinstance(tag, str))
                if isinstance(personalities, list)
                else (entry.get("Gender") or ""),
            }
        )
    return voices


async def synthesize_edge(text: str, *, voice: str, rate: str) -> bytes:
    """One complete MP3 from Microsoft's endpoint. No credential at all.

    `Communicate.stream()` yields `audio` frames as they arrive; they are
    concatenated here for the same complete-file reason piper's are.
    """
    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=rate,
        connect_timeout=10,
        receive_timeout=int(EDGE_TIMEOUT_S),
    )
    frames: list[bytes] = []
    async for chunk in communicate.stream():
        if isinstance(chunk, dict) and chunk.get("type") == "audio":
            data = chunk.get("data")
            if isinstance(data, (bytes, bytearray)):
                frames.append(bytes(data))
    return b"".join(frames)


# ---------------------------------------------------------------------------
# Request / shared resolution
# ---------------------------------------------------------------------------


class SpeakRequest(BaseModel):
    """Body for `POST /api/speak`.

    Closed schema like every other body in this service: a typo'd field is a
    422, never a silently-dropped instruction. `voice` is the one thing a
    request may choose -- the provider is the owner's setting, and a request
    that could pick one would let a client route the owner's text to a
    service they had not chosen.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    #: Optional. A voice id from `GET /api/speak/voices`. Absent means the
    #: server-configured `TTS_VOICE`, and an unknown one is a 422 rather than
    #: a quiet substitution -- hearing a different voice with no explanation
    #: is the TTS-shaped version of a silent fallback.
    voice: str | None = None

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        return reject_blank_text(value)


def resolve_provider(settings: Settings) -> TTSProvider:
    """The configured provider, or a 503 that names what is wrong.

    Three refusals, each naming its own fix: an unknown name, a provider this
    gateway does not implement, and an implemented provider whose package is
    absent. None of them falls through to a provider that would work.
    """
    name = settings.tts_provider.strip().lower()
    spec = TTS_PROVIDERS.get(name)
    if spec is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"TTS_PROVIDER is {settings.tts_provider!r}; expected one of "
                f"{', '.join(TTS_PROVIDER_NAMES)}"
            ),
        )
    if not spec.shipped:
        raise HTTPException(
            status_code=503,
            detail=(
                f"TTS_PROVIDER is {spec.name!r} but {spec.missing}. This "
                f"gateway ships {' and '.join(SHIPPED_PROVIDERS)} and never "
                "falls back to a provider you did not choose."
            ),
        )
    if spec.name == "piper" and not PIPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "local speech synthesis is unavailable: the piper-tts package "
                "is not installed in this gateway's environment "
                "(`uv add piper-tts`), and this gateway never falls back to a "
                "provider you did not choose"
            ),
        )
    if spec.name == "edge" and not EDGE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "TTS_PROVIDER is 'edge' but the edge-tts package is not "
                "installed in this gateway's environment (`uv add edge-tts`). "
                "This is exactly the state a provider name alone cannot tell "
                "you about, which is why GET /api/speak/voices probes instead "
                "of reporting the setting back."
            ),
        )
    return spec


async def available_voices(settings: Settings, spec: TTSProvider) -> list[dict[str, Any]]:
    """What the configured provider really has. Probed, never hardcoded.

    A transport failure asking the cloud provider is a **502**: the engine is
    installed and configured, and the network is what went wrong.
    """
    if spec.name == "piper":
        return await asyncio.to_thread(piper_installed_voices, piper_voice_dir(settings))
    try:
        return await list_edge_voices()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"could not ask the edge speech service which voices it "
                f"serves: {exc.__class__.__name__}: {exc}"
            ),
        ) from exc


def default_voice_id(voices: list[dict[str, Any]], spec: TTSProvider) -> str | None:
    """Which voice to use when the owner has configured none.

    The engine's own documented default **if the probe actually found it**,
    else the first voice in the list. The order matters and is measured:
    `edge` lists 322 voices ordered by locale, so "the first one" is
    `af-ZA-AdriNeural` -- Afrikaans, for an owner who set nothing and speaks
    English. Preferring the engine's own default fixes that without inventing
    a catalog, because the hint is only ever used when the live probe
    confirms it exists.
    """
    if not voices:
        return None
    if spec.preferred_voice:
        for voice in voices:
            if voice.get("id") == spec.preferred_voice:
                return spec.preferred_voice
    return str(voices[0]["id"])


def resolve_voice(
    voices: list[dict[str, Any]],
    *,
    spec: TTSProvider,
    requested: str | None,
    configured: str,
) -> str:
    """Which voice id to speak with: the request's, the setting's, or the first.

    * A **requested** voice the provider does not have is a **422** listing a
      few that it does -- a client error, and one the app can correct.
    * A **configured** voice the provider does not have is a **503**: the
      owner's setting names something that is not there, which is server
      configuration, not a bad request. Naming it is the fix.
    * Neither set, and voices exist: `default_voice_id()` decides. That is a
      default *within* the chosen provider, not a fallback to another one, and
      the id used is reported back in `X-TTS-Voice` so nothing is silent.
    * No voices at all: a **503** naming how to get one.
    """
    if not voices:
        if spec.name == "piper":
            raise HTTPException(
                status_code=503,
                detail=(
                    "no piper voice is installed. Download one into "
                    "TTS_PIPER_VOICE_DIR, e.g. `uv run python -m "
                    "piper.download_voices --download-dir data/piper-voices "
                    "en_US-lessac-medium`"
                ),
            )
        raise HTTPException(
            status_code=503,
            detail=f"the {spec.name} speech service listed no voices",
        )

    known = {voice["id"] for voice in voices}
    if requested and requested.strip():
        wanted = requested.strip()
        if wanted not in known:
            sample = ", ".join(sorted(known)[:8])
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{spec.name} has no voice {wanted!r}. Available ids "
                    f"include: {sample}. GET /api/speak/voices lists them all."
                ),
            )
        return wanted

    chosen = configured.strip()
    if chosen:
        if chosen not in known:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"TTS_VOICE is {chosen!r} but {spec.name} has no such "
                    "voice. Change it in Settings > Models & voice, or reset "
                    "it to fall back to TTS_VOICE in .env."
                ),
            )
        return chosen
    fallback = default_voice_id(voices, spec)
    assert fallback is not None  # `voices` is non-empty above
    return fallback


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@speak_router.post("/speak")
async def speak(body: SpeakRequest) -> Response:
    """Synthesise `text` and answer with the audio bytes (P5-10).

    Body: `{"text": str, "voice": str?}`. Answers the audio itself -- not
    JSON, not base64 -- with the provider's own `Content-Type`
    (`audio/wav` for piper, `audio/mpeg` for edge) and a `Content-Length`,
    because the client writes it to a file and plays it.

    Three response headers carry the provenance the app shows, and are the
    reason nothing here has to be inferred from the request: `X-TTS-Provider`,
    `X-TTS-Voice`, and `X-TTS-Characters`.

    Provider and voice come from server settings (`TTS_*`) unless the body
    names a voice; a request can never choose the provider. The full error
    contract is in the module docstring -- and the one that matters most is
    that there isn't a silent fallback in it: an engine that cannot run says
    so with a 503 and the app reads the reply on-device instead.
    """
    settings: Settings = get_settings()
    spec = resolve_provider(settings)

    text = body.text
    cap = settings.tts_max_input_chars
    if len(text) > cap:
        raise HTTPException(
            status_code=413,
            detail=(
                f"text is {len(text)} characters; the speech cap is {cap} "
                "(TTS_MAX_INPUT_CHARS). Speak it a sentence or a section at a "
                "time -- that is faster to first audio anyway."
            ),
        )

    voices = await available_voices(settings, spec)
    voice_id = resolve_voice(voices, spec=spec, requested=body.voice, configured=settings.tts_voice)

    if spec.name == "piper":
        model_path = piper_voice_dir(settings) / f"{voice_id}.onnx"
        try:
            audio = await asyncio.to_thread(
                synthesize_piper,
                text,
                model_path=model_path,
                length_scale=settings.tts_piper_length_scale or None,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"piper could not synthesise the text with voice "
                    f"{voice_id!r}: {exc.__class__.__name__}: {exc}"
                ),
            ) from exc
    else:
        try:
            audio = await synthesize_edge(text, voice=voice_id, rate=settings.tts_edge_rate)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"the edge speech service failed on voice {voice_id!r}: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            ) from exc

    if not audio:
        # An engine that answers with zero bytes has failed, and saying so is
        # the difference between the app falling back and the app playing
        # silence -- which the owner would read as a broken button.
        raise HTTPException(
            status_code=502,
            detail=(
                f"{spec.name} returned no audio for {len(text)} characters with voice {voice_id!r}"
            ),
        )

    logger.info(
        "spoke %d chars as %d bytes via %s (%s)",
        len(text),
        len(audio),
        spec.name,
        voice_id,
    )
    return Response(
        content=audio,
        media_type=spec.mime,
        headers={
            "Content-Length": str(len(audio)),
            "X-TTS-Provider": spec.name,
            "X-TTS-Voice": voice_id,
            "X-TTS-Characters": str(len(text)),
        },
    )


@speak_router.get("/speak/voices")
async def speak_voices() -> dict[str, Any]:
    """Which voices the configured provider **actually** has (P5-10).

    Response: `{"provider", "provider_label", "local", "keyless", "voice",
    "count", "voices": [{"id", "name", "locale", "provider", "description"}],
    "detail"}`.

    Probed, not declared. For `piper` that is a scan of `TTS_PIPER_VOICE_DIR`
    for `.onnx` files that have their `.onnx.json` beside them; for `edge` it
    is a live query to the service. The owner already hit the failure this
    prevents: a configuration reporting `tts.provider = edge` while
    `edge_tts` was not installed -- a claim about a capability nothing could
    deliver.

    **An installed engine with no voices is a 200 with an empty list**, not a
    503: "there is nothing here yet" is an answer a picker can render, and
    `detail` says how to fix it. Only a missing *engine* (or an unknown
    provider) is a 503, and a provider that failed while being asked is a 502
    -- the same three-way honesty `POST /api/config/providers/probe` keeps.
    """
    settings: Settings = get_settings()
    spec = resolve_provider(settings)
    voices = await available_voices(settings, spec)

    configured = settings.tts_voice.strip()
    known = {voice["id"] for voice in voices}
    # Resolved through exactly the same rule `POST /api/speak` uses, so the
    # picker can never show one default while the button speaks another.
    selected = configured if configured in known else default_voice_id(voices, spec)

    if not voices and spec.name == "piper":
        detail = (
            f"no piper voice is installed in {piper_voice_dir(settings)}. "
            "Download one with `uv run python -m piper.download_voices "
            "--download-dir data/piper-voices en_US-lessac-medium`."
        )
    elif not voices:
        detail = f"the {spec.name} speech service listed no voices"
    elif configured and configured not in known:
        detail = (
            f"TTS_VOICE is {configured!r}, which {spec.name} does not have; "
            f"{selected!r} would be used instead. Pick one below to fix it."
        )
    else:
        detail = f"{len(voices)} voice(s) available from {spec.name}"

    return {
        "provider": spec.name,
        "provider_label": spec.label,
        "local": spec.local,
        "keyless": spec.keyless,
        "voice": selected,
        "count": len(voices),
        "voices": voices,
        "detail": detail,
    }


__all__ = [
    "EDGE_AVAILABLE",
    "EDGE_TIMEOUT_S",
    "MAX_VOICES",
    "PIPER_AVAILABLE",
    "SHIPPED_PROVIDERS",
    "TTS_PROVIDERS",
    "TTS_PROVIDER_NAMES",
    "SpeakRequest",
    "TTSProvider",
    "available_voices",
    "default_voice_id",
    "list_edge_voices",
    "piper_installed_voices",
    "piper_voice_dir",
    "resolve_provider",
    "resolve_voice",
    "speak",
    "speak_router",
    "speak_voices",
    "synthesize_edge",
    "synthesize_piper",
]
