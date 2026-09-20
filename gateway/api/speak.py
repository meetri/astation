"""Text-to-speech: `POST /api/speak` and `GET /api/speak/voices` (P5-10)."""

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

speak_router = APIRouter(tags=["speak"])

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


class TTSProvider:
    """One synthesis backend, and what it needs before it can speak."""

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
        self.preferred_voice = preferred_voice


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
        # Edge lists by locale, so "the first voice" is Afrikaans; used only if probed.
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

TTS_PROVIDER_NAMES: tuple[str, ...] = tuple(TTS_PROVIDERS)

SHIPPED_PROVIDERS: tuple[str, ...] = tuple(
    name for name, spec in TTS_PROVIDERS.items() if spec.shipped
)

EDGE_TIMEOUT_S = 30.0

MAX_VOICES = 500


# Held across the load: two concurrent first requests must not both load the model.
_VOICE_LOCK = threading.Lock()
_VOICES: dict[str, Any] = {}


def piper_voice_dir(settings: Settings) -> Path:
    """The configured voice directory, resolved against the service root."""
    return Path(settings.tts_piper_voice_dir).expanduser()


def piper_installed_voices(directory: Path) -> list[dict[str, Any]]:
    """Every usable piper voice in `directory`. **Scanned, never assumed.**"""
    if not directory.is_dir():
        return []
    voices: list[dict[str, Any]] = []
    for model in sorted(directory.glob("*.onnx")):
        config = model.with_suffix(".onnx.json")
        # A lone .onnx is a half-finished download: the .json holds the sample rate.
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
    """One complete WAV, synthesised on this machine's CPU."""
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


async def list_edge_voices() -> list[dict[str, Any]]:
    """Ask Microsoft which voices it serves. **A live query, not a table.**"""
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
    """One complete MP3 from Microsoft's endpoint. No credential at all."""
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


# No provider field on purpose: a request must not route the text to another service.
class SpeakRequest(BaseModel):
    """Body for `POST /api/speak`."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    voice: str | None = None

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        return reject_blank_text(value)


def resolve_provider(settings: Settings) -> TTSProvider:
    """The configured provider, or a 503 that names what is wrong."""
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
    """What the configured provider really has. Probed, never hardcoded."""
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
    """Which voice to use when the operator has configured none."""
    if not voices:
        return None
    if spec.preferred_voice:
        for voice in voices:
            if voice.get("id") == spec.preferred_voice:
                return spec.preferred_voice
    return str(voices[0]["id"])


# A requested voice is 422, a configured one 503: a bad setting is not a bad request.
def resolve_voice(
    voices: list[dict[str, Any]],
    *,
    spec: TTSProvider,
    requested: str | None,
    configured: str,
) -> str:
    """Which voice id to speak with: the request's, the setting's, or the first."""
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
    assert fallback is not None
    return fallback


@speak_router.post("/speak")
async def speak(body: SpeakRequest) -> Response:
    """Synthesise `text` and answer with the audio bytes (P5-10)."""
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
    """Which voices the configured provider **actually** has (P5-10)."""
    settings: Settings = get_settings()
    spec = resolve_provider(settings)
    voices = await available_voices(settings, spec)

    configured = settings.tts_voice.strip()
    known = {voice["id"] for voice in voices}
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
