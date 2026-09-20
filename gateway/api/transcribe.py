"""Speech-to-text: `POST /api/transcribe` (P5-2a)."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Form, HTTPException, UploadFile

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

transcribe_router = APIRouter(tags=["transcribe"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Enforced for the local provider only; a cloud provider exposes no pre-decode duration.
MAX_AUDIO_DURATION_S = 150.0

_UPLOAD_CHUNK_BYTES = 64 * 1024

_CLOUD_TIMEOUT_S = 120.0

_FALLBACK_AUDIO_MIME = "application/octet-stream"

try:  # pragma: no cover - exercised via the flag, not the import machinery
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    FASTER_WHISPER_AVAILABLE = True
except ImportError:  # pragma: no cover - dep is installed in this project
    WhisperModel = None  # type: ignore[assignment]
    FASTER_WHISPER_AVAILABLE = False


class AudioTooLongError(Exception):
    """The decoded clip exceeds `MAX_AUDIO_DURATION_S` (local provider)."""

    def __init__(self, duration_s: float) -> None:
        self.duration_s = duration_s
        super().__init__(f"audio is {duration_s:.1f}s long; the cap is {MAX_AUDIO_DURATION_S:.0f}s")


def combined_vocab_hint(server_hint: str, request_hint: str | None) -> str | None:
    """Server-configured jargon first, the request's hint appended."""
    parts = [p.strip() for p in (server_hint, request_hint or "") if p and p.strip()]
    return ", ".join(parts) or None


# Held across the load so a concurrent first request waits instead of loading its own.
_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_NAME: str | None = None


def _get_local_model(model_name: str) -> Any:
    """The process-wide WhisperModel, loaded at most once per model name."""
    global _MODEL, _MODEL_NAME
    with _MODEL_LOCK:
        if _MODEL is None or model_name != _MODEL_NAME:
            logger.info("loading faster-whisper model %r (first use)", model_name)
            _MODEL = WhisperModel(model_name, device="cpu", compute_type="int8")
            _MODEL_NAME = model_name
        return _MODEL


# Module-level so tests can replace it with a fake engine instead of downloading a model.
def transcribe_local(
    audio: bytes,
    *,
    model_name: str,
    language: str | None,
    initial_prompt: str | None,
) -> dict[str, Any]:
    """Synchronous faster-whisper transcription (run via `asyncio.to_thread`)."""
    model = _get_local_model(model_name)
    segments, info = model.transcribe(
        io.BytesIO(audio),
        language=language or None,
        initial_prompt=initial_prompt,
    )
    # info.duration is known before any segment decodes; the cap must be checked here.
    duration = getattr(info, "duration", None)
    if duration is not None and duration > MAX_AUDIO_DURATION_S:
        raise AudioTooLongError(float(duration))
    text = "".join(segment.text for segment in segments).strip()
    return {
        "text": text,
        "language": getattr(info, "language", None),
        "duration_s": float(duration) if duration is not None else None,
    }


_CLOUD_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_field": "stt_openai_key",
        "key_env": "STT_OPENAI_KEY",
        "model_field": "stt_openai_model",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_field": "stt_groq_key",
        "key_env": "STT_GROQ_KEY",
        "model_field": "stt_groq_model",
    },
}


async def transcribe_cloud(
    provider: str,
    audio: bytes,
    *,
    filename: str,
    mime_type: str,
    model: str,
    api_key: str,
    language: str | None,
    prompt: str | None,
) -> dict[str, Any]:
    """One multipart POST to `{base}/audio/transcriptions`, Bearer-keyed."""
    config = _CLOUD_PROVIDERS[provider]
    url = f"{config['base_url']}/audio/transcriptions"
    data: dict[str, str] = {"model": model, "response_format": "verbose_json"}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    try:
        async with httpx.AsyncClient(timeout=_CLOUD_TIMEOUT_S) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                data=data,
                files={"file": (filename, audio, mime_type)},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"could not reach the {provider} transcription API: {exc}",
        ) from exc
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"the {provider} transcription API answered "
                f"HTTP {response.status_code}: {response.text[:300]}"
            ),
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"the {provider} transcription API answered non-JSON",
        ) from exc
    duration = payload.get("duration")
    return {
        "text": payload.get("text") or "",
        "language": payload.get("language"),
        "duration_s": float(duration) if isinstance(duration, (int, float)) else None,
    }


async def _read_capped(file: UploadFile) -> bytes:
    """The whole upload, streamed with the cap enforced mid-stream."""
    buffer = bytearray()
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"audio exceeds the {MAX_AUDIO_BYTES} B cap",
            )
    if not buffer:
        raise HTTPException(status_code=422, detail="audio upload is empty (0 bytes)")
    return bytes(buffer)


def _transcription_response(result: dict[str, Any], *, provider: str, model: str) -> dict[str, Any]:
    return {
        "transcription": {
            "text": result.get("text") or "",
            "language": result.get("language"),
            "duration_s": result.get("duration_s"),
            "provider": provider,
            "model": model,
        }
    }


# MIME and extension are deliberately not gated; the decoder decides what is audio.
@transcribe_router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    vocab_hint: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Transcribe one uploaded audio clip (P5-2a)."""
    settings: Settings = get_settings()
    audio = await _read_capped(file)
    hint = combined_vocab_hint(settings.stt_vocab_hint, vocab_hint)
    language = settings.stt_language.strip() or None
    provider = settings.stt_provider.strip().lower() or "local"

    if provider == "local":
        if not FASTER_WHISPER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail=(
                    "local transcription is unavailable: the faster-whisper "
                    "package is not installed in this gateway's environment "
                    "(`uv add faster-whisper`), and this gateway never falls "
                    "back to a cloud provider silently"
                ),
            )
        model_name = settings.stt_local_model
        try:
            result = await asyncio.to_thread(
                transcribe_local,
                audio,
                model_name=model_name,
                language=language,
                initial_prompt=hint,
            )
        except AudioTooLongError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # A decode failure here means undecodable bytes, a client error: 422, not 500.
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(f"could not transcribe the audio: {exc.__class__.__name__}: {exc}"),
            ) from exc
        return _transcription_response(result, provider="local", model=model_name)

    if provider in _CLOUD_PROVIDERS:
        config = _CLOUD_PROVIDERS[provider]
        api_key = getattr(settings, config["key_field"]).get_secret_value().strip()
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"STT_PROVIDER is {provider!r} but {config['key_env']} is "
                    "not set; add the key to the gateway's .env or switch "
                    "STT_PROVIDER back to 'local'"
                ),
            )
        model = getattr(settings, config["model_field"])
        result = await transcribe_cloud(
            provider,
            audio,
            filename=file.filename or "audio.m4a",
            mime_type=(file.content_type or _FALLBACK_AUDIO_MIME),
            model=model,
            api_key=api_key,
            language=language,
            prompt=hint,
        )
        return _transcription_response(result, provider=provider, model=model)

    raise HTTPException(
        status_code=503,
        detail=(
            f"STT_PROVIDER is {settings.stt_provider!r}; expected one of 'local', 'openai', 'groq'"
        ),
    )


__all__ = [
    "FASTER_WHISPER_AVAILABLE",
    "MAX_AUDIO_BYTES",
    "MAX_AUDIO_DURATION_S",
    "AudioTooLongError",
    "combined_vocab_hint",
    "transcribe_audio",
    "transcribe_cloud",
    "transcribe_local",
    "transcribe_router",
]
