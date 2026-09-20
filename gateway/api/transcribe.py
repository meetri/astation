"""Speech-to-text: `POST /api/transcribe` (P5-2a).

One route: a multipart audio upload (m4a/aac, wav, mp3, ogg -- whatever the
phone's recorder produces) in, a transcription out:

    {"transcription": {"text", "language", "duration_s", "provider", "model"}}

Provider abstraction mirroring Hermes's own transcription-tool conventions
(`STT_*` settings, provider names `local` | `openai` | `groq`):

* **`local` (the default)** runs faster-whisper in-process. The model is a
  lazy singleton behind a `threading.Lock` -- two concurrent first requests
  must not both load a multi-hundred-MB model -- and the transcription itself
  runs via `asyncio.to_thread`, so a 30-second decode never blocks the event
  loop the rest of the gateway is serving on. faster-whisper decodes through
  its bundled PyAV, so no system ffmpeg is needed. If `faster_whisper` is not
  importable the answer is an honest 503 naming the missing package -- never
  a silent fallback to a cloud provider the operator did not configure.
* **`openai` / `groq`** are keyed passthroughs: an httpx multipart POST to
  `{base}/audio/transcriptions` with a Bearer key. A missing key is an honest
  503 with the env var named.

Vocabulary biasing -- the direct answer to the operator's technical-words
complaint: the server-configured `STT_VOCAB_HINT` and the request's optional
`vocab_hint` form field are combined (server hint first, request hint
appended) and passed to Whisper as `initial_prompt` (local) / `prompt`
(cloud), which biases decoding toward the jargon it names.

Bounds, honestly stated:

* **25 MB upload cap, enforced mid-stream** (413 past it, same `_capped`
  pattern as `api/attachments.py`); an empty upload is a 422.
* **~150 s duration cap, local only.** faster-whisper's
  `model.transcribe(...)` returns `(segments_generator, info)` and
  `info.duration` is known *before* any segment is decoded, so an over-long
  clip is refused (422) before the expensive part starts. For the cloud
  providers there is no pre-decode duration to inspect -- the 25 MB byte cap
  is the only bound this gateway can actually enforce on them, and this
  docstring says so rather than pretending otherwise.

The declared MIME/extension is deliberately NOT a gate: pickers and recorders
lie about audio container types constantly, and the decoder (PyAV locally,
Whisper's own upstream) is the authority on whether the bytes are decodable
audio. Undecodable bytes come back as a 422 naming the decode failure.

Tests inject a fake engine by monkeypatching the module-level
`transcribe_local` function (and `FASTER_WHISPER_AVAILABLE` for the
missing-package path), so no test downloads a Whisper model or touches the
network.
"""

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

#: Authenticated routes (mounted under `/api` in `api.main`).
transcribe_router = APIRouter(tags=["transcribe"])

#: Upload size cap (task spec: 25 MB). Enforced while streaming the upload,
#: so an oversized body is refused at the cap, not after buffering.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

#: Duration cap (task spec: ~2.5 min). Enforced for the local provider only
#: -- see the module docstring for why the cloud providers can't have one.
MAX_AUDIO_DURATION_S = 150.0

#: Chunk size when reading the inbound upload stream (attachments pattern).
_UPLOAD_CHUNK_BYTES = 64 * 1024

#: Cloud passthrough request timeout. Whisper-as-a-service on a <=25 MB clip
#: is tens of seconds worst case; 120 s is generous without hanging forever.
_CLOUD_TIMEOUT_S = 120.0

_FALLBACK_AUDIO_MIME = "application/octet-stream"

#: Import guard for the default provider. A flag rather than an inline
#: try/import in the route so (a) the import cost is paid once, and (b) tests
#: can monkeypatch the "not installed" state without uninstalling anything.
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


# ---------------------------------------------------------------------------
# Vocab hint
# ---------------------------------------------------------------------------


def combined_vocab_hint(server_hint: str, request_hint: str | None) -> str | None:
    """Server-configured jargon first, the request's hint appended.

    Joined with ", " -- both are glossary-style term lists and Whisper's
    prompt biasing reads a comma-separated glossary naturally. None (rather
    than "") when there is nothing to say, so the engines receive no prompt
    argument at all instead of an empty one.
    """
    parts = [p.strip() for p in (server_hint, request_hint or "") if p and p.strip()]
    return ", ".join(parts) or None


# ---------------------------------------------------------------------------
# Local engine: faster-whisper, lazy singleton, injectable
# ---------------------------------------------------------------------------

#: Guards the singleton load: two concurrent first requests must not both
#: load the model (a multi-hundred-MB allocation each). Held across the load,
#: so the second request waits for the first's model instead of duplicating it.
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


def transcribe_local(
    audio: bytes,
    *,
    model_name: str,
    language: str | None,
    initial_prompt: str | None,
) -> dict[str, Any]:
    """Synchronous faster-whisper transcription (run via `asyncio.to_thread`).

    Module-level on purpose: this is the injection point the tests replace
    with a fake engine, so the whole HTTP surface is testable without a model
    download.

    faster-whisper returns `(segments_generator, info)` with `info.duration`
    available BEFORE any segment is decoded, so the duration cap is enforced
    here, before the expensive part -- `AudioTooLongError` if the clip is
    over `MAX_AUDIO_DURATION_S`.
    """
    model = _get_local_model(model_name)
    segments, info = model.transcribe(
        io.BytesIO(audio),
        language=language or None,
        initial_prompt=initial_prompt,
    )
    duration = getattr(info, "duration", None)
    if duration is not None and duration > MAX_AUDIO_DURATION_S:
        raise AudioTooLongError(float(duration))
    text = "".join(segment.text for segment in segments).strip()
    return {
        "text": text,
        "language": getattr(info, "language", None),
        "duration_s": float(duration) if duration is not None else None,
    }


# ---------------------------------------------------------------------------
# Cloud engines: keyed passthrough to an OpenAI-shaped /audio/transcriptions
# ---------------------------------------------------------------------------

#: base URL + which settings fields hold the key/model, per cloud provider.
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
    """One multipart POST to `{base}/audio/transcriptions`, Bearer-keyed.

    `response_format=verbose_json` so the answer carries `language` and
    `duration` alongside `text`; both are read tolerantly (null when absent)
    since only `text` is contractual across providers.
    """
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
        # The upstream error body is short structured JSON in practice and
        # never echoes the Bearer key; bounded anyway.
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


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


async def _read_capped(file: UploadFile) -> bytes:
    """The whole upload, streamed with the cap enforced mid-stream.

    Same shape as `api/attachments.py`'s `_capped`: the 413 fires at the
    first chunk past the cap, not after the body has been buffered. 422 on
    empty -- zero bytes of audio is a client error, not a transcription.
    """
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


@transcribe_router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    vocab_hint: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Transcribe one uploaded audio clip (P5-2a).

    Multipart field `file` (audio: m4a/aac, wav, mp3, ogg), optional form
    field `vocab_hint` -- extra jargon appended to the server-configured
    `STT_VOCAB_HINT` for Whisper prompt biasing. Provider/model come from
    server settings (`STT_PROVIDER`, default `local`); a request cannot
    choose them. Bounds and error contract are in the module docstring:
    413 past 25 MB, 422 on empty/undecodable/over-150s audio (duration cap
    local-only), 503 for a missing engine or key -- never a silent fallback.
    """
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
        except Exception as exc:
            # PyAV raising here almost always means the bytes are not
            # decodable audio -- a client error, named honestly either way.
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
