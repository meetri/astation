"""Research Gateway settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import runtime_config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _REPO_ROOT / ".env"

ENV_FILE = _ENV_FILE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hermes_scheme: str = Field(alias="HERMES_SCHEME", default="http")
    hermes_host: str = Field(alias="HERMES_HOST", default="127.0.0.1")
    hermes_port: int = Field(alias="HERMES_PORT", default=9119)
    hermes_username: str = Field(alias="HERMES_USERNAME", default="")
    hermes_password: SecretStr = Field(alias="HERMES_PASSWORD", default=SecretStr(""))

    research_gateway_username: str = Field(alias="RESEARCH_GATEWAY_USERNAME", default="")
    research_gateway_password: SecretStr = Field(
        alias="RESEARCH_GATEWAY_PASSWORD", default=SecretStr("")
    )

    hermes_sandbox_root: str = Field(alias="HERMES_SANDBOX_ROOT", default="/opt/data")

    artifact_ignore_dirs: str = Field(alias="ARTIFACT_IGNORE_DIRS", default="")
    artifact_ignore_globs: str = Field(alias="ARTIFACT_IGNORE_GLOBS", default="")
    hermes_sandbox_denylist_dirs: str = Field(alias="HERMES_SANDBOX_DENYLIST_DIRS", default="")
    hermes_sandbox_denylist_globs: str = Field(alias="HERMES_SANDBOX_DENYLIST_GLOBS", default="")

    hermes_attachment_dir: str = Field(
        alias="HERMES_ATTACHMENT_DIR", default="/opt/data/attachments"
    )

    research_gateway_view_roots: str = Field(alias="RESEARCH_GATEWAY_VIEW_ROOTS", default="")

    audit_clickhouse_url: str = Field(alias="AUDIT_CLICKHOUSE_URL", default="")
    audit_clickhouse_user: str = Field(alias="AUDIT_CLICKHOUSE_USER", default="reader")
    audit_clickhouse_password: SecretStr = Field(
        alias="AUDIT_CLICKHOUSE_PASSWORD", default=SecretStr("")
    )
    audit_ingest_url: str = Field(alias="AUDIT_INGEST_URL", default="")
    audit_host_label: str = Field(alias="AUDIT_HOST_LABEL", default="")
    audit_query_max_rows: int = Field(alias="AUDIT_QUERY_MAX_ROWS", default=10_000)
    audit_query_timeout_s: float = Field(alias="AUDIT_QUERY_TIMEOUT_S", default=30.0)

    research_gateway_public_base_url: str = Field(
        alias="RESEARCH_GATEWAY_PUBLIC_BASE_URL", default=""
    )

    stt_provider: str = Field(alias="STT_PROVIDER", default="local")
    stt_local_model: str = Field(alias="STT_LOCAL_MODEL", default="small")
    stt_language: str = Field(alias="STT_LANGUAGE", default="en")
    stt_vocab_hint: str = Field(alias="STT_VOCAB_HINT", default="")
    stt_openai_key: SecretStr = Field(alias="STT_OPENAI_KEY", default=SecretStr(""))
    stt_groq_key: SecretStr = Field(alias="STT_GROQ_KEY", default=SecretStr(""))
    stt_openai_model: str = Field(alias="STT_OPENAI_MODEL", default="whisper-1")
    stt_groq_model: str = Field(alias="STT_GROQ_MODEL", default="whisper-large-v3-turbo")

    rewrite_base_url: str = Field(alias="REWRITE_BASE_URL", default="")
    rewrite_model: str = Field(alias="REWRITE_MODEL", default="anthropic/claude-3.5-haiku")
    rewrite_api_key: SecretStr = Field(alias="REWRITE_API_KEY", default=SecretStr(""))
    rewrite_profile: str = Field(alias="REWRITE_PROFILE", default="")
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
    rewrite_timeout_s: float = Field(alias="REWRITE_TIMEOUT_S", default=90.0)
    rewrite_max_tokens: int = Field(alias="REWRITE_MAX_TOKENS", default=2400)
    rewrite_disable_thinking: bool = Field(alias="REWRITE_DISABLE_THINKING", default=False)

    rewrite_prompt_dir: str = Field(alias="REWRITE_PROMPT_DIR", default="")
    rewrite_prompt_file_ttl_s: float = Field(alias="REWRITE_PROMPT_FILE_TTL_S", default=15.0)

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
    handoff_last_messages: int = Field(alias="HANDOFF_LAST_MESSAGES", default=12, ge=1, le=200)

    project_workspace_subdir: str = Field(alias="PROJECT_WORKSPACE_SUBDIR", default="")

    converse_base_url: str = Field(alias="CONVERSE_BASE_URL", default="")
    converse_model: str = Field(alias="CONVERSE_MODEL", default="llama3.2:3b")
    converse_api_key: SecretStr = Field(alias="CONVERSE_API_KEY", default=SecretStr(""))
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
    converse_max_context_chars: int = Field(alias="CONVERSE_MAX_CONTEXT_CHARS", default=6000)
    converse_anchor_units: int = Field(alias="CONVERSE_ANCHOR_UNITS", default=3)
    converse_timeout_s: float = Field(alias="CONVERSE_TIMEOUT_S", default=60.0)
    converse_max_tokens: int = Field(alias="CONVERSE_MAX_TOKENS", default=800)
    converse_temperature: float = Field(alias="CONVERSE_TEMPERATURE", default=0.0)
    converse_disable_thinking: bool = Field(alias="CONVERSE_DISABLE_THINKING", default=False)

    tts_provider: str = Field(alias="TTS_PROVIDER", default="piper")
    tts_voice: str = Field(alias="TTS_VOICE", default="")
    tts_piper_voice_dir: str = Field(alias="TTS_PIPER_VOICE_DIR", default="./data/piper-voices")
    tts_piper_length_scale: float = Field(alias="TTS_PIPER_LENGTH_SCALE", default=1.0)
    tts_edge_rate: str = Field(alias="TTS_EDGE_RATE", default="+0%")
    tts_max_input_chars: int = Field(alias="TTS_MAX_INPUT_CHARS", default=6000)

    research_gateway_db_path: str = Field(
        alias="RESEARCH_GATEWAY_DB_PATH", default="./data/research-gateway.db"
    )
    research_gateway_artifact_root: str = Field(
        alias="RESEARCH_GATEWAY_ARTIFACT_ROOT", default="./data/artifacts"
    )
    research_gateway_runtime_config_path: str = Field(
        alias="RESEARCH_GATEWAY_RUNTIME_CONFIG_PATH",
        default="./data/runtime-config.json",
    )

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

    research_gateway_profile_reconcile_interval_s: int = Field(
        alias="RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S", default=0, ge=0
    )

    research_gateway_profile_docker_exec_target: str = Field(
        alias="RESEARCH_GATEWAY_PROFILE_DOCKER_EXEC_TARGET", default=""
    )

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
            f"artifact_ignore_dirs={self.artifact_ignore_dirs!r}, "
            f"artifact_ignore_globs={self.artifact_ignore_globs!r}, "
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
            f"audit_clickhouse_url={self.audit_clickhouse_url!r}, "
            f"audit_clickhouse_user={self.audit_clickhouse_user!r}, "
            f"audit_clickhouse_password=SecretStr('**********'), "
            f"audit_ingest_url={self.audit_ingest_url!r}, "
            f"audit_host_label={self.audit_host_label!r}, "
            f"audit_query_max_rows={self.audit_query_max_rows!r}, "
            f"audit_query_timeout_s={self.audit_query_timeout_s!r}, "
            f"research_gateway_public_base_url={self.research_gateway_public_base_url!r}, "
            f"research_gateway_db_path={self.research_gateway_db_path!r}, "
            f"research_gateway_artifact_root={self.research_gateway_artifact_root!r}, "
            f"research_gateway_runtime_config_path="
            f"{self.research_gateway_runtime_config_path!r})"
        )

    __str__ = __repr__


def get_settings() -> Settings:
    """Return a freshly-loaded Settings instance, overlay applied."""
    base = Settings()
    overlay = runtime_config.read_overlay(base.research_gateway_runtime_config_path)
    if not overlay.values:
        return base
    try:
        return Settings(**overlay.values)
    except Exception as exc:
        logger.error(
            "the runtime config overlay at %s could not be applied (%s); "
            "falling back to .env for every value. Keys in the overlay: %s",
            base.research_gateway_runtime_config_path,
            exc.__class__.__name__,
            sorted(overlay.values),
        )
        return base
