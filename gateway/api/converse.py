"""Ask a question about your own research: `POST /api/converse` (G-12)."""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.orm import Session as OrmSession

from api.rewrite import (
    build_headers,
    content_from_completion,
    endpoint_url,
    provider_label,
)
from config.settings import Settings, get_settings
from domain.converse import ContextUnit, Selection, render_context, select_context
from domain.converse_context import (  # noqa: F401  (re-exported; see module docstring)
    _EVENT_TEXT_KEYS,
    _LEAD_RE,
    _SOURCES_LINE_RE,
    _SOURCES_TAIL_RE,
    _STRAY_LEAD_RE,
    _TEXT_MIME_PREFIXES,
    _TEXT_MIME_VALUES,
    _THINK_RE,
    ANSWER_TOKEN,
    EVENT_LABELS,
    MAX_ARTIFACT_CANDIDATES,
    MAX_DOCUMENT_BYTES,
    MAX_EVENT_CANDIDATES,
    MAX_RUN_CANDIDATES,
    MAX_TRANSCRIPT_ROWS,
    NARRATIVE_EVENT_TYPES,
    REFUSAL_TOKEN,
    ParsedAnswer,
    artifact_units,
    assemble_document_context,
    assemble_project_context,
    assemble_session_context,
    citations_for,
    document_units,
    event_units,
    parse_answer,
    run_units,
    transcript_units,
)
from domain.db import columns_present, schema_checked_db
from domain.hermes_runtime import (
    _validate_stored_session_id,
)
from domain.models import Artifact, Project

logger = logging.getLogger(__name__)

converse_router = APIRouter(tags=["converse"])


class ConverseRequest(BaseModel):
    """Body for `POST /api/converse`. Closed schema, like every body here."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    scope: Literal["session", "project", "document"]
    # A Hermes STORED session id, never a live handle and never a workspace sess_ id.
    session_id: str | None = None
    project_id: str | None = None
    artifact_id: str | None = None

    @field_validator("question")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _exactly_the_scopes_id(self) -> ConverseRequest:
        required = SCOPE_ID_FIELD[self.scope]
        for name in SCOPE_ID_FIELD.values():
            value = getattr(self, name)
            if name == required:
                if value is None or not str(value).strip():
                    raise ValueError(f"scope {self.scope!r} needs a non-empty {name}")
            elif value is not None:
                raise ValueError(
                    f"scope {self.scope!r} does not use {name}; it answers a "
                    f"different question and would be silently ignored"
                )
        return self


SCOPE_ID_FIELD: dict[str, str] = {
    "session": "session_id",
    "project": "project_id",
    "document": "artifact_id",
}


# Question last: it stays in view after thousands of characters of records.
def build_user_message(question: str, selection: Selection) -> str:
    """The user turn: the numbered context, then the question."""
    return (
        "CONTEXT -- records from the user's own research gateway:\n\n"
        f"{render_context(selection)}\n\n"
        f"QUESTION: {question.strip()}"
    )


# Separate from rewrite's payload to force temperature: the 0.8 default invents.
def build_payload(
    user_message: str,
    *,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """The chat-completions body."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        **({"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}),
    }


async def answer_via_chat_completions(
    user_message: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool = False,
) -> str:
    """One POST to `{base}/chat/completions`; the raw answer text out."""
    url = endpoint_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                url,
                headers=build_headers(api_key),
                json=build_payload(
                    user_message,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    disable_thinking=disable_thinking,
                ),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"could not reach the converse endpoint at {url}: {exc}",
        ) from exc
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"the converse endpoint answered HTTP {response.status_code}: {response.text[:300]}"
            ),
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"the converse endpoint answered non-JSON: {response.text[:300]}",
        ) from exc
    return content_from_completion(payload, label="converse")


_converse_db = schema_checked_db(
    "converse_schema_verified",
    lambda engine: (
        columns_present(engine, "runs", "runtime_session_id")
        and columns_present(engine, "artifacts", "status")
    ),
)


@converse_router.post("/converse")
async def converse(
    body: ConverseRequest,
    request: Request,
    db: OrmSession = Depends(_converse_db),
) -> dict[str, Any]:
    """Answer a spoken question about the operator's own research (G-12)."""
    settings: Settings = get_settings()
    base_url = settings.converse_base_url.strip()
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail=(
                "asking questions about your research is not configured: "
                "CONVERSE_BASE_URL is empty. Point it at any OpenAI-compatible "
                "/chat/completions endpoint (e.g. http://<ollama-host>:11434/v1 for "
                "Ollama) in the gateway's .env or from the app's "
                "provider settings. This gateway never falls back to a provider "
                "you did not configure."
            ),
        )
    model = settings.converse_model.strip()
    if not model:
        raise HTTPException(
            status_code=503,
            detail=(
                "asking questions about your research is not configured: "
                "CONVERSE_MODEL is empty. Set it to a model the configured "
                "endpoint serves (e.g. llama3.2:3b)."
            ),
        )

    question = body.question
    max_question = settings.converse_max_question_chars
    if len(question) > max_question:
        raise HTTPException(
            status_code=413,
            detail=(
                f"question is {len(question)} characters; the cap is "
                f"{max_question} (CONVERSE_MAX_QUESTION_CHARS)"
            ),
        )

    scope_id = str(getattr(body, SCOPE_ID_FIELD[body.scope])).strip()
    units = await _assemble(request, db, body.scope, scope_id)

    selection = select_context(
        units,
        question,
        budget_chars=settings.converse_max_context_chars,
        anchor_units=settings.converse_anchor_units,
    )
    context_report = {
        "scope": body.scope,
        "scope_id": scope_id,
        "strategy": "recency-anchor + idf-weighted question overlap",
        "units": len(selection.units),
        "units_considered": selection.considered,
        "units_matching_question": selection.matched,
        "chars": selection.chars,
        "budget_chars": selection.budget_chars,
        "budget_exhausted": selection.truncated,
        "anchor_units": settings.converse_anchor_units,
        "question_terms": list(selection.question_terms),
    }

    # Not a fallback: with nothing recorded there is nothing to answer from, so no call.
    if not selection.units:
        logger.info(
            "converse refused before calling the model: no context units for %s %s",
            body.scope,
            scope_id,
        )
        return {
            "answer": _empty_context_answer(body.scope),
            "citations": [],
            "context": {**context_report, "model_called": False},
            "provider": provider_label(base_url),
            "model": model,
            "refused": True,
        }

    raw = await answer_via_chat_completions(
        build_user_message(question, selection),
        base_url=base_url,
        model=model,
        api_key=settings.converse_api_key.get_secret_value().strip(),
        system_prompt=settings.converse_prompt,
        timeout_s=settings.converse_timeout_s,
        max_tokens=settings.converse_max_tokens,
        temperature=settings.converse_temperature,
        disable_thinking=settings.converse_disable_thinking,
    )
    parsed = parse_answer(raw)
    citations = citations_for(selection, parsed)
    logger.info(
        "converse answered a %s question from %d/%d units (%d chars) via %s "
        "(%s); refused=%s, sources_declared=%s, protocol_followed=%s",
        body.scope,
        len(selection.units),
        selection.considered,
        selection.chars,
        provider_label(base_url),
        model,
        parsed.refused,
        parsed.sources_declared,
        parsed.protocol_followed,
    )
    return {
        "answer": parsed.text,
        "citations": citations,
        "context": {
            **context_report,
            "model_called": True,
            "sources_declared": parsed.sources_declared,
            # False means the token protocol was ignored, so refused is not reliable.
            "protocol_followed": parsed.protocol_followed,
        },
        "provider": provider_label(base_url),
        "model": model,
        "refused": parsed.refused,
    }


def _empty_context_answer(scope: str) -> str:
    """The spoken refusal for a scope with nothing recorded in it."""
    subject = {
        "session": "this session",
        "project": "this project",
        "document": "this document",
    }.get(scope, "that")
    return (
        f"I have nothing recorded for {subject}, so there is nothing I can "
        "answer from. Nothing was guessed."
    )


async def _assemble(
    request: Request, db: OrmSession, scope: str, scope_id: str
) -> list[ContextUnit]:
    """Candidate units for a scope, or a 404 naming what was not found."""
    if scope == "session":
        return await assemble_session_context(request, db, _validate_stored_session_id(scope_id))
    if scope == "project":
        project = db.get(Project, scope_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"no project with id {scope_id!r}")
        return assemble_project_context(db, project)
    artifact = db.get(Artifact, scope_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"no artifact with id {scope_id!r}")
    return assemble_document_context(request, db, artifact)


__all__ = [
    "ANSWER_TOKEN",
    "EVENT_LABELS",
    "MAX_DOCUMENT_BYTES",
    "NARRATIVE_EVENT_TYPES",
    "REFUSAL_TOKEN",
    "SCOPE_ID_FIELD",
    "ConverseRequest",
    "ParsedAnswer",
    "answer_via_chat_completions",
    "artifact_units",
    "assemble_document_context",
    "assemble_project_context",
    "assemble_session_context",
    "build_payload",
    "build_user_message",
    "citations_for",
    "converse",
    "converse_router",
    "document_units",
    "event_units",
    "parse_answer",
    "run_units",
    "transcript_units",
]
