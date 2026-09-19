"""Ask a question about your own research: `POST /api/converse` (G-12).

The owner's ask, stated repeatedly: *"Ultimately I'd still like to be able to
have a conversation realtime about the project / session / document."* This is
the server half of it -- one route that takes a spoken question and a scope and
answers in two or three sentences a voice can read out, with citations.

    {"question": str,
     "scope": "session" | "project" | "document",
     "session_id"?, "project_id"?, "artifact_id"?}
      -> {"answer", "citations": [...], "context": {...},
          "provider", "model", "refused"}

## Why this cannot go through the Hermes agent

Not a preference -- three measurements, the same three that killed the
Hermes-routed rewrite (2026-09-01 review, ask #4). Real agent turn latency on
the live instance is **2.5 to 15 minutes**; B-62 records completion signals
arriving **8 to 10 minutes late**; and `prompt.submit` would write the
question into the owner's own transcript as if they had asked the agent it.
Conversation needs seconds. So this is the second consumer of the review's
**gateway-resident utility model** reframe (item G-12): something fast beside
something slow, reading the durable state the gateway already keeps. Measured
on this machine 2026-09-02: `llama3.2:3b` on Ollama answers in about 2.8
seconds.

## The engineering problem is context selection, not latency

A benchmark of this exact task, on a plain "last N messages" window, produced a
**confident, well-formed, entirely fabricated answer**: asked about a topic the
window did not contain, an 8B model asserted a conclusion it had no basis for,
while a 3B on the same window correctly said the topic was not there. **A
confident liar with a good voice is worse than no feature.** Everything below
is arranged around that:

* **The window is assembled, not sliced.** `domain/converse_context.py`
  gathers the candidates and `domain/converse.py` scores every
  candidate record from the gateway's own store by IDF-weighted overlap with
  the question, fills the budget round-robin across record kinds, and **never
  selects a record that matches nothing** -- so an unanswerable question
  produces a SHORT window rather than a padded one. A small recency anchor is
  the only unearned part, and it is configurable down to zero.
* **Every unit carries the id it will be cited by.** The response's
  `citations` are real `row_id` / `run_...` / `art_...` values with the terms
  that matched, so a claim can be traced to a record.
* **Refusal is the cheap default.** Measured: every model tried refuses in
  under a second, so refusing costs nothing. The prompt says to prefer it, and
  a context with no units at all never reaches the model at all.
* **The refusal is a token, not a vibe.** Every reply must open with
  `ANSWER:` or `NOANSWER:` and the route parses that; nothing infers refusal
  from wording. An earlier automated "did it refuse?" classifier on this task
  keyword-matched the prose and was wrong twice. When the token is missing the
  response says `context.protocol_followed: false` rather than guessing.

## Measured on the owner's own records, 2026-09-02

29 questions across all three scopes on real sessions -- 14 the records answer,
11 they genuinely do not, 4 whose premise the records contradict -- with every
answer read rather than keyword-scored. On the shipped defaults
(`llama3.2:3b`): **1 confabulation in 29** (3.4%), 10 of 11 absent questions
structurally refused, 3 of 4 false premises declined, protocol followed 29/29,
median 0.9-2.5 s. On `qwen3:8b` through the same endpoint: **0 in 29**, at a
13.9 s median. On `gemma3:4b`: roughly 6 in 15 adversarial questions -- so
model choice dominates every other knob here, which is exactly why it is a
registry setting the owner can change from the app.

## Error contract

| Status | When |
|---|---|
| 404 | unknown `project_id` / `artifact_id`, or a stored session id Hermes does not have |
| 413 | `question` longer than `CONVERSE_MAX_QUESTION_CHARS` (default 600) |
| 422 | blank `question`, unknown `scope`, a missing or mismatched id for the scope |
| 502 | endpoint unreachable, non-2xx, non-JSON, 200-with-no-content; or Hermes failing on the transcript read |
| 503 | `CONVERSE_BASE_URL` / `CONVERSE_MODEL` unset (feature disabled), or the workspace schema is unmigrated |

Never a silent fallback to a provider the owner did not configure -- the
`api/transcribe.py` rule, inherited whole.

The candidate assembly per source and per scope, and the answer parsing, are
`domain/converse_context.py` (CLEANUP_PLAN step 3.5), re-exported here; this
module keeps the request body, the prompt, the provider call and the route.
"""

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

#: Authenticated routes (mounted under `/api` in `api.main`).
converse_router = APIRouter(tags=["converse"])

# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class ConverseRequest(BaseModel):
    """Body for `POST /api/converse`. Closed schema, like every body here.

    **Exactly one id, and it must be the one the scope uses.** A `project_id`
    sent with `scope: "session"` is a 422 rather than a silently-ignored field:
    the two answer different questions, and quietly answering the other one is
    the same class of failure as a silent provider fallback.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    scope: Literal["session", "project", "document"]
    #: Hermes STORED session id (`20260829_182532_991e3f`) -- never a live
    #: handle, never a workspace `sess_...` id.
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


#: scope -> the body field carrying its id. One source of truth for the
#: validator, the assembler and the response's `scope_id`.
SCOPE_ID_FIELD: dict[str, str] = {
    "session": "session_id",
    "project": "project_id",
    "document": "artifact_id",
}


# ---------------------------------------------------------------------------
# Prompt + provider
# ---------------------------------------------------------------------------


def build_user_message(question: str, selection: Selection) -> str:
    """The user turn: the numbered context, then the question.

    Question LAST, deliberately: it is the instruction the model should still
    have in view after reading several thousand characters of records, and a
    question buried above them is measurably easier to drift from.
    """
    return (
        "CONTEXT -- records from the user's own research gateway:\n\n"
        f"{render_context(selection)}\n\n"
        f"QUESTION: {question.strip()}"
    )


def build_payload(
    user_message: str,
    *,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """The chat-completions body.

    Separate from `api/rewrite.py::build_payload` for one reason worth the
    duplication: **`temperature` is sent explicitly.** Ollama's default is 0.8,
    and for a task whose entire value is that it does not invent, leaving
    sampling creativity at a chat default is a correctness bug, not a style
    choice. `stream` is absent (default false): the app wants one finished
    string to hand to the synthesizer.
    """
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
    """One POST to `{base}/chat/completions`; the raw answer text out.

    Module-level on purpose: this is the injection point the route tests
    replace with a fake, exactly as `api/rewrite.py` exposes
    `rewrite_via_chat_completions`. Every failure is a 502 naming what
    happened, and the 200-with-no-content guard is `api/rewrite.py`'s own --
    reused rather than re-implemented, because OpenRouter's HTTP-200-with-an-
    `error`-body is the same trap on this route as on that one.
    """
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


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------


#: A DB session, with an unmigrated workspace turned into a 503
#: (`domain.db.schema_checked_db`). Same shape as `api.runs._runs_db` /
#: `api.artifacts._artifacts_db`: this route reads `runs.runtime_session_id`
#: and `artifacts.status`, so it needs both migrations and says so rather
#: than 500-ing on a missing column.
_converse_db = schema_checked_db(
    "converse_schema_verified",
    lambda engine: (
        columns_present(engine, "runs", "runtime_session_id")
        and columns_present(engine, "artifacts", "status")
    ),
)


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@converse_router.post("/converse")
async def converse(
    body: ConverseRequest,
    request: Request,
    db: OrmSession = Depends(_converse_db),
) -> dict[str, Any]:
    """Answer a spoken question about the owner's own research (G-12).

    Body: `{"question", "scope", "session_id"|"project_id"|"artifact_id"}` --
    exactly the one id the scope uses; another one is a 422, not a silently
    ignored field.

    Response: `{"answer", "citations", "context", "provider", "model",
    "refused"}`. `answer` is always the thing to speak, refusal included, so a
    client can hand it straight to the synthesizer without branching. `context`
    reports what was selected and why -- the question terms, how many candidate
    records existed, how many matched, how many made the window and how many
    characters they cost -- so a bad answer can be diagnosed from the same JSON
    that produced it.

    **A window with no units never reaches the model.** That is not a fallback,
    it is the cheapest correct refusal: nothing was recorded, so there is
    nothing to answer from, and `context.units` proves it.

    Endpoint, model, prompt, budget and timeout all come from server settings
    (`CONVERSE_*`, runtime-configurable from the app); a request cannot choose
    them, the same rule `POST /api/rewrite` and `POST /api/transcribe` follow.
    The full error contract is in the module docstring.
    """
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

    if not selection.units:
        # The cheapest correct refusal: nothing was recorded for this scope, so
        # there is nothing an answer could rest on. Measured, refusing costs
        # under a second even when a model IS asked -- so not asking is free.
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
        # May be empty -- `build_headers` then sends no Authorization header,
        # which is the normal case for Ollama and every local server.
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
            # False means the model ignored the ANSWER:/NOANSWER: protocol, so
            # `refused` is not trustworthy for this reply. Reported rather than
            # papered over -- see `parse_answer`.
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
