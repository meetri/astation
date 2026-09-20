"""Assembling the context window for `POST /api/converse`, and parsing the answer.

The record-gathering half of the feature: every candidate a scope can offer
(transcript rows, run events, runs, artifacts, a document's text) becomes a
`ContextUnit` carrying the id it will be cited by, per source and per scope;
the model's reply is then parsed for the forced `ANSWER:` / `NOANSWER:` token
and its `SOURCES:` line turned into real citations. Scoring and selection --
which of those units earn a place in the window -- is `domain/converse.py`,
pure and I/O-free; this module is the one that reads the store and, for
`session` scope, Hermes. The prompt, the provider call and the route are
`api/converse.py`, which re-exports every name here (CLEANUP_PLAN step 3.5).

## What each scope reads

| Scope | Primary narrative source | Also assembled |
|---|---|---|
| `session` | the session's transcript rows (user + assistant) | that session's runs, its `message.completed` / `status.update` / `clarify.requested` events, its artifacts |
| `project` | the project's runs' assistant replies | the project row, its filed sessions, its artifacts |
| `document` | the artifact's own text | the artifact row and its producing run/session |

Transcript rows are the one source that is not in the gateway's own tables --
`messages` is empty by design, Hermes owns the transcript -- so `session` scope
reads them through exactly the same path `GET /api/sessions/{id}/messages`
uses (cached live handle, or one `session.resume`). That read is the only
Hermes round trip this route makes, it is **read-only**, and no prompt is ever
submitted to a session.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError
from domain.converse import ContextUnit, Selection
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _with_live_handle,
    _with_reconnect,
)
from domain.models import Artifact, Project, Run, RunEvent, Session
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

#: The two opening words the prompt forces, one of which must start every
#: reply. **Parsed, not inferred** -- see the module docstring and
#: `Settings.converse_prompt` for the measurement behind making it a forced
#: binary rather than a refusal-only sentinel.
REFUSAL_TOKEN = "NOANSWER"
ANSWER_TOKEN = "ANSWER"

#: A leading `ANSWER:` / `NOANSWER:`, tolerating the markdown a small model
#: sometimes wraps it in (`**ANSWER:**`). `NOANSWER` is listed first so the
#: alternation cannot match the `ANSWER` inside it.
_LEAD_RE = re.compile(r"^[\s*#>_`-]*(?P<token>NOANSWER|ANSWER)\b[\s*_`]*:?[\s*_`]*", re.IGNORECASE)

#: A protocol token that opens a LATER line. Measured on a real reply: a model
#: answered `NOANSWER: ...` and then repeated itself as `ANSWER: ...` two lines
#: down, and the second token would have been read aloud. Only the OPENING
#: tokens decide `refused` -- this strips the leftovers from the spoken text
#: without touching that decision, and it is line-anchored so the word inside a
#: sentence is never disturbed.
_STRAY_LEAD_RE = re.compile(
    r"^[ \t*#>_`-]*(?:NOANSWER|ANSWER)[ \t*_`]*:[ \t*_`]*",
    re.IGNORECASE | re.MULTILINE,
)

#: The declaration the model names its sources on. Two accepted shapes, both
#: measured on real replies:
#:
#: * a WHOLE LINE starting with `SOURCES:` -- everything after it on that line
#:   is protocol, however malformed (`SOURCES: 2, 4 -- 1, 5` was observed);
#: * `SOURCES:` at the very END of a line, with nothing after it but `none`
#:   or number-and-punctuation noise -- because a small model very often tacks
#:   it onto the last sentence instead of giving it a line, and because the
#:   noise is real: `SOURCES: 3, 5 -- 3, 5` was measured on a live reply.
#:
#: The second shape still refuses to eat WORDS, so `SOURCES:` used mid-sentence
#: with prose after it stays in the spoken text where it belongs.
_SOURCES_LINE_RE = re.compile(r"^[ \t]*SOURCES[ \t]*:.*$", re.IGNORECASE | re.MULTILINE)
_SOURCES_TAIL_RE = re.compile(
    r"[ \t]*SOURCES[ \t]*:[ \t]*(?:none|[0-9,.\-–— \t]*)$",
    re.IGNORECASE | re.MULTILINE,
)

#: A reasoning model served through an OpenAI-compatible shim can put its
#: thinking in `content` inside these tags. Stripped before anything is parsed
#: or spoken -- reading a chain of thought aloud is not the feature.
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

#: How many records each source contributes as CANDIDATES, before selection.
#: These are the pull limits, not the window: selection then keeps only what
#: the question earns plus the anchor. Generous enough that the record which
#: answers a question is very likely among them, small enough that assembling
#: them is a handful of indexed queries and one transcript read.
MAX_TRANSCRIPT_ROWS = 400
MAX_RUN_CANDIDATES = 60
MAX_EVENT_CANDIDATES = 400
MAX_ARTIFACT_CANDIDATES = 60

#: Run-event types worth putting in front of a model. `message.completed` is
#: the turn's final answer (B-38: exactly one per turn, carrying
#: `final_response`); the other three are the moments a human would call
#: notable. Deliberately EXCLUDED: `session.usage` (pure counters),
#: `tool.generating`/`tool.started`/`tool.completed` (2,873 rows on the live
#: DB against 157 completions -- they would swamp the candidate pool and their
#: text is mostly file content already available as artifacts) and
#: `message.interim` (a superseded draft of the completion beside it, so
#: including it means offering the model two versions of the same claim and
#: letting it pick).
NARRATIVE_EVENT_TYPES = (
    "message.completed",
    "status.update",
    "clarify.requested",
    "approval.requested",
)

#: Keys carrying the human-readable text of an event payload, in the order
#: they are looked for. `question` is a clarify, `command`/`description` an
#: approval request.
_EVENT_TEXT_KEYS = ("text", "final_response", "question", "description", "command")

#: How much of a document artifact is read off disk as candidate text. The
#: selector condenses whatever it gets; this bounds the *read*, so a 200 MB
#: ingested file cannot turn a 3-second answer into a disk stall.
MAX_DOCUMENT_BYTES = 200_000

#: MIME prefixes/values whose bytes are plain text. Anything else is described
#: by its metadata rather than decoded -- an answer built from mojibake is
#: worse than one built from "a 4 megabyte PNG called spectrogram".
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_VALUES = frozenset(
    {
        "application/json",
        "application/x-yaml",
        "application/yaml",
        "application/xml",
        "application/javascript",
        "application/x-python",
        "application/x-python-code",
        "application/toml",
        "application/x-sh",
    }
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _when(value: datetime | None) -> tuple[int, float]:
    """A total, exception-free sort key for a possibly-missing timestamp.

    `(0, epoch)` for a real time, `(-1, 0.0)` for none, so rows with no
    timestamp sort first and nothing ever compares a naive datetime with an
    aware one. SQLite hands back naive values and Postgres would hand back
    aware ones; sorting on the raw column would work today and raise
    `TypeError` the day the store changes.
    """
    if value is None:
        return (-1, 0.0)
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return (0, aware.timestamp())


def _text_of(value: Any) -> str:
    """A row/payload field as text, without inventing a shape.

    A string is itself; a list of parts contributes its string members; a dict
    contributes its string values. Anything else is dropped rather than
    `str()`-ed, because `{'tool_id': ...}` rendered as Python source is noise
    the selector would then score.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_text_of(item) for item in value if _text_of(item))
    if isinstance(value, dict):
        return "\n".join(piece for piece in (_text_of(item) for item in value.values()) if piece)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _is_texty(mime_type: str) -> bool:
    mime = (mime_type or "").split(";")[0].strip().lower()
    return mime.startswith(_TEXT_MIME_PREFIXES) or mime in _TEXT_MIME_VALUES


def _say_bytes(size: int | None) -> str:
    """A size the way a person says it -- the answer is going to a voice."""
    if size is None:
        return "an unrecorded size"
    if size < 1024:
        return f"{size} bytes"
    if size < 1024 * 1024:
        return f"about {size / 1024:.0f} kilobytes"
    return f"about {size / (1024 * 1024):.1f} megabytes"


# ---------------------------------------------------------------------------
# Candidate assembly, per source
# ---------------------------------------------------------------------------


def transcript_units(messages: Any, *, limit: int = MAX_TRANSCRIPT_ROWS) -> list[ContextUnit]:
    """Citable units from a Hermes transcript. **The B-34 rule still holds.**

    A transcript row has exactly one guaranteed key, `role`; 58% of real rows
    are tool calls with no `text`, no `row_id` and no `timestamp` at all. So
    nothing here requires a key, nothing invents one, and a row that is not a
    dict is skipped rather than crashing the route.

    Only `user` and `assistant` rows become units, and only ones carrying both
    a `row_id` and some text -- **a unit that cannot be cited has no business
    in the window**, and a tool row has no `row_id` to cite. What a tool
    actually produced reaches the model through the artifact and run-event
    sources instead, which do have ids.
    """
    if not isinstance(messages, (list, tuple)):
        return []
    units: list[ContextUnit] = []
    for row in messages:
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        if role not in ("user", "assistant"):
            continue
        row_id = row.get("row_id")
        if isinstance(row_id, bool) or not isinstance(row_id, (int, str)):
            continue
        ref = str(row_id).strip()
        if not ref:
            continue
        text = _text_of(row.get("text")).strip()
        if not text:
            continue
        timestamp = row.get("timestamp")
        units.append(
            ContextUnit(
                kind="message",
                ref=ref,
                label="what you said" if role == "user" else "the agent's reply",
                text=text,
                timestamp=timestamp if isinstance(timestamp, str) else None,
                detail={"role": role},
                primary=True,
            )
        )
    # Newest `limit` rows, back into chronological order. The pull is bounded
    # here rather than in selection because a 16,000-message session would
    # otherwise cost an IDF pass over the whole history for one question.
    return units[-limit:]


def event_units(db: OrmSession, run_ids: list[str], *, limit: int) -> list[ContextUnit]:
    """Narrative run events for `run_ids`, oldest first, cited by run id + seq.

    These are the gateway's OWN durable rows -- 14,103 of them on the live DB
    -- and `message.completed` in particular carries the agent's final answer
    for a turn, which is the single richest thing this service stores. See
    `NARRATIVE_EVENT_TYPES` for what is deliberately left out and why.
    """
    if not run_ids:
        return []
    rows = (
        db.execute(
            select(RunEvent)
            .where(
                RunEvent.run_id.in_(run_ids),
                RunEvent.event_type.in_(NARRATIVE_EVENT_TYPES),
            )
            .order_by(RunEvent.timestamp.desc(), RunEvent.seq.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    units: list[ContextUnit] = []
    for event in sorted(rows, key=lambda e: (_when(e.timestamp), e.seq)):
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        text = ""
        for key in _EVENT_TEXT_KEYS:
            text = _text_of(payload.get(key)).strip()
            if text:
                break
        if not text:
            continue
        units.append(
            ContextUnit(
                kind="run_event",
                ref=event.run_id,
                label=EVENT_LABELS.get(event.event_type, event.event_type),
                text=text,
                timestamp=iso_z(event.timestamp),
                detail={"seq": event.seq, "event_type": event.event_type},
            )
        )
    return units


#: Spoken-friendly names for the event types that reach the window. The label
#: is in the prompt, so it has to read as something a person would say.
EVENT_LABELS = {
    "message.completed": "the agent's final answer for a turn",
    "status.update": "a session status note",
    "clarify.requested": "a question the agent asked you",
    "approval.requested": "a command the agent asked permission to run",
}


def run_units(runs: list[Run]) -> list[ContextUnit]:
    """One unit per run: what happened, when, and how it ended.

    Cheap, small, and the only source that can answer "how many turns" or
    "did anything get interrupted" -- questions no transcript row states.
    """
    units: list[ContextUnit] = []
    for run in sorted(runs, key=lambda r: (_when(r.started_at), r.id)):
        started = iso_z(run.started_at)
        ended = iso_z(run.ended_at)
        parts = [f"A {run.kind} run that is {run.status}."]
        if started:
            parts.append(f"It started at {started}.")
        if ended:
            parts.append(f"It ended at {ended}.")
        if run.runtime_session_id:
            parts.append(f"It belongs to session {run.runtime_session_id}.")
        note = run.command_json if isinstance(run.command_json, dict) else {}
        explanation = note.get("stale_close_note")
        if isinstance(explanation, str) and explanation:
            parts.append(explanation)
        units.append(
            ContextUnit(
                kind="run",
                ref=run.id,
                label="a recorded run",
                text=" ".join(parts),
                timestamp=started,
                detail={"status": run.status},
            )
        )
    return units


def artifact_units(artifacts: list[Artifact]) -> list[ContextUnit]:
    """One unit per artifact, from its metadata alone -- never its bytes.

    Bytes are read only for `scope: "document"`, where the artifact IS the
    subject. In a session or project window an artifact is a fact about what
    was produced ("a 4 megabyte PNG called spectrogram, written to this path"),
    and reading 622 files off disk to answer one spoken question would trade
    the whole latency budget for material the question rarely wants.
    """
    units: list[ContextUnit] = []
    for artifact in sorted(artifacts, key=lambda a: (_when(a.created_at), a.id)):
        title = artifact.title or (artifact.source_path or "").rsplit("/", 1)[-1]
        parts = [
            f"An artifact called {title or 'untitled'}, "
            f"of type {artifact.mime_type}, {_say_bytes(artifact.size_bytes)}."
        ]
        if artifact.source_path:
            parts.append(f"It was produced at the path {artifact.source_path}.")
        if artifact.status != "available":
            parts.append(f"Its content is {artifact.status}.")
        metadata = artifact.metadata_json if isinstance(artifact.metadata_json, dict) else {}
        for key in ("caption", "description", "summary", "note"):
            extra = _text_of(metadata.get(key)).strip()
            if extra:
                parts.append(extra)
        units.append(
            ContextUnit(
                kind="artifact",
                ref=artifact.id,
                label="a file this research produced",
                text=" ".join(parts),
                timestamp=iso_z(artifact.created_at),
                detail={
                    "title": title or None,
                    "source_path": artifact.source_path,
                    "mime_type": artifact.mime_type,
                },
            )
        )
    return units


def document_units(artifact: Artifact, text: str) -> list[ContextUnit]:
    """The artifact's own text, split into citable chunks.

    Chunked rather than fed whole because the selector's job is to put the
    relevant PART of a document in the window; one 200 KB unit would be
    condensed by paragraph anyway, and chunking makes each piece separately
    scoreable and separately citable (`art_...` plus its chunk number).
    """
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not blocks:
        return []
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) > 1500:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    title = artifact.title or (artifact.source_path or artifact.id)
    return [
        ContextUnit(
            kind="artifact",
            ref=artifact.id,
            label=f"part {number} of the document {title}",
            text=chunk,
            timestamp=iso_z(artifact.created_at),
            detail={"chunk": number, "source_path": artifact.source_path},
            primary=True,
        )
        for number, chunk in enumerate(chunks, start=1)
    ]


def _order(units: list[ContextUnit]) -> list[ContextUnit]:
    """Stamp the global append index the recency anchor reads."""
    for index, unit in enumerate(units):
        unit.order = index
    return units


# ---------------------------------------------------------------------------
# Scope assembly
# ---------------------------------------------------------------------------


async def assemble_session_context(
    request: Request, db: OrmSession, stored_session_id: str
) -> list[ContextUnit]:
    """Everything the gateway knows about one Hermes session.

    Order matters: sources are appended oldest-narrative-first so `order`
    ranks them, and the transcript goes LAST so the recency anchor lands on
    the freshest thing said rather than on an artifact row.
    """
    runs = list(
        db.execute(
            select(Run)
            .where(Run.runtime_session_id == stored_session_id)
            .order_by(Run.started_at.desc(), Run.id)
            .limit(MAX_RUN_CANDIDATES)
        )
        .scalars()
        .all()
    )
    artifacts = list(
        db.execute(
            select(Artifact)
            .join(Run, Artifact.producing_run_id == Run.id)
            .where(Run.runtime_session_id == stored_session_id)
            .order_by(Artifact.created_at.desc(), Artifact.id)
            .limit(MAX_ARTIFACT_CANDIDATES)
        )
        .scalars()
        .all()
    )
    units: list[ContextUnit] = []
    units += run_units(runs)
    units += artifact_units(artifacts)
    units += event_units(db, [run.id for run in runs], limit=MAX_EVENT_CANDIDATES)
    units += transcript_units(await _read_transcript(request, stored_session_id))
    return _order(units)


async def _read_transcript(request: Request, stored_session_id: str) -> Any:
    """The session's messages, through the same path `GET /messages` uses.

    Read-only: a cached live handle when one exists on this connection, one
    `session.resume` otherwise, then `session.history`. **No prompt is ever
    submitted from this route**, to this session or any other.

    `messages` is empty (not an error) if the gateway has no adapter wired --
    the route still answers from runs, events and artifacts, which are its own.
    """
    adapter: HermesAdapter | None = getattr(request.app.state, "hermes_adapter", None)
    if adapter is None:  # pragma: no cover - the app always wires one
        return []
    cache = getattr(request.app.state, "live_handle_cache", None)
    try:
        _live_id, history = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(adapter, cache, stored_session_id, adapter.session_history),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_session_id) from exc
    if not isinstance(history, dict):
        return []
    return history.get("messages") or []


def assemble_project_context(db: OrmSession, project: Project) -> list[ContextUnit]:
    """Everything the gateway knows about one project.

    No transcript: a project spans many sessions and reading each one's history
    would turn a 3-second answer into a dozen Hermes round trips. The
    `message.completed` events ARE the agent's final answers for those
    sessions, they are already in this gateway's own tables, and they are the
    primary narrative source here for exactly that reason.
    """
    runs = list(
        db.execute(
            select(Run)
            .where(Run.project_id == project.id)
            .order_by(Run.started_at.desc(), Run.id)
            .limit(MAX_RUN_CANDIDATES)
        )
        .scalars()
        .all()
    )
    artifacts = list(
        db.execute(
            select(Artifact)
            .where(Artifact.project_id == project.id)
            .order_by(Artifact.created_at.desc(), Artifact.id)
            .limit(MAX_ARTIFACT_CANDIDATES)
        )
        .scalars()
        .all()
    )
    sessions = list(
        db.execute(select(Session).where(Session.project_id == project.id)).scalars().all()
    )

    description = (project.description or "").strip()
    overview = [f"The project is called {project.title}."]
    if description:
        overview.append(description)
    if sessions:
        titles = ", ".join(
            session.title or session.runtime_session_id or session.id for session in sessions[:20]
        )
        overview.append(f"It has {len(sessions)} filed session(s): {titles}.")
    overview.append(f"It has {len(runs)} recorded run(s) and {len(artifacts)} artifact(s).")

    units: list[ContextUnit] = [
        ContextUnit(
            kind="project",
            ref=project.id,
            label="the project record",
            text=" ".join(overview),
            timestamp=iso_z(project.created_at),
        )
    ]
    units += run_units(runs)
    units += artifact_units(artifacts)
    units += event_units(db, [run.id for run in runs], limit=MAX_EVENT_CANDIDATES)
    return _order(units)


def assemble_document_context(
    request: Request, db: OrmSession, artifact: Artifact
) -> list[ContextUnit]:
    """The artifact itself, plus where it came from.

    The provenance units matter as much as the text: "which session made this"
    is a question the document alone cannot answer, and `producing_run_id` is
    the primary provenance link on the row (never an `artifact_links` mirror).
    """
    units: list[ContextUnit] = artifact_units([artifact])
    run = db.get(Run, artifact.producing_run_id) if artifact.producing_run_id else None
    if run is not None:
        units += run_units([run])
        units += event_units(db, [run.id], limit=MAX_EVENT_CANDIDATES)
    units += document_units(artifact, _read_document_text(request, artifact))
    return _order(units)


def _read_document_text(request: Request, artifact: Artifact) -> str:
    """The artifact's bytes as text, bounded, or `""`.

    Every failure is an empty string, never an exception: a document whose
    bytes are gone still has metadata and provenance worth answering from, and
    "the file is unavailable" is a better answer than a 500. Non-text MIME
    types are not decoded at all -- an answer built from mojibake is worse than
    one built from the metadata.
    """
    if artifact.status != "available" or not artifact.storage_key:
        return ""
    if not _is_texty(artifact.mime_type):
        return ""
    store = getattr(request.app.state, "artifact_store", None)
    if store is None:  # pragma: no cover - the app always wires one
        return ""
    try:
        path = store.path_for_key(artifact.storage_key)
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            raw = handle.read(MAX_DOCUMENT_BYTES)
    except (OSError, ValueError):
        logger.warning("could not read artifact %s for a converse document window", artifact.id)
        return ""
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Parsing the model's answer
# ---------------------------------------------------------------------------


class ParsedAnswer(BaseModel):
    """What came back, decoded. Nothing here guesses at intent."""

    text: str
    refused: bool
    #: Context numbers the model said it used, `()` when it declared none.
    sources: tuple[int, ...]
    #: Whether a parseable `SOURCES:` declaration was present at all --
    #: reported so a model that ignores the protocol is visible rather than
    #: silently downgrading the citations to "everything we sent".
    sources_declared: bool
    #: Whether the reply opened with `ANSWER:` or `NOANSWER:` as the prompt
    #: requires. **False means `refused` is not trustworthy for this reply**,
    #: and the response says so rather than pretending otherwise: a model that
    #: ignores the protocol may have refused in prose the route deliberately
    #: does not try to read.
    protocol_followed: bool


def parse_answer(raw: str) -> ParsedAnswer:
    """Split the model's reply into spoken text, refusal flag and source numbers.

    **The refusal is a token, not a vibe.** `NOANSWER` must be the first word
    of the reply; nothing here infers refusal from wording. An earlier
    automated "did it refuse?" classifier on this task keyword-matched the
    prose and was wrong twice in a small sample, so this route will not do the
    same thing at runtime -- it reports `protocol_followed: false` instead and
    lets the caller see that the signal is missing.

    Both opening tokens and the `SOURCES:` declaration are protocol rather than
    prose, so they are stripped from the spoken text. `<think>` blocks go
    first -- a reasoning model behind an OpenAI-compatible shim can put its
    chain of thought in `content`, and reading that aloud is not the feature.
    """
    text = _THINK_RE.sub("", raw or "").strip()

    sources: list[int] = []
    declared = False
    for pattern in (_SOURCES_LINE_RE, _SOURCES_TAIL_RE):
        for match in pattern.finditer(text):
            declared = True
            for token in re.findall(r"\d+", match.group(0)):
                number = int(token)
                if number not in sources:
                    sources.append(number)
        text = pattern.sub("", text).strip()

    # **Every** leading protocol token is consumed, not just the first.
    # Measured on the operator's records: the model
    # wrote `ANSWER: NOANSWER. The context does not contain ...` -- a refusal
    # wearing an answer's hat. Stripping only the first token reported that as
    # an ANSWER and left the word NOANSWER in the text to be read aloud, which
    # is the exact failure this route exists to prevent. So the tokens are
    # collected and **any** NOANSWER among them makes it a refusal.
    tokens: list[str] = []
    for _ in range(4):  # bounded; each iteration consumes at least one token
        lead = _LEAD_RE.match(text)
        if lead is None:
            break
        tokens.append(lead.group("token").upper())
        text = text[lead.end() :].strip()
    refused = REFUSAL_TOKEN in tokens
    # A repeat that opens a LATER line is leftover protocol too, and must not
    # be spoken; it never changes the decision the opening tokens already made.
    text = _STRAY_LEAD_RE.sub("", text).strip()
    # A refusal whose reason did not survive as a SENTENCE gets the standard
    # one. Measured: one reply left the single word "CONTEXT" behind after the
    # protocol tokens came off, and a voice saying "context" is worse than a
    # voice saying nothing was found. Two words is the floor -- a refusal
    # reason shorter than that is not a reason.
    if refused and len(text.split()) < 2:
        text = "I could not find anything in your records that answers that."
    return ParsedAnswer(
        text=text,
        refused=refused,
        sources=tuple(sources),
        sources_declared=declared,
        protocol_followed=bool(tokens),
    )


def citations_for(selection: Selection, parsed: ParsedAnswer) -> list[dict[str, Any]]:
    """The records the answer rests on, as citable ids.

    When the model declared valid source numbers, only those units are cited
    and each is marked `declared: true`. When it declared none -- or numbers
    that name nothing -- **every unit in the window is returned instead**, with
    `declared: false`, because "here is everything the answer could have come
    from" is honest where a silently empty citation list is not.

    A refusal cites nothing: no claim was made, so nothing supports one.
    """
    if parsed.refused:
        return []
    wanted = set(parsed.sources)
    chosen = [entry for entry in selection.units if entry.number in wanted]
    declared = bool(chosen)
    if not chosen:
        chosen = list(selection.units)
    return [
        {
            "n": entry.number,
            "kind": entry.unit.kind,
            "ref": entry.unit.ref,
            "label": entry.unit.label,
            "timestamp": entry.unit.timestamp,
            "matched_terms": list(entry.matched_terms),
            "selected_by": entry.reason,
            "chars": len(entry.text),
            "declared": declared,
            **entry.unit.detail,
        }
        for entry in chosen
    ]
