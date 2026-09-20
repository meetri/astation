"""Assembling the context window for `POST /api/converse`, and parsing the answer."""

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

REFUSAL_TOKEN = "NOANSWER"
ANSWER_TOKEN = "ANSWER"

# NOANSWER precedes ANSWER in the alternation; reversed it would match the ANSWER inside it.
_LEAD_RE = re.compile(r"^[\s*#>_`-]*(?P<token>NOANSWER|ANSWER)\b[\s*_`]*:?[\s*_`]*", re.IGNORECASE)

# Line-anchored, and it never changes the refusal decision the opening tokens already made.
_STRAY_LEAD_RE = re.compile(
    r"^[ \t*#>_`-]*(?:NOANSWER|ANSWER)[ \t*_`]*:[ \t*_`]*",
    re.IGNORECASE | re.MULTILINE,
)

_SOURCES_LINE_RE = re.compile(r"^[ \t]*SOURCES[ \t]*:.*$", re.IGNORECASE | re.MULTILINE)
_SOURCES_TAIL_RE = re.compile(
    r"[ \t]*SOURCES[ \t]*:[ \t]*(?:none|[0-9,.\-–— \t]*)$",
    re.IGNORECASE | re.MULTILINE,
)

_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

MAX_TRANSCRIPT_ROWS = 400
MAX_RUN_CANDIDATES = 60
MAX_EVENT_CANDIDATES = 400
MAX_ARTIFACT_CANDIDATES = 60

# Excludes tool.*, session.usage and message.interim on purpose: they swamp or duplicate the pool.
NARRATIVE_EVENT_TYPES = (
    "message.completed",
    "status.update",
    "clarify.requested",
    "approval.requested",
)

_EVENT_TEXT_KEYS = ("text", "final_response", "question", "description", "command")

MAX_DOCUMENT_BYTES = 200_000

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


def _when(value: datetime | None) -> tuple[int, float]:
    """A total, exception-free sort key for a possibly-missing timestamp."""
    if value is None:
        return (-1, 0.0)
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return (0, aware.timestamp())


def _text_of(value: Any) -> str:
    """A row/payload field as text, without inventing a shape."""
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


def transcript_units(messages: Any, *, limit: int = MAX_TRANSCRIPT_ROWS) -> list[ContextUnit]:
    """Citable units from a Hermes transcript. **The B-34 rule still holds.**"""
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
    return units[-limit:]


def event_units(db: OrmSession, run_ids: list[str], *, limit: int) -> list[ContextUnit]:
    """Narrative run events for `run_ids`, oldest first, cited by run id + seq."""
    if not run_ids:
        return []
    rows = (
        db.execute(
            select(RunEvent)
            .where(
                RunEvent.run_id.in_(run_ids),
                RunEvent.event_type.in_(NARRATIVE_EVENT_TYPES),
            )
            # DESC plus limit takes the NEWEST rows; ordering ascending in SQL would take the oldest.
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


EVENT_LABELS = {
    "message.completed": "the agent's final answer for a turn",
    "status.update": "a session status note",
    "clarify.requested": "a question the agent asked you",
    "approval.requested": "a command the agent asked permission to run",
}


def run_units(runs: list[Run]) -> list[ContextUnit]:
    """One unit per run: what happened, when, and how it ended."""
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
    """One unit per artifact, from its metadata alone -- never its bytes."""
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
    """The artifact's own text, split into citable chunks."""
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


async def assemble_session_context(
    request: Request, db: OrmSession, stored_session_id: str
) -> list[ContextUnit]:
    """Everything the gateway knows about one Hermes session."""
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
    """The session's messages, through the same path `GET /messages` uses."""
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
    """Everything the gateway knows about one project."""
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
    """The artifact itself, plus where it came from."""
    units: list[ContextUnit] = artifact_units([artifact])
    run = db.get(Run, artifact.producing_run_id) if artifact.producing_run_id else None
    if run is not None:
        units += run_units([run])
        units += event_units(db, [run.id], limit=MAX_EVENT_CANDIDATES)
    units += document_units(artifact, _read_document_text(request, artifact))
    return _order(units)


def _read_document_text(request: Request, artifact: Artifact) -> str:
    """The artifact's bytes as text, bounded, or `""`."""
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


class ParsedAnswer(BaseModel):
    """What came back, decoded. Nothing here guesses at intent."""

    text: str
    refused: bool
    sources: tuple[int, ...]
    sources_declared: bool
    protocol_followed: bool


def parse_answer(raw: str) -> ParsedAnswer:
    """Split the model's reply into spoken text, refusal flag and source numbers."""
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

    # Every leading token is consumed, not just the first; any NOANSWER among them is a refusal.
    tokens: list[str] = []
    for _ in range(4):
        lead = _LEAD_RE.match(text)
        if lead is None:
            break
        tokens.append(lead.group("token").upper())
        text = text[lead.end() :].strip()
    refused = REFUSAL_TOKEN in tokens
    text = _STRAY_LEAD_RE.sub("", text).strip()
    # Two-word floor: a one-word remnant is not a reason, so the standard sentence replaces it.
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
    """The records the answer rests on, as citable ids."""
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
