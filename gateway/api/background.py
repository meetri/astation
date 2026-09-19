"""Background tasks: submit route, ledger lists, and the completion pipeline (P2-1, fixes B-42).

Why a ledger at all: `prompt.background` is fire-and-forget and **Hermes
keeps no trace of it** (PV "Phase 2 probe", measured live 2026-08-30). The
submit returns `{"task_id": "bg_..."}` and then the wire is silent -- no
events, no pollable status, no transcript row, and a background-only session
is never persisted. Completion is exactly one `background.complete`
`{task_id, text}` event, delivered only to connections attached to the
session at that instant. Until P2-1 the gateway dropped that event unmapped
(B-42), so the app could never learn a task finished. The `background_tasks`
table (written at submit, updated on completion) plus that one event are the
background pill's entire data source.

Three cooperating pieces; the first lives here, the other two in
`domain/background_ledger.py` (CLEANUP_PLAN step 3.5) and are re-exported from
this module with their constants:

1. **Routes** (`background_router`, mounted on the authenticated `/api`
   router in `api.main`): submit a task, list a session's tasks, list all
   tasks. Same conventions as every other route -- stored ids in, live
   handles resolved per call via the shared `_with_live_handle()`, 404/502
   mapping via `_http_error_from_hermes`, 503 for an unmigrated DB.
2. **`BackgroundLedger`** -- the event-side orchestration, wired to
   `EventBroadcaster` by `api.main.lifespan`.
3. **Transcript injection** (`append_finished_background_results`) on every
   transcript read.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError
from api.validators import reject_blank_text, reject_rewind_fields
from domain import background_tasks as ledger_ops
from domain.background_ledger import (  # noqa: F401  (re-exported; see module docstring)
    BACKGROUND_COMPLETED_EVENT_TYPE,
    ORPHANED_EXPLANATION,
    SYNTHESIZED_FROM_FIELD,
    SYNTHESIZED_FROM_VALUE,
    SYNTHESIZED_MESSAGE_EVENT_TYPE,
    BackgroundLedger,
    append_finished_background_results,
    synthesized_transcript_row,
    task_row,
)
from domain.db import schema_checked_db, table_present
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
)
from domain.timeutil import iso_z

logger = logging.getLogger(__name__)

background_router = APIRouter(tags=["background"])

# ---------------------------------------------------------------------------
# Request/response shapes
# ---------------------------------------------------------------------------


class BackgroundSubmission(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/background`.

    Same closed schema and blank-text rules as `TurnSubmission`
    (`api.main`), for the same reasons: `text` is the only thing a client
    can put on the wire, a whitespace-only prompt must never start a real
    task (B-24), and the rewind/truncate fields are rejected by name even
    though `prompt.background` is not known to read them -- "not known to"
    is not "does not", and the guard costs nothing.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        return reject_blank_text(
            value,
            message=(
                "text must contain non-whitespace characters: a blank prompt "
                "would still start a real background task on the Hermes session"
            ),
        )

    @model_validator(mode="before")
    @classmethod
    def _reject_rewind_fields(cls, data: Any) -> Any:
        return reject_rewind_fields(data, submission="background")


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------


#: A DB session, with "you never ran the migration" turned into a 503
#: (`domain.db.schema_checked_db`). Mirrors `api.projects.workspace_db` but
#: verifies the *ledger* table: `schema_is_present` predates this feature and
#: only checks the Phase 1 tables, so a database migrated to the previous head
#: would pass it and then 500 on the first ledger query.
_background_db = schema_checked_db(
    "background_schema_verified", lambda engine: table_present(engine, "background_tasks")
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@background_router.post("/sessions/{stored_session_id}/background")
async def submit_background_task(
    stored_session_id: str,
    submission: BackgroundSubmission,
    request: Request,
    db: OrmSession = Depends(_background_db),
) -> dict:
    """Submit a `prompt.background` task and write its ledger row.

    `{stored_session_id}` is the **STORED / durable** id, resolved to a live
    handle through the shared `_with_live_handle()` exactly like
    `POST /turns`. The ledger row is written *before* the response goes out,
    against the STORED id -- the completion event carries only a live handle
    (which may be re-minted by then), so the submit-time record here is the
    only reliable session attribution the task will ever have.

    Returns 502 if Hermes acknowledges without a usable `task_id`: without
    one, the completion event could never be matched, which is B-42 with
    extra steps -- the task may be running, but this gateway cannot track it
    and says so instead of pretending.

    NOTE (measured, PV "Phase 2 probe"): a 200 here means *accepted*, and
    that is all the wire can ever say. No progress events will follow; the
    next observable fact about this task is its single
    `background.complete` event, surfaced via the ledger as
    `state: finished` and injected into the session timeline.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    adapter: HermesAdapter = request.app.state.hermes_adapter
    cache = request.app.state.live_handle_cache
    try:
        live_id, acknowledgement = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: adapter.prompt_background(live, submission.text),
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    task_id = acknowledgement.get("task_id") if isinstance(acknowledgement, dict) else None
    if not isinstance(task_id, str) or not task_id:
        raise HTTPException(
            status_code=502,
            detail=(
                "Hermes accepted the background prompt but returned no task_id; "
                "the task cannot be tracked (its completion event could never be "
                f"matched). Raw acknowledgement keys: "
                f"{sorted(acknowledgement) if isinstance(acknowledgement, dict) else type(acknowledgement).__name__}"
            ),
        )

    generation = getattr(adapter, "connection_generation", None)
    row = ledger_ops.record_submitted(
        db,
        task_id=task_id,
        stored_session_id=stored_id,
        prompt_text=submission.text,
        connection_generation=generation if isinstance(generation, int) else None,
    )
    db.commit()

    return {
        "task_id": row.task_id,
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "state": row.state,
        "submitted_at": iso_z(row.submitted_at),
    }


@background_router.get("/sessions/{stored_session_id}/background")
async def list_session_background_tasks(
    stored_session_id: str,
    db: OrmSession = Depends(_background_db),
) -> dict:
    """This session's ledger, newest submit first -- the per-session pill/sheet feed.

    Served entirely from the gateway's own ledger: Hermes has nothing to ask
    (no status surface exists for background tasks -- measured). No Hermes
    round trip, so this works even while Hermes is unreachable, which is
    exactly when the owner wants to know what was in flight.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    tasks = ledger_ops.tasks_for_session(db, stored_id)
    return {
        "stored_session_id": stored_id,
        "tasks": [task_row(task) for task in tasks],
    }


@background_router.get("/background")
async def list_all_background_tasks(db: OrmSession = Depends(_background_db)) -> dict:
    """The global ledger, newest submit first, bounded."""
    tasks = ledger_ops.all_tasks(db)
    return {"tasks": [task_row(task) for task in tasks]}
