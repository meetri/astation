"""Background tasks: submit route, ledger lists, and the completion pipeline (P2-1, fixes B-42)."""

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


class BackgroundSubmission(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/background`."""

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


_background_db = schema_checked_db(
    "background_schema_verified", lambda engine: table_present(engine, "background_tasks")
)


@background_router.post("/sessions/{stored_session_id}/background")
async def submit_background_task(
    stored_session_id: str,
    submission: BackgroundSubmission,
    request: Request,
    db: OrmSession = Depends(_background_db),
) -> dict:
    """Submit a `prompt.background` task and write its ledger row."""
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
    """This session's ledger, newest submit first -- the per-session pill/sheet feed."""
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
