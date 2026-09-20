"""Instance-shaped Hermes surfaces: the profile lens, vitals, session rename."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError, HermesRPCError

from api.projects import (
    HERMES_RUNTIME,
    _find_filing,
    file_stored_session,
    workspace_db,
)
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _is_session_not_found,
    _rpc_error_code,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)
from domain.instance_config import (
    HERMES_CONFIG_GET_ARGV,
    SESSIONS_AUTO_PRUNE_KEY,
    SESSIONS_RETENTION_DAYS_KEY,
    InstanceConfigCache,
    _instance_config_cache,
)
from domain.models import Project, Run
from domain.snapshot_builder import SnapshotStorageError, take_snapshot

logger = logging.getLogger(__name__)

instance_router = APIRouter(tags=["instance"])

APPROVALS_MODE_KEY = "approvals.mode"

OBSERVED_APPROVAL_MODES: dict[str, bool] = {"smart": True}

MAX_TITLE_CHARS = 200

HERMES_NOTHING_TO_BRANCH_CODE = 4008

SESSION_BRANCH_HONOURS_TITLE = True

# session.branch reads "name"; a "title" param is accepted and silently ignored.
SESSION_BRANCH_TITLE_PARAM = "name"

HERMES_SESSIONS_DELETE_ARGV: tuple[str, ...] = ("sessions", "delete")

# Pin only, never unpin: the same flag drives a UI sidebar and who set it is unknowable here.
HERMES_SESSIONS_PIN_ARGV: tuple[str, ...] = ("sessions", "pin")

# cli.exec gives the process no stdin, so an interactive confirmation would hang the delete.
HERMES_ASSUME_YES = "--yes"

MAX_STORED_ID_CHARS = 128

# Blocks option injection: the id becomes an argv token, so a leading '-' would read as an option.
_ARGV_SAFE_STORED_ID = re.compile(rf"[A-Za-z0-9][A-Za-z0-9_.-]{{0,{MAX_STORED_ID_CHARS - 1}}}")


def _validate_stored_session_id_for_argv(stored_session_id: str) -> str:
    """`_validate_stored_session_id()` plus the argv-shape gate above."""
    stored_id = _validate_stored_session_id(stored_session_id)
    if not _ARGV_SAFE_STORED_ID.fullmatch(stored_id):
        raise HTTPException(
            status_code=422,
            detail=(
                f"stored_session_id {stored_id!r} is not a usable Hermes stored "
                "session id: it must start with a letter or digit and contain "
                f"only letters, digits, '_', '.' or '-' (max {MAX_STORED_ID_CHARS} "
                "characters). This id becomes an argument to the `hermes` CLI, "
                "so the shape is enforced rather than passed through."
            ),
        )
    return stored_id


class SessionTitle(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/title`."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)

    @field_validator("title")
    @classmethod
    def _reject_blank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must contain non-whitespace characters")
        return value


class SessionFork(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/fork`. Optional."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=MAX_TITLE_CHARS)

    @field_validator("title")
    @classmethod
    def _reject_blank_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title must contain non-whitespace characters")
        return value


@instance_router.get("/profiles")
async def list_profiles(request: Request) -> dict:
    """Every Hermes profile ("agent"/"channel"), Hermes's payload verbatim,
    each with the gateway's own live connection status added.
    """
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        result = await _with_reconnect(request.app.state, adapter, adapter.profiles_list)
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    profiles = result.get("profiles") if isinstance(result, dict) else None
    profiles = profiles if isinstance(profiles, list) else []

    manager = getattr(request.app.state, "profile_connection_manager", None)
    connected_by_name = (
        {row["name"]: row["connected"] for row in manager.list_profiles()}
        if manager is not None
        else {}
    )
    annotated = [
        {**row, "connected": connected_by_name.get(row.get("name"))}
        if isinstance(row, dict)
        else row
        for row in profiles
    ]
    return {
        "profiles": annotated,
        "session_create_honours_profile": False,
        "session_create_supports_profile": True,
    }


@instance_router.get("/vitals")
async def get_vitals(request: Request) -> dict:
    """What approval policy this Hermes instance is running under -- and whether it prunes."""
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: adapter.config_get(APPROVALS_MODE_KEY),
        )
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    value: Any = result.get("value") if isinstance(result, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=502,
            detail=(
                f"config.get {{'key': {APPROVALS_MODE_KEY!r}}} returned no usable "
                f"string value (got {value!r}); the key is on Hermes's verified "
                "allowlist, so this is an upstream shape change"
            ),
        )
    mode = value.strip()
    config_cache = _instance_config_cache(request.app.state)
    config_cache.ensure_running(request.app.state)
    return {
        "vitals": {
            "approvals_mode": mode,
            "approvals_mode_known": mode in OBSERVED_APPROVAL_MODES,
            "auto_approves_dangerous_commands": OBSERVED_APPROVAL_MODES.get(mode),
            **config_cache.snapshot(),
        }
    }


@instance_router.post("/sessions/{stored_session_id}/title")
async def rename_session(
    stored_session_id: str,
    body: SessionTitle,
    request: Request,
    profile: str = Query(default="default"),
) -> dict:
    """Rename one session -- the review's ask #2, the buildable half."""
    stored_id = _validate_stored_session_id(stored_session_id)
    title = body.title.strip()
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: adapter.session_title(live, title),
                profile=profile,
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    echoed = result.get("title") if isinstance(result, dict) else None
    pending = result.get("pending") if isinstance(result, dict) else None
    logger.info("renamed session %s", stored_id)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "title": echoed if isinstance(echoed, str) and echoed else title,
        "pending": pending if isinstance(pending, bool) else None,
        "title_result": result,
    }


@instance_router.post("/sessions/{stored_session_id}/fork", status_code=201)
async def fork_session(
    stored_session_id: str,
    request: Request,
    body: SessionFork | None = None,
    profile: str = Query(default="default"),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """Fork a conversation: copy its transcript into a NEW session, same project."""
    stored_id = _validate_stored_session_id(stored_session_id)
    branch_params: dict[str, Any] = {}
    if body is not None and body.title is not None:
        if not SESSION_BRANCH_HONOURS_TITLE:  # pragma: no cover - measured True
            raise HTTPException(
                status_code=422,
                detail=(
                    "session.branch does not accept a name for the fork on this "
                    "instance; it is named '<parent title> #N'. Fork first, "
                    "then POST /api/sessions/{stored_id}/title to rename it."
                ),
            )
        branch_params[SESSION_BRANCH_TITLE_PARAM] = body.title.strip()

    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)
    try:
        _live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: adapter.session_branch(live, **branch_params),
                profile=profile,
            ),
        )
    except HermesError as exc:
        raise _fork_http_error(exc, stored_id) from exc

    payload = result if isinstance(result, dict) else {}
    fork_stored = payload.get("stored_session_id")
    if not isinstance(fork_stored, str) or not fork_stored.strip():
        raise HTTPException(
            status_code=502,
            detail=(
                "session.branch returned no stored_session_id for the fork "
                f"(got {fork_stored!r}); the measured shape carries both id "
                "spaces, so this is an upstream change"
            ),
        )
    fork_stored = fork_stored.strip()

    fork_live = payload.get("session_id")
    hermes_parent = payload.get("parent")
    parent_stored = (
        hermes_parent.strip()
        if isinstance(hermes_parent, str) and hermes_parent.strip()
        else stored_id
    )
    title = payload.get("title")
    title = title if isinstance(title, str) and title else None
    message_count = payload.get("message_count")
    if isinstance(message_count, bool) or not isinstance(message_count, int):
        message_count = None

    filing = _file_fork_with_its_parent(db, parent_stored, fork_stored, title)
    logger.info(
        "forked session %s -> %s (project=%r)",
        stored_id,
        fork_stored,
        filing["project_id"],
    )
    return {
        "stored_session_id": fork_stored,
        "live_session_id": fork_live if isinstance(fork_live, str) and fork_live else None,
        "source_stored_session_id": stored_id,
        "parent_stored_session_id": (
            hermes_parent if isinstance(hermes_parent, str) and hermes_parent else None
        ),
        "title": title,
        "message_count": message_count,
        **filing,
    }


def _fork_http_error(exc: HermesError, stored_id: str) -> HTTPException:
    """`_http_error_from_hermes()` plus the one code only forking can raise."""
    if isinstance(exc, HermesRPCError) and _rpc_error_code(exc) == HERMES_NOTHING_TO_BRANCH_CODE:
        return HTTPException(
            status_code=422,
            detail=(
                f"session {stored_id!r} has no messages: a session must have at "
                "least one message before it can be forked, because a fork is "
                f"a copy of its transcript (Hermes: {exc})"
            ),
        )
    return _http_error_from_hermes(exc, stored_id)


def _file_fork_with_its_parent(
    db: OrmSession, parent_stored_id: str, fork_stored_id: str, title: str | None
) -> dict[str, Any]:
    """File the fork into whatever project its parent is filed in."""
    summary: dict[str, Any] = {
        "project_id": None,
        "project_title": None,
        "workspace_session_id": None,
        "filed": False,
        "filing_error": None,
    }
    try:
        parent_filing = _find_filing(db, HERMES_RUNTIME, parent_stored_id)
        if parent_filing is None:
            return summary
        project = db.get(Project, parent_filing.project_id)
        if project is None:  # pragma: no cover - FK makes this unreachable
            summary["filing_error"] = (
                f"the parent is filed into project {parent_filing.project_id!r}, "
                "which no longer exists"
            )
            return summary
        filing_row, _created = file_stored_session(db, project, fork_stored_id, title)
        summary["project_id"] = project.id
        summary["project_title"] = project.title
        summary["workspace_session_id"] = filing_row["workspace_session_id"]
        summary["filed"] = True
    except HTTPException as exc:
        db.rollback()
        summary["filing_error"] = str(exc.detail)
        logger.warning(
            "forked session %s could not be filed alongside its parent %s: %s",
            fork_stored_id,
            parent_stored_id,
            exc.detail,
        )
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        db.rollback()
        summary["filing_error"] = str(exc)
        logger.warning(
            "forked session %s exists on Hermes but could not be filed",
            fork_stored_id,
            exc_info=True,
        )
    return summary


def _cleanup_workspace_filing(db: OrmSession, stored_id: str) -> dict[str, Any]:
    """Drop the workspace's own row for a session Hermes no longer has."""
    summary: dict[str, Any] = {
        "workspace_filing_deleted": False,
        "workspace_session_id": None,
        "project_id": None,
        "runs_detached": 0,
        "workspace_cleanup_error": None,
    }
    try:
        filing = _find_filing(db, HERMES_RUNTIME, stored_id)
        if filing is None:
            return summary
        summary["workspace_session_id"] = filing.id
        summary["project_id"] = filing.project_id
        detached = db.execute(
            update(Run).where(Run.session_id == filing.id).values(session_id=None)
        )
        summary["runs_detached"] = detached.rowcount or 0
        db.delete(filing)
        db.commit()
        summary["workspace_filing_deleted"] = True
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        db.rollback()
        summary["workspace_cleanup_error"] = str(exc)
        logger.warning(
            "Hermes session %s was deleted but its workspace filing row could "
            "not be removed; it will render as missing until unfiled by hand",
            stored_id,
            exc_info=True,
        )
    return summary


@instance_router.delete("/sessions/{stored_session_id}")
async def delete_session(
    stored_session_id: str,
    request: Request,
    force: bool = Query(default=False),
    profile: str = Query(default="default"),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """**Delete one session from Hermes's own store -- after copying it.**"""
    stored_id = _validate_stored_session_id_for_argv(stored_session_id)
    profile_argv = ["-p", profile] if profile and profile != "default" else []
    argv = [*profile_argv, *HERMES_SESSIONS_DELETE_ARGV, stored_id, HERMES_ASSUME_YES]
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)

    snapshot_id: str | None = None
    if not force:
        try:
            snapshot = await take_snapshot(
                request.app.state, stored_id, reason="pre_delete", db=db, profile=profile
            )
        except HermesError as exc:
            if _is_session_not_found(exc):
                raise _http_error_from_hermes(exc, stored_id) from exc
            return _snapshot_blocked_delete(stored_id, str(exc))
        except SnapshotStorageError as exc:
            return _snapshot_blocked_delete(stored_id, str(exc))
        snapshot_id = snapshot.id
        logger.warning(
            "pre-delete snapshot %s taken for session %s (%d rows)",
            snapshot_id,
            stored_id,
            snapshot.message_rows,
        )
    else:
        logger.warning("delete of %s FORCED without a snapshot (?force=true)", stored_id)

    logger.warning(
        "issuing IRREVERSIBLE Hermes session delete for stored id %s on profile %r (argv=%r)",
        stored_id,
        profile,
        argv,
    )
    try:
        result = await _with_reconnect(request.app.state, adapter, lambda: adapter.cli_exec(argv))
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    payload = result if isinstance(result, dict) else {}
    code = payload.get("code")
    output = payload.get("output")
    output_text = output if isinstance(output, str) else ""
    blocked = payload.get("blocked") is True
    # bool is an int subclass, so an explicit isinstance check is needed to reject code: true.
    succeeded = not blocked and isinstance(code, int) and not isinstance(code, bool) and code == 0
    if not succeeded:
        hint = payload.get("hint")
        said = output_text or (str(hint) if hint else "") or "no output"
        raise HTTPException(
            status_code=502,
            detail=(
                f"`hermes sessions delete {stored_id}` did not succeed "
                f"(blocked={payload.get('blocked')!r}, code={code!r}): {said}"
            ),
        )

    logger.warning("DELETED Hermes session %s from the instance", stored_id)
    live_handle_closed = await _close_live_handle(request.app.state, adapter, cache, stored_id)
    cleanup = _cleanup_workspace_filing(db, stored_id)
    return {
        "stored_session_id": stored_id,
        "deleted": True,
        "snapshot_id": snapshot_id,
        "live_handle_closed": live_handle_closed,
        "cli_code": code,
        "cli_output": output_text,
        **cleanup,
    }


def _snapshot_blocked_delete(stored_id: str, error: str) -> JSONResponse:
    """The 409 for "could not copy it, so did not delete it". Top-level keys, by contract."""
    return JSONResponse(
        status_code=409,
        content={
            "detail": (
                f"the pre-delete snapshot of session {stored_id!r} failed, so the "
                "session was NOT deleted; retry, or repeat with ?force=true to "
                "delete without a copy"
            ),
            "snapshot_error": error,
        },
    )


async def _close_live_handle(
    app_state: Any, adapter: HermesAdapter, cache: Any, stored_id: str
) -> bool | None:
    """`session.close` the handle we hold for `stored_id`, then forget it. Never raises."""
    live_id = cache.get(stored_id) if cache is not None else None
    closed: bool | None = None
    if live_id is not None:
        try:
            result = await _with_reconnect(
                app_state, adapter, lambda: adapter.session_close(live_id)
            )
            closed = (result.get("closed") is True) if isinstance(result, dict) else False
        except Exception as exc:
            logger.info(
                "session.close after deleting %s raised; the zombie handle stays "
                "in active_list until the connection drops: %s",
                stored_id,
                exc,
            )
            closed = None
    if cache is not None:
        cache.discard(stored_id)
    return closed


__all__ = [
    "APPROVALS_MODE_KEY",
    "HERMES_ASSUME_YES",
    "HERMES_CONFIG_GET_ARGV",
    "HERMES_NOTHING_TO_BRANCH_CODE",
    "HERMES_SESSIONS_DELETE_ARGV",
    "HERMES_SESSIONS_PIN_ARGV",
    "MAX_STORED_ID_CHARS",
    "MAX_TITLE_CHARS",
    "OBSERVED_APPROVAL_MODES",
    "SESSIONS_AUTO_PRUNE_KEY",
    "SESSIONS_RETENTION_DAYS_KEY",
    "SESSION_BRANCH_HONOURS_TITLE",
    "SESSION_BRANCH_TITLE_PARAM",
    "InstanceConfigCache",
    "SessionFork",
    "SessionTitle",
    "delete_session",
    "fork_session",
    "get_vitals",
    "instance_router",
    "list_profiles",
    "rename_session",
]
