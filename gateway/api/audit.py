"""The audit surface: what actually happened on the host, for the app and for AI tooling."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from domain.audit_store import (
    READABLE_TABLES,
    AuditQueryRejected,
    AuditRecorder,
    AuditStore,
    AuditUnavailable,
)
from domain.db import columns_present, schema_checked_db
from domain.models import Run

logger = logging.getLogger(__name__)

audit_router = APIRouter(tags=["audit"])

_audit_db = schema_checked_db(
    "audit_runs_schema_verified",
    lambda engine: columns_present(engine, "runs", "runtime_session_id"),
)

_FALLBACK_WINDOW_S = 3600

_HEALTH_CLASSES = ("host_exec", "host_file", "host_net", "host_beat")


def _store(request: Request) -> AuditStore:
    store = getattr(request.app.state, "audit_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="the audit store is not wired into this gateway build",
        )
    return store


def _recorder(request: Request) -> AuditRecorder | None:
    return getattr(request.app.state, "audit_recorder", None)


def _caller(request: Request) -> str:
    """Who is asking. `agent:<profile>` for tooling, `app:<user>` for a person."""
    header = request.headers.get("x-trg-audit-caller", "").strip()
    if header:
        return header[:120]
    user = getattr(getattr(request, "state", None), "username", None)
    return f"app:{user}" if user else "app"


async def _run_read(
    request: Request,
    *,
    route: str,
    sql: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Execute, record, and translate failures into HTTP. One path for all routes."""
    store = _store(request)
    recorder = _recorder(request)
    caller = _caller(request)
    started = time.perf_counter()
    try:
        result = await store.execute(sql, params=params)
    except AuditUnavailable as exc:
        if recorder:
            await recorder.record(
                caller=caller,
                route=route,
                sql=sql,
                rows_out=0,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuditQueryRejected as exc:
        if recorder:
            await recorder.record(
                caller=caller,
                route=route,
                sql=sql,
                rows_out=0,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if recorder:
        await recorder.record(
            caller=caller,
            route=route,
            sql=sql,
            rows_out=result.row_count,
            duration_ms=result.duration_ms,
        )
    return result


@audit_router.get("/audit/health")
async def audit_health(request: Request) -> dict[str, Any]:
    """Is the pipeline alive, and how far behind is it?"""
    store = _store(request)
    recorder = _recorder(request)
    if not store.configured:
        return {
            "status": "not_configured",
            "reason": "AUDIT_CLICKHOUSE_URL is unset; see audit/README.md",
            "endpoint": None,
            "classes": [],
        }

    sql = f"""
        SELECT audit_class, host, rows, age
        FROM (
          {
        " UNION ALL ".join(
            f"SELECT '{name}' AS audit_class, host, count() AS rows, "
            f"dateDiff('second', max(ts), now()) AS age "
            f"FROM audit.{name} GROUP BY host"
            for name in _HEALTH_CLASSES
        )
    }
        )
        ORDER BY audit_class, host
    """
    try:
        result = await store.execute(sql)
    except (AuditUnavailable, AuditQueryRejected) as exc:
        return {
            "status": "unreachable",
            "reason": str(exc),
            "endpoint": store.endpoint,
            "classes": [],
        }

    classes = [
        {
            "class": row["audit_class"],
            "host": row["host"],
            "rows": int(row["rows"]),
            "newest_age_s": int(row["age"]),
        }
        for row in result.rows
    ]
    beats = [c for c in classes if c["class"] == "host_beat"]

    stale = [c for c in beats if c["newest_age_s"] > 900]
    if not beats:
        status = "no_data"
    elif stale:
        status = "stale"
    else:
        status = "ok"
    return {
        "status": status,
        "endpoint": store.endpoint,
        "classes": classes,
        "hosts": sorted({c["host"] for c in classes}),
        "recorder": (recorder.stats if recorder else None),
        "attribution": "tool_call",
        "note": (
            "A session's timeline shows only work that session's own tool calls "
            "started. The host's other activity is not included."
        ),
    }


def _windows_for_session(db: OrmSession, stored_session_id: str) -> list[dict[str, Any]]:
    """The time ranges a session was working, from this gateway's run ledger."""
    runs = (
        db.execute(
            select(Run)
            # `runtime_session_id`, not `session_id`: the latter is a filing FK, never a stored id.
            .where(Run.runtime_session_id == stored_session_id)
            .order_by(Run.started_at.desc())
            .limit(50)
        )
        .scalars()
        .all()
    )
    windows: list[dict[str, Any]] = []
    for run in runs:
        if not run.started_at:
            continue
        ended = run.ended_at
        windows.append(
            {
                "run_id": run.id,
                "started_at": run.started_at.isoformat(),
                "ended_at": ended.isoformat() if ended else None,
                "status": run.status,
                "open": ended is None,
            }
        )
    return windows


def _profile_of_session(db: OrmSession, stored_session_id: str, rows: list[dict[str, Any]]) -> str:
    """Which profile this session runs on."""
    # The audit rows' own profile is authoritative; the run ledger's profile column can be wrong.
    for row in rows:
        profile = str(row.get("row_profile") or "").strip()
        if profile:
            return profile
    record = db.execute(
        select(Run.profile)
        .where(Run.runtime_session_id == stored_session_id)
        .where(Run.profile.is_not(None))
        .order_by(Run.started_at.desc())
        .limit(1)
    ).first()
    return str(record[0]) if record and record[0] else ""


def _unambiguous_windows(
    db: OrmSession, stored_session_id: str, profile: str, windows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Split a session's run windows by whether anything else could have caused
    the host activity inside them.
    """
    if not profile:
        return [], {w["run_id"] for w in windows}
    others = db.execute(
        select(Run.runtime_session_id, Run.started_at, Run.ended_at)
        .where(Run.profile == profile)
        .where(Run.runtime_session_id.is_not(None))
        .where(Run.runtime_session_id != stored_session_id)
        .where(Run.started_at.is_not(None))
        .order_by(Run.started_at.desc())
        .limit(2000)
    ).all()
    clean: list[dict[str, Any]] = []
    ambiguous: set[str] = set()
    for window in windows:
        start, end = window["started_at"], window["ended_at"]
        if end is None:

            # An open run has no end; bounding it at now() would sweep in anything started since.
            ambiguous.add(window["run_id"])
            continue
        overlap = any(
            other_start.isoformat() < end and start < (other_end or other_start).isoformat()
            for _, other_start, other_end in others
            if other_start is not None
        )
        (ambiguous.add(window["run_id"]) if overlap else clean.append(window))
    return clean, ambiguous


@audit_router.get("/audit/sessions/{stored_session_id}/timeline")
async def session_timeline(
    stored_session_id: str,
    request: Request,
    run_id: str | None = Query(None, description="narrow to one run"),
    limit: int = Query(500, ge=1, le=5000),
    db: OrmSession = Depends(_audit_db),
) -> dict[str, Any]:
    """What THIS SESSION did on the host. Nothing else."""
    windows = _windows_for_session(db, stored_session_id)
    if run_id:
        windows = [w for w in windows if w["run_id"] == run_id]
        if not windows:
            raise HTTPException(status_code=404, detail=f"no run {run_id} on this session")

    params: dict[str, Any] = {"session": stored_session_id, "limit": limit}

    def window_clause(column: str = "ts") -> str:
        """The optional single-run filter, for whichever alias holds `ts`."""
        if not (run_id and windows):
            return ""
        if windows[0]["ended_at"] is None:
            return f" AND {column} BETWEEN parseDateTimeBestEffort({{start:String}}) AND now()"
        return (
            f" AND {column} BETWEEN parseDateTimeBestEffort({{start:String}}) "
            "AND parseDateTimeBestEffort({end:String})"
        )

    if run_id and windows:
        params["start"] = windows[0]["started_at"]
        if windows[0]["ended_at"] is not None:
            params["end"] = windows[0]["ended_at"]


    # Qualified: the tool branch self-joins, so a bare `ts` is ambiguous and the query fails.
    tool_window = window_clause("t.ts")
    host_window = window_clause()
    sql = f"""
        -- `t.ts AS ts`, not a bare `t.ts`: a UNION takes its column NAMES
        -- from the first branch, so an unaliased qualified column would
        -- name every row's timestamp `t.ts` and the grouping and ordering
        -- below, which read `ts`, would silently see nothing.
        SELECT 'tool' AS kind, t.ts AS ts, toString(t.tool_name) AS subject,
               t.args AS detail, '' AS container_id, toUInt32(0) AS uid,
               '' AS exec_id, t.command AS extra,
               'agent' AS source, t.tool_call_id AS ref,
               toString(t.profile) AS row_profile,
               r.took_ms AS duration_ms, toString(r.outcome) AS status
        FROM audit.agent_tool AS t
        LEFT JOIN (
            SELECT tool_call_id,
                   max(duration_ms) AS took_ms,
                   argMax(status, ts) AS outcome
            FROM audit.agent_tool
            WHERE stored_session_id = {{session:String}} AND phase = 'result'
              AND tool_call_id != ''
            GROUP BY tool_call_id
        ) AS r ON t.tool_call_id = r.tool_call_id
        WHERE t.stored_session_id = {{session:String}}
          AND t.phase != 'result'{tool_window}
        UNION ALL
        SELECT 'exec' AS kind, ts, binary AS subject, arguments AS detail,
               container_id, uid, exec_id, toString(tool_name) AS extra,
               'host' AS source, tool_call_id AS ref, toString(profile) AS row_profile,
               toUInt32(0) AS duration_ms, '' AS status
        FROM audit.exec_attributed
        WHERE stored_session_id = {{session:String}}{host_window}
        UNION ALL
        SELECT 'file' AS kind, f.ts, f.path AS subject, f.op AS detail,
               f.container_id, f.uid, f.exec_id,
               toString(f.path_confidence) AS extra,
               'host' AS source, '' AS ref, '' AS row_profile,
               toUInt32(0) AS duration_ms, '' AS status
        FROM audit.host_file AS f
        INNER JOIN (
            SELECT DISTINCT exec_id FROM audit.exec_attributed
            WHERE stored_session_id = {{session:String}} AND exec_id != ''
        ) AS ae ON f.exec_id = ae.exec_id
        UNION ALL
        SELECT 'net' AS kind, n.ts,
               concat(n.daddr, ':', toString(n.dport)) AS subject,
               n.direction AS detail, n.container_id, n.uid, n.exec_id,
               n.binary AS extra,
               'host' AS source, '' AS ref, '' AS row_profile,
               toUInt32(0) AS duration_ms, '' AS status
        FROM audit.host_net AS n
        INNER JOIN (
            SELECT DISTINCT exec_id FROM audit.exec_attributed
            WHERE stored_session_id = {{session:String}} AND exec_id != ''
        ) AS ae ON n.exec_id = ae.exec_id
        ORDER BY ts
        LIMIT {{limit:UInt32}}
    """
    result = await _run_read(request, route="sessions.timeline", sql=sql, params=params)

    inferred_rows: list[dict[str, Any]] = []
    ambiguous: set[str] = set()
    profile = _profile_of_session(db, stored_session_id, result.rows)
    clean_windows, ambiguous = _unambiguous_windows(db, stored_session_id, profile, windows)
    if profile and clean_windows:
        for window in clean_windows[:10]:
            inferred_params = {
                "session": stored_session_id,
                "pfx": f"/profiles/{profile}/",
                "start": window["started_at"],
                "end": window["ended_at"],
                "limit": max(1, limit // 4),
            }
            inferred_sql = """
                SELECT 'exec' AS kind, e.ts, e.binary AS subject,
                       e.arguments AS detail, e.container_id, e.uid, e.exec_id,
                       e.parent_binary AS extra, 'inferred' AS source, '' AS ref,
                       '' AS row_profile, toUInt32(0) AS duration_ms, '' AS status
                FROM audit.host_exec AS e
                WHERE e.ts BETWEEN parseDateTimeBestEffort({start:String})
                              AND parseDateTimeBestEffort({end:String})
                  AND (position(e.arguments, {pfx:String}) > 0
                       OR position(e.cwd, {pfx:String}) > 0
                       OR position(e.parent_arguments, {pfx:String}) > 0)
                  AND e.exec_id NOT IN (
                      SELECT exec_id FROM audit.exec_attributed
                      WHERE stored_session_id = {session:String} AND exec_id != ''
                  )
                ORDER BY e.ts
                LIMIT {limit:UInt32}
            """
            try:
                extra = await _run_read(
                    request,
                    route="sessions.timeline.inferred",
                    sql=inferred_sql,
                    params=inferred_params,
                )
            except HTTPException:

                # An enhancement tier: a failure here must not fail the tiers above.
                break
            inferred_rows.extend(extra.rows)

    all_rows = list(result.rows) + inferred_rows

    if not all_rows:
        return {
            "session_id": stored_session_id,
            "correlation": "none",
            "windows": windows,
            "runs": [],
            "sources": {"agent_claim": 0, "host_observed": 0, "inferred": 0},
            "note": (
                "Nothing has been traced to this session. Either it made no tool calls, "
                "or they happened before session attribution was capturing. Only this "
                "session's own work is shown here -- the host's other activity is "
                "deliberately not included."
            ),
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(all_rows, key=lambda r: str(r.get("ts") or "")):
        groups.setdefault(_run_for(row.get("ts"), windows), []).append(row)

    runs_out: list[dict[str, Any]] = []
    for window in windows:
        rows = groups.pop(window["run_id"], [])
        if rows:
            runs_out.append({**window, **_group(rows, result.truncated)})
    for leftover_id, rows in groups.items():
        runs_out.append(
            {
                "run_id": leftover_id,
                "started_at": rows[0].get("ts", ""),
                "ended_at": rows[-1].get("ts"),
                "status": None,
                "open": False,
                **_group(rows, result.truncated),
            }
        )

    agent_rows = sum(1 for row in all_rows if row.get("source") == "agent")
    inferred_count = sum(1 for row in all_rows if row.get("source") == "inferred")
    host_rows = len(all_rows) - agent_rows - inferred_count
    inferred_note = (
        " Rows marked as started by this profile are a weaker inference: the process "
        "named this profile and ran inside this session's turn, and no other session "
        "of the same profile was running at the time. They are shown because a tool "
        "that runs no shell command can still start processes, and that work was "
        "otherwise invisible here."
        if inferred_count
        else ""
    )
    if host_rows:
        correlation = "tool_call"
        correlation_note = (
            "Two kinds of evidence, kept apart. Rows marked as observed are processes "
            "this session's own tool calls started, and the files and connections those "
            "processes produced -- the kernel recorded them independently. Rows marked "
            "as the agent's account are the tool calls themselves, as the agent reported "
            "them. The host's other work is not included." + inferred_note
        )
    else:
        correlation = "agent_only"
        correlation_note = (
            "These are this session's own tool calls, as the agent reported them. The "
            "kernel recorded nothing to corroborate them, which is the expected result "
            "for work that only reads: the file sensor captures writes, and a tool that "
            "reads a file or searches for one leaves no trace it can see. Treat these "
            "rows as the agent's account rather than independent observation." + inferred_note
        )

    return {
        "session_id": stored_session_id,
        "correlation": correlation,
        "correlation_note": correlation_note,
        "sources": {
            "agent_claim": agent_rows,
            "host_observed": host_rows,
            "inferred": inferred_count,
        },
        "ambiguous_runs": sorted(ambiguous),
        "windows": windows,
        "runs": runs_out,
    }


def _run_for(timestamp: Any, windows: list[dict[str, Any]]) -> str:
    """Which run's window contains this event, or `attributed` for none."""
    stamp = _comparable(timestamp)
    for window in windows:
        start = _comparable(window.get("started_at"))
        end = window.get("ended_at")
        if start and stamp >= start and (end is None or stamp <= _comparable(end)):
            return str(window["run_id"])
    return "attributed"


def _comparable(value: Any) -> str:
    """One string shape for the two timestamp formats that meet here."""
    # One side separates date and time with `T`, the other a space, and "T" sorts above " ".
    text = str(value or "").strip().replace("T", " ")
    for suffix in ("Z", "+00:00"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.rstrip()


def _group(rows: list[dict[str, Any]], truncated: bool) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return {"counts": counts, "events": rows, "truncated": truncated}


_MAX_RESULT_CHARS = 200_000


@audit_router.get("/audit/tool-calls/{tool_call_id}/output")
async def tool_call_output(
    tool_call_id: str,
    request: Request,
    stored_session_id: str = Query(..., description="the session the call belongs to"),
) -> dict[str, Any]:
    """What a tool actually returned, from the CONVERSATION store."""
    store = getattr(request.app.state, "chat_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="the chat store is not available")
    try:
        found = store.tool_result(
            stored_session_id=stored_session_id,
            tool_call_id=tool_call_id,
            max_chars=_MAX_RESULT_CHARS,
        )
    except Exception as exc:
        logger.warning("tool output lookup failed: %s", exc)
        raise HTTPException(status_code=503, detail="could not read the chat store") from exc
    if found is None:
        raise HTTPException(
            status_code=404,
            detail="no recorded output for this tool call",
        )
    return {
        **found,
        "source": "chat_store",
        "evidence": (
            "Recorded as part of the conversation, not by the audit trail. The audit "
            "archive holds no tool output by design. This copy lives in a database the "
            "gateway can rewrite, and it was never passed through the audit path's "
            "secret masking."
        ),
    }


@audit_router.get("/audit/files")
async def audit_files(
    request: Request,
    path_prefix: str = Query("", description="only paths starting with this"),
    since_minutes: int = Query(60, ge=1, le=60 * 24 * 30),
    op: str | None = Query(None, description="open_write | unlink | rename | write"),
    limit: int = Query(200, ge=1, le=5000),
) -> dict[str, Any]:
    """What changed on disk, newest first."""
    clauses = ["ts > now() - INTERVAL {since:UInt32} MINUTE"]
    params: dict[str, Any] = {"since": since_minutes, "limit": limit}
    if path_prefix:
        clauses.append("startsWith(path, {prefix:String})")
        params["prefix"] = path_prefix
    if op:
        clauses.append("op = {op:String}")
        params["op"] = op
    sql = f"""
        SELECT ts, host, op, path, path_to, path_confidence,
               binary, uid, container_id, exec_id, redacted
        FROM audit.host_file
        WHERE {" AND ".join(clauses)}
        ORDER BY ts DESC
        LIMIT {{limit:UInt32}}
    """
    result = await _run_read(request, route="files", sql=sql, params=params)
    return {
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "duration_ms": result.duration_ms,
    }


@audit_router.get("/audit/net")
async def audit_net(
    request: Request,
    since_minutes: int = Query(60, ge=1, le=60 * 24 * 30),
    direction: str | None = Query(None, description="outbound | inbound | close"),
    dport: int | None = Query(None, ge=0, le=65535),
    daddr: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
) -> dict[str, Any]:
    """Who this host talked to. Loopback is excluded at the sensor."""
    clauses = ["ts > now() - INTERVAL {since:UInt32} MINUTE"]
    params: dict[str, Any] = {"since": since_minutes, "limit": limit}
    if direction:
        clauses.append("direction = {direction:String}")
        params["direction"] = direction
    if dport is not None:
        clauses.append("dport = {dport:UInt32}")
        params["dport"] = dport
    if daddr:
        clauses.append("daddr = {daddr:String}")
        params["daddr"] = daddr
    sql = f"""
        SELECT ts, host, direction, binary, uid,
               saddr, sport, daddr, dport, container_id, exec_id
        FROM audit.host_net
        WHERE {" AND ".join(clauses)}
        ORDER BY ts DESC
        LIMIT {{limit:UInt32}}
    """
    result = await _run_read(request, route="net", sql=sql, params=params)
    return {
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "duration_ms": result.duration_ms,
    }


class AuditQueryBody(BaseModel):
    """`extra="forbid"`, like every body in this service: an unknown field is a
    422 rather than a silently ignored instruction."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, max_length=20_000)
    params: dict[str, str | int | float] = Field(default_factory=dict)


@audit_router.post("/audit/query")
async def audit_query(
    request: Request,
    body: AuditQueryBody = Body(...),
) -> dict[str, Any]:
    """One read-only SELECT against the audit schema."""
    result = await _run_read(request, route="query", sql=body.sql, params=dict(body.params))
    return {
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "max_rows": _store(request).max_rows,
        "duration_ms": result.duration_ms,
        "statistics": result.statistics,
    }


@audit_router.get("/audit/schema")
async def audit_schema(request: Request) -> dict[str, Any]:
    """The tables and columns a query may use."""
    names = ", ".join(f"'{t}'" for t in READABLE_TABLES)
    sql = f"""
        SELECT table, name, type
        FROM system.columns
        WHERE database = 'audit' AND table IN ({names})
        ORDER BY table, position
    """
    try:
        result = await _run_read(request, route="schema", sql=sql)
    except HTTPException as exc:
        if exc.status_code == 422:
            return {
                "tables": [{"table": t, "columns": []} for t in READABLE_TABLES],
                "note": (
                    "The audit credential cannot read system.columns, so only table "
                    "names are listed. See audit/clickhouse/schema.sql for the columns."
                ),
            }
        raise
    tables: dict[str, list[dict[str, str]]] = {}
    for row in result.rows:
        tables.setdefault(row["table"], []).append({"name": row["name"], "type": row["type"]})
    return {
        "tables": [{"table": t, "columns": cols} for t, cols in sorted(tables.items())],
        "readable": list(READABLE_TABLES),
    }
