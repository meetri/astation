"""Read-only access to the audit store, for the app and for AI tooling."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

READABLE_TABLES = (
    "host_exec",
    "host_exit",
    "host_file",
    "host_net",
    "host_auth",
    "host_container",
    "host_beat",
    "agent_tool",
    "agent_llm",
    "agent_session",
    "audit_query",
)

_ALLOWED_STARTS = ("select", "with")

_FORBIDDEN_FUNCTIONS = (
    "url",
    "s3",
    "file",
    "remote",
    "remotesecure",
    "mysql",
    "postgresql",
    "jdbc",
    "odbc",
    "hdfs",
    "executable",
    "input",
    "cluster",
    "clusterallreplicas",
    "azureblobstorage",
    "gcs",
    "deltalake",
    "hudi",
    "iceberg",
    "redis",
    "mongodb",
    "sqlite",
    "dictionary",
    "merge",
)

_COMMENT = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)
_STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'")


class AuditUnavailable(RuntimeError):
    """The store could not be reached or is not configured."""


class AuditQueryRejected(ValueError):
    """The SQL is not a single plain read. Answered as 422, and recorded."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    duration_ms: int
    truncated: bool = False
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)


def strip_sql_noise(sql: str) -> str:
    """Remove comments and string literals, for structural checks only."""
    return _STRINGS.sub("''", _COMMENT.sub(" ", sql))


def validate_select(sql: str) -> str:
    """Return the statement, or raise `AuditQueryRejected` explaining why not."""
    if not sql or not sql.strip():
        raise AuditQueryRejected("the query is empty")

    stripped = strip_sql_noise(sql).strip()
    if not stripped:
        raise AuditQueryRejected("the query is only comments")

    body = stripped.rstrip().rstrip(";")
    if ";" in body:
        raise AuditQueryRejected(
            "only one statement is allowed; a chained statement was refused "
            "before it reached the store"
        )

    first = body.lstrip().split(None, 1)[0].lower() if body.lstrip() else ""
    if first not in _ALLOWED_STARTS:
        raise AuditQueryRejected(
            f"only SELECT (or WITH) is allowed here; this query starts with {first.upper() or '?'}. "
            "The audit store is read-only by design and the database user cannot write either."
        )

    lowered = body.lower()
    for name in _FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{re.escape(name)}\s*\(", lowered):
            raise AuditQueryRejected(
                f"the table function {name}() is not available here: it reads from outside "
                "the audit store, which this credential cannot do"
            )

    if re.search(r"\binto\s+outfile\b", lowered):
        raise AuditQueryRejected("INTO OUTFILE writes to disk and is not allowed")
    if re.search(r"\bformat\s+\w+\s*$", lowered):
        raise AuditQueryRejected(
            "a trailing FORMAT clause is set by the gateway; remove it from the query"
        )
    return sql.strip().rstrip(";")


def _has_own_limit(statement: str) -> bool:
    """Does the statement already end in a LIMIT of its own?"""
    tail = strip_sql_noise(statement).lower().strip()
    return bool(re.search(r"\blimit\s+(\d+|\{\s*\w+\s*:\s*\w+\s*\})\s*$", tail))


class AuditStore:
    """A thin, read-only client for the audit store's HTTP interface."""

    def __init__(
        self,
        *,
        endpoint: str,
        user: str = "reader",
        password: str = "",
        database: str = "audit",
        max_rows: int = 10_000,
        timeout_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = (endpoint or "").rstrip("/")
        self._user = user
        self._password = password
        self._database = database
        self._max_rows = max_rows
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    @property
    def configured(self) -> bool:
        return bool(self._endpoint)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def max_rows(self) -> int:
        return self._max_rows

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s + 5)
        return self._client

    async def execute(self, sql: str, *, params: dict[str, Any] | None = None) -> QueryResult:
        """Run one validated read. Raises `AuditQueryRejected` / `AuditUnavailable`."""
        statement = validate_select(sql)
        if not self.configured:
            raise AuditUnavailable(
                "the audit store is not configured (AUDIT_CLICKHOUSE_URL is unset). "
                "See audit/README.md to deploy it."
            )

        query = (
            f"{statement}\nFORMAT JSON"
            if _has_own_limit(statement)
            else f"{statement}\nLIMIT {self._max_rows + 1}\nFORMAT JSON"
        )

        request_params: dict[str, Any] = {"database": self._database}
        for key, value in (params or {}).items():
            request_params[f"param_{key}"] = value

        started = time.perf_counter()
        try:
            response = await self._http().post(
                self._endpoint,
                content=query.encode("utf-8"),
                params=request_params,
                headers={
                    "X-ClickHouse-User": self._user,
                    "X-ClickHouse-Key": self._password,
                },
                timeout=self._timeout_s + 5,
            )
        except httpx.TimeoutException as exc:
            raise AuditUnavailable(
                f"the audit store did not answer within {self._timeout_s:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise AuditUnavailable(f"the audit store is unreachable: {type(exc).__name__}") from exc

        duration_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code == 403:
            raise AuditQueryRejected(
                "the audit credential refused this query. It may only SELECT from the "
                f"audit database: {_clickhouse_message(response.text)}"
            )
        if response.status_code >= 400:
            message = _clickhouse_message(response.text)
            if response.status_code in (404, 502, 503, 504):
                raise AuditUnavailable(
                    f"the audit store answered {response.status_code}: {message}"
                )
            raise AuditQueryRejected(f"the audit store rejected the query: {message}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise AuditUnavailable("the audit store returned a body that is not JSON") from exc

        rows = payload.get("data") or []
        columns = [c["name"] for c in (payload.get("meta") or [])]
        truncated = len(rows) > self._max_rows
        if truncated:
            rows = rows[: self._max_rows]
        return QueryResult(
            columns=columns,
            rows=rows,
            duration_ms=duration_ms,
            truncated=truncated,
            statistics=payload.get("statistics") or {},
        )


class AuditRecorder:
    """Writes `audit.audit_query` rows: who asked this store what."""

    def __init__(
        self,
        *,
        ingest_url: str,
        host: str = "",
        timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = (ingest_url or "").rstrip("/")
        self._host = host or "gateway"
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None
        self.stats: dict[str, int] = {"recorded": 0, "failed": 0, "skipped": 0}
        self._warned = False

    @property
    def configured(self) -> bool:
        return bool(self._url)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def record(
        self,
        *,
        caller: str,
        route: str,
        sql: str,
        rows_out: int,
        duration_ms: int,
        error: str = "",
    ) -> None:
        if not self.configured:
            self.stats["skipped"] += 1
            if not self._warned:
                self._warned = True
                logger.warning(
                    "audit: no ingest endpoint configured, so audit queries are NOT being "
                    "recorded. Set AUDIT_INGEST_URL to close that gap."
                )
            return
        import json as _json

        line = _json.dumps(
            {
                "audit_class": "audit_query",
                "host": self._host,
                "caller": caller,
                "route": route,
                "sql": (sql or "")[:4000],
                "rows_out": rows_out,
                "duration_ms": duration_ms,
                "error": (error or "")[:1000],
            }
        )
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self._timeout_s)
            response = await self._client.post(
                self._url, content=line.encode("utf-8"), timeout=self._timeout_s
            )
            if response.status_code >= 400:
                self.stats["failed"] += 1
                logger.warning("audit: ingest answered %s recording a query", response.status_code)
            else:
                self.stats["recorded"] += 1
        except Exception as exc:
            self.stats["failed"] += 1
            logger.warning("audit: could not record a query: %s", type(exc).__name__)


def _clickhouse_message(body: str) -> str:
    """The first useful line of a ClickHouse error, without the stack."""
    text = (body or "").strip()
    if not text:
        return "no detail"
    for line in text.splitlines():
        if "DB::Exception" in line or "Code:" in line:
            return line.strip()[:300]
    return text.splitlines()[0][:300]
