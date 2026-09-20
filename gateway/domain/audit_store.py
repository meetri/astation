"""Read-only access to the audit store, for the app and for AI tooling.

The audit store (`audit-setup/` in this repo) records every process, file mutation
and network connection on a host. It lives in ClickHouse on the collector, its
schema is `audit-setup/clickhouse/schema.sql`, and this module is the ONLY way the
gateway touches it.

Three rules shape everything here, and each exists because the alternative
fails quietly:

**Read-only, enforced twice.** The gateway connects as the ClickHouse `reader`
user, which is granted `SELECT` on `audit.*` and nothing else, with
`readonly=1` and row/time caps declared as *constraints* so a query cannot
raise its own ceiling. That is the real control; it is enforced by the database
whatever this process does. `validate_select` is the second layer: it rejects
anything that is not a single SELECT before a request is made, so a caller gets
a clear 422 instead of a database error, and a statement-chaining attempt never
reaches the wire at all.

**Unreachable is not empty.** A store that cannot be reached raises
`AuditUnavailable`, which the routes turn into a 503 naming the reason. It
never degrades to an empty result: "nothing happened" and "we could not look"
are opposite answers to a security question, and a UI that cannot tell them
apart is worse than no UI.

**Every read is itself recorded.** `audit.audit_query` holds who asked what,
how long it took and how many rows came back. The analyst is audited by the
thing it analyses. Recording is best-effort and never fails a caller's query --
but a failure to record is logged, because silent gaps in that table would
defeat its purpose.

Correlation, and its honest limit: until the session-capture hooks ship, a
host event carries no session id. `session_windows` therefore correlates by
TIME and CONTAINER, using the run ledger this gateway already keeps -- a run's
start and end bound the work it did. That is a real answer available today, and
every row it returns is labelled `correlation="time_window"` so nobody mistakes
it for the exact attribution that `pre_tool_call` will provide later.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: Tables a caller may read. The ClickHouse user is granted `audit.*` and this
#: list is the gateway's own narrower view of it, so a table added to the store
#: for some other purpose is not automatically exposed through the app.
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

#: Statement starts that are reads. Anything else is refused outright.
_ALLOWED_STARTS = ("select", "with")

#: Table functions that read from OUTSIDE the store. The `reader` user is not
#: granted them, so these are already refused by ClickHouse; naming them here
#: turns a confusing permission error into a clear message, and makes the
#: intent reviewable.
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
    """The store could not be reached or is not configured.

    Deliberately distinct from "no rows": the routes answer 503, never 200 with
    an empty list. Conflating the two would let a dead collector read as a
    quiet week.
    """


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
    """Remove comments and string literals, for structural checks only.

    Both are removed before looking for statement separators or function names,
    so a semicolon inside a quoted string is not mistaken for a second
    statement and a table name mentioned in a comment is not mistaken for a
    call. The ORIGINAL sql is what gets executed; this is only ever inspected.
    """
    return _STRINGS.sub("''", _COMMENT.sub(" ", sql))


def validate_select(sql: str) -> str:
    """Return the statement, or raise `AuditQueryRejected` explaining why not.

    This is the gateway's own check, in front of a database user that already
    cannot write. It exists to give callers a precise reason, and to stop a
    chained statement before it is sent rather than relying on the server to
    refuse the second half.
    """
    if not sql or not sql.strip():
        raise AuditQueryRejected("the query is empty")

    stripped = strip_sql_noise(sql).strip()
    if not stripped:
        raise AuditQueryRejected("the query is only comments")

    # One statement. A trailing semicolon is fine; anything after it is not.
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
        # `\b<name>\s*(` -- a call, not a column that merely contains the word.
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
    """Does the statement already end in a LIMIT of its own?

    It must recognise a PARAMETERISED limit (`LIMIT {limit:UInt32}`), not just a
    literal. A check that only matched digits appended a second LIMIT to every
    route that binds its own, producing `LIMIT {limit:UInt32} LIMIT 10001` and a
    syntax error from the store -- which surfaced as a 422 on /files and /net
    against the deployed instance, 2026-09-20.
    """
    tail = strip_sql_noise(statement).lower().strip()
    return bool(re.search(r"\blimit\s+(\d+|\{\s*\w+\s*:\s*\w+\s*\})\s*$", tail))


class AuditStore:
    """A thin, read-only client for the audit store's HTTP interface.

    Constructed once at startup and shared. `configured` is False when no
    endpoint is set, which is the normal state for an install that has not
    deployed the audit stack -- every route then answers 503 with that reason
    rather than pretending the store is empty.
    """

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
        """Run one validated read. Raises `AuditQueryRejected` / `AuditUnavailable`.

        Parameters are passed to ClickHouse as `param_<name>` and referenced in
        SQL as `{name:Type}`, so a caller's value is never concatenated into the
        statement. Every route that takes user input uses this rather than
        building SQL by hand.
        """
        # Validate BEFORE checking whether a store exists. A write or a chained
        # statement is wrong whatever the store's state, and answering
        # "unavailable" would send someone to debug a healthy collector over a
        # typo. It also means the refusal rules hold on an install that has no
        # audit stack at all.
        statement = validate_select(sql)
        if not self.configured:
            raise AuditUnavailable(
                "the audit store is not configured (AUDIT_CLICKHOUSE_URL is unset). "
                "See audit/README.md to deploy it."
            )

        # One row over the cap, so a full page is distinguishable from a page
        # that happens to be exactly the cap.
        query = (
            f"{statement}\nFORMAT JSON"
            if _has_own_limit(statement)
            else f"{statement}\nLIMIT {self._max_rows + 1}\nFORMAT JSON"
        )

        # ONLY the database. No settings.
        #
        # The reader profile is `readonly=1`, and ClickHouse refuses ANY
        # setting change in that mode -- including LOWERING a cap. Sending
        # max_result_rows here, even set to a stricter value than the profile's
        # own, failed every query with:
        #   Code: 164 ... Cannot modify 'max_result_rows' setting in readonly mode
        # Measured against the deployed store, 2026-09-20.
        #
        # Which is the right behaviour, and it makes the point this module
        # already claimed: the caps belong to the database user, declared as
        # constraints in audit/clickhouse/users.d/audit.xml, and are not this
        # process's to set. The row cap is still applied as a LIMIT in the
        # statement above, which is a query construct rather than a setting.
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
            # The credential is read-only on purpose; say which rule was hit.
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
    """Writes `audit.audit_query` rows: who asked this store what.

    Separate from `AuditStore` because it is the opposite direction and uses a
    different credential path. The gateway's store credential is read-only and
    *cannot* write this table -- by design -- so the row goes to the shipper's
    ingest endpoint, the same route every other audit event takes, and lands
    through the same transform and the same two sinks.

    Never raises. A read must not fail because its record failed; but a
    recording failure is counted and logged, because a silently unrecorded
    query defeats the point of the table.
    """

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
                # Bounded: a pathological query must not become a pathological row.
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
