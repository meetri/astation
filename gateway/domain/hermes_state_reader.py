"""Read-only backfill reader for a Hermes profile's own `state.db`."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PINNED_MESSAGES_COLUMNS: tuple[str, ...] = (
    "id",
    "session_id",
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "effect_disposition",
    "timestamp",
    "token_count",
    "finish_reason",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "platform_message_id",
    "observed",
    "active",
    "compacted",
    "api_content",
    "display_kind",
    "display_metadata",
)


class HermesSchemaDriftError(RuntimeError):
    """Raised when a `state.db`'s `messages` table no longer matches
    `PINNED_MESSAGES_COLUMNS` -- a Hermes upgrade changed the schema this
    reader was built against, per §8's risk row. The fix is to re-measure the
    live schema and update this module deliberately, not to loosen the guard.
    """


@dataclass(frozen=True)
class BackfillMessage:
    """One `state.db` `messages` row, mapped onto `chat_messages` (§4) field
    names. `seq`, `profile`, `stored_session_id`, and `created_at` are not
    here -- those are the receiving `ChatStore`'s job, not the reader's.
    """

    hermes_row_id: int
    role: str
    text: str | None
    reasoning: str | None
    tool_name: str | None
    tool_call_id: str | None
    tool_args_json: str | None
    compacted: bool
    source: str = "backfill"


class ChatStoreProtocol(Protocol):
    """The two operations the sweep needs from a `ChatStore`-shaped object."""

    def max_hermes_row_id(self, *, profile: str, stored_session_id: str) -> int | None:
        """Highest `hermes_row_id` already captured for this (profile,
        session), by any source (`submit` / `live` / `backfill`) -- `None` if
        nothing has ever been captured for it."""
        ...

    def append_message(
        self, *, profile: str, stored_session_id: str, message: BackfillMessage
    ) -> None:
        """Append one row. Assigns `seq`, stamps `created_at`, and is the
        thing responsible for the `(profile, stored_session_id, seq)`
        uniqueness constraint (§4) -- this module calls it once per row, in
        ascending `hermes_row_id` order, and never for a row id it already
        read as the high-water mark."""
        ...


@contextmanager
def open_state_db_readonly(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open `db_path` with SQLite's own `mode=ro` URI flag -- refuses to
    create the file if it's missing and the connection cannot write to it at
    the driver level, not just by convention. Never point this at a live
    Hermes `state.db` from an agent session; this module is exercised only
    against fixture databases built in tests."""
    resolved = Path(db_path).resolve()
    uri = f"file:{resolved}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def assert_messages_schema(conn: sqlite3.Connection) -> None:
    """Fail loudly if `messages`'s actual columns disagree with
    `PINNED_MESSAGES_COLUMNS` -- order and membership both, since a Hermes
    upgrade could reorder columns without renaming any of them."""
    rows = conn.execute("PRAGMA table_info(messages)").fetchall()
    actual = tuple(row["name"] for row in rows)
    if actual != PINNED_MESSAGES_COLUMNS:
        raise HermesSchemaDriftError(
            "hermes state.db 'messages' table columns drifted from the "
            "pinned Hermes 0.20.5 schema.\n"
            f"expected: {PINNED_MESSAGES_COLUMNS}\n"
            f"actual:   {actual}"
        )


def list_session_ids(conn: sqlite3.Connection) -> list[str]:
    """Every session id present in this profile's `sessions` table."""
    rows = conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    return [row["id"] for row in rows]


def fetch_new_rows(
    conn: sqlite3.Connection, session_id: str, since_hermes_row_id: int
) -> list[sqlite3.Row]:
    """Rows for `session_id` newer than `since_hermes_row_id`, oldest first."""
    return conn.execute(
        "SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id",
        (session_id, since_hermes_row_id),
    ).fetchall()


def to_backfill_message(row: sqlite3.Row) -> BackfillMessage:
    """Map one raw `messages` row onto the `chat_messages` field shape."""
    reasoning = row["reasoning"] or row["reasoning_content"]
    return BackfillMessage(
        hermes_row_id=row["id"],
        role=row["role"],
        text=row["content"],
        reasoning=reasoning,
        tool_name=row["tool_name"],
        tool_call_id=row["tool_call_id"],
        tool_args_json=row["tool_calls"],
        compacted=bool(row["compacted"]),
    )


@dataclass(frozen=True)
class SweepResult:
    profile: str
    sessions_scanned: int
    rows_appended: int


def sweep_profile_state_db(
    db_path: str | Path, profile: str, chat_store: ChatStoreProtocol
) -> SweepResult:
    """Backfill sweep for one profile's `state.db` (§3, §4's "Backfill" row)."""
    with open_state_db_readonly(db_path) as conn:
        assert_messages_schema(conn)
        session_ids = list_session_ids(conn)
        rows_appended = 0
        for session_id in session_ids:
            watermark = chat_store.max_hermes_row_id(profile=profile, stored_session_id=session_id)
            since = watermark if watermark is not None else 0
            for row in fetch_new_rows(conn, session_id, since):
                message = to_backfill_message(row)
                chat_store.append_message(
                    profile=profile, stored_session_id=session_id, message=message
                )
                rows_appended += 1
        return SweepResult(
            profile=profile, sessions_scanned=len(session_ids), rows_appended=rows_appended
        )
