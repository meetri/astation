"""SQLAlchemy 2.0 declarative models for the Research Gateway schema.

Source of truth: `docs/ARCHITECTURE.md` §13 (Database Schema v1). One table
per top-level block in that section; field names/types map directly onto
what's listed there. `*_json` columns use the generic `JSON` type (portable
across SQLite for now and Postgres later, per the doc's stated migration
path).

IDs are opaque strings (e.g. `proj_...`, `art_...`) per the doc's worked
examples (§5, §8), not autoincrement integers — `new_id()` mints them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: The only agent runtime this build talks to. Stored on every filing row so a
#: second runtime can be added later without the rows already written becoming
#: ambiguous -- and so the `UNIQUE (runtime, runtime_session_id)` constraint
#: means what it says: one workspace session per stored session *per runtime*.
#: (Was `api.projects.HERMES_RUNTIME`, which still re-exports it.)
HERMES_RUNTIME = "hermes"


def new_id(prefix: str) -> str:
    """Mint an opaque, prefixed, sortable-enough ID (e.g. 'proj_3f9a1c2b...')."""
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("proj"))
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    memory_namespace: Mapped[str | None] = mapped_column(String, nullable=True)
    #: The file-browser bookmark: an absolute sandbox
    #: path the browser opens at, or NULL for "open at the root". Nothing but
    #: the browser reads it -- deliberately NOT the project's instructions,
    #: which are a `HERMES.md` file (`domain/project_workspace.py`).
    folder_path: Mapped[str | None] = mapped_column(String, nullable=True)
    #: The one artifact this project is currently ABOUT -- the report, the
    #: figure, the thing someone would open first.
    #:
    #: **Deliberately NOT a foreign key.** `artifacts.project_id` already
    #: points here, so a second key pointing back makes the two tables
    #: circular, and SQLAlchemy cannot order a cycle for `create_all` /
    #: `drop_all` (caught by `test_an_unmigrated_database_says_so`, which
    #: tears the schema down). The constraint would buy little anyway: there
    #: is no delete for an artifact anywhere in this system, and the read path
    #: already renders `null` for a row it cannot find
    #: (`api/projects.py::_pinned_artifact_json`).
    pinned_artifact_id: Mapped[str | None] = mapped_column(String, nullable=True)
    default_profile_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("agent_profiles.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Session(Base):
    """A workspace session, plus its mapping onto a runtime's own session.

    Hermes (and, by assumption, any comparable runtime) exposes **two
    distinct session id spaces** — see `docs/PROTOCOL_VERIFIED.md`, section
    "Session resume & the two id spaces" (verified live 2026-08-29). Keeping
    them in two separate columns is deliberate: conflating them is what
    `docs/ARCHITECTURE.md` §21 warns about ("Do not let Hermes session IDs
    become primary keys for your research workspace"), and at runtime it
    surfaces as a bare `[4001] session not found`.

    `sessions.id` (this table's PK) is always our own opaque `sess_...` id.
    Neither runtime id is ever a primary or foreign key.
    """

    __tablename__ = "sessions"
    __table_args__ = (
        # A given stored runtime session maps to at most one workspace
        # Session row per runtime *and profile* -- a stored id is
        # only unique within a profile, so two profiles could otherwise
        # collide on the same id. NULLs compare as distinct in both SQLite
        # and Postgres, so any number of rows may still be unbound (not yet
        # attached to a runtime session).
        UniqueConstraint(
            "runtime",
            "profile",
            "runtime_session_id",
            name="uq_sessions_runtime_profile_runtime_session_id",
        ),
        # Every Phase 1 read of this table filters on `project_id`: the
        # per-project session list, the session count on the Project Library
        # screen, and the unfile-everything sweep in `DELETE /api/projects/{id}`.
        # The unique constraint above already indexes
        # `(runtime, runtime_session_id)`, which is what the All-sessions filing
        # lookup uses; this is the other access path and it had nothing.
        Index("ix_sessions_project_id", "project_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("sess"))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), nullable=False)
    runtime: Mapped[str] = mapped_column(String, nullable=False)

    #: The Hermes profile this session's runtime ids belong to --
    #: "default", "kimi25", "qwen38-flash", ... `runtime_session_id` is only
    #: unique *within* a profile, so this has to travel with the session
    #: rather than be inferred; every session-scoped route resolves its
    #: adapter from this, not from the request's own default connection.
    #: `server_default` only, no Python-side `default=`, on purpose
    #: (`test_schema.py`'s deployed-head migration test, B-116's own failure
    #: class): a client-side default makes SQLAlchemy name this column in
    #: every INSERT it generates, including a Core insert simulating a
    #: pre-migration row that has no such column yet. `server_default` is
    #: DDL only, so both a fresh `create_all()` test DB and the real Alembic
    #: migration give an un-set row "default" without this column ever
    #: needing to appear in a hand-written INSERT.
    profile: Mapped[str] = mapped_column(String, nullable=False, server_default="default")

    #: The runtime's **STORED / durable** session id — e.g. Hermes
    #: "20260829_182532_991e3f", from `session.list` -> `id`,
    #: `session.create` -> `stored_session_id`, or `session.active_list` ->
    #: `session_key`. It survives a Hermes restart and a gateway reconnect,
    #: so this is the durable runtime mapping and the only runtime id worth
    #: persisting. **This column NEVER holds a live handle.** It is also not
    #: directly usable for `session.history` / `prompt.submit` /
    #: `session.interrupt` / `session.activate` — those need the live handle,
    #: obtained by `session.resume {"session_id": <this value>}`.
    #: NULL means the workspace session has no runtime session yet.
    runtime_session_id: Mapped[str | None] = mapped_column(String, nullable=True)

    #: The runtime's **LIVE / ephemeral** handle — e.g. Hermes "5bfd9de6",
    #: from `session.create` -> `session_id` or `session.resume` ->
    #: `session_id`. Cached here only to avoid a redundant `session.resume`
    #: within one connection window.
    #:
    #: **NOT valid across a Hermes restart or a gateway reconnect** — the
    #: same stored id resolves to a *different* live handle on a later
    #: connection (verified: `5bfd9de6` then `d8779141`). Treat a value read
    #: from this column as a hint, not a fact: it must be re-resolved via
    #: `session.resume` whenever the connection it was minted on is gone.
    #: **Never use it as a key, a foreign key, a join column, or a lookup
    #: index** — it is not unique over time and it is not stable.
    runtime_live_session_id: Mapped[str | None] = mapped_column(String, nullable=True)

    title: Mapped[str | None] = mapped_column(String, nullable=True)
    branch_parent_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("sessions.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    #: **User intent, not a hide flag** (P6-3, `docs/SESSION_ARCHIVE_DESIGN.md`
    #: §2/§6.1). NULL = active. Set by `POST /api/sessions/{stored}/archive`
    #: only *after* a `SessionSnapshot` row for the session has been committed,
    #: so an archived session always has a durable copy behind it; cleared by
    #: `unarchive`. Moves the row between a project's Active and Archived
    #: sections and nothing else -- the session still appears in
    #: `GET /api/sessions` (the All-sessions guarantee) and Hermes is never
    #: told. This is the column the 2026-09-01 decision declined while
    #: "archive" meant a workspace-local soft-hide over nothing; it exists now
    #: because it rides on a real copy rather than standing in for one.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SessionSnapshot(Base):
    """A gateway-owned, complete, point-in-time copy of one Hermes session (P6-3).

    Hermes owns every transcript and, until this table, the gateway kept no
    copy at all (`messages` has 0 rows by P2-2 design; `run_events` only sees
    turns made through this gateway while connected). A snapshot is the
    durability the operator asked for on 2026-09-03 -- "keep all session data
    including AI chat and reasoning" -- taken through the ordinary
    `session.resume` path and stored **verbatim**: every row, every role,
    `reasoning`, `args`, `context`, `display_kind`, in Hermes's order, with no
    B-86 dedup (an archive is storage, not bandwidth). See
    `docs/SESSION_ARCHIVE_DESIGN.md` §2 and §6.1/§6.2.

    **This row is the index; the bytes are elsewhere.** The gzipped JSON
    document lives in the content-addressed `ArtifactStore`
    (`RESEARCH_GATEWAY_ARTIFACT_ROOT`) under `storage_key`, the same store
    artifacts and attachments use. `checksum` is the sha256 of the
    *uncompressed* document (what a reader verifies after gunzip);
    `content_checksum` covers only the transcript rows + background results
    (what the sweep compares to skip an unchanged session); `storage_key` is
    what the store returned for the gzip bytes.

    **Both FKs are `ON DELETE SET NULL`, and that is the whole cleanup
    story.** `project_id` is copied from the filing row at snapshot time and
    `workspace_session_id` points at the filing row itself; when a project is
    deleted or a session is unfiled/deleted, SQLite (with `PRAGMA
    foreign_keys=ON`, `domain/db.py`) nulls the reference and the snapshot
    row survives. No route in this wave deletes a snapshot. `project_id` is
    nullable for the same reason `runs.project_id` and `artifacts.project_id`
    are (P2-2d): unfiled sessions are the normal state and a pre-delete
    snapshot of one must still be possible.

    `stored_session_id` is the Hermes **STORED / durable** id -- never a live
    handle, never a key into Hermes, never a primary key here (two-id-spaces
    rule, `docs/PROTOCOL_VERIFIED.md`).

    `list_message_count` is `session.list.message_count` at snapshot time and
    is a **change flag only, never a size**: measured 2026-09-03, the list
    said 104 for a session whose resume returned 1,379 rows. The sweep
    compares "differs", never "greater". `message_rows` is the positional
    truth: `len(messages) + len(background_results)` as stored.
    """

    __tablename__ = "session_snapshots"
    __table_args__ = (
        # The two read paths: "snapshots of this session, newest first" (the
        # per-session list, the sweep's latest-snapshot lookup) and "every
        # snapshot filed under this project" (the Archived section).
        Index("ix_session_snapshots_stored_taken", "stored_session_id", "taken_at"),
        Index("ix_session_snapshots_project_id", "project_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("snap"))
    project_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    workspace_session_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    #: Hermes STORED session id. Never a live handle.
    stored_session_id: Mapped[str] = mapped_column(String, nullable=False)
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: manual | pre_delete | sweep | pre_compress (`api.snapshots.SNAPSHOT_REASONS`).
    reason: Mapped[str] = mapped_column(String, nullable=False)
    #: Hermes's title at snapshot time (the `session.list` row's), if known.
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    #: len(messages) + len(background_results) -- the positional truth.
    message_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    #: `session.list.message_count` at snapshot time. A change flag, not a size.
    list_message_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Verbatim from `session.resume`; NULL when Hermes sent none.
    messages_omitted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    #: Uncompressed document size.
    raw_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Gzipped size on disk.
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    #: sha256 of the UNCOMPRESSED document.
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    #: sha256 of json.dumps(messages + background_results, sort_keys=True).
    content_checksum: Mapped[str] = mapped_column(String, nullable=False)
    #: Exactly what `ArtifactStore.write_stream` returned for the gzip bytes.
    storage_key: Mapped[str] = mapped_column(String, nullable=False)
    #: The `session.list` row verbatim, or NULL.
    hermes_meta_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)


class ChatMessage(Base):
    """One durable chat-history row (P1 chat-history redesign, B-136).

    See `docs/CHAT_HISTORY_DESIGN.md` §3/§4 for the full design this
    implements verbatim. Replaces the `messages`/`turns` tables this
    migration drops (0 rows since P2-2 -- see `events/persistence.py`'s
    module docstring for why that Phase-0 machinery was cut rather than
    wired up: Hermes owns the transcript, and nothing read that copy). This
    table is different in kind: it is populated from each Hermes **profile**'s
    own live event stream (`domain/profile_connection.py`), one row per
    captured event, and it is what `GET /api/sessions/{id}/chat` reads from
    -- Hermes's own transcript is never queried for that route.

    **Uniqueness is `(profile, stored_session_id, seq)`, never
    `stored_session_id` alone.** Profiles are separate Hermes runtimes with
    separate id spaces -- two
    profiles could in principle mint the same stored id, and this table must
    not collide if they do.

    `seq` is assigned BY THE GATEWAY (`ChatStore`), monotonically per
    `(profile, stored_session_id)` -- it is not any ordering Hermes provides.
    `hermes_row_id` is that profile's own `state.db` `messages.id` **when
    known**; most live-captured rows do not carry one, because Hermes's push
    events (`message.interim`, `tool.complete`, `message.complete`) do not
    include it on the wire -- only the transcript rows `session.resume` /
    `session.history` return do. Backfill
    (reading a profile's `state.db` directly) is expected to be the path that
    populates it. `source` records which path wrote the row: `submit` (the
    app's own turn-submit route, before Hermes is called), `live` (captured
    off a `ProfileConnection`'s event stream), or `backfill`.

    `role` is `user | assistant | tool | system | marker`. `marker` is a
    non-content bookmark row: a compaction notice (`status.update
    kind=compacted`; compaction never deletes Hermes's own rows (§1.4), so
    this records that the event happened without rewriting anything around
    it), a memory-recall notice, or a background-process notice (B-191;
    `ChatStore.notice_kind` tells them apart for the app).

    `reasoning` is never truncated (D-3): whatever `message.completed`
    carried is stored in full.
    """

    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint(
            "profile",
            "stored_session_id",
            "seq",
            name="uq_chat_messages_profile_session_seq",
        ),
        Index("ix_chat_messages_profile", "profile"),
        Index("ix_chat_messages_stored_session_id", "stored_session_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("cm"))
    #: Hermes profile name ("default", "qwen38-flash", ...). Never inferred.
    profile: Mapped[str] = mapped_column(String, nullable=False)
    #: Hermes STORED session id -- unique only WITHIN a profile (see above).
    stored_session_id: Mapped[str] = mapped_column(String, nullable=False)
    turn_id: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Monotonic per (profile, stored_session_id), assigned by `ChatStore`.
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_args_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    tool_result_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    #: That profile's own `state.db` `messages.id`, when known -- see class docstring.
    hermes_row_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: submit | live | backfill -- see class docstring.
    source: Mapped[str] = mapped_column(String, nullable=False)
    compacted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ContextSnapshot(Base):
    __tablename__ = "context_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ctxsnap"))
    # No FK constraint here (deliberately): `turns.context_snapshot_id` is the
    # canonical FK direction (per ARCHITECTURE.md §13, only that side is
    # annotated "FK"). turn<->context_snapshot is a mutual reference; adding
    # a constraint on both sides creates an unresolvable table-creation-order
    # cycle for a single-column-at-a-time schema, so this side is a plain,
    # unenforced back-reference.
    turn_id: Mapped[str | None] = mapped_column(String, nullable=True)
    policy_id: Mapped[str | None] = mapped_column(String, nullable=True)
    token_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    digest: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ContextItem(Base):
    __tablename__ = "context_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ctxitem"))
    snapshot_id: Mapped[str] = mapped_column(
        String, ForeignKey("context_snapshots.id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String, nullable=False)
    source_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    token_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    provenance_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)


class Run(Base):
    """One Hermes turn, observed on the event stream (P2-2).

    **Flat, one row per turn, no parent/child tree.** The wire never produces
    child runs (a `prompt.background` task emits nothing while it runs and a
    delegation is a different surface entirely -- PV "Phase 2 probe"), so the
    §9 run *tree* is explicitly cut this phase; `parent_run_id` stays in the
    schema for a future runtime that has real sub-runs but is never written.

    **`project_id` is nullable (P2-2d).** A run inherits its project from the
    workspace Session row that *files* its Hermes session -- and filing is
    optional and additive (`api/projects.py`): most real sessions (every
    spike, everything unfiled) have no Session row at all. NULL means
    "unfiled", exactly the same answer `GET /api/sessions` gives
    (`project_id: null`). The rejected alternative -- a synthetic "unfiled"
    project row -- would surface a fake project in the Project Library
    (`GET /api/projects` lists every row), break its session-count semantics,
    and give `DELETE /api/projects/{id}` ("unfile everything") an undefined
    meaning for a project that *is* the absence of filing.

    **`runtime_session_id` is the attribution that always exists.** It is the
    Hermes **STORED / durable** session id (never a live handle -- the same
    column contract as `sessions.runtime_session_id`), recorded when the run
    is opened. `session_id`/`project_id` may both be NULL for an unfiled
    session, but every run still names the Hermes session it belongs to, so
    `GET /api/runs?session=<stored id>` works for spike sessions too.
    """

    __tablename__ = "runs"
    __table_args__ = (
        # The two P2-2e list filters. `runtime_session_id` is how the app
        # asks "this conversation's runs" (it holds stored ids for every
        # session, filed or not); `project_id` is the project-scoped view.
        Index("ix_runs_runtime_session_id", "runtime_session_id"),
        Index("ix_runs_project_id", "project_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("run"))
    project_id: Mapped[str | None] = mapped_column(String, ForeignKey("projects.id"), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String, ForeignKey("sessions.id"), nullable=True)
    #: Hermes STORED session id (durable, two-id-spaces rule). Nullable only
    #: because rows predating this column exist in principle; the recorder
    #: always writes it.
    runtime_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Which Hermes profile's connection this turn was observed on.
    #: Recorded at open time because it is the only moment it is known for
    #: certain -- the run's own event stream is what identifies it, and
    #: `session_id` is NULL for every unfiled session, so there is no row to
    #: read it back off later. The read path needs it for exactly one thing:
    #: `_reconcile_stale_running` asks Hermes "is this session still working",
    #: and asking the WRONG connection answers "never heard of it", which
    #: closes a genuinely-running turn as stale.
    #:
    #: `server_default` and no client-side `default=`, deliberately (B-116's
    #: class of trap): a client-side default makes SQLAlchemy name this column
    #: in every generated INSERT, including the Core inserts `test_schema.py`
    #: uses to simulate the PRE-migration schema.
    profile: Mapped[str] = mapped_column(String, nullable=False, server_default="default")
    #: Free-form, unenforced -- the `turns` table it originally pointed at was
    #: retired by the chat-history migration (B-136, `ChatMessage` above) and
    #: this column is, and always was, never written by any code path (the
    #: wire produces no turn concept -- see this class's own docstring on
    #: `parent_run_id`). Kept only for backward-reading old rows.
    turn_id: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_run_id: Mapped[str | None] = mapped_column(String, ForeignKey("runs.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    command_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RunEvent(Base):
    """Append-only per-run event log (P2-2). See `events/persistence.py`.

    `seq` is the **per-run, restart-surviving cursor** (P2-2c): allocated at
    persist time, monotonic within its run, and stored here -- so
    `GET /api/runs/{id}/events?after_seq=N` stays valid across any number of
    gateway restarts. It is NOT the broadcaster's global stream counter,
    which resets with the process and must never be used as a durable
    cursor.
    """

    __tablename__ = "run_events"
    __table_args__ = (
        # The one read path: events for a run, ordered by / filtered on seq.
        Index("ix_run_events_run_id_seq", "run_id", "seq"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("revt"))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BackgroundTask(Base):
    """The gateway's ledger of `prompt.background` tasks (P2-1, fixes B-42).

    This table exists because Hermes keeps NOTHING about a background task
    (PV "Phase 2 probe", 2026-08-30): the submit returns a `task_id` and the
    completion is a single `background.complete` event -- no transcript row,
    no pollable status, and a background-only session is never persisted at
    all. The submit-side row written here plus that one completion event are
    the background pill's *entire* data source, so the ledger is written at
    submit time, not lazily.

    `task_id` is Hermes's own id (`bg_` + 6 hex) and is the primary key --
    the completion event carries `{task_id, text}` and nothing else, so it is
    the only join key that exists. This is a deliberate, documented exception
    to the "runtime ids are never workspace primary keys" rule
    (`ARCHITECTURE.md` §21): unlike a session id there is no stable/ephemeral
    split here, just one opaque token that appears exactly twice on the wire.
    6 hex chars is a small space, so `record_submitted` upserts (with a
    warning) instead of trusting uniqueness across Hermes restarts.

    States (`domain/background_tasks.py` owns the transitions):

    * ``running``  -- submitted, no completion seen.
    * ``finished`` -- `background.complete` observed; `result_text` holds the
      task's one-and-only result (already markdown).
    * ``orphaned`` -- a gateway restart or Hermes reconnect happened while
      this row was ``running``. Measured (PV "Phase 2a probes", 5 runs): the
      completion event is delivered only to connections attached to the
      session at the instant it fires, never buffered or replayed -- so after
      a reconnect the outcome is genuinely unknown. The work itself most
      likely finished (marker-file proof); what may be lost is only the
      announcement. ``orphaned -> finished`` is a legal transition: the
      gateway re-resumes running-task sessions on every reconnect, and a
      completion that fires after that re-attach still arrives.
    """

    __tablename__ = "background_tasks"
    __table_args__ = (
        # The per-session list (pill, sheet, transcript injection) filters on
        # the stored session id; the orphan sweep and the reconnect rescue
        # filter on state.
        Index("ix_background_tasks_stored_session_id", "stored_session_id"),
        Index("ix_background_tasks_state", "state"),
    )

    #: Hermes's `bg_...` task id -- see class docstring for why it is the PK.
    task_id: Mapped[str] = mapped_column(String, primary_key=True)

    #: The Hermes **STORED / durable** session id, recorded at submit time.
    #: The completion event is stamped with only a LIVE handle (which may be
    #: unmapped by then), so this column is what attributes a result to a
    #: session. NULL only for a completion observed with no matching submit
    #: (a task submitted outside this gateway) whose live handle could not be
    #: attributed. **Never a live handle, never a workspace `sess_...` id.**
    stored_session_id: Mapped[str | None] = mapped_column(String, nullable=True)

    #: The prompt text submitted, verbatim. NULL for unmatched completions.
    prompt_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    #: running | finished | orphaned -- see class docstring.
    state: Mapped[str] = mapped_column(String, nullable=False, default="running")

    #: The `text` from `background.complete` -- markdown, and the only copy
    #: anywhere (Hermes writes no transcript row for a background turn).
    result_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: `HermesAdapter.connection_generation` at submit. Diagnostic: a
    #: completion is only deliverable while the submit-time attachment (or a
    #: later re-resume) holds, so "which connection was this submitted on"
    #: is the honest context for an orphaned row. Never used for resolution.
    connection_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Artifact(Base):
    """A durable file the gateway ingested from the Hermes sandbox (P3-1.0).

    **`project_id` is nullable (P3-1.0, mirroring P2-2d).** An artifact
    inherits its project from the workspace Session row that *files* the
    Hermes session it was produced in -- and filing is optional and additive
    (`api/projects.py`): most real sessions (every spike, everything unfiled)
    have no Session row at all. Unfiled spike sessions are this project's
    standard test path, so the very first live ingestion test
    would crash on a NOT NULL constraint. NULL means "unfiled", exactly the
    same answer `GET /api/sessions` and `runs.project_id` give. The rejected
    alternative -- a synthetic "unfiled" project row -- fails here for the
    same reasons it failed for runs (fake project in the Project Library,
    broken counts, undefined delete semantics).

    **`storage_key` is nullable.** An `unavailable` artifact was never
    fetched from the sandbox, so it has no storage key yet; the key is
    written only when the bytes actually land under
    `RESEARCH_GATEWAY_ARTIFACT_ROOT`.

    **`status` is `available` | `unavailable`** (server default `available`).
    `unavailable` is P3-1's designed failure path: the ingestion saw a
    structured file signal but could not fetch the bytes (sandbox file gone,
    download route failed) -- the row still exists so the app can show what
    *was* produced, with `source_path` naming where a refresh should look.

    **`source_path` is the absolute Hermes sandbox path the artifact was
    fetched from** (indexed). It answers "was this already ingested" (dedup
    lookup on ingest) and is the anchor for a future refresh/re-fetch;
    without it neither is queryable.

    **Provenance: `producing_run_id` is the PRIMARY provenance link.**
    Ingestion writes it directly on this row when the producing run is known
    (it is NULL for artifacts registered outside any run, e.g. a manual
    Tier-1 "save to library"). `artifact_links` is for *additional* relations
    only -- `input` / `evidence` / `attachment` / `reference` (§5), written
    when an artifact is later attached to, cited by, or fed into some other
    entity. The two are not duplicates: never write a `produced`-style
    artifact_link to mirror `producing_run_id`.
    """

    __tablename__ = "artifacts"
    __table_args__ = (
        # The P3-1 dedup/refresh lookup: "is this sandbox path already
        # ingested?" runs on every structured file signal.
        Index("ix_artifacts_source_path", "source_path"),
        Index("ix_artifacts_archived_at", "archived_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("art"))
    project_id: Mapped[str | None] = mapped_column(String, ForeignKey("projects.id"), nullable=True)
    producing_run_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("runs.id"), nullable=True
    )
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    #: NULL until the bytes are actually stored -- see class docstring.
    storage_key: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String, nullable=True)
    #: available | unavailable -- see class docstring.
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="available", server_default="available"
    )
    #: Absolute Hermes sandbox path the bytes came from -- see class docstring.
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    #: When the operator archived this artifact; NULL means visible. A timestamp
    #: rather than a boolean: it orders "recently archived", and un-archiving
    #: is a null write rather than a state machine.
    #:
    #: **Archive is not delete.** The bytes, the provenance and every
    #: transcript chip that opens this artifact keep working; it is hidden
    #: from the library's default listings and from the bookmark shelf, and
    #: reachable again through "Show archived". There is deliberately no
    #: delete for an artifact anywhere in this system
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Tag(Base):
    """One tag in the shared vocabulary.

    **One vocabulary for projects and artifacts**, not one per owner kind:
    this workspace is one project holding 95% of everything, and a tag that
    meant different things on a project and on a file would be two
    vocabularies wearing one name.

    `name` is unique and stored already-normalized
    (`domain/tags.py::normalize_tag_name`) -- lowercased, whitespace
    collapsed, a small alphabet. That uniqueness is what makes creating a tag
    idempotent, and what stops the library growing `Results` beside `results`:
    two rows that render identically in a chip and filter to different sets.
    """

    __tablename__ = "tags"
    __table_args__ = (
        UniqueConstraint("name", name="uq_tags_name"),
        Index("ix_tags_name", "name"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("tag"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArtifactTag(Base):
    """An artifact carries a tag. Cascades from both sides: deleting a tag
    detaches it everywhere and never deletes the artifact."""

    __tablename__ = "artifact_tags"
    __table_args__ = (Index("ix_artifact_tags_tag_id", "tag_id"),)

    artifact_id: Mapped[str] = mapped_column(
        String, ForeignKey("artifacts.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[str] = mapped_column(
        String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProjectTag(Base):
    """A project carries a tag. Same table shape and same cascade rule as
    `ArtifactTag`, deliberately: one vocabulary, two owners."""

    __tablename__ = "project_tags"
    __table_args__ = (Index("ix_project_tags_tag_id", "tag_id"),)

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[str] = mapped_column(
        String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Collection(Base):
    """A named, ordered, cross-project set of artifacts the operator curates.

    **Ordered is what makes this not a tag.** "Figures for the L328 paper" has
    a figure 1 and a figure 2; a tag has no such thing. `position` is the
    decision, not a chronology, which is why it is an integer the operator can
    rewrite rather than an `added_at` sort.

    **Cross-project, like the bookmark shelf.** This workspace is one project
    holding 95% of everything, so "which project" answers nothing about where
    a curated set belongs.
    """

    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("name", name="uq_collections_name"),
        Index("ix_collections_updated_at", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("col"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class CollectionArtifact(Base):
    """One artifact's membership of one collection, at one position.

    Cascades from both sides: deleting a collection never deletes an artifact,
    and an artifact that goes away leaves no dangling membership.
    """

    __tablename__ = "collection_artifacts"
    __table_args__ = (Index("ix_collection_artifacts_order", "collection_id", "position"),)

    collection_id: Mapped[str] = mapped_column(
        String, ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True
    )
    artifact_id: Mapped[str] = mapped_column(
        String, ForeignKey("artifacts.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArtifactBookmark(Base):
    """One star, on one artifact, **in one project**.

    Owner ask 2026-09-19: *"I just created a new project and I see bookmarked
    artifacts from other projects."* A star used to live on the artifact row
    itself (`artifacts.bookmarked_at`), which made it one global shelf shown
    in every scope; it is now a decision taken inside a scope, so the same
    file can be starred in the project that produced it and not in the one
    that is merely reading it.

    **`scope` is a project id, or `""` for the unfiled scope.** Empty string
    rather than NULL because SQLite treats NULLs as distinct in a unique
    index: `(artifact, NULL)` could be inserted twice and the shelf would
    show the same row twice. A non-null sentinel makes the uniqueness real,
    and the composite primary key is what enforces "starred once per scope".

    Cascades from the artifact: there is no delete for an artifact in this
    system, but if one ever goes it leaves no dangling star behind. The scope
    is deliberately NOT a foreign key to `projects` -- deleting a project is
    contractually one DELETE of our own rows (`api/projects.py`), and a
    cascade here would make it two.
    """

    __tablename__ = "artifact_bookmarks"
    __table_args__ = (
        # The shelf query: one scope's stars, newest first. Read on every
        # project open, so it gets its own index rather than a table scan.
        Index("ix_artifact_bookmarks_scope", "scope", "bookmarked_at"),
    )

    artifact_id: Mapped[str] = mapped_column(
        String, ForeignKey("artifacts.id", ondelete="CASCADE"), primary_key=True
    )
    #: The project this star was made in, or `""` for the unfiled scope.
    scope: Mapped[str] = mapped_column(String, primary_key=True)
    #: When the operator last said this matters -- the shelf's order. Re-starring
    #: refreshes it, so "un-star then star again" reads as a fresh entry.
    bookmarked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArtifactLink(Base):
    """Additional artifact relations ONLY -- never primary provenance.

    `Artifact.producing_run_id` is the primary provenance link and lives on
    the artifact row itself; rows here express the *other* ways an artifact
    relates to an entity (`input` / `evidence` / `attachment` / `reference`,
    §5) and are written when such a relation is created, not at ingestion.
    """

    __tablename__ = "artifact_links"

    artifact_id: Mapped[str] = mapped_column(String, ForeignKey("artifacts.id"), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String, primary_key=True)
    entity_id: Mapped[str] = mapped_column(String, primary_key=True)
    relation: Mapped[str] = mapped_column(String, primary_key=True)


class Attachment(Base):
    """The ledger for one composer attachment's two-turn journey to the
    sandbox (P3-3/P3-7).

    Hermes has NO upload endpoint (`/api/files*` is GET-only, measured PV
    "Phase 3 probe"), so the only way user-picked bytes reach the sandbox is
    the P3-0c-measured chain: the gateway stores the upload and serves it at
    an unguessable capability URL, a PRIMING TURN asks the agent to `curl` it
    into the sandbox, the gateway verifies the fetched copy byte-exact, and
    only then references it into the conversation (`image.attach` for images;
    a path reference for documents). That takes minutes (~2.5-3.5 min
    measured floor, 15 min worst case), so the whole flow is asynchronous and
    this row is what the app polls to show an honest "preparing attachment"
    state -- ledger-style, the same reasoning as `BackgroundTask`.

    States (`api/attachments.py::AttachmentOrchestrator` owns transitions):

    * ``uploaded``  -- bytes are in the gateway store; nothing sent to Hermes.
    * ``priming``   -- the fetch turn was submitted; waiting for the sandbox
      copy to appear and verify byte-exact (checksum match via
      `/api/files/download`).
    * ``attaching`` -- verified; `image.attach` is being issued (images only).
    * ``attached``  -- terminal success. For an image, `attach_result_json`
      holds Hermes's own `image.attach` answer; for a document, the verified
      `sandbox_path` is the deliverable and the app includes it in the user's
      next message.
    * ``failed``    -- terminal failure; `detail` says why (submit refused,
      verify timeout, checksum mismatch, attach error).
    * ``orphaned``  -- the gateway restarted mid-flow; the outcome is
      genuinely unknown (same honesty rule as `BackgroundTask.orphaned`).
      Unlike a background task there is no rescue: re-attach by re-uploading.

    `stored_session_id` is the Hermes **STORED** id (never a live handle,
    never a workspace `sess_...` id) -- live handles are re-resolved at every
    step. Bytes live in the shared content-addressed `ArtifactStore`
    (`storage_key`), so an attachment costs nothing extra when the same file
    is attached twice. `serve_token` is the unguessable capability the
    unauthenticated serve route is keyed on: the priming turn's `curl` runs
    on the Hermes host without the gateway's Basic-auth credential, and the
    token (a) never appears in logs, (b) grants exactly one file, and (c) is
    useless once the row is terminal (the serve route refuses).
    """

    __tablename__ = "attachments"
    __table_args__ = (
        # The app's per-session poll/list, and the serve route's token lookup.
        Index("ix_attachments_stored_session_id", "stored_session_id"),
        Index("ix_attachments_serve_token", "serve_token", unique=True),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("attach"))
    #: Hermes STORED session id the attachment is destined for.
    stored_session_id: Mapped[str] = mapped_column(String, nullable=False)
    #: Hermes profile that owns the stored id. Background work may run long
    #: after the upload request, so routing must travel with the ledger row.
    profile: Mapped[str] = mapped_column(String, nullable=False, default="default")
    #: Sanitized original filename (display + sandbox target naming).
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    #: sha256 of the uploaded bytes -- the verify step's ground truth.
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    #: Key into the shared content-addressed artifact store.
    storage_key: Mapped[str] = mapped_column(String, nullable=False)
    #: "image" (image/* -> image.attach) | "document" (path reference).
    kind: Mapped[str] = mapped_column(String, nullable=False)
    #: Absolute sandbox path the priming turn curls the file to.
    sandbox_path: Mapped[str] = mapped_column(String, nullable=False)
    #: Capability token for the unauthenticated serve route.
    serve_token: Mapped[str] = mapped_column(String, nullable=False)
    #: uploaded | priming | attaching | attached | failed | orphaned.
    state: Mapped[str] = mapped_column(String, nullable=False, default="uploaded")
    #: Human-readable progress/error detail for the app's quiet status line.
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Hermes's `image.attach` result (images) / verify metadata.
    attach_result_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AgentProfile(Base):
    __tablename__ = "agent_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("profile"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    runtime: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_policy_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    context_policy_id: Mapped[str | None] = mapped_column(String, nullable=True)
    memory_policy_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
