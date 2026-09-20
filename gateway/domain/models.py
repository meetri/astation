"""SQLAlchemy 2.0 declarative models for the Research Gateway schema."""

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
    folder_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Not a ForeignKey on purpose: artifacts.project_id points back, and the cycle breaks create_all.
    pinned_artifact_id: Mapped[str | None] = mapped_column(String, nullable=True)
    default_profile_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("agent_profiles.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Session(Base):
    """A workspace session, plus its mapping onto a runtime's own session."""

    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint(
            "runtime",
            "profile",
            "runtime_session_id",
            name="uq_sessions_runtime_profile_runtime_session_id",
        ),
        Index("ix_sessions_project_id", "project_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("sess"))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), nullable=False)
    runtime: Mapped[str] = mapped_column(String, nullable=False)

    # server_default only; a client-side default would name this column in every generated INSERT.
    profile: Mapped[str] = mapped_column(String, nullable=False, server_default="default")

    runtime_session_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # A hint, not a fact: invalid after any reconnect, and never usable as a key or join column.
    runtime_live_session_id: Mapped[str | None] = mapped_column(String, nullable=True)

    title: Mapped[str | None] = mapped_column(String, nullable=True)
    branch_parent_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("sessions.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SessionSnapshot(Base):
    """A gateway-owned, complete, point-in-time copy of one Hermes session (P6-3)."""

    __tablename__ = "session_snapshots"
    __table_args__ = (
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
    stored_session_id: Mapped[str] = mapped_column(String, nullable=False)
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    message_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    list_message_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    messages_omitted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    raw_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    content_checksum: Mapped[str] = mapped_column(String, nullable=False)
    storage_key: Mapped[str] = mapped_column(String, nullable=False)
    hermes_meta_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)


class ChatMessage(Base):
    """One durable chat-history row (P1 chat-history redesign, B-136)."""

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
    profile: Mapped[str] = mapped_column(String, nullable=False)
    stored_session_id: Mapped[str] = mapped_column(String, nullable=False)
    turn_id: Mapped[str | None] = mapped_column(String, nullable=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_args_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    tool_result_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    hermes_row_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    compacted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ContextSnapshot(Base):
    __tablename__ = "context_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ctxsnap"))
    # Deliberately not a ForeignKey: the mutual turn<->snapshot reference would be a creation cycle.
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
    """One Hermes turn, observed on the event stream (P2-2)."""

    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_runtime_session_id", "runtime_session_id"),
        Index("ix_runs_project_id", "project_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("run"))
    project_id: Mapped[str | None] = mapped_column(String, ForeignKey("projects.id"), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String, ForeignKey("sessions.id"), nullable=True)
    runtime_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    profile: Mapped[str] = mapped_column(String, nullable=False, server_default="default")
    # Never written by any code path; kept only so old rows still read back.
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
    """Append-only per-run event log (P2-2). See `events/persistence.py`."""

    __tablename__ = "run_events"
    __table_args__ = (
        Index("ix_run_events_run_id_seq", "run_id", "seq"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("revt"))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BackgroundTask(Base):
    """The gateway's ledger of `prompt.background` tasks (P2-1, fixes B-42)."""

    __tablename__ = "background_tasks"
    __table_args__ = (
        Index("ix_background_tasks_stored_session_id", "stored_session_id"),
        Index("ix_background_tasks_state", "state"),
    )

    task_id: Mapped[str] = mapped_column(String, primary_key=True)

    stored_session_id: Mapped[str | None] = mapped_column(String, nullable=True)

    prompt_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    state: Mapped[str] = mapped_column(String, nullable=False, default="running")

    result_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    connection_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Artifact(Base):
    """A durable file the gateway ingested from the Hermes sandbox (P3-1.0)."""

    __tablename__ = "artifacts"
    __table_args__ = (
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
    storage_key: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="available", server_default="available"
    )
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Tag(Base):
    """One tag in the shared vocabulary."""

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
    """A named, ordered, cross-project set of artifacts the operator curates."""

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
    """One artifact's membership of one collection, at one position."""

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
    """One star, on one artifact, **in one project**."""

    __tablename__ = "artifact_bookmarks"
    __table_args__ = (
        Index("ix_artifact_bookmarks_scope", "scope", "bookmarked_at"),
    )

    artifact_id: Mapped[str] = mapped_column(
        String, ForeignKey("artifacts.id", ondelete="CASCADE"), primary_key=True
    )
    scope: Mapped[str] = mapped_column(String, primary_key=True)
    bookmarked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArtifactLink(Base):
    """Additional artifact relations ONLY -- never primary provenance."""

    __tablename__ = "artifact_links"

    artifact_id: Mapped[str] = mapped_column(String, ForeignKey("artifacts.id"), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String, primary_key=True)
    entity_id: Mapped[str] = mapped_column(String, primary_key=True)
    relation: Mapped[str] = mapped_column(String, primary_key=True)


class Attachment(Base):
    """The ledger for one composer attachment's two-turn journey to the
    sandbox (P3-3/P3-7).
    """

    __tablename__ = "attachments"
    __table_args__ = (
        Index("ix_attachments_stored_session_id", "stored_session_id"),
        Index("ix_attachments_serve_token", "serve_token", unique=True),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("attach"))
    stored_session_id: Mapped[str] = mapped_column(String, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False, default="default")
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    storage_key: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    sandbox_path: Mapped[str] = mapped_column(String, nullable=False)
    serve_token: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False, default="uploaded")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
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
