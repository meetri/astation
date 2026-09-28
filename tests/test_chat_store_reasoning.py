"""`ChatStore.capture_event` must not drop a `message.completed` turn's
reasoning when Hermes carries it under `reasoning_content` instead of
`reasoning` -- the same fallback every other reasoning reader in this
codebase (`transcript.py._measure_reasoning`, the app's
`SessionMessage.reasoningText`, `hermes_state_reader.to_backfill_message`)
already applies. Without it, the chat-store page -- the app's primary
transcript source -- silently serves `reasoning: null` for a turn that
plainly streamed reasoning live.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))

from domain.chat_store import ChatStore  # noqa: E402
from domain.models import Base  # noqa: E402

PROFILE = "default"
STORED_ID = "s_test"
STORED_ID_FIELD = "_stored_session_id"


@pytest.fixture()
def chat_store() -> ChatStore:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return ChatStore(factory)


def _message_completed_envelope(payload: dict) -> tuple[SimpleNamespace, dict]:
    canonical = SimpleNamespace(type="message.completed", payload=payload)
    envelope = {"payload": payload, "run_id": "run_1"}
    return canonical, envelope


def _rows(chat_store: ChatStore) -> list[dict]:
    return chat_store.page(profile=PROFILE, stored_session_id=STORED_ID, limit=50)


def test_reasoning_key_is_captured(chat_store: ChatStore) -> None:
    payload = {
        STORED_ID_FIELD: STORED_ID,
        "text": "Hello",
        "reasoning": "the primary key's text",
    }
    canonical, envelope = _message_completed_envelope(payload)
    chat_store.capture_event(PROFILE, canonical, envelope)

    [row] = _rows(chat_store)
    assert row["reasoning"] == "the primary key's text"


def test_reasoning_content_fallback_is_captured(chat_store: ChatStore) -> None:
    """The bug: a model that only sends `reasoning_content` used to leave
    this row's `reasoning` permanently `None` in the chat store, even though
    Hermes's own transcript, the light-listing `has_reasoning` flag, and the
    backfill sweep all recognize `reasoning_content` as real reasoning."""
    payload = {
        STORED_ID_FIELD: STORED_ID,
        "text": "Hello",
        "reasoning_content": "the fallback key's text",
    }
    canonical, envelope = _message_completed_envelope(payload)
    chat_store.capture_event(PROFILE, canonical, envelope)

    [row] = _rows(chat_store)
    assert row["reasoning"] == "the fallback key's text"


def test_reasoning_key_wins_over_reasoning_content(chat_store: ChatStore) -> None:
    payload = {
        STORED_ID_FIELD: STORED_ID,
        "text": "Hello",
        "reasoning": "primary",
        "reasoning_content": "fallback",
    }
    canonical, envelope = _message_completed_envelope(payload)
    chat_store.capture_event(PROFILE, canonical, envelope)

    [row] = _rows(chat_store)
    assert row["reasoning"] == "primary"


def test_no_reasoning_at_all_stays_none(chat_store: ChatStore) -> None:
    payload = {STORED_ID_FIELD: STORED_ID, "text": "Hello"}
    canonical, envelope = _message_completed_envelope(payload)
    chat_store.capture_event(PROFILE, canonical, envelope)

    [row] = _rows(chat_store)
    assert row["reasoning"] is None


# --- Reasoning that only ever streamed as `reasoning.delta` frames ---------------------------


def _capture(chat_store: ChatStore, event_type: str, payload: dict, run_id: str = "run_1") -> str | None:
    payload = {STORED_ID_FIELD: STORED_ID, **payload}
    canonical = SimpleNamespace(type=event_type, payload=payload)
    return chat_store.capture_event(PROFILE, canonical, {"payload": payload, "run_id": run_id})


def test_streamed_reasoning_lands_on_the_completed_row(chat_store: ChatStore) -> None:
    """The bug: reasoning streamed as deltas, `message.completed` carried none, and the row
    the app reloads after the turn replaced the reply that had shown it -- `reasoning: null`."""
    assert _capture(chat_store, "reasoning.delta", {"text": "First, "}) is None
    _capture(chat_store, "reasoning.delta", {"text": "think."})
    _capture(chat_store, "message.completed", {"text": "Answer"})

    [row] = _rows(chat_store)
    assert row["text"] == "Answer"
    assert row["reasoning"] == "First, think."


def test_payload_reasoning_wins_over_streamed(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "partial"})
    _capture(chat_store, "message.completed", {"text": "Answer", "reasoning": "authoritative"})

    [row] = _rows(chat_store)
    assert row["reasoning"] == "authoritative"


def test_streamed_reasoning_goes_with_its_segment(chat_store: ChatStore) -> None:
    """Reasoning before a tool call belongs to the sealed segment, the way the app draws it
    live; the final reply gets only what streamed after."""
    _capture(chat_store, "reasoning.delta", {"text": "before tool"})
    _capture(chat_store, "message.interim", {"text": "Let me check."})
    _capture(chat_store, "reasoning.delta", {"text": "after tool"})
    _capture(chat_store, "message.completed", {"text": "Done."})

    segment, final = _rows(chat_store)
    assert (segment["text"], segment["reasoning"]) == ("Let me check.", "before tool")
    assert (final["text"], final["reasoning"]) == ("Done.", "after tool")


def test_streamed_reasoning_is_held_per_run(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "run one"}, run_id="run_1")
    _capture(chat_store, "reasoning.delta", {"text": "run two"}, run_id="run_2")
    _capture(chat_store, "message.completed", {"text": "A"}, run_id="run_2")
    _capture(chat_store, "message.completed", {"text": "B"}, run_id="run_1")

    first, second = _rows(chat_store)
    assert (first["text"], first["reasoning"]) == ("A", "run two")
    assert (second["text"], second["reasoning"]) == ("B", "run one")


def test_held_reasoning_is_spent_once(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "once"})
    _capture(chat_store, "message.completed", {"text": "A"})
    _capture(chat_store, "message.completed", {"text": "B"})

    first, second = _rows(chat_store)
    assert first["reasoning"] == "once"
    assert second["reasoning"] is None
