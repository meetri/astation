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


# --- Multi-step turns: every step's reasoning is kept ------------------------------------------


def test_each_tool_step_keeps_its_own_reasoning(chat_store: ChatStore) -> None:
    """The bug: Hermes's `message.complete` repeats only the turn's last reasoning block, and
    it used to replace everything streamed across the turn's earlier tool-call steps."""
    _capture(chat_store, "reasoning.delta", {"text": "step one thinking"})
    _capture(chat_store, "tool.started", {"tool_id": "t1", "name": "read"})
    _capture(chat_store, "tool.completed", {"tool_id": "t1", "name": "read", "result": "x"})
    _capture(chat_store, "reasoning.delta", {"text": "step two thinking"})
    _capture(chat_store, "tool.started", {"tool_id": "t2", "name": "grep"})
    _capture(chat_store, "tool.completed", {"tool_id": "t2", "name": "grep", "result": "y"})
    _capture(chat_store, "reasoning.delta", {"text": "final thinking"})
    _capture(chat_store, "message.completed", {"text": "Answer", "reasoning": "final thinking"})

    rows = _rows(chat_store)
    assert [(r["role"], r["text"], r["reasoning"]) for r in rows] == [
        ("assistant", "", "step one thinking"),
        ("tool", None, None),
        ("assistant", "", "step two thinking"),
        ("tool", None, None),
        ("assistant", "Answer", "final thinking"),
    ]


def test_tool_started_returns_no_row_id(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "thinking"})
    assert _capture(chat_store, "tool.started", {"tool_id": "t1", "name": "read"}) is None


def test_parallel_tool_calls_write_one_reasoning_row(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "plan"})
    _capture(chat_store, "tool.started", {"tool_id": "t1", "name": "read"})
    _capture(chat_store, "tool.started", {"tool_id": "t2", "name": "read"})
    _capture(chat_store, "message.completed", {"text": "Done."})

    rows = _rows(chat_store)
    assert [(r["text"], r["reasoning"]) for r in rows] == [("", "plan"), ("Done.", None)]


def test_payload_repeating_an_earlier_step_is_not_written_twice(chat_store: ChatStore) -> None:
    """A final step with no reasoning of its own: Hermes's payload falls back to the tool
    step's block, which is already on that step's row."""
    _capture(chat_store, "reasoning.delta", {"text": "only thinking"})
    _capture(chat_store, "tool.started", {"tool_id": "t1", "name": "read"})
    _capture(chat_store, "message.completed", {"text": "Done.", "reasoning": "only thinking"})

    step, final = _rows(chat_store)
    assert step["reasoning"] == "only thinking"
    assert final["reasoning"] is None


def test_streamed_text_wins_when_it_holds_the_payload(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "long streamed reasoning, then the tail"})
    _capture(chat_store, "message.completed", {"text": "A", "reasoning": "then the tail"})

    [row] = _rows(chat_store)
    assert row["reasoning"] == "long streamed reasoning, then the tail"


def test_payload_wins_when_it_holds_the_stream(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "the start"})
    _capture(chat_store, "message.completed", {"text": "A", "reasoning": "the start and the rest"})

    [row] = _rows(chat_store)
    assert row["reasoning"] == "the start and the rest"


# --- The reasoning archive -------------------------------------------------------------------


def _archive(chat_store: ChatStore) -> list[dict]:
    return chat_store.reasoning_archive(profile=PROFILE, stored_session_id=STORED_ID)


def test_streamed_reasoning_is_archived_while_it_streams(chat_store: ChatStore) -> None:
    """A gateway restart mid-turn must not lose what already streamed."""
    _capture(chat_store, "reasoning.delta", {"text": "first chunk"})

    [entry] = _archive(chat_store)
    assert (entry["source"], entry["text"], entry["sealed"]) == ("stream", "first chunk", False)


def test_archive_seals_each_step_against_its_row(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "step one"})
    _capture(chat_store, "tool.started", {"tool_id": "t1", "name": "read"})
    _capture(chat_store, "reasoning.delta", {"text": "step two"})
    row_id = _capture(chat_store, "message.completed", {"text": "Answer"})

    entries = _archive(chat_store)
    assert [(e["step"], e["text"], e["sealed"]) for e in entries] == [
        (0, "step one", True),
        (1, "step two", True),
    ]
    step_row, final_row = _rows(chat_store)
    assert entries[0]["chat_message_id"] == step_row["id"]
    assert entries[1]["chat_message_id"] == final_row["id"] == row_id


def test_archive_keeps_a_payload_that_differs_from_the_stream(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "partial"})
    _capture(chat_store, "message.completed", {"text": "A", "reasoning": "authoritative"})

    texts = {(e["source"], e["text"]) for e in _archive(chat_store)}
    assert texts == {("stream", "partial"), ("payload", "authoritative")}


def test_archive_flushes_are_throttled_but_sealing_writes_everything(chat_store: ChatStore) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "a"})
    _capture(chat_store, "reasoning.delta", {"text": "b"})
    [entry] = _archive(chat_store)
    assert entry["text"] == "a"  # second chunk waits for the flush interval

    _capture(chat_store, "message.completed", {"text": "A"})
    [entry] = _archive(chat_store)
    assert (entry["text"], entry["sealed"]) == ("ab", True)


def test_turn_completed_hook_fires(chat_store: ChatStore) -> None:
    seen: list[tuple[str, str]] = []
    chat_store.on_turn_completed = lambda profile, stored_id: seen.append((profile, stored_id))
    _capture(chat_store, "message.completed", {"text": "A"})
    assert seen == [(PROFILE, STORED_ID)]
