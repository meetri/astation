"""`reconcile_reasoning`: archive every reasoning body in Hermes's transcript, and give a chat
row the full body when it holds none or only part of it -- never a shorter or unrelated one."""

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
from domain.reasoning_reconcile import reasoning_body, reconcile_reasoning  # noqa: E402

PROFILE = "default"
STORED_ID = "s_test"

LONG_STEP = "I should read the config first, then check which profile is active."


@pytest.fixture()
def factory() -> sessionmaker:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@pytest.fixture()
def chat_store(factory: sessionmaker) -> ChatStore:
    return ChatStore(factory)


def _capture(chat_store: ChatStore, event_type: str, payload: dict, run_id: str = "run_1") -> None:
    payload = {"_stored_session_id": STORED_ID, **payload}
    canonical = SimpleNamespace(type=event_type, payload=payload)
    chat_store.capture_event(PROFILE, canonical, {"payload": payload, "run_id": run_id})


def _rows(chat_store: ChatStore) -> list[dict]:
    return chat_store.page(profile=PROFILE, stored_session_id=STORED_ID, limit=50)


def _hermes(*rows: dict) -> list[dict]:
    return [{"role": "user", "text": "go"}, *rows]


def test_every_hermes_body_is_archived_once(factory, chat_store) -> None:
    messages = _hermes(
        {"role": "assistant", "text": "", "row_id": 11, "reasoning": "step one"},
        {"role": "tool", "name": "read"},
        {"role": "assistant", "text": "Answer", "row_id": 12, "reasoning": "step two"},
    )
    first = reconcile_reasoning(factory, PROFILE, STORED_ID, messages)
    again = reconcile_reasoning(factory, PROFILE, STORED_ID, messages)

    assert first.archived == 2
    assert again.archived == 0
    archived = chat_store.reasoning_archive(profile=PROFILE, stored_session_id=STORED_ID)
    assert {(e["hermes_row_id"], e["text"]) for e in archived} == {(11, "step one"), (12, "step two")}


def test_a_clipped_reasoning_only_row_gets_the_full_body(factory, chat_store) -> None:
    """A reconnect lost the start of the stream: the row holds only the tail."""
    _capture(chat_store, "reasoning.delta", {"text": LONG_STEP[20:]})
    _capture(chat_store, "tool.started", {"tool_id": "t1", "name": "read"})
    _capture(chat_store, "message.completed", {"text": "Answer"})

    result = reconcile_reasoning(
        factory, PROFILE, STORED_ID,
        _hermes(
            {"role": "assistant", "text": "", "row_id": 1, "reasoning": LONG_STEP},
            {"role": "assistant", "text": "Answer", "row_id": 2},
        ),
    )
    assert result.upgraded == 1
    assert _rows(chat_store)[0]["reasoning"] == LONG_STEP


def test_a_reply_with_no_reasoning_is_filled_by_its_unique_text(factory, chat_store) -> None:
    _capture(chat_store, "message.completed", {"text": "Here is the answer."})
    reconcile_reasoning(
        factory, PROFILE, STORED_ID,
        _hermes({"role": "assistant", "text": "Here is the answer.", "row_id": 3, "reasoning": "why"}),
    )
    assert _rows(chat_store)[0]["reasoning"] == "why"


def test_an_ambiguous_text_is_not_filled(factory, chat_store) -> None:
    _capture(chat_store, "message.completed", {"text": "Done."}, run_id="r1")
    reconcile_reasoning(
        factory, PROFILE, STORED_ID,
        _hermes(
            {"role": "assistant", "text": "Done.", "row_id": 1, "reasoning": "turn one"},
            {"role": "user", "text": "again"},
            {"role": "assistant", "text": "Done.", "row_id": 2, "reasoning": "turn two"},
        ),
    )
    assert _rows(chat_store)[0]["reasoning"] is None


def test_reasoning_is_never_shortened_or_replaced(factory, chat_store) -> None:
    _capture(chat_store, "reasoning.delta", {"text": "A whole turn of reasoning, all steps."})
    _capture(chat_store, "message.completed", {"text": "Answer"})
    reconcile_reasoning(
        factory, PROFILE, STORED_ID,
        _hermes({"role": "assistant", "text": "Answer", "row_id": 5, "reasoning": "all steps."}),
    )
    assert _rows(chat_store)[0]["reasoning"] == "A whole turn of reasoning, all steps."


def test_structured_reasoning_is_read() -> None:
    assert reasoning_body(
        {"reasoning_details": [{"type": "reasoning.text", "text": "one"}, {"data": "opaque"}]}
    ) == "one"
    assert reasoning_body(
        {"codex_reasoning_items": [
            {"type": "reasoning", "summary": [{"text": "a"}, {"text": "b"}], "encrypted_content": "x"}
        ]}
    ) == "a\n\nb"
    assert reasoning_body({"reasoning": "", "reasoning_content": "fallback"}) == "fallback"
    assert reasoning_body({}) is None


def test_a_text_repeated_by_a_row_without_reasoning_is_still_ambiguous(factory, chat_store) -> None:
    _capture(chat_store, "message.completed", {"text": "Done."}, run_id="r1")
    reconcile_reasoning(
        factory, PROFILE, STORED_ID,
        _hermes(
            {"role": "assistant", "text": "Done.", "row_id": 1},
            {"role": "user", "text": "again"},
            {"role": "assistant", "text": "Done.", "row_id": 2, "reasoning": "turn two"},
        ),
    )
    assert _rows(chat_store)[0]["reasoning"] is None
