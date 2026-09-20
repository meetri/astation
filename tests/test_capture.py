"""Tests for the hook enrichment layer."""

from __future__ import annotations

import importlib.util
import queue
import sys
import time
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("trg_capture", PLUGIN_DIR / "capture.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trg_capture"] = mod
    spec.loader.exec_module(mod)
    return mod


capture = _load()


class FakeRecorder:
    def __init__(self, open_runs=()):
        self.events: list[tuple] = []
        self._open = set(open_runs)

    def has_open_run(self, stored_id, profile="default"):
        return stored_id in self._open

    def handle_event(self, canonical, envelope, profile):
        self.events.append((canonical.type, canonical.payload, profile))


class FakeState:
    def __init__(self, recorder):
        self.run_recorder = recorder


def drain_for(recorder) -> capture.HookDrain:
    return capture.HookDrain(queue.Queue(), FakeState(recorder), profile="default")


def test_turn_map_carries_session_and_model_forward():
    """post_api_request has neither; both must come from pre_llm_call."""
    m = capture.TurnMap()
    m.remember("t1", session_id="20260919_1", model="qwen3.8-27b")
    assert m.get("t1")["session_id"] == "20260919_1"
    assert m.get("t1")["model"] == "qwen3.8-27b"


def test_turn_map_is_bounded():
    m = capture.TurnMap(max_entries=10)
    for i in range(50):
        m.remember(f"t{i}", session_id="s")
    assert len(m) <= 11


def test_turn_map_expires_old_turns():
    m = capture.TurnMap(ttl_s=0.01)
    m.remember("t1", session_id="s")
    time.sleep(0.05)
    assert m.get("t1") == {}


def test_turn_map_ignores_a_missing_turn_id():
    m = capture.TurnMap()
    m.remember(None, session_id="s")
    assert len(m) == 0
    assert m.get(None) == {}


def test_usage_payload_maps_both_key_spellings_profile_stats_reads():
    out = capture.usage_payload(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "prompt_tokens": 100,
            "total_tokens": 120,
            "request_count": 1,
        },
        "qwen",
    )
    assert out["prompt"] == 100 and out["input"] == 100
    assert out["completion"] == 20 and out["output"] == 20
    assert out["total"] == 120 and out["calls"] == 1
    assert out["model"] == "qwen"


def test_usage_payload_keeps_cache_tokens_the_wire_never_reported():
    out = capture.usage_payload(
        {
            "input_tokens": 5,
            "output_tokens": 1,
            "cache_read_tokens": 32567,
            "cache_write_tokens": 7,
        },
        None,
    )
    assert out["cache_read"] == 32567
    assert out["cache_write"] == 7


def test_usage_payload_omits_model_when_unknown_rather_than_guessing():
    out = capture.usage_payload({"input_tokens": 1, "output_tokens": 1}, None)
    assert "model" not in out


def test_usage_is_attributed_through_the_turn_map():
    rec = FakeRecorder()
    d = drain_for(rec)
    d.handle(
        {"hook": "pre_llm_call", "kwargs": {"turn_id": "t1", "session_id": "S1", "model": "qwen"}}
    )
    d.handle(
        {
            "hook": "post_api_request",
            "kwargs": {"turn_id": "t1", "usage": {"input_tokens": 10, "output_tokens": 2}},
        }
    )
    assert len(rec.events) == 1
    etype, payload, _ = rec.events[0]
    assert etype == "session.usage"
    assert payload["_stored_session_id"] == "S1"
    assert payload["usage"]["model"] == "qwen"
    assert d.stats["usage_recorded"] == 1


def test_usage_marks_itself_per_request_not_cumulative():
    """The wire's session.usage is cumulative and gets SUBTRACTED. These are
    per request and must be SUMMED; a database holding both needs the flag."""
    rec = FakeRecorder()
    d = drain_for(rec)
    d.handle({"hook": "pre_llm_call", "kwargs": {"turn_id": "t1", "session_id": "S1"}})
    d.handle(
        {
            "hook": "post_api_request",
            "kwargs": {"turn_id": "t1", "usage": {"input_tokens": 1, "output_tokens": 1}},
        }
    )
    assert rec.events[0][1]["_per_request"] is True


def test_usage_without_a_session_is_counted_not_silently_dropped():
    """HAZARD 1: post_api_request carries no session_id. Both consumers drop an
    event with no _stored_session_id without a word."""
    rec = FakeRecorder()
    d = drain_for(rec)
    d.handle(
        {"hook": "post_api_request", "kwargs": {"turn_id": "unknown", "usage": {"input_tokens": 1}}}
    )
    assert rec.events == []
    assert d.stats["dropped_no_session"] == 1


def test_empty_usage_records_nothing():
    rec = FakeRecorder()
    d = drain_for(rec)
    d.handle({"hook": "pre_llm_call", "kwargs": {"turn_id": "t1", "session_id": "S1"}})
    d.handle({"hook": "post_api_request", "kwargs": {"turn_id": "t1", "usage": {}}})
    assert rec.events == []


def test_session_end_closes_a_run_that_is_still_open():
    rec = FakeRecorder(open_runs={"S1"})
    d = drain_for(rec)
    d.handle(
        {
            "hook": "on_session_end",
            "kwargs": {"session_id": "S1", "turn_id": "t1", "completed": True},
        }
    )
    assert rec.events and rec.events[0][0] == "message.completed"
    assert rec.events[0][1]["_safety_net"] is True
    assert d.stats["runs_closed"] == 1


def test_session_end_does_nothing_when_no_run_is_open():
    """The socket's own close carries status/error this hook lacks, so the
    safety net must never pre-empt a close that already happened."""
    rec = FakeRecorder(open_runs=set())
    d = drain_for(rec)
    d.handle(
        {
            "hook": "on_session_end",
            "kwargs": {"session_id": "S1", "turn_id": "t1", "completed": True},
        }
    )
    assert rec.events == []
    assert d.stats["runs_closed"] == 0


def test_a_second_close_for_the_same_turn_is_refused():
    """HAZARD 2: message.completed is both run-opening and run-closing, so a
    duplicate finds no open run, qualifies as opening, and mints a phantom
    single-event run."""
    rec = FakeRecorder(open_runs={"S1"})
    d = drain_for(rec)
    ev = {
        "hook": "on_session_end",
        "kwargs": {"session_id": "S1", "turn_id": "t1", "completed": True},
    }
    d.handle(ev)
    d.handle(ev)
    assert len(rec.events) == 1
    assert d.stats["duplicate_close"] == 1


def test_interrupted_is_carried_through_the_safety_net():
    rec = FakeRecorder(open_runs={"S1"})
    d = drain_for(rec)
    d.handle(
        {
            "hook": "on_session_end",
            "kwargs": {
                "session_id": "S1",
                "turn_id": "t1",
                "completed": False,
                "interrupted": True,
            },
        }
    )
    assert rec.events[0][1]["status"] == "interrupted"


def test_session_end_without_a_session_id_is_ignored():
    rec = FakeRecorder(open_runs={"S1"})
    d = drain_for(rec)
    d.handle({"hook": "on_session_end", "kwargs": {"turn_id": "t1", "completed": True}})
    assert rec.events == []


def test_an_unknown_hook_is_ignored_not_an_error():
    rec = FakeRecorder()
    d = drain_for(rec)
    d.handle({"hook": "some_future_hook", "kwargs": {"x": 1}})
    assert rec.events == []


def test_drain_survives_a_recorder_that_raises():
    class Exploding(FakeRecorder):
        def handle_event(self, *a, **k):
            raise RuntimeError("boom")

    rec = Exploding(open_runs={"S1"})
    d = drain_for(rec)
    with pytest.raises(RuntimeError):
        d.handle(
            {
                "hook": "on_session_end",
                "kwargs": {"session_id": "S1", "turn_id": "t1", "completed": True},
            }
        )
    assert d.stats["errors"] == 0


def test_missing_recorder_is_tolerated():
    class NoRecorder:
        run_recorder = None

    d = capture.HookDrain(queue.Queue(), NoRecorder(), profile="default")
    d.handle({"hook": "pre_llm_call", "kwargs": {"turn_id": "t1", "session_id": "S1"}})
    d.handle(
        {"hook": "post_api_request", "kwargs": {"turn_id": "t1", "usage": {"input_tokens": 1}}}
    )
    assert d.stats["usage_recorded"] == 1


def test_prompt_source_returns_the_text_pre_llm_call_carried():
    """This is what replaces a whole session.resume per foreign turn."""
    d = drain_for(FakeRecorder())
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t1", "session_id": "S1", "user_message": "typed in the TUI"},
        }
    )
    assert d.prompt_source("default", "S1") == ("typed in the TUI", "t1")


def test_prompt_source_is_consumed_on_read():
    """A prompt answers for exactly ONE run. Left in place, the next foreign
    turn on the same session would be written with the previous turn's text --
    and the store dedups per run, so the wrong text would be committed."""
    d = drain_for(FakeRecorder())
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t1", "session_id": "S1", "user_message": "first"},
        }
    )
    assert d.prompt_source("default", "S1") == ("first", "t1")
    assert d.prompt_source("default", "S1") is None


def test_prompt_source_returns_none_for_an_unknown_session():
    d = drain_for(FakeRecorder())
    assert d.prompt_source("default", "never-seen") is None


def test_prompt_source_ignores_an_empty_prompt():
    d = drain_for(FakeRecorder())
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t1", "session_id": "S1", "user_message": "   "},
        }
    )
    assert d.prompt_source("default", "S1") is None


def test_prompt_source_keeps_only_the_newest_prompt_per_session():
    d = drain_for(FakeRecorder())
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t1", "session_id": "S1", "user_message": "old"},
        }
    )
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t2", "session_id": "S1", "user_message": "new"},
        }
    )
    assert d.prompt_source("default", "S1") == ("new", "t2")


def test_prompt_source_expires_a_stale_prompt():
    """A prompt older than the turn window is not evidence about the run
    opening now; the resume path should run instead."""
    d = drain_for(FakeRecorder())
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t1", "session_id": "S1", "user_message": "ancient"},
        }
    )
    d._latest_prompt["S1"] = ("ancient", "t1", time.time() - capture.TURN_TTL_S - 1)
    assert d.prompt_source("default", "S1") is None


def test_prompt_source_separates_sessions():
    d = drain_for(FakeRecorder())
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t1", "session_id": "S1", "user_message": "one"},
        }
    )
    d.handle(
        {
            "hook": "pre_llm_call",
            "kwargs": {"turn_id": "t2", "session_id": "S2", "user_message": "two"},
        }
    )
    assert d.prompt_source("default", "S2") == ("two", "t2")
    assert d.prompt_source("default", "S1") == ("one", "t1")
