"""The session-attribution forwarder: what turns "something ran on this
machine" into "this session ran this".

It sits in the agent's own process, fed by a hook that runs INSIDE a turn, so
the tests are weighted towards the ways that placement goes wrong:

* it must never raise, whatever it is handed, because Hermes swallows hook
  exceptions and a crash here would be both silent and inside the user's turn;
* it must never grow without bound, and a drop must be COUNTED, because an
  audit channel that quietly stops is worse than one that is visibly broken;
* it must not leak a secret into a store that is archived beyond deletion.

The join key gets its own tests. `command` is what ties a tool call to a kernel
exec record, and getting it wrong does not fail loudly -- it just produces a
timeline that is quietly empty.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "trg_audit_forwarder_under_test", Path(__file__).resolve().parents[1] / "audit_forwarder.py"
)
af = importlib.util.module_from_spec(_SPEC)
sys.modules["trg_audit_forwarder_under_test"] = af
_SPEC.loader.exec_module(af)


def forwarder(url: str = ""):
    """Never started: the sending thread is not what these tests are about."""
    return af.AuditForwarder(ingest_url=url, host_label="hermes")


# --- the join key ---------------------------------------------------------


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"command": "git status"}, "git status"),
        ({"cmd": "ls -la"}, "ls -la"),
        ({"script": "echo hi"}, "echo hi"),
        ({"shell_command": "pwd"}, "pwd"),
        ({"command": "  spaced  "}, "spaced"),
    ],
)
def test_the_command_is_found_under_any_of_the_names_tools_use(args, expected):
    assert af.command_of("terminal", args) == expected


@pytest.mark.parametrize(
    "args",
    [None, "a string", {}, {"path": "/tmp/x"}, {"command": ""}, {"command": "   "}, 42],
)
def test_a_tool_that_runs_no_command_yields_no_join_key(args):
    """Those rows are still recorded -- they simply join to nothing, which is
    honest. Inventing a key would attach a session to a process it never ran."""
    assert af.command_of("read_file", args) == ""


def test_a_very_long_command_is_bounded():
    command = af.command_of("terminal", {"command": "x" * 99_999})
    assert len(command) <= af.MAX_ARG_CHARS


# --- arguments ------------------------------------------------------------


def test_arguments_are_summarised_deterministically():
    """Sorted keys, so the same call produces the same row and a diff between
    two rows means the call really differed."""
    first, _ = af.summarize_args({"b": 2, "a": 1})
    second, _ = af.summarize_args({"a": 1, "b": 2})
    assert first == second


def test_a_huge_argument_is_truncated_and_says_so():
    summary, _ = af.summarize_args({"body": "y" * 50_000})
    assert len(summary) < 50_000
    assert "chars]" in summary, "a truncated value must say it was truncated"


def test_unserialisable_arguments_do_not_raise():
    class Awkward:
        def __repr__(self):
            return "<awkward>"

    summary, _ = af.summarize_args({"thing": Awkward()})
    assert summary


# --- secrets --------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAIOSFODNN7EXAMPLE",
        "sk-abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz12",
        "password=hunter2",
    ],
)
def test_a_secret_never_leaves_this_process_intact(secret):
    """The audit archive cannot be edited or deleted once written, so a secret
    that reaches it is there for the life of the retention window."""
    summary, changed = af.summarize_args({"command": f"curl -H '{secret}'"})
    assert secret not in summary
    assert changed


def test_a_recorded_row_is_flagged_when_something_was_masked():
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_call(
        {
            "session_id": "20260920_000000_aaaaaa",
            "tool_name": "terminal",
            "args": {"command": "export TOKEN=ghp_abcdefghijklmnopqrstuvwxyz12"},
        }
    )
    row = f._drain_batch()[0]
    assert row["redacted"] == 1
    assert "ghp_" not in row["command"]


# --- what a row carries ---------------------------------------------------


def test_a_tool_call_carries_the_ids_that_make_the_join_possible():
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_call(
        {
            "session_id": "20260920_040242_d0346c",
            "turn_id": "turn_7",
            "tool_call_id": "call_9",
            "tool_name": "terminal",
            "args": {"command": "git status"},
            "profile": "gpt-sol",
        }
    )
    row = f._drain_batch()[0]
    assert row["audit_class"] == "agent_tool"
    assert row["stored_session_id"] == "20260920_040242_d0346c"
    assert row["turn_id"] == "turn_7"
    assert row["tool_call_id"] == "call_9"
    assert row["command"] == "git status"
    assert row["profile"] == "gpt-sol"
    assert row["host"] == "hermes"
    assert row["ts"]


def test_a_call_with_no_session_is_counted_not_silently_dropped():
    """It would mean Hermes changed what it passes to the hook. A quietly
    empty attribution table is the exact failure this feature exists to
    prevent, so the gap is visible on /health."""
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_call({"tool_name": "terminal", "args": {"command": "ls"}})
    assert f.stats["no_session"] == 1
    assert f._drain_batch() == []


# --- it must never break a turn -------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"session_id": None},
        {"session_id": 12345, "args": object()},
        {"session_id": "s", "args": {"command": None}},
    ],
)
def test_no_input_can_make_recording_raise(payload):
    """This runs inside the agent's turn and Hermes swallows hook exceptions,
    so a raise here would be both costly and invisible."""
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_call(payload)  # must not raise


def test_the_queue_is_bounded_and_overflow_is_counted():
    f = forwarder("http://ingest.invalid/ingest")
    for i in range(af.MAX_QUEUED + 25):
        f.record_tool_call({"session_id": "s", "tool_name": "t", "args": {"command": f"cmd{i}"}})
    assert len(f._queue) == af.MAX_QUEUED
    assert f.stats["dropped"] >= 25, "an overflowing audit channel must say so"


def test_an_unconfigured_forwarder_does_not_pretend_to_work():
    f = forwarder("")
    assert not f.configured
    f.start()  # no thread, no error
    f.close()


def test_a_session_event_is_recorded_with_its_session():
    f = forwarder("http://ingest.invalid/ingest")
    f.record_session_event("start", {"session_id": "s1", "cwd": "/opt/data", "profile": "default"})
    row = f._drain_batch()[0]
    assert row["audit_class"] == "agent_session"
    assert row["event"] == "start"
    assert row["stored_session_id"] == "s1"


# --- which agent did this -------------------------------------------------


def test_the_processes_own_profile_labels_rows_hermes_leaves_unlabelled():
    """Hermes does not pass `profile` to `pre_tool_call`; on the deploy host 44
    of 46 stored rows carried an empty one. Each `gateway run` process serves
    exactly one profile and `register()` is handed its name, so the fallback is
    what makes "which agent did this" answerable in the store itself."""
    f = af.AuditForwarder(ingest_url="http://ingest.invalid/ingest", profile="gpt-astra")
    f.record_tool_call({"session_id": "s1", "tool_name": "terminal", "args": {"command": "ls"}})
    assert f._drain_batch()[0]["profile"] == "gpt-astra"


def test_a_profile_hermes_does_pass_still_wins_over_the_fallback():
    """The fallback fills a gap; it must never overwrite the truth."""
    f = af.AuditForwarder(ingest_url="http://ingest.invalid/ingest", profile="gpt-astra")
    f.record_tool_call({"session_id": "s1", "profile": "claude", "args": {"command": "ls"}})
    assert f._drain_batch()[0]["profile"] == "claude"


def test_session_events_carry_the_same_fallback():
    f = af.AuditForwarder(ingest_url="http://ingest.invalid/ingest", profile="grok4-6")
    f.record_session_event("start", {"session_id": "s1"})
    assert f._drain_batch()[0]["profile"] == "grok4-6"


def test_no_profile_anywhere_is_an_empty_string_not_a_crash():
    f = af.AuditForwarder(ingest_url="http://ingest.invalid/ingest")
    f.record_tool_call({"session_id": "s1", "args": {"command": "ls"}})
    assert f._drain_batch()[0]["profile"] == ""


# --- how long it took, and nothing that could carry a payload -------------


def test_a_result_row_carries_the_duration_and_the_outcome():
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_result(
        {
            "session_id": "20260920_040242_d0346c",
            "turn_id": "turn_7",
            "tool_call_id": "call_9",
            "tool_name": "terminal",
            "duration_ms": 1234,
            "status": "ok",
        }
    )
    row = f._drain_batch()[0]
    assert row["phase"] == "result"
    assert row["duration_ms"] == 1234
    assert row["status"] == "ok"
    assert row["tool_call_id"] == "call_9", "the join key back to the call row"


def test_the_tools_output_never_leaves_this_process():
    """The operator's decision, 2026-09-20: duration yes, results no. Hermes hands
    the whole result to this hook and a file read returns the file. The archive
    is under compliance-mode object lock, so anything written there cannot be
    deleted for the retention window."""
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_result(
        {
            "session_id": "s1",
            "tool_call_id": "c1",
            "tool_name": "read_file",
            "duration_ms": 5,
            "status": "ok",
            "result": "BEGIN RSA PRIVATE KEY hunter2 the entire file contents",
            "error_message": "failed reading /etc/shadow: root:$6$secrethash",
        }
    )
    row = f._drain_batch()[0]
    serialised = json.dumps(row)
    assert "RSA PRIVATE KEY" not in serialised
    assert "secrethash" not in serialised
    assert "result" not in row
    assert "error_message" not in row, "it can quote the content that failed"


def test_a_call_row_is_marked_as_the_call_half():
    """The call row is written before the tool runs and must exist whatever
    happens next. A tool that hangs never reaches `post_tool_call`."""
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_call({"session_id": "s1", "tool_name": "terminal", "args": {"command": "ls"}})
    assert f._drain_batch()[0]["phase"] == "call"


@pytest.mark.parametrize("duration", [None, "", "abc", -5, object()])
def test_a_nonsense_duration_becomes_zero_rather_than_raising(duration):
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_result(
        {"session_id": "s1", "tool_call_id": "c", "duration_ms": duration}
    )
    assert f._drain_batch()[0]["duration_ms"] == 0


def test_a_result_with_no_session_is_counted_like_a_call_with_none():
    f = forwarder("http://ingest.invalid/ingest")
    f.record_tool_result({"tool_call_id": "c", "duration_ms": 1})
    assert f.stats["no_session"] == 1
    assert f._drain_batch() == []


def test_a_result_row_carries_the_processes_profile_too():
    f = forwarder("http://ingest.invalid/ingest")
    f._profile = "gpt-astra"
    f.record_tool_result({"session_id": "s1", "tool_call_id": "c", "duration_ms": 1})
    assert f._drain_batch()[0]["profile"] == "gpt-astra"
