"""Run-event retention: bound the table's growth without losing anything the product reads."""
import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from domain.db import make_engine, make_sessionmaker
from domain.models import Base, Run, RunEvent, utcnow
from domain.run_event_retention import (
    PRUNED_PAYLOAD,
    PRUNABLE_EVENT_TYPES,
    RunEventRetention,
    prune_run_events,
)

BIG = {'output': 'x' * 2000}


@pytest.fixture
def factory(tmp_path):
    engine = make_engine(str(tmp_path / 'research.db'))
    Base.metadata.create_all(engine)
    yield make_sessionmaker(engine)
    engine.dispose()


def add_run(factory, run_id, *, status, age_days, events):
    when = utcnow() - timedelta(days=age_days)
    with factory() as db:
        db.add(Run(id=run_id, kind='turn', status=status, started_at=when))
        db.flush()
        for seq, (event_type, payload) in enumerate(events, start=1):
            db.add(RunEvent(id=f'{run_id}-{seq}', run_id=run_id, seq=seq, event_type=event_type,
                            payload_json=payload, timestamp=when))
        db.commit()


def payloads(factory, run_id):
    with factory() as db:
        rows = db.execute(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)).scalars()
        return [(r.seq, r.event_type, r.payload_json) for r in rows]


MIXED = [
    ('message.started', {'turn': 1}),
    ('tool.started', {'args': BIG}),
    ('tool.completed', BIG),
    ('subagent.tool', BIG),
    ('session.usage', {'input_tokens': 123456, 'output_tokens': 7890, 'total': 131346}),
    ('message.completed', {'text': 'a real answer ' * 20}),
    ('approval.requested', {'command': 'rm -rf build/ ' * 5}),
]


def test_old_finished_run_loses_only_bulky_tool_payloads(factory):
    add_run(factory, 'old', status='completed', age_days=120, events=MIXED)

    assert prune_run_events(factory, retention_days=90) == 3

    after = payloads(factory, 'old')
    assert [(seq, t) for seq, t, _ in after] == [(i, t) for i, (t, _) in enumerate(MIXED, start=1)]  # rows + order intact
    by_type = {t: p for _, t, p in after}
    for pruned in ('tool.started', 'tool.completed', 'subagent.tool'):
        assert by_type[pruned] == PRUNED_PAYLOAD
    for kept in ('message.started', 'session.usage', 'message.completed', 'approval.requested'):
        assert by_type[kept] == dict(MIXED)[kept]


def test_usage_messages_and_approvals_are_never_prunable():
    for event_type in ('session.usage', 'message.completed', 'message.interim', 'message.started',
                       'approval.requested', 'approval.resolved', 'clarify.requested', 'status.update'):
        assert event_type not in PRUNABLE_EVENT_TYPES


def test_recent_runs_are_left_alone(factory):
    add_run(factory, 'recent', status='completed', age_days=10, events=MIXED)
    assert prune_run_events(factory, retention_days=90) == 0
    assert payloads(factory, 'recent')[2][2] == BIG


@pytest.mark.parametrize('status', ['running', 'queued'])
def test_a_run_that_may_still_be_writing_is_never_touched(factory, status):
    add_run(factory, 'open', status=status, age_days=400, events=MIXED)
    assert prune_run_events(factory, retention_days=90) == 0
    assert payloads(factory, 'open')[2][2] == BIG


@pytest.mark.parametrize('status', ['failed', 'interrupted', 'completed'])
def test_every_finished_status_is_eligible(factory, status):
    add_run(factory, 'done', status=status, age_days=200, events=MIXED)
    assert prune_run_events(factory, retention_days=90) == 3


def test_a_second_pass_does_nothing(factory):
    add_run(factory, 'old', status='completed', age_days=120, events=MIXED)
    assert prune_run_events(factory, retention_days=90) == 3
    assert prune_run_events(factory, retention_days=90) == 0


def test_zero_days_keeps_everything(factory):
    add_run(factory, 'old', status='completed', age_days=900, events=MIXED)
    assert prune_run_events(factory, retention_days=0) == 0
    assert payloads(factory, 'old')[2][2] == BIG


def test_it_works_through_many_rows_in_small_batches(factory):
    add_run(factory, 'busy', status='completed', age_days=200,
            events=[('tool.completed', BIG) for _ in range(23)])
    assert prune_run_events(factory, retention_days=90, batch_rows=5) == 23
    assert all(p == PRUNED_PAYLOAD for _, _, p in payloads(factory, 'busy'))


def test_small_payloads_are_not_worth_pruning(factory):
    add_run(factory, 'tiny', status='completed', age_days=200, events=[('tool.started', {'n': 1})])
    assert prune_run_events(factory, retention_days=90) == 0


@pytest.mark.asyncio
async def test_background_task_prunes_then_stops_cleanly(factory):
    add_run(factory, 'old', status='completed', age_days=120, events=MIXED)
    retention = RunEventRetention(factory, retention_days=90, interval_s=3600, startup_delay_s=0)
    retention.start()
    for _ in range(100):
        if payloads(factory, 'old')[2][2] == PRUNED_PAYLOAD:
            break
        await asyncio.sleep(0.02)
    assert payloads(factory, 'old')[2][2] == PRUNED_PAYLOAD
    await retention.close()
    await retention.close()  # idempotent


@pytest.mark.asyncio
async def test_disabled_retention_starts_no_task(factory):
    retention = RunEventRetention(factory, retention_days=0, startup_delay_s=0)
    retention.start()
    assert retention.enabled is False and retention._task is None
    await retention.close()


@pytest.mark.asyncio
async def test_a_failing_pass_does_not_kill_the_loop(factory, monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('database is locked')

    monkeypatch.setattr('domain.run_event_retention.prune_run_events', boom)
    retention = RunEventRetention(factory, retention_days=90, interval_s=0.01, startup_delay_s=0)
    retention.start()
    for _ in range(100):
        if len(calls) >= 3:
            break
        await asyncio.sleep(0.02)
    assert len(calls) >= 3  # kept trying after the first failure
    await retention.close()
