"""Shared-connection ownership regressions; only temporary SQLite databases."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from api.chat import session_chat
from domain.foreign_prompt_capture import ForeignPromptCapture
from domain.models import Base, Project, Session, Run, RunEvent
from domain.chat_store import ChatStore
from domain.live_handles import LiveHandleCache
from domain.event_stream import EventBroadcaster
from domain.run_recorder import RunRecorder
from api.bootstrap import wire_event_stream
from api.sessions import submit_turn, TurnSubmission


class Adapter:
    connection_generation = 1
    is_connected = True

    async def resume_session(self, stored_id, **kwargs):
        self.resumed_profile = kwargs.get('profile', 'default')
        return {'session_id': 'live-' + self.resumed_profile}

    async def prompt_submit(self, live_id, text):
        return {'status': 'streaming'}


def raw(kind='message.complete', live='live-gpt-astra', stored='same-id', **extra):
    return {'method': kind, 'params': {'session_id': live, 'stored_session_id': stored, **extra}}


@pytest.fixture
def state(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'workspace.db'),
                           connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        db.add(Project(id='p', title='temporary'))
        db.add_all([Session(id='s-'+p, project_id='p', runtime='hermes', profile=p,
                            runtime_session_id='same-id') for p in ('default', 'gpt-astra')])
        db.commit()
    adapter = Adapter()
    state = SimpleNamespace(db_sessions=factory, chat_store=ChatStore(factory), hermes_adapter=adapter,
                            live_handle_cache=LiveHandleCache(adapter), shared_profile_handle_caches={})
    yield state
    engine.dispose()


def test_no_completion_id_prompt_answer_replay_through_history_route(state, monkeypatch, tmp_path):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    capture = ForeignPromptCapture(state, state.chat_store)
    recorder = RunRecorder(state.db_sessions, on_run_opened=capture.handle_run_opened)
    broadcaster = wire_event_stream(state, state.hermes_adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), recorder)
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    fixture = json.loads((Path(__file__).parent / 'fixtures/split-profile-sanitized.json').read_text())
    # Replay prompt/answer projection; tool/subagent ordering is a separate client gate.
    expected = [r for r in fixture if r['role'] in ('user', 'assistant')]
    ids = []
    async def replay():
        for row in expected:
            if row['role'] == 'user':
                ack = await submit_turn('same-id', TurnSubmission(text=row['text']), request, profile='gpt-astra')
                ids.append(ack['chat_message_id'])
            else:
                event = raw(text=row['text'])  # deliberately no completion/message ID
                broadcaster._forward_one(event)
                frame = broadcaster._history[-1][1]
                assert frame['payload']['_profile'] == 'gpt-astra'
                ids.append(frame['payload']['chat_row_id'])
        return await session_chat('same-id', request, profile='gpt-astra', before=None, limit=200)
    result = asyncio.run(replay())
    rows = result['messages']
    assert [(r['role'], r['text']) for r in rows] == [(r['role'], r['text']) for r in expected]
    assert [r['id'] for r in rows] == ids
    assert len(set(ids)) == len(expected) == 23
    assert all(r['turn_id'] for r in rows)
    assert state.chat_store.page(profile='default', stored_session_id='same-id') == []
    with state.db_sessions() as db:
        assert all(r.profile == 'gpt-astra' and r.session_id == 's-gpt-astra' for r in db.scalars(select(Run)))
    (tmp_path / 'canonical-history.json').write_text(json.dumps(result, default=str, indent=2))


def test_shared_capture_uses_canonical_profile_before_broadcast(state, monkeypatch):
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    recorder = RunRecorder(state.db_sessions)
    broadcaster = wire_event_stream(state, state.hermes_adapter, state.live_handle_cache,
                                   SimpleNamespace(handle_completed=lambda e: None), recorder)
    async def submit():
        return await submit_turn('same-id', TurnSubmission(text='User1'),
                                 SimpleNamespace(app=SimpleNamespace(state=state)), profile='gpt-astra')
    ack = asyncio.run(submit())
    assert state.hermes_adapter.resumed_profile == 'gpt-astra'
    broadcaster._forward_one(raw('message.complete', text='Assistant1'))
    page = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
    assert [r['text'] for r in page] == ['User1', 'Assistant1']
    assert page[0]['id'] == ack['chat_message_id']
    frame = broadcaster._history[-1][1]
    assert frame['payload']['_profile'] == 'gpt-astra'
    assert frame['session_id'] == 's-gpt-astra'
    assert frame['payload']['chat_row_id'] == page[1]['id']
    assert state.chat_store.page(profile='default', stored_session_id='same-id') == []
    with state.db_sessions() as db:
        run = db.scalar(select(Run))
        assert (run.profile, run.session_id) == ('gpt-astra', 's-gpt-astra')


@pytest.mark.parametrize('case', ['ambiguous-live', 'ambiguous-stored', 'unknown', 'conflicting-pair', 'stale-generation', 'observed-only'])
def test_shared_binding_rejects_unproved_or_ambiguous_owner(state, case):
    other = LiveHandleCache(state.hermes_adapter)
    state.shared_profile_handle_caches['gpt-astra'] = other
    event = raw(text='must not escape')
    if case == 'ambiguous-live':
        state.live_handle_cache.put('same-id', 'live-gpt-astra')
        other.put('same-id', 'live-gpt-astra')
    elif case == 'ambiguous-stored':
        state.live_handle_cache.put('same-id', 'live-default')
        other.put('same-id', 'live-gpt-astra')
        event = raw('session.title', live='same-id', stored=None, title='private')
    elif case == 'conflicting-pair':
        other.put('different-id', 'live-gpt-astra')
    elif case == 'stale-generation':
        other.put('same-id', 'live-gpt-astra')
        state.hermes_adapter.connection_generation += 1
    elif case == 'observed-only':
        state.live_handle_cache.observe_live_mapping('same-id', 'live-gpt-astra')
    seen = []
    broadcaster = EventBroadcaster(state.hermes_adapter, state.live_handle_cache,
        profile_caches=lambda: state.shared_profile_handle_caches,
        on_canonical_event=lambda *args: seen.append(args))
    broadcaster._forward_one(event)
    assert seen == []
    # A gateway-only control frame may request recovery, but private data never escapes.
    assert not any(f['payload'].get('text') or f['payload'].get('title') for _, f, _ in broadcaster._history)


def test_lazy_watch_in_other_profile_does_not_suppress_capture(state, monkeypatch):
    from domain.watch_sessions import watched
    monkeypatch.setattr(EventBroadcaster, 'start', lambda self: None)
    other = LiveHandleCache(state.hermes_adapter)
    other.put('same-id', 'live-gpt-astra')
    state.shared_profile_handle_caches['gpt-astra'] = other
    broadcaster = wire_event_stream(state, state.hermes_adapter, state.live_handle_cache,
        SimpleNamespace(handle_completed=lambda e: None), RunRecorder(state.db_sessions))
    watched.watch('same-id')  # default-profile lazy watch
    try:
        broadcaster._forward_one(raw(text='visible nondefault answer'))
        rows = state.chat_store.page(profile='gpt-astra', stored_session_id='same-id')
        assert [r['text'] for r in rows] == ['visible nondefault answer']
        assert rows[0]['turn_id'] is not None
    finally:
        watched.clear()
