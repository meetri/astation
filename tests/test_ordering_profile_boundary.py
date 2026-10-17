"""Isolated diagnostics against frozen installed code; never opens a user's DB."""
import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from domain.models import Base, Project, Session, Run
from domain.chat_store import ChatStore
from domain.run_recorder import RunRecorder

@pytest.fixture
def factory():
    engine = create_engine('sqlite://', poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()

def test_run_identity_must_include_profile(factory):
    with factory() as db:
        db.add(Project(id='p', title='isolated diagnostic'))
        db.add_all([Session(id='s-'+p, project_id='p', runtime='hermes', profile=p,
                            runtime_session_id='same-id') for p in ('default', 'gpt-astra')])
        db.commit()
    recorder = RunRecorder(factory)
    envelopes = []
    for profile in ('default', 'gpt-astra'):
        payload = {'_stored_session_id': 'same-id', '_profile': profile, 'text': 'diagnostic'}
        canonical = SimpleNamespace(type='message.started', timestamp=datetime.now(timezone.utc))
        envelope = {'payload': payload}
        recorder.handle_event(canonical, envelope, profile)
        envelopes.append(envelope)
    assert envelopes[0]['session_id'] == 's-default'
    assert envelopes[1]['session_id'] == 's-gpt-astra'
    assert envelopes[0]['run_id'] != envelopes[1]['run_id']
    assert recorder.open_run_count == 2
    for profile in ('default', 'gpt-astra'):
        assert recorder.has_open_run('same-id', profile)
        envelope = {'payload': {'_stored_session_id': 'same-id', 'request_id': 'q-'+profile}}
        recorder.handle_event(SimpleNamespace(type='approval.requested', timestamp=datetime.now(timezone.utc)), envelope, profile)
        assert envelope['run_id'] == envelopes[('default', 'gpt-astra').index(profile)]['run_id']
        assert recorder.stored_id_for_request('q-'+profile, profile) == 'same-id'
        assert recorder.record_synthetic('same-id', 'approval.resolved', {'request_id': 'q-'+profile}, profile)
    complete = {'payload': {'_stored_session_id': 'same-id', 'text': 'done'}}
    recorder.handle_event(SimpleNamespace(type='message.completed', timestamp=datetime.now(timezone.utc)), complete, 'default')
    assert not recorder.has_open_run('same-id', 'default')
    assert recorder.has_open_run('same-id', 'gpt-astra')
    assert complete['run_id'] == envelopes[0]['run_id']
    with factory() as db:
        assert db.get(Run, envelopes[0]['run_id']).status == 'completed'
        assert db.get(Run, envelopes[1]['run_id']).status == 'running'

def test_exact_profile_history_does_not_merge_same_id(factory):
    store = ChatStore(factory)
    for profile in ('default', 'gpt-astra'):
        payload = {'_stored_session_id': 'same-id', 'text': profile+' reply'}
        store.capture_event(profile, SimpleNamespace(type='message.completed'), {'payload':payload, 'run_id':profile})
    assert [r['text'] for r in store.page(profile='gpt-astra', stored_session_id='same-id')] == ['gpt-astra reply']
    assert [r['text'] for r in store.page(profile='default', stored_session_id='same-id')] == ['default reply']
