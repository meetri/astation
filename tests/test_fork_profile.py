"""Forks must inherit a filing only from their own Hermes profile."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as OrmSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
from api import instance
from domain.models import Base, Project, Session


@pytest.mark.parametrize("collision", [False, True])
def test_fork_files_with_its_own_profile(monkeypatch, collision):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with OrmSession(engine) as db:
        default_project = Project(id="proj_default", title="Other agent")
        selected_project = Project(id="proj_selected", title="Selected agent")
        db.add_all([default_project, selected_project])
        db.flush()
        db.add(Session(id="sess_selected", project_id=selected_project.id,
                       runtime="hermes", profile="gpt-sol", runtime_session_id="same-id"))
        if collision:
            db.add(Session(id="sess_default", project_id=default_project.id,
                           runtime="hermes", profile="default", runtime_session_id="same-id"))
        db.commit()

        async def branched(_state, _adapter, _call):
            return "live-handle", {"stored_session_id": "fork-id", "parent": "same-id"}

        monkeypatch.setattr(instance, "resolve_profile_adapter", lambda *_: object())
        monkeypatch.setattr(instance, "resolve_live_handle_cache", lambda *_: object())
        monkeypatch.setattr(instance, "_with_reconnect", branched)
        request = SimpleNamespace(app=SimpleNamespace(state=object()))
        result = asyncio.run(instance.fork_session(
            "same-id", request, body=None, profile="gpt-sol", db=db
        ))

        assert result["filed"] is True
        assert result["project_id"] == selected_project.id
        row = db.scalar(select(Session).where(Session.runtime_session_id == "fork-id"))
        assert row is not None
        assert row.profile == "gpt-sol"
        assert row.project_id == selected_project.id
