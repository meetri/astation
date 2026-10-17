"""Opt-in integration against the installed Hermes host source, no network/data."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_real_host_registry_and_plugin_adapter(tmp_path, monkeypatch):
    pytest.importorskip('hermes_constants')
    from tui_gateway import server
    from hermes_constants import get_default_hermes_root
    path = Path(__file__).resolve().parents[1] / 'dashboard/plugin_api.py'
    spec = importlib.util.spec_from_file_location('contract_plugin', path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    adapter = object.__new__(plugin._build_adapter_class())
    adapter._settings = SimpleNamespace(research_gateway_db_path=str(tmp_path/'research.db'))
    monkeypatch.setattr(server, '_sessions', {
        'diagnostic-a': {'session_key': 'duplicate', 'profile_home': str(get_default_hermes_root()/'profiles/gpt-astra')},
        'diagnostic-b': {'session_key': 'duplicate', 'profile_home': str(get_default_hermes_root())},
    })
    assert adapter.event_identity_epoch
    assert adapter.event_recovery_path == tmp_path/'research.event-inbox.sqlite'
    assert adapter.resolve_event_identity(None, 'diagnostic-a') == ('duplicate', 'diagnostic-a', 'gpt-astra')
    assert adapter.resolve_event_identity(None, 'diagnostic-b') == ('duplicate', 'diagnostic-b', 'default')
    assert adapter.resolve_event_identity('duplicate', None) is None
    assert adapter.resolve_event_identity('wrong', 'diagnostic-a') is None


def test_registry_reused_handle_cannot_claim_older_event():
    pytest.importorskip('hermes_constants')
    from domain.host_identity import resolve_host_identity
    import threading
    host = SimpleNamespace(_sessions_lock=threading.Lock(), _sessions={
        'reused': {'session_key': 'same-id', 'created_at': 200, 'profile_home': '/isolated/profiles/gpt-astra'},
    }, _session_lookup_key=lambda session: session['session_key'])
    assert resolve_host_identity(host, 'same-id', 'reused', observed_at=100) is None
    assert resolve_host_identity(host, 'same-id', 'reused', observed_at=300) == ('same-id', 'reused', 'gpt-astra')
