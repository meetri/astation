"""Read the hosting Hermes registry; never resume/probe another profile.

Contract verified against Hermes f97608f: server._sessions maps live handle
-> session dict, session_key/agent.session_id is the stored ID, profile_home
is the resolved session owner (None means the routing process home).
"""
from __future__ import annotations


def resolve_host_identity(host, stored_id, live_id, *, observed_at=None):
    from hermes_constants import profile_name_for_home, get_routing_process_hermes_home
    if not live_id:
        # A bare stored ID can exist in more than one profile.
        return None
    with host._sessions_lock:
        session = host._sessions.get(live_id)
        if session is None:
            return None
        if observed_at is not None:
            created_at = session.get('created_at')
            # A later registry incarnation cannot claim an older held frame.
            if not isinstance(created_at, (int, float)) or created_at > observed_at:
                return None
        stored = host._session_lookup_key(session)
        home = session.get('profile_home') or get_routing_process_hermes_home()
        profile = profile_name_for_home(home)
        if not stored or not profile or (stored_id and stored != stored_id):
            return None
        return stored, live_id, profile
