"""Time formatting shared by every router that serves a timestamp.

One `iso_z` replaces the eight identical `_iso` copies that grew one router at
a time (`api/projects.py`, `runs.py`, `artifacts.py`, `attachments.py`,
`background.py`, `converse.py`, `snapshots.py`, `snapshot_sweep.py` --
CLEANUP_PLAN step 3.3). "Now" is `domain.models.utcnow`, which the ORM
defaults already use; there is no second copy of it here on purpose.
"""

from __future__ import annotations

from datetime import UTC, datetime


def iso_z(value: datetime | None) -> str | None:
    """UTC ISO-8601 with a `Z`, or None.

    SQLite has no timezone type, so a `DateTime(timezone=True)` column reads
    back naive even though `utcnow()` wrote UTC. Stamping the `Z` here rather
    than shipping a naive string keeps the client from having to guess -- a
    naive timestamp rendered as local time is off by hours, silently.
    """
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")
