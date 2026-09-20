"""Time formatting shared by every router that serves a timestamp."""

from __future__ import annotations

from datetime import UTC, datetime


def iso_z(value: datetime | None) -> str | None:
    """UTC ISO-8601 with a `Z`, or None."""
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")
