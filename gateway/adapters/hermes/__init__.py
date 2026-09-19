"""Hermes TUI Gateway adapter.

See `docs/PROTOCOL_VERIFIED.md` for the verified wire protocol this
implements, and `client.HermesAdapter`'s docstring for the public interface.
"""

from __future__ import annotations

from .client import HermesAdapter
from .exceptions import (
    HermesAuthError,
    HermesConnectionError,
    HermesError,
    HermesProtocolError,
    HermesRPCError,
)
from .models import APPROVAL_CHOICE_FOR_APPROVED, APPROVAL_CHOICES, REWIND_FIELDS, RewindRequest

__all__ = [
    "APPROVAL_CHOICES",
    "APPROVAL_CHOICE_FOR_APPROVED",
    "REWIND_FIELDS",
    "HermesAdapter",
    "HermesAuthError",
    "HermesConnectionError",
    "HermesError",
    "HermesProtocolError",
    "HermesRPCError",
    "RewindRequest",
]
