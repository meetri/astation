"""Small typed payloads for `HermesAdapter`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REWIND_FIELDS = frozenset(
    {
        "truncate_before_row_id",
        "truncate_before_user_ordinal",
        "confirm_truncate",
        "confirm_empty_truncate",
    }
)

APPROVAL_CHOICES: tuple[str, ...] = ("once", "session", "always", "deny")

APPROVAL_CHOICE_FOR_APPROVED = {True: "once", False: "deny"}


@dataclass(frozen=True, slots=True)
class RewindRequest:
    """Explicit request to rewind/edit/regenerate session history."""

    truncate_before_row_id: str | None = None
    truncate_before_user_ordinal: int | None = None
    confirm_truncate: bool = False
    confirm_empty_truncate: bool = False

    def to_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"confirm_truncate": self.confirm_truncate}
        if self.truncate_before_row_id is not None:
            params["truncate_before_row_id"] = self.truncate_before_row_id
        if self.truncate_before_user_ordinal is not None:
            params["truncate_before_user_ordinal"] = self.truncate_before_user_ordinal
        if self.confirm_empty_truncate:
            params["confirm_empty_truncate"] = self.confirm_empty_truncate
        return params
