"""Small typed payloads for `HermesAdapter`.

`RewindRequest` exists to make the destructive rewind/edit fields impossible
to pass by accident. See `docs/PROTOCOL_VERIFIED.md` §"Rewind / edit
semantics": a `prompt.submit` carrying `truncate_before_row_id` /
`truncate_before_user_ordinal` (+ `confirm_truncate`, optionally
`confirm_empty_truncate`) performs a destructive rewrite of session history
on the real server. `HermesAdapter.prompt_submit()` only ever emits these
fields when the caller passes an explicit `RewindRequest` via the separate
`rewind=` keyword — never via a generic `**params` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Field names exactly as documented in PROTOCOL_VERIFIED.md. Kept in one place
# so `HermesAdapter.prompt_submit()` can check an incoming `**extra_params`
# dict against this set and refuse to send it, instead of relying on nobody
# ever copy-pasting one of these names into a shared kwargs dict.
REWIND_FIELDS = frozenset(
    {
        "truncate_before_row_id",
        "truncate_before_user_ordinal",
        "confirm_truncate",
        "confirm_empty_truncate",
    }
)

# The four answers `approval.respond` accepts, verified live 2026-08-30 and
# recorded in `docs/PROTOCOL_VERIFIED.md` ("`approval.request` — the frame,
# verbatim"). **An approval is not a boolean.** Hermes computes the subset it
# is willing to offer per request and puts it on the event's `choices` key;
# this is the full vocabulary those subsets are drawn from.
#
#   once     allow this one invocation
#   session  allow for the rest of this live session
#   always   persist to the permanent allowlist
#   deny     refuse (also Hermes's own default for a missing `choice`)
#
# Modelling the prompt as approve/deny throws away `session` and `always` —
# the two answers that stop the operator being asked the same question forever.
APPROVAL_CHOICES: tuple[str, ...] = ("once", "session", "always", "deny")

# What a boolean maps onto when a client can only express yes/no. `True` is
# deliberately the *narrowest* yes: a client that cannot say which yes it means
# must not be silently granted the permanent one.
APPROVAL_CHOICE_FOR_APPROVED = {True: "once", False: "deny"}


@dataclass(frozen=True, slots=True)
class RewindRequest:
    """Explicit request to rewind/edit/regenerate session history.

    Only constructed by code implementing an actual rewind UI action (see
    `docs/ARCHITECTURE.md` §21 on not letting a lower layer improvise a
    destructive action on the user's behalf). `confirm_truncate` defaults to
    `False` on purpose — the caller must consciously opt in even after
    choosing to build a `RewindRequest` at all.
    """

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
