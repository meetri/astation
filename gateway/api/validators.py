"""Request-body validation rules shared by more than one router.

Each rule is a plain function a pydantic `field_validator` / `model_validator`
method delegates to, so the model keeps its own method (pydantic needs one on
the class) but the check and its wording are written once. They replaced the
per-router copies in `api/sessions.py`, `api/background.py`, `api/speak.py`
and `api/rewrite.py` (CLEANUP_PLAN step 3.3); every message those copies
produced is preserved verbatim through the `message` / `submission` parameters.
"""

from __future__ import annotations

from typing import Any

from adapters.hermes import REWIND_FIELDS

#: The default refusal for a whitespace-only `text` field.
BLANK_TEXT_MESSAGE = "text must contain non-whitespace characters"


def reject_blank_text(value: str, *, message: str = BLANK_TEXT_MESSAGE) -> str:
    """Reject whitespace-only text with a 422, before anything upstream is touched.

    Returns the caller's string **verbatim** when it passes: the check is
    `value.strip()`, but silently rewriting a user's text is not a route's
    job, and the transcript should hold what was sent. `min_length=1` alone
    accepts `"   "` and `"\\n"` (B-24); this is what actually refuses them.
    """
    if not value.strip():
        raise ValueError(message)
    return value


def reject_rewind_fields(data: Any, *, submission: str) -> Any:
    """Refuse any `prompt.submit` rewind/truncate key smuggled into a body.

    `truncate_before_row_id`, `truncate_before_user_ordinal`,
    `confirm_truncate` and `confirm_empty_truncate` perform a *destructive
    rewrite* of the user's real session history (`docs/PROTOCOL_VERIFIED.md`,
    "Rewind / edit semantics"); they belong only on an explicit,
    user-initiated rewind and must never be forwardable from an HTTP body.
    Runs as a `mode="before"` model validator so the rewind case gets a
    message that names the hazard instead of pydantic's generic "extra inputs
    are not permitted". `submission` names the body ("turn", "background").
    """
    if isinstance(data, dict):
        smuggled = REWIND_FIELDS & data.keys()
        if smuggled:
            raise ValueError(
                f"rewind-only field(s) {sorted(smuggled)} are not accepted on a "
                f"{submission} submission: they destructively rewrite session history "
                "and are never forwarded from an HTTP body"
            )
    return data
