"""Instance-shaped Hermes surfaces: the profile lens, vitals, session rename.

Three thin routes added by the 2026-09-01 review. None of them owns a gateway
table (so neither `api/runs.py` nor `api/artifacts.py` is their home) and none
is a slash command or a human-in-the-loop prompt (so neither `api/commands.py`
nor `api/prompts.py` is either). They are grouped here because they share one
property: each is a passthrough of a Hermes method whose wire shape was
measured live on 2026-09-01 (`docs/PROTOCOL_VERIFIED.md`, "Profiles" and
"`session.title` and `cli.exec -p`").

* **`GET /api/profiles`** -- `profiles.list`, verbatim. The owner's
  "agents"/"channels". A **lens, not a picker**: `session.create` accepts
  `profile` and `model` and silently ignores both (measured with a
  deliberately bogus model id -- the session was still created and still
  reported `qwen3.8-27b`), so nothing above this may offer "create a session
  as gpu-b" or "switch this session's agent". What IS buildable is browse +
  filter + label, which is why `GET /api/sessions?profile=` exists
  (`api/main.py`) and why this response states the constraint explicitly.
* **`GET /api/vitals`** -- the instance's `approvals.mode`. The live instance
  runs `smart`, whose LLM auto-approved four `DANGEROUS_PATTERNS` commands
  (B-44), and until now the app never said so. Read-only on purpose: this
  gateway wraps `config.get` and deliberately does NOT wrap `config.set` --
  changing the owner's approval policy from a phone is not a feature anyone
  asked for, and the failure mode of getting it wrong is unbounded.
* **`POST /api/sessions/{stored_id}/title`** -- rename, the one verified
  write in the review's ask #2. Session-scoped but a pure Hermes passthrough,
  the same way `api/background.py` and `api/attachments.py` host their own
  `/sessions/{id}/...` routes.
* **`POST /api/sessions/{stored_id}/fork`** -- conversation forking, the
  owner's 2026-09-02 ask ("if we do fork I want the fork to be under the same
  project"). `session.branch` upstream, then the new stored id is filed into
  **whatever project the parent is filed in**, through `api/projects.py`'s own
  `file_stored_session()`. An unfiled parent yields an unfiled fork; no
  project is invented. See `fork_session()` for the contract and
  `docs/PROTOCOL_VERIFIED.md` ("Session branching") for the measured shape it
  is built on.
* **`DELETE /api/sessions/{stored_id}`** -- the one genuinely destructive
  route in this service. It deletes the session from **Hermes's own store**
  via `cli.exec {"argv": ["sessions", "delete", <id>, "--yes"]}` -- but since
  P6-3 it **snapshots first** (`api/snapshots.py`, reason `pre_delete`), so
  "irreversible" now means "removed from Hermes, kept in the gateway". See
  `delete_session()` for the full contract and the reasons it is shaped the
  way it is.

**Archive lives in `api/snapshots.py`, not here, and it is not a hide flag.**
Hermes's `hermes sessions archive` is still **filter-only** (no single-session
positional, re-verified 2026-09-03), so it is not wrapped and never will be.
The owner declined "archive" on 2026-09-01 while it meant a workspace-local
soft-hide over nothing; on 2026-09-03 they reopened it as *durability*: a
gateway-owned complete copy of the transcript, reasoning included, filed into
a project (`docs/SESSION_ARCHIVE_DESIGN.md`). `sessions.archived_at` exists
because it rides on that copy. What this module contributes is
`HERMES_SESSIONS_PIN_ARGV` -- the deployed CLI's per-session `sessions pin`,
which archive rides along with so auto-prune cannot take an archived session
-- and the argv validator both routes share.

**There is deliberately NO bulk delete and no prune.** `hermes sessions
delete` takes one session id and this gateway exposes exactly that: one
session, explicitly named, per request. A filter-shaped destructive route
(`archive`'s own shape) is how a mistargeted sweep destroys research nobody
meant to touch, and `tests/test_instance.py` asserts no such route exists.

**The two-id-space trap lives here.** `session.title` takes the **LIVE**
handle: measured on a throwaway spike, `{"session_id": <STORED>}` answers
`[4001] session not found` and `{"session_id": <LIVE>}` answers
`{"pending": false, "title": ...}`. (The `session.title` *event* carries the
stored id -- that is what the older note in PV described, and it is a
different direction of travel.) So the route takes the stored id from the URL
like every other session route and resolves the handle through the shared
`_with_live_handle()` / `_with_reconnect()` machinery. No live handle is ever
accepted from a client or persisted.

All three are mounted on the authenticated `/api` router in `api.main`, so
Basic auth is inherited by construction.

The vitals cache (`InstanceConfigCache`, the two `hermes config get` values
served without waiting) is `domain/instance_config.py` (CLEANUP_PLAN step
3.5), re-exported here.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession

from adapters.hermes import HermesAdapter, HermesError, HermesRPCError

# `_find_filing` and `file_stored_session` are imported rather than
# re-implemented on purpose: the first encodes the schema's
# `UNIQUE (runtime, runtime_session_id)` rule ("the one workspace row for a
# stored runtime session"), the second encodes the whole
# idempotent-file / 409-if-elsewhere / unique-constraint-race path. A second
# copy of either here is exactly the drift `api/projects.py`'s docstring warns
# about.
from api.projects import (
    HERMES_RUNTIME,
    _find_filing,
    file_stored_session,
    workspace_db,
)
from domain.hermes_runtime import (
    _http_error_from_hermes,
    _is_session_not_found,
    _rpc_error_code,
    _validate_stored_session_id,
    _with_live_handle,
    _with_reconnect,
    resolve_live_handle_cache,
    resolve_profile_adapter,
)
from domain.instance_config import (
    HERMES_CONFIG_GET_ARGV,
    SESSIONS_AUTO_PRUNE_KEY,
    SESSIONS_RETENTION_DAYS_KEY,
    InstanceConfigCache,
    _instance_config_cache,
)
from domain.models import Project, Run
from domain.snapshot_builder import SnapshotStorageError, take_snapshot

logger = logging.getLogger(__name__)

instance_router = APIRouter(tags=["instance"])

#: The `config.get` key this gateway reads for the vitals label. On Hermes's
#: narrow allowlist (measured: `profile`, `approvals.mode` and
#: `diagnostics.share_nous` work; `model`, `models`, `profiles` and friends
#: answer `[4002] unknown config key`).
APPROVALS_MODE_KEY = "approvals.mode"

#: Approval modes whose behaviour this project has actually OBSERVED, mapped
#: to "does it approve dangerous commands without asking the human".
#:
#: `smart` is measured, not inferred: on the live instance its LLM
#: auto-approved four commands matching Hermes's own `DANGEROUS_PATTERNS`
#: (B-44). Every other mode name is left out deliberately -- an unobserved
#: mode answers `null` ("this gateway does not know"), never a guessed
#: `false`, because a false negative here is exactly the reassurance the
#: owner must not be given.
OBSERVED_APPROVAL_MODES: dict[str, bool] = {"smart": True}

#: Rename cap. Hermes's own titles are short one-liners (the live instance's
#: longest is well under this); the cap exists so a runaway client cannot
#: push an unbounded string through `session.title`.
MAX_TITLE_CHARS = 200

#: Hermes's "there is nothing here to fork" answer, measured live 2026-09-02:
#: `session.branch` on a session with no turns returns
#: `[4008] nothing to branch -- send a message first`. It sits in neither
#: `_HERMES_SESSION_NOT_FOUND_CODES` bucket -- the session exists and is
#: perfectly healthy, it just has no transcript to copy -- so it maps to a
#: **422**, not a 404 and not a 502. See `_fork_http_error()`.
HERMES_NOTHING_TO_BRANCH_CODE = 4008

#: Whether `session.branch` honours a caller-supplied name for the fork.
#: **Measured live 2026-09-02, and the answer is yes** -- the `/branch [name]`
#: slash usage was right. It had never been reached because the empty-session
#: `[4008]` fired first. See `SESSION_BRANCH_TITLE_PARAM` for the trap.
SESSION_BRANCH_HONOURS_TITLE = True

#: **`name`, and only `name`.** Measured live 2026-09-02 on the same session,
#: back to back:
#:
#:     {"session_id": <LIVE>, "name":  "fork-name-probe-XYZ"}
#:         -> {"title": "fork-name-probe-XYZ"}          honoured
#:     {"session_id": <LIVE>, "title": "fork-title-probe-XYZ"}
#:         -> {"title": "<parent title> #3"}            SILENTLY IGNORED
#:
#: which is why this is a named constant rather than a literal at the call
#: site: every other Hermes session method spells this `title`, so `title` is
#: exactly the wrong guess here and it fails by quietly naming the fork
#: something else. The HTTP body still says `title` -- that is this gateway's
#: own vocabulary, shared with `POST /sessions/{id}/title` -- and the
#: translation happens in one place.
SESSION_BRANCH_TITLE_PARAM = "name"

#: The `hermes` CLI subcommand `DELETE /api/sessions/{id}` runs, minus the id.
#: Measured live 2026-09-01: `hermes sessions delete [-h] [--yes] session_id`
#: -- a **single session id positional**, which is why one-session deletion is
#: buildable at all (`sessions archive`, by contrast, is filter-only). Kept as
#: a constant so the test that asserts the exact argv is asserting against the
#: same literal the route sends.
HERMES_SESSIONS_DELETE_ARGV: tuple[str, ...] = ("sessions", "delete")

#: The `hermes` CLI subcommand `POST /api/sessions/{id}/archive`
#: (`api/snapshots.py`) runs best-effort after its snapshot, minus the id.
#: Measured live 2026-09-03 (PV "Session archive probe"): `sessions pin <id…>`
#: exists on the deployed binary, answers `Pinned session '<id>'.`, and a
#: pinned session is "exempt from the `sessions.auto_archive` stale sweep and
#: always appear[s] in listings". `unpin` is deliberately never sent: the same
#: flag drives Hermes Desktop's Pinned sidebar and this gateway cannot know
#: who set it. Same reason as the delete argv for being a constant: the test
#: asserting the exact argv asserts against the literal the route sends.
HERMES_SESSIONS_PIN_ARGV: tuple[str, ...] = ("sessions", "pin")

#: `--yes` -- the CLI's own "do not prompt" flag. `cli.exec` gives the spawned
#: process no stdin to answer an interactive confirmation with, so without
#: this the delete would hang or be refused rather than run.
HERMES_ASSUME_YES = "--yes"

#: Longest stored session id accepted on the destructive path. Real Hermes
#: stored ids are 22 characters (`20260829_182532_991e3f`); this is generous
#: headroom, not a fit, and exists so an unbounded string can never become a
#: process argument.
MAX_STORED_ID_CHARS = 128

#: **A security boundary, not cosmetics.** A stored id from the URL is
#: interpolated into an `argv` **process argument list** (`cli.exec`), where
#: every element is a real token parsed by the `hermes` CLI's own argparse.
#: There is no shell, so shell metacharacters are inert -- the live risk is
#: **option injection**: a value beginning with `-` is read as an OPTION
#: rather than as the `session_id` positional, which is how a request for one
#: session turns into a differently-shaped CLI invocation. This pattern
#: therefore requires an alphanumeric first character and allows only
#: `[A-Za-z0-9_.-]` after it -- no leading dash, no whitespace, no `=`, no
#: path separators, no NUL, nothing that argparse reads as anything but a
#: plain positional.
_ARGV_SAFE_STORED_ID = re.compile(rf"[A-Za-z0-9][A-Za-z0-9_.-]{{0,{MAX_STORED_ID_CHARS - 1}}}")


def _validate_stored_session_id_for_argv(stored_session_id: str) -> str:
    """`_validate_stored_session_id()` plus the argv-shape gate above.

    Two checks, not one. The shared validator is still the first thing that
    runs (empty/whitespace -> 422, and the stripped value is what everything
    downstream sees), because a destructive route must not have its own
    private idea of what a stored id is. The pattern check is the extra
    obligation this route carries and the other session routes do not: their
    id goes into a JSON-RPC *param*, this one goes into a process *argument
    list*.

    Rejects with 422 -- a malformed id is a client error, and it is refused
    **before any `cli.exec` call is made at all**, which
    `tests/test_instance.py` asserts by checking the adapter was never
    touched.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    if not _ARGV_SAFE_STORED_ID.fullmatch(stored_id):
        raise HTTPException(
            status_code=422,
            detail=(
                f"stored_session_id {stored_id!r} is not a usable Hermes stored "
                "session id: it must start with a letter or digit and contain "
                f"only letters, digits, '_', '.' or '-' (max {MAX_STORED_ID_CHARS} "
                "characters). This id becomes an argument to the `hermes` CLI, "
                "so the shape is enforced rather than passed through."
            ),
        )
    return stored_id


class SessionTitle(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/title`.

    Closed schema, non-blank, capped. A whitespace-only title is a 422 rather
    than a rename to nothing: Hermes accepts what it is given, and a session
    with a blank title is unfindable in the app's own list.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)

    @field_validator("title")
    @classmethod
    def _reject_blank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must contain non-whitespace characters")
        return value


class SessionFork(BaseModel):
    """Body for `POST /api/sessions/{stored_session_id}/fork`. Optional.

    The whole body is optional -- forking takes no input beyond the session
    being forked, and Hermes names the fork `"<parent title> #2"` itself.

    `title` exists only for the case where `session.branch` turns out to
    honour the name the `/branch [name]` slash usage suggests. While
    `SESSION_BRANCH_HONOURS_TITLE` is False the route answers 422 rather than
    accepting the field and dropping it: a closed schema that silently ignores
    what it was given is how a client comes to believe it named something it
    did not.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=MAX_TITLE_CHARS)

    @field_validator("title")
    @classmethod
    def _reject_blank_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title must contain non-whitespace characters")
        return value


@instance_router.get("/profiles")
async def list_profiles(request: Request) -> dict:
    """Every Hermes profile ("agent"/"channel"), Hermes's payload verbatim,
    each with the gateway's own live connection status added (B-136).

    Response: `{"profiles": [...], "session_create_honours_profile": false,
    "session_create_supports_profile": true}`. The `profiles` list is
    `profiles.list`'s own result with one key added per entry -- `connected`,
    from `ProfileConnectionManager.list_profiles()`
    (`domain/profile_connection.py`) keyed by `name`; `null` for a profile
    the manager has no connection tracked for at all (its reconciliation
    timer, `RESEARCH_GATEWAY_PROFILE_RECONCILE_INTERVAL_S`, may simply not
    have run yet). Every other key is `profiles.list`'s own, untouched --
    `path`, `is_default`, `model`, `provider`, `description`, `display_name`,
    `skill_count`, `last_session`
    (`{id, title, preview, started_at, last_active, message_count}`),
    `worker_session`, `canonical_session`, `ui_meta_revisions`, `has_avatar`.

    **Two related but distinct facts, both still true, now both named
    explicitly:**

    * `session_create_honours_profile` (constant `false`) -- Hermes's own
      `session.create` RPC accepts a `profile` param and silently ignores it
      (measured, PV "Profiles"). Unchanged; still means nothing above this
      route may pass `profile` to that RPC expecting it to route the call.
    * `session_create_supports_profile` (constant `true`, B-136) --
      `POST /projects/{id}/sessions/new` *can* create a session under a
      named profile now, not via that ignored RPC param but by sending
      `session.create` out on that profile's own connection
      (`resolve_profile_adapter`, `domain/hermes_runtime.py`) instead of the
      default one. A profile that isn't connected (`connected: false` or
      `null` above) is a 503 from that route, never a silent fall-back.

    502 for any Hermes failure -- this is a live read with nothing to cache.
    """
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        result = await _with_reconnect(request.app.state, adapter, adapter.profiles_list)
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    profiles = result.get("profiles") if isinstance(result, dict) else None
    profiles = profiles if isinstance(profiles, list) else []

    manager = getattr(request.app.state, "profile_connection_manager", None)
    connected_by_name = (
        {row["name"]: row["connected"] for row in manager.list_profiles()}
        if manager is not None
        else {}
    )
    annotated = [
        {**row, "connected": connected_by_name.get(row.get("name"))}
        if isinstance(row, dict)
        else row
        for row in profiles
    ]
    return {
        "profiles": annotated,
        # Measured, not assumed -- see the docstring and PV "Profiles".
        "session_create_honours_profile": False,
        "session_create_supports_profile": True,
    }


@instance_router.get("/vitals")
async def get_vitals(request: Request) -> dict:
    """What approval policy this Hermes instance is running under -- and whether it prunes.

    Response::

        {"vitals": {"approvals_mode": "smart",
                    "approvals_mode_known": true,
                    "auto_approves_dangerous_commands": true,
                    "sessions_auto_prune": false,
                    "sessions_retention_days": 90}}

    * `approvals_mode` -- Hermes's own string, verbatim, from
      `config.get {"key": "approvals.mode"}`.
    * `approvals_mode_known` -- whether this project has actually observed
      that mode's behaviour (`OBSERVED_APPROVAL_MODES`).
    * `auto_approves_dangerous_commands` -- `true` for `smart` (B-44:
      measured, its LLM auto-approved four `DANGEROUS_PATTERNS` commands),
      and **`null`, never `false`, for any mode this project has not
      measured**. A client should treat null as "unknown, say so", not as
      "safe".

    * `sessions_auto_prune` / `sessions_retention_days` (P6-3) -- Hermes's
      own `config get` answers, **from `InstanceConfigCache`**, never awaited
      here (each `cli.exec` is ~3-4 s). `null` until the first background
      pass lands, and `null` for anything the parse rules do not accept. On
      the owner's instance, measured 2026-09-03: `false` / `90`.

    Read-only. There is no companion write route and there is not going to be
    one from a phone (module docstring).

    502 for any Hermes failure on `approvals.mode`, including a `config.get`
    result with no `value` -- a shape surprise on a verified-allowlisted key is
    an upstream problem worth seeing, not something to paper over with a
    null. The two cached keys never cause a 502: they are best-effort by
    design and say `null` instead.
    """
    adapter: HermesAdapter = request.app.state.hermes_adapter
    try:
        result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: adapter.config_get(APPROVALS_MODE_KEY),
        )
    except HermesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    value: Any = result.get("value") if isinstance(result, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=502,
            detail=(
                f"config.get {{'key': {APPROVALS_MODE_KEY!r}}} returned no usable "
                f"string value (got {value!r}); the key is on Hermes's verified "
                "allowlist, so this is an upstream shape change"
            ),
        )
    mode = value.strip()
    # The adapter is connected by now (the `config.get` above went through), so
    # this is "after the adapter first connects" for the refresher; it starts
    # the background pass and returns whatever the cache holds right now.
    config_cache = _instance_config_cache(request.app.state)
    config_cache.ensure_running(request.app.state)
    return {
        "vitals": {
            "approvals_mode": mode,
            "approvals_mode_known": mode in OBSERVED_APPROVAL_MODES,
            "auto_approves_dangerous_commands": OBSERVED_APPROVAL_MODES.get(mode),
            **config_cache.snapshot(),
        }
    }


@instance_router.post("/sessions/{stored_session_id}/title")
async def rename_session(stored_session_id: str, body: SessionTitle, request: Request) -> dict:
    """Rename one session -- the review's ask #2, the buildable half.

    `{stored_session_id}` is the **STORED / durable** id, like every other
    session route here. `session.title` itself needs the **LIVE** handle
    (verified live 2026-09-01: a stored id answers `[4001] session not
    found`), which is resolved through the shared `_with_live_handle()`:
    cache hit on the current Hermes connection, else one `session.resume`,
    with a single self-healing re-resolve if Hermes has forgotten the handle.

    Body: `{"title": "..."}` -- non-blank, at most `MAX_TITLE_CHARS`
    characters, sent stripped. Anything else is a 422.

    Response: `{"stored_session_id", "live_session_id", "title",
    "pending", "title_result"}`. `title` is Hermes's echoed title when it
    sends one, else the title we submitted; `pending` is Hermes's own boolean
    (`false` on the measured rename) or null if it sent none; `title_result`
    is the raw result, verbatim.

    404 for an unknown stored id, 502 for any other Hermes failure.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    title = body.title.strip()
    adapter: HermesAdapter = request.app.state.hermes_adapter
    cache = request.app.state.live_handle_cache
    try:
        live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: adapter.session_title(live, title),
            ),
        )
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    echoed = result.get("title") if isinstance(result, dict) else None
    pending = result.get("pending") if isinstance(result, dict) else None
    logger.info("renamed session %s", stored_id)
    return {
        "stored_session_id": stored_id,
        "live_session_id": live_id,
        "title": echoed if isinstance(echoed, str) and echoed else title,
        "pending": pending if isinstance(pending, bool) else None,
        "title_result": result,
    }


@instance_router.post("/sessions/{stored_session_id}/fork", status_code=201)
async def fork_session(
    stored_session_id: str,
    request: Request,
    body: SessionFork | None = None,
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """Fork a conversation: copy its transcript into a NEW session, same project.

    The owner's requirement, verbatim: *"Can you check what it will take to
    support context / conversation forking? If we do fork I want the fork to
    be under the same project."* The second sentence is the whole reason this
    route exists rather than the app calling `session.branch` and shrugging --
    Hermes has no idea what a project is, so filing the fork where its parent
    lives is the gateway's job.

    **What a fork IS, exactly** (measured 2026-09-02, PV "Session branching"):
    a new, independent Hermes session pre-loaded with a **complete, verbatim
    copy** of the source's transcript -- every row `session.history` returns
    for the parent, same order, `row_id`s included, final assistant reply
    included. **There is no truncation.** The two sessions then diverge and
    nothing syncs between them: this is a copy, not a live branch.

    That corrects the earlier "the branch is one message SHORTER than its
    source" reading, which was a comparison between two different counters:
    Hermes's `message_count` (from `session.list` / `session.resume` /
    `session.history`) can exceed `len(messages)` by one on a session whose
    last turn was interrupted, while the branch's `message_count` always
    equals its own `len(messages)`. Measured row-for-row on five sources,
    settled and interrupted: `branch.messages == history.messages`, zero
    delta. See PV for the numbers.

    **Optional body:** `{"title": "..."}` names the fork. Honoured -- measured
    live -- but only through Hermes's `name` parameter; a `title` parameter is
    silently ignored upstream (`SESSION_BRANCH_TITLE_PARAM`). Omit the body
    entirely and Hermes names the fork `"<parent title> #2"` itself.

    **Contract.** `{stored_session_id}` is the STORED / durable id, like every
    other session route here. `session.branch` itself takes the **LIVE**
    handle (a stored id answers `[4001] session not found`), resolved through
    the shared `_with_live_handle()` -- cache hit on the current connection,
    else one `session.resume`, with a single self-healing re-resolve.

    Response (201)::

        {"stored_session_id":        <STORED id of the FORK>,
         "live_session_id":          <LIVE handle of the fork, or null>,
         "source_stored_session_id": <the id in the URL>,
         "parent_stored_session_id": <Hermes's own `parent`, or null>,
         "title":                    "<parent title> #2",
         "message_count":            <int, or null>,
         "project_id":               <str or NULL>,
         "project_title":            <str or NULL>,
         "workspace_session_id":     <str or null>,
         "filed":                    <bool>,
         "filing_error":             <str or null>}

    `project_id` is the point: the fork is filed into **whatever project the
    parent is filed in**, through `api/projects.py`'s own
    `file_stored_session()` -- the same idempotent/409/race-retry path
    `POST /projects/{id}/sessions` uses, not a second copy of it. **If the
    parent is UNFILED the fork is unfiled too** and `project_id` is `null`.
    That is the normal state (`api/projects.py` rule 3), not a failure, and no
    project is invented for it.

    **Hermes's `messages` are deliberately NOT echoed.** The branch result
    carries the whole copied transcript -- 1.6 MB on a real session (B-01) --
    and the client already has `GET /api/sessions/{id}/messages` for that. The
    counts and the ids are what a client needs to navigate to the fork.

    **Errors, honestly.** `[4008] nothing to branch` -> **422**: an empty
    session genuinely cannot be forked and the message says so, which is also
    why the app hides the action on a session with no messages. `[4001]` ->
    404 (the shared mapping). Anything else -> 502 carrying Hermes's own
    words. A fork that Hermes refused is never reported as a success, and a
    fork Hermes made but the gateway could not file is reported as made **and
    unfiled**, with the reason -- never as filed.
    """
    stored_id = _validate_stored_session_id(stored_session_id)
    branch_params: dict[str, Any] = {}
    if body is not None and body.title is not None:
        if not SESSION_BRANCH_HONOURS_TITLE:  # pragma: no cover - measured True
            raise HTTPException(
                status_code=422,
                detail=(
                    "session.branch does not accept a name for the fork on this "
                    "instance; it is named '<parent title> #N'. Fork first, "
                    "then POST /api/sessions/{stored_id}/title to rename it."
                ),
            )
        # `name`, not `title` -- the parameter Hermes actually reads. See
        # SESSION_BRANCH_TITLE_PARAM: `title` is accepted and silently ignored,
        # which is the worst of the two failure modes.
        branch_params[SESSION_BRANCH_TITLE_PARAM] = body.title.strip()

    adapter: HermesAdapter = request.app.state.hermes_adapter
    cache = request.app.state.live_handle_cache
    try:
        _live_id, result = await _with_reconnect(
            request.app.state,
            adapter,
            lambda: _with_live_handle(
                adapter,
                cache,
                stored_id,
                lambda live: adapter.session_branch(live, **branch_params),
            ),
        )
    except HermesError as exc:
        raise _fork_http_error(exc, stored_id) from exc

    payload = result if isinstance(result, dict) else {}
    fork_stored = payload.get("stored_session_id")
    if not isinstance(fork_stored, str) or not fork_stored.strip():
        # Without the fork's durable id there is nothing to file and nothing
        # the app could navigate to -- the branch may well exist upstream, but
        # this gateway cannot honestly say where. A shape surprise on a
        # verified method is an upstream problem worth seeing.
        raise HTTPException(
            status_code=502,
            detail=(
                "session.branch returned no stored_session_id for the fork "
                f"(got {fork_stored!r}); the measured shape carries both id "
                "spaces, so this is an upstream change"
            ),
        )
    fork_stored = fork_stored.strip()

    fork_live = payload.get("session_id")
    hermes_parent = payload.get("parent")
    # Hermes's own `parent` when it sent one -- it is the authoritative record
    # of what was actually branched -- else the id from the URL, which is the
    # session we resolved. Both are STORED ids.
    parent_stored = (
        hermes_parent.strip()
        if isinstance(hermes_parent, str) and hermes_parent.strip()
        else stored_id
    )
    title = payload.get("title")
    title = title if isinstance(title, str) and title else None
    message_count = payload.get("message_count")
    if isinstance(message_count, bool) or not isinstance(message_count, int):
        message_count = None

    filing = _file_fork_with_its_parent(db, parent_stored, fork_stored, title)
    logger.info(
        "forked session %s -> %s (project=%r)",
        stored_id,
        fork_stored,
        filing["project_id"],
    )
    return {
        "stored_session_id": fork_stored,
        "live_session_id": fork_live if isinstance(fork_live, str) and fork_live else None,
        "source_stored_session_id": stored_id,
        "parent_stored_session_id": (
            hermes_parent if isinstance(hermes_parent, str) and hermes_parent else None
        ),
        "title": title,
        "message_count": message_count,
        **filing,
    }


def _fork_http_error(exc: HermesError, stored_id: str) -> HTTPException:
    """`_http_error_from_hermes()` plus the one code only forking can raise.

    `[4008] nothing to branch -- send a message first` (measured live
    2026-09-02) is not an upstream failure and not a missing session: it is a
    request for something that cannot exist, i.e. a **422**. Mapping it to 502
    would tell the owner the gateway is broken when the honest answer is "this
    conversation has no turns yet".
    """
    if isinstance(exc, HermesRPCError) and _rpc_error_code(exc) == HERMES_NOTHING_TO_BRANCH_CODE:
        return HTTPException(
            status_code=422,
            detail=(
                f"session {stored_id!r} has no messages: a session must have at "
                "least one message before it can be forked, because a fork is "
                f"a copy of its transcript (Hermes: {exc})"
            ),
        )
    return _http_error_from_hermes(exc, stored_id)


def _file_fork_with_its_parent(
    db: OrmSession, parent_stored_id: str, fork_stored_id: str, title: str | None
) -> dict[str, Any]:
    """File the fork into whatever project its parent is filed in.

    The owner's actual requirement. Reuses `api/projects.py`'s
    `file_stored_session()` rather than writing a second filing path, for the
    same reason `_find_filing` is imported rather than re-implemented: the
    idempotency, the 409-on-already-filed and the unique-constraint race
    retry are all encoded there, and a second copy drifts.

    **An unfiled parent means an unfiled fork.** `project_id: null` is the
    honest answer and the normal state (`api/projects.py` rule 3); inventing
    a project to put the fork in would be this gateway deciding how someone
    organizes their own research.

    Never raises. The fork already exists upstream, so a filing failure is
    *reported* (`filed: false` plus `filing_error`) rather than turned into an
    error that tells the caller the whole fork failed -- which would leave a
    real session on the instance that the app believes does not exist.
    """
    summary: dict[str, Any] = {
        "project_id": None,
        "project_title": None,
        "workspace_session_id": None,
        "filed": False,
        "filing_error": None,
    }
    try:
        parent_filing = _find_filing(db, HERMES_RUNTIME, parent_stored_id)
        if parent_filing is None:
            return summary  # unfiled parent -> unfiled fork. Not a failure.
        project = db.get(Project, parent_filing.project_id)
        if project is None:  # pragma: no cover - FK makes this unreachable
            summary["filing_error"] = (
                f"the parent is filed into project {parent_filing.project_id!r}, "
                "which no longer exists"
            )
            return summary
        filing_row, _created = file_stored_session(db, project, fork_stored_id, title)
        summary["project_id"] = project.id
        summary["project_title"] = project.title
        summary["workspace_session_id"] = filing_row["workspace_session_id"]
        summary["filed"] = True
    except HTTPException as exc:
        # The only one `file_stored_session()` raises is the 409 for a stored
        # id already filed somewhere else -- impossible for an id Hermes just
        # minted, but a fork that exists must not be reported as a failure.
        db.rollback()
        summary["filing_error"] = str(exc.detail)
        logger.warning(
            "forked session %s could not be filed alongside its parent %s: %s",
            fork_stored_id,
            parent_stored_id,
            exc.detail,
        )
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        db.rollback()
        summary["filing_error"] = str(exc)
        logger.warning(
            "forked session %s exists on Hermes but could not be filed",
            fork_stored_id,
            exc_info=True,
        )
    return summary


def _cleanup_workspace_filing(db: OrmSession, stored_id: str) -> dict[str, Any]:
    """Drop the workspace's own row for a session Hermes no longer has.

    Called only *after* Hermes has confirmed the delete. Without it the app
    shows a ghost: `GET /api/projects/{id}/sessions` renders every filed row
    whether or not Hermes still lists it (`missing: true`), which is the right
    answer for a session that vanished on its own and the wrong one for a
    session this gateway just deleted on the user's instruction.

    **Deletes exactly the filing row, and detaches what points at it.**
    `runs.session_id` is a real FK to `sessions.id` and SQLite enforces it
    (`PRAGMA foreign_keys=ON`, `domain/db.py`), so the runs recorded for this
    session are set back to `session_id = NULL` first -- the same "unfiled"
    state a run gets when its session was never filed (P2-2d). They keep their
    `runtime_session_id`, so `GET /api/runs?session=<stored id>` still finds
    the history, and they keep their `project_id`, because the work really did
    happen inside that project and the project still exists.

    **Nothing else is deleted.** Run events, artifacts, background-task
    results and attachment ledger rows are durable research output that
    outlives the conversation that produced it -- deleting a session is not a
    licence to reap them, and none of them renders as a session in the app.

    Returns a summary; never raises. The Hermes delete already happened and is
    irreversible, so a local cleanup failure must be *reported*, not turned
    into a 500 that tells the caller the whole operation failed.
    """
    summary: dict[str, Any] = {
        "workspace_filing_deleted": False,
        "workspace_session_id": None,
        "project_id": None,
        "runs_detached": 0,
        "workspace_cleanup_error": None,
    }
    try:
        filing = _find_filing(db, HERMES_RUNTIME, stored_id)
        if filing is None:
            # Unfiled -- the normal state (`api/projects.py` rule 3). Nothing
            # to clean up and nothing wrong.
            return summary
        summary["workspace_session_id"] = filing.id
        summary["project_id"] = filing.project_id
        detached = db.execute(
            update(Run).where(Run.session_id == filing.id).values(session_id=None)
        )
        summary["runs_detached"] = detached.rowcount or 0
        db.delete(filing)
        db.commit()
        summary["workspace_filing_deleted"] = True
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        db.rollback()
        summary["workspace_cleanup_error"] = str(exc)
        logger.warning(
            "Hermes session %s was deleted but its workspace filing row could "
            "not be removed; it will render as missing until unfiled by hand",
            stored_id,
            exc_info=True,
        )
    return summary


@instance_router.delete("/sessions/{stored_session_id}")
async def delete_session(
    stored_session_id: str,
    request: Request,
    force: bool = Query(default=False),
    profile: str = Query(default="default"),
    db: OrmSession = Depends(workspace_db),
) -> dict:
    """**Delete one session from Hermes's own store -- after copying it.**

    `?profile=` (B-152) names the agent whose store holds the session. A
    stored id is only unique *within* a profile and every Hermes call below
    -- the pre-delete snapshot's resume, `cli.exec sessions delete`, the
    `session.close` -- goes to that profile's own connection through
    `resolve_profile_adapter`; before this the delete always ran on the
    default connection, so a non-default agent's session came back
    `502 ... Session '<id>' not found` no matter what.

    This is not the "unfile" of `DELETE /api/projects/{id}/sessions/{id}`,
    which removes a workspace index row and leaves the conversation intact.
    This deletes the conversation from the instance. Since P6-3 it is no
    longer the one route with no undo *and no copy*: the order is
    **snapshot -> delete -> close**, and the response names the copy.

    **Contract.** `{stored_session_id}` is the STORED / durable id, like every
    other session route here. Response:
    `{"stored_session_id", "deleted": true, "snapshot_id": "snap_…"|null,
    "live_handle_closed": true|false|null, ...}` -- plus the workspace cleanup
    summary (`workspace_filing_deleted`, `workspace_session_id`, `project_id`,
    `runs_detached`, `workspace_cleanup_error`) and the CLI's own `cli_code` /
    `cli_output`.

    **1. Snapshot first** (`api.snapshots.take_snapshot`, reason
    `pre_delete`), unless `?force=true`. The snapshot resolves the session
    through one real `session.resume` and stores the full transcript,
    reasoning included, under whatever project the session is filed in (or
    unfiled). If Hermes says the session does not exist (`[4007]`/`[4001]`)
    that is a **404**, exactly as before. **Any other snapshot failure is a
    409** `{"detail", "snapshot_error"}` and **nothing is deleted** -- the app
    shows the reason and offers "delete anyway", which is this same route
    with `?force=true`: a second, distinct confirmation, never the default.

    **2. `cli.exec {"argv": ["sessions", "delete", <stored id>, "--yes"]}`** --
    the `hermes` CLI's own `sessions delete [-h] [--yes] session_id`, measured
    live 2026-09-01. Single positional; the stored id is the id space the CLI
    works in. The id is validated by `_validate_stored_session_id_for_argv()`
    before *anything* -- including the snapshot -- runs, and a malformed id is
    a 422 with neither `session.resume` nor `cli.exec` called. **A non-zero
    `code` is a failure, never a silent success:** `cli.exec` returns 200 with
    `{"blocked", "code", "output"}` regardless; only `blocked: false` and
    `code == 0` counts as deleted, and only then does anything below run.

    **3. Close the live handle, then drop it, then clean up.** Measured
    2026-09-03 (PV "Three-eyes follow-up"): deleting underneath a live handle
    is clean upstream, but the handle the snapshot's resume minted stays in
    `session.active_list` as a **zombie** (`session.history` answers
    `count: 0`). So after a confirmed delete: best-effort `session.close
    {"session_id": <that live handle>}` (`{"closed": true}` -> `true`; any
    other answer -> `false`; no handle known or the call raised -> `null`),
    then `live_handle_cache.discard(stored_id)` so nothing can reuse it, then
    `_cleanup_workspace_filing()` as before. The snapshot row is untouched by
    the cleanup: its `workspace_session_id` FK is `ON DELETE SET NULL`.

    Logged at WARNING on both sides of the CLI call: once when the delete is
    issued (so the record exists even if the process dies mid-flight) and once
    when Hermes confirms it.
    """
    stored_id = _validate_stored_session_id_for_argv(stored_session_id)
    argv = [*HERMES_SESSIONS_DELETE_ARGV, stored_id, HERMES_ASSUME_YES]
    adapter: HermesAdapter = resolve_profile_adapter(request.app.state, profile)
    cache = resolve_live_handle_cache(request.app.state, profile)

    snapshot_id: str | None = None
    if not force:
        try:
            snapshot = await take_snapshot(
                request.app.state, stored_id, reason="pre_delete", db=db, profile=profile
            )
        except HermesError as exc:
            if _is_session_not_found(exc):
                raise _http_error_from_hermes(exc, stored_id) from exc
            return _snapshot_blocked_delete(stored_id, str(exc))
        except SnapshotStorageError as exc:
            return _snapshot_blocked_delete(stored_id, str(exc))
        snapshot_id = snapshot.id
        logger.warning(
            "pre-delete snapshot %s taken for session %s (%d rows)",
            snapshot_id,
            stored_id,
            snapshot.message_rows,
        )
    else:
        logger.warning("delete of %s FORCED without a snapshot (?force=true)", stored_id)

    logger.warning(
        "issuing IRREVERSIBLE Hermes session delete for stored id %s on profile %r (argv=%r)",
        stored_id,
        profile,
        argv,
    )
    try:
        result = await _with_reconnect(request.app.state, adapter, lambda: adapter.cli_exec(argv))
    except HermesError as exc:
        raise _http_error_from_hermes(exc, stored_id) from exc

    payload = result if isinstance(result, dict) else {}
    code = payload.get("code")
    output = payload.get("output")
    output_text = output if isinstance(output, str) else ""
    blocked = payload.get("blocked") is True
    # `bool` is an `int` subclass -- `code: true` must not pass for a zero exit.
    succeeded = not blocked and isinstance(code, int) and not isinstance(code, bool) and code == 0
    if not succeeded:
        # The CLI's own words, verbatim. `hint` is what Hermes sends instead of
        # `output` when it refuses to run the command at all (measured shape,
        # PV "Phase 3 probe"), so a blocked delete still says why.
        hint = payload.get("hint")
        said = output_text or (str(hint) if hint else "") or "no output"
        raise HTTPException(
            status_code=502,
            detail=(
                f"`hermes sessions delete {stored_id}` did not succeed "
                f"(blocked={payload.get('blocked')!r}, code={code!r}): {said}"
            ),
        )

    logger.warning("DELETED Hermes session %s from the instance", stored_id)
    live_handle_closed = await _close_live_handle(request.app.state, adapter, cache, stored_id)
    cleanup = _cleanup_workspace_filing(db, stored_id)
    return {
        "stored_session_id": stored_id,
        "deleted": True,
        "snapshot_id": snapshot_id,
        "live_handle_closed": live_handle_closed,
        "cli_code": code,
        "cli_output": output_text,
        **cleanup,
    }


def _snapshot_blocked_delete(stored_id: str, error: str) -> JSONResponse:
    """The 409 for "could not copy it, so did not delete it". Top-level keys, by contract."""
    return JSONResponse(
        status_code=409,
        content={
            "detail": (
                f"the pre-delete snapshot of session {stored_id!r} failed, so the "
                "session was NOT deleted; retry, or repeat with ?force=true to "
                "delete without a copy"
            ),
            "snapshot_error": error,
        },
    )


async def _close_live_handle(
    app_state: Any, adapter: HermesAdapter, cache: Any, stored_id: str
) -> bool | None:
    """`session.close` the handle we hold for `stored_id`, then forget it. Never raises.

    The handle comes from `LiveHandleCache` -- the pre-delete snapshot's resume
    put it there on this connection -- so no second resolution happens and a
    stored id is never sent where a live handle belongs. `None` when no handle
    is cached (a `?force=true` delete of a session nobody opened) or when the
    call raised; the cache entry is discarded in every case.
    """
    live_id = cache.get(stored_id) if cache is not None else None
    closed: bool | None = None
    if live_id is not None:
        try:
            result = await _with_reconnect(
                app_state, adapter, lambda: adapter.session_close(live_id)
            )
            closed = (result.get("closed") is True) if isinstance(result, dict) else False
        except Exception as exc:
            logger.info(
                "session.close after deleting %s raised; the zombie handle stays "
                "in active_list until the connection drops: %s",
                stored_id,
                exc,
            )
            closed = None
    if cache is not None:
        cache.discard(stored_id)
    return closed


__all__ = [
    "APPROVALS_MODE_KEY",
    "HERMES_ASSUME_YES",
    "HERMES_CONFIG_GET_ARGV",
    "HERMES_NOTHING_TO_BRANCH_CODE",
    "HERMES_SESSIONS_DELETE_ARGV",
    "HERMES_SESSIONS_PIN_ARGV",
    "MAX_STORED_ID_CHARS",
    "MAX_TITLE_CHARS",
    "OBSERVED_APPROVAL_MODES",
    "SESSIONS_AUTO_PRUNE_KEY",
    "SESSIONS_RETENTION_DAYS_KEY",
    "SESSION_BRANCH_HONOURS_TITLE",
    "SESSION_BRANCH_TITLE_PARAM",
    "InstanceConfigCache",
    "SessionFork",
    "SessionTitle",
    "delete_session",
    "fork_session",
    "get_vitals",
    "instance_router",
    "list_profiles",
    "rename_session",
]
