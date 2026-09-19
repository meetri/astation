"""A project's on-disk workspace: the instructions file, and where it lives.

Owner ask 2026-09-07: a project carries **instructions** that every session
created under it shares, whatever profile owns that session -- *"I want ...
different sessions owned by different profiles to be able to join the same
project and have the same foundation in terms of what they're working on."*

The mechanism, measured live before it was built
(`docs/PROJECT_INSTRUCTIONS_DESIGN.md`): Hermes bakes a session's system
prompt at `session.create` from a `HERMES.md` found at the session's working
directory. So the gateway keeps one folder per project under the sandbox
root, holds the instructions in its `HERMES.md`, and creates every session
filed under the project with `cwd` pointing there. Hermes loads the file into
that session's system prompt itself -- no per-turn cost, nothing in the
transcript, identical for every profile.

**The file is the single source of truth.** There is no `instructions` column:
the owner edits `HERMES.md` with the app's own file editor
(`PUT /api/sandbox/text`), and a DB mirror would only be a second copy to
drift. "Tied to the project" is expressed by the path -- the folder is keyed
by the project's id and goes when the project goes.

**Instructions are frozen at session creation** (measured: an edited
`HERMES.md` did not change a session even on a cold resume -- Hermes stores a
`system_prompt_hash` at create). So an edit reaches sessions created *after*
it, which is the behaviour the owner chose ("new sessions only"). The
`folder_path` file-browser bookmark is a separate concern entirely
(`Project.folder_path`); nothing here reads it.
"""

from __future__ import annotations

import posixpath
from typing import Any

from adapters.hermes.exceptions import HermesError
from domain.sandbox_fs import SandboxFS
from domain.sandbox_paths import is_under_root

#: Per-project workspace directories live here, relative to the sandbox root:
#: `<root>/astation/projects/<project_id>/`. Alongside the speech
#: prompts' `astation/prompts` (`domain/prompt_files.py`) -- one
#: gateway-owned corner of the sandbox, not scattered.
DEFAULT_WORKSPACE_SUBDIR = "astation/projects"

#: The instructions file's name inside a project's workspace directory. This
#: is the name Hermes looks for FIRST when building a system prompt (ahead of
#: AGENTS.md / CLAUDE.md), and -- unlike AGENTS.md -- it does NOT flip the
#: session into "coding project" posture (measured; `agent/coding_context.py`
#: treats AGENTS.md as a project-root marker, HERMES.md as plain context).
INSTRUCTIONS_FILENAME = "HERMES.md"


def _sandbox_root(settings: Any) -> str:
    return (getattr(settings, "hermes_sandbox_root", "") or "/opt/data").rstrip("/")


def workspace_subdir(settings: Any) -> str:
    """The `<subdir>` under the sandbox root, `PROJECT_WORKSPACE_SUBDIR` when
    set, else the default. Trailing/leading slashes trimmed so it always
    joins cleanly."""
    configured = (getattr(settings, "project_workspace_subdir", "") or "").strip().strip("/")
    return configured or DEFAULT_WORKSPACE_SUBDIR


def workspace_dir(settings: Any, project_id: str) -> str:
    """The absolute sandbox path of a project's workspace directory.

    Pure -- no I/O. `project_id` is a gateway-minted `proj_...` slug (opaque,
    filesystem-safe by construction), so it is joined directly.
    """
    return posixpath.join(_sandbox_root(settings), workspace_subdir(settings), project_id)


def instructions_path(settings: Any, project_id: str) -> str:
    """The absolute sandbox path of a project's `HERMES.md`. Pure."""
    return posixpath.join(workspace_dir(settings, project_id), INSTRUCTIONS_FILENAME)


def is_within_sandbox(settings: Any, project_id: str) -> bool:
    """Whether the computed workspace path really sits under the sandbox root.

    A guard, not an expectation: the default subdir is always inside the root,
    but a misconfigured `PROJECT_WORKSPACE_SUBDIR` (an absolute path, a `..`)
    must not let the gateway `mkdir`/write outside it. Every writing/reading
    helper below checks this first and no-ops rather than reaching upstream.
    """
    root = posixpath.normpath(_sandbox_root(settings))
    return is_under_root(posixpath.normpath(workspace_dir(settings, project_id)), root)


async def ensure_instructions_file(fs: SandboxFS, settings: Any, project_id: str) -> str:
    """Make the project's workspace directory and seed an empty `HERMES.md` if
    absent; return the file's absolute path.

    Called by the app's "Instructions" door before it opens the editor: the
    editor saves through `PUT /api/sandbox/text`, which cannot CREATE a file
    (404 on a missing one) and cannot create a parent directory. This does
    both, idempotently -- `files_mkdir` is `mkdir -p`, and the file is only
    written when it does not already exist, so re-opening never clobbers what
    the owner wrote.

    The seed is deliberately EMPTY. Hermes's `_load_hermes_md` returns nothing
    for an empty file, so a project whose instructions were opened but never
    written adds nothing to any session's system prompt -- the same as having
    no instructions at all.
    """
    if not is_within_sandbox(settings, project_id):
        raise ValueError(
            f"project workspace for {project_id!r} resolves outside the sandbox root "
            f"-- check PROJECT_WORKSPACE_SUBDIR ({workspace_subdir(settings)!r})"
        )
    directory = workspace_dir(settings, project_id)
    path = instructions_path(settings, project_id)

    mkdir_response = await fs.files_mkdir(directory)
    if mkdir_response.status_code not in (200, 201):
        raise HermesError(
            f"could not create project workspace {directory!r}: HTTP {mkdir_response.status_code}"
        )

    if not await instructions_file_exists(fs, settings, project_id):
        write_response = await fs.fs_write_text(path, "")
        if write_response.status_code not in (200, 201):
            raise HermesError(f"could not seed {path!r}: HTTP {write_response.status_code}")
    return path


async def instructions_file_exists(fs: SandboxFS, settings: Any, project_id: str) -> bool:
    """Whether a `HERMES.md` is present in the project's workspace.

    Read-text 200 means present (empty or not); 404 means absent. Any other
    status, or a transport failure, is treated as "absent" -- the caller
    (session creation) uses this only to DECIDE whether to point `cwd` at the
    workspace, and the safe answer to "is there anything to load" when we
    cannot tell is no (leave `cwd` at Hermes's default rather than send a
    session into a folder we could not confirm).
    """
    if not is_within_sandbox(settings, project_id):
        return False
    try:
        response = await fs.fs_read_text(instructions_path(settings, project_id))
    except HermesError:
        return False
    return response.status_code == 200
