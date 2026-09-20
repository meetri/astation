"""A project's on-disk workspace: the instructions file, and where it lives."""

from __future__ import annotations

import posixpath
from typing import Any

from adapters.hermes.exceptions import HermesError
from domain.sandbox_fs import SandboxFS
from domain.sandbox_paths import is_under_root

DEFAULT_WORKSPACE_SUBDIR = "astation/projects"

INSTRUCTIONS_FILENAME = "HERMES.md"


INSTRUCTIONS_SEED = """## Filing what you produce

Files you write in this folder are kept automatically. Put anything worth
keeping in a folder that says what it is, and it files itself:

- `figures/` — plots, diagrams, images
- `results/` — measurements, metrics, tables
- `reports/` — write-ups meant to be read
- `data/` — datasets and derived data
- `notes/` — working notes

Name the experiment with a folder above those — `l328-sweep/figures/roc.png`
files that image under both `figure` and `l328-sweep`, so every figure from
that run is one tap away later. Scratch files need no folder; leave them at
the top and nothing is claimed about them.
"""


def _sandbox_root(settings: Any) -> str:
    return (getattr(settings, "hermes_sandbox_root", "") or "/opt/data").rstrip("/")


def workspace_subdir(settings: Any) -> str:
    """The `<subdir>` under the sandbox root, `PROJECT_WORKSPACE_SUBDIR` when
    set, else the default. Trailing/leading slashes trimmed so it always
    joins cleanly."""
    configured = (getattr(settings, "project_workspace_subdir", "") or "").strip().strip("/")
    return configured or DEFAULT_WORKSPACE_SUBDIR


def workspace_dir(settings: Any, project_id: str) -> str:
    """The absolute sandbox path of a project's workspace directory."""
    return posixpath.join(_sandbox_root(settings), workspace_subdir(settings), project_id)


def instructions_path(settings: Any, project_id: str) -> str:
    """The absolute sandbox path of a project's `HERMES.md`. Pure."""
    return posixpath.join(workspace_dir(settings, project_id), INSTRUCTIONS_FILENAME)


def is_within_sandbox(settings: Any, project_id: str) -> bool:
    """Whether the computed workspace path really sits under the sandbox root."""
    root = posixpath.normpath(_sandbox_root(settings))
    return is_under_root(posixpath.normpath(workspace_dir(settings, project_id)), root)


async def ensure_instructions_file(fs: SandboxFS, settings: Any, project_id: str) -> str:
    """Make the project's workspace directory and seed an empty `HERMES.md` if
    absent; return the file's absolute path.
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
        write_response = await fs.fs_write_text(path, INSTRUCTIONS_SEED)
        if write_response.status_code not in (200, 201):
            raise HermesError(f"could not seed {path!r}: HTTP {write_response.status_code}")
    return path


async def instructions_file_exists(fs: SandboxFS, settings: Any, project_id: str) -> bool:
    """Whether a `HERMES.md` is present in the project's workspace."""
    if not is_within_sandbox(settings, project_id):
        return False
    try:
        response = await fs.fs_read_text(instructions_path(settings, project_id))
    except HermesError:
        return False
    return response.status_code == 200
