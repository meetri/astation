"""Filing an artifact from where the agent put it (Phase C, §5.3).

The cheapest leverage in the artifact plan: **the producer knows what a file
IS at the moment it writes it.** Asking a person to classify it later, from a
list of thousands, is the expensive way to learn the same fact.

The agent cannot call the gateway's own API -- it has no credential for it and
no tool that speaks it -- so the convention it CAN follow is the one thing it
already controls: the path it writes to. Every session created under a project
runs with its `cwd` set to that project's workspace folder
(`docs/PROJECT_INSTRUCTIONS_DESIGN.md` §3), so a relative write lands there,
and the folders it chooses are the filing.

    <workspace>/<project_id>/figures/roc.png       -> tags: figure
    <workspace>/<project_id>/l328-sweep/figures/roc.png -> tags: figure, l328-sweep

Two deliberate limits keep this from inventing a vocabulary:

* **Roles are a CLOSED set.** Only the five folder names below become a role
  tag. `misc/` is a folder, not a tag, and stays one.
* **One experiment tag at most**, taken from the first directory under the
  project workspace, and only if it normalizes as a tag name at all. A folder
  called `Run 7 (final!!)` is filed as a folder and not as a tag, rather than
  becoming a tag nobody can type.

Nothing here reaches outside a project workspace. The owner's existing library
-- 2,443 files under the sandbox root -- is untouched by this, because none of
it is under `astation/projects/<id>/`.
"""

from __future__ import annotations

import posixpath

from domain.tags import TagNameError, normalize_tag_name

#: Folder name -> role tag. Singular and plural both, because both are what a
#: person (or an agent) actually types, and a convention that fails on
#: `figures/` because it wanted `figure/` is a convention nobody keeps.
ROLE_FOLDERS: dict[str, str] = {
    "figure": "figure",
    "figures": "figure",
    "result": "result",
    "results": "result",
    "report": "report",
    "reports": "report",
    "data": "data",
    "datasets": "data",
    "note": "note",
    "notes": "note",
}

#: The roles, deduplicated and in the order the instructions list them. Used
#: by the seeded instructions text so the two can never drift.
ROLE_TAGS: tuple[str, ...] = ("figure", "result", "report", "data", "note")


def project_relative_parts(source_path: str, workspace_root: str) -> tuple[str, list[str]] | None:
    """`(project_id, [directory components])` for a path inside a project
    workspace, else `None`.

    The filename is not a component: a file called `results.md` sitting in the
    project root is a file, not a result folder, and reading its name as a
    role would tag by coincidence.
    """
    if not source_path or not workspace_root:
        return None
    root = posixpath.normpath(workspace_root).rstrip("/")
    normalized = posixpath.normpath(source_path)
    if not normalized.startswith(root + "/"):
        return None
    parts = [part for part in normalized[len(root) + 1 :].split("/") if part]
    if len(parts) < 2:  # just `<project_id>` or `<project_id>/file`
        return (parts[0], []) if parts else None
    return parts[0], parts[1:-1]


def filing_tags(source_path: str, workspace_root: str) -> list[str]:
    """The tags a path in a project workspace files itself under.

    Role tags first (they are the closed, shared vocabulary), then the
    experiment. Sorted within each group so the result is stable, which is
    what lets a test assert it and a reader predict it.
    """
    parts = project_relative_parts(source_path, workspace_root)
    if parts is None:
        return []
    _project_id, directories = parts
    roles = sorted({ROLE_FOLDERS[d.lower()] for d in directories if d.lower() in ROLE_FOLDERS})
    experiment = _experiment_tag(directories)
    return roles + ([experiment] if experiment and experiment not in roles else [])


def _experiment_tag(directories: list[str]) -> str | None:
    """The first non-role directory, as a tag -- or nothing.

    "Nothing" is a real answer here. A directory that cannot be a tag name
    (punctuation, too long, a leading dash) is left as a folder rather than
    forced into the vocabulary: the folder view already shows it, and a tag
    the owner cannot type is worse than no tag.
    """
    for directory in directories:
        if directory.lower() in ROLE_FOLDERS:
            continue
        try:
            return normalize_tag_name(directory)
        except TagNameError:
            return None
    return None
