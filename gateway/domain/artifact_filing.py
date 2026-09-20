"""Filing an artifact from where the agent put it (Phase C, §5.3)."""

from __future__ import annotations

import posixpath

from domain.tags import TagNameError, normalize_tag_name

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

ROLE_TAGS: tuple[str, ...] = ("figure", "result", "report", "data", "note")


def project_relative_parts(source_path: str, workspace_root: str) -> tuple[str, list[str]] | None:
    """`(project_id, [directory components])` for a path inside a project
    workspace, else `None`.
    """
    if not source_path or not workspace_root:
        return None
    root = posixpath.normpath(workspace_root).rstrip("/")
    normalized = posixpath.normpath(source_path)
    if not normalized.startswith(root + "/"):
        return None
    parts = [part for part in normalized[len(root) + 1 :].split("/") if part]
    if len(parts) < 2:
        return (parts[0], []) if parts else None
    return parts[0], parts[1:-1]


def filing_tags(source_path: str, workspace_root: str) -> list[str]:
    """The tags a path in a project workspace files itself under."""
    parts = project_relative_parts(source_path, workspace_root)
    if parts is None:
        return []
    _project_id, directories = parts
    roles = sorted({ROLE_FOLDERS[d.lower()] for d in directories if d.lower() in ROLE_FOLDERS})
    experiment = _experiment_tag(directories)
    return roles + ([experiment] if experiment and experiment not in roles else [])


def _experiment_tag(directories: list[str]) -> str | None:
    """The first non-role directory, as a tag -- or nothing."""
    for directory in directories:
        if directory.lower() in ROLE_FOLDERS:
            continue
        try:
            return normalize_tag_name(directory)
        except TagNameError:
            return None
    return None
