"""Directory Handler — Tier 1 primer for LangGoap.

A minimal end-to-end GOAP loop adapted from GOApy's ``directory_handler``
example (``research/repos/GOApy/examples/directory_handler``).

World state models a workspace directory and a marker file inside it.
The planner discovers that it must create the directory *before* it can
create the token, and the executor performs the real filesystem
operations.  This exercises the complete planning + execution loop with
side-effecting actions against the real OS — no mocks.

Usage
-----
The helpers take a ``workspace`` path so the notebook and the tests can
both run against a ``tempfile.TemporaryDirectory`` and never touch
``/tmp``.

>>> import tempfile
>>> from pathlib import Path
>>> from tutorial_examples.directory_handler import (
...     directory_handler_actions,
...     initial_world_state,
... )
>>> with tempfile.TemporaryDirectory() as tmp:
...     workspace = Path(tmp) / "goap_workspace"
...     actions = directory_handler_actions(workspace)
...     state = initial_world_state(workspace)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgoap import ActionSpec


def _workspace_exists(workspace: Path) -> bool:
    return workspace.is_dir()


def _token_exists(workspace: Path) -> bool:
    return (workspace / ".token").is_file()


def initial_world_state(workspace: Path) -> dict[str, Any]:
    """Sense the filesystem once and return the starting world state.

    LangGoap's executor reads the state keys directly, so we hand the
    planner concrete ``bool``s rather than GOApy's string enums.
    """
    return {
        "workspace": str(workspace),
        "workspace_exists": _workspace_exists(workspace),
        "token_exists": _token_exists(workspace),
    }


def create_workspace(ws: dict[str, Any]) -> dict[str, Any]:
    """Create the workspace directory and re-sense its state."""
    workspace = Path(ws["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    return {
        "workspace_exists": True,
        "token_exists": _token_exists(workspace),
    }


def create_token(ws: dict[str, Any]) -> dict[str, Any]:
    """Create the marker file inside the workspace."""
    workspace = Path(ws["workspace"])
    token = workspace / ".token"
    token.touch(exist_ok=True)
    return {"token_exists": True}


def directory_handler_actions(workspace: Path) -> list[ActionSpec]:
    """Return the ActionSpec list for the directory-handler pipeline.

    Two actions:

    - ``create_workspace`` — requires ``workspace_exists=False``,
      sets ``workspace_exists=True``.
    - ``create_token`` — requires the workspace to exist and the token
      to be missing, sets ``token_exists=True``.

    The ``workspace`` argument is captured by the execute closures so
    different test runs can target different temporary directories.
    """
    del workspace  # closures use ws["workspace"] instead
    return [
        ActionSpec(
            name="create_workspace",
            preconditions={"workspace_exists": False},
            effects={"workspace_exists": True},
            execute=create_workspace,
        ),
        ActionSpec(
            name="create_token",
            preconditions={"workspace_exists": True, "token_exists": False},
            effects={"token_exists": True},
            execute=create_token,
        ),
    ]
