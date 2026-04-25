"""Module-reference loader for CLI commands.

Handles ``module.path:variable`` references, allowing CLI commands
to load Python objects (actions, goals, plans, world states) from
user code without requiring them to be installed as packages.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec


def load_python_variable(ref: str) -> Any:
    """Load a Python variable from a ``module:variable`` reference.

    The current working directory is added to ``sys.path`` so that
    local modules can be imported without installation.

    Args:
        ref: A string in the form ``module.path:variable_name``.

    Returns:
        The referenced Python object.

    Raises:
        ValueError: If *ref* does not contain exactly one colon.
        ImportError: If the module cannot be imported.
        AttributeError: If the variable is not found in the module.
    """
    if ":" not in ref:
        raise ValueError(
            f"Invalid reference {ref!r}. Expected 'module.path:variable' "
            f"(e.g. 'my_actions:ACTIONS')."
        )
    parts = ref.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid reference {ref!r}. Expected 'module.path:variable' "
            f"(e.g. 'my_actions:ACTIONS')."
        )
    module_path, var_name = parts

    # Ensure cwd is on sys.path for local module imports.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)
    return getattr(module, var_name)


def load_actions(ref: str) -> list[ActionSpec]:
    """Load and validate a list of :class:`ActionSpec` from a module reference.

    Args:
        ref: A ``module:variable`` reference pointing to a
            ``list[ActionSpec]``.

    Returns:
        The loaded action list.

    Raises:
        TypeError: If the loaded object is not a list of ``ActionSpec``.
    """
    obj = load_python_variable(ref)
    if not isinstance(obj, list):
        raise TypeError(
            f"Expected a list[ActionSpec] from {ref!r}, got {type(obj).__name__}."
        )
    for i, item in enumerate(obj):
        if not isinstance(item, ActionSpec):
            raise TypeError(
                f"Item {i} in {ref!r} is {type(item).__name__}, expected ActionSpec."
            )
    return obj


def load_goal(ref: str) -> GoalSpec:
    """Load and validate a :class:`GoalSpec` from a module reference.

    Args:
        ref: A ``module:variable`` reference pointing to a ``GoalSpec``.

    Returns:
        The loaded goal specification.

    Raises:
        TypeError: If the loaded object is not a ``GoalSpec``.
    """
    obj = load_python_variable(ref)
    if not isinstance(obj, GoalSpec):
        raise TypeError(f"Expected a GoalSpec from {ref!r}, got {type(obj).__name__}.")
    return obj


def load_world_state(ref: str | None) -> dict[str, Any]:
    """Load a world-state dict from a module reference or JSON file.

    Args:
        ref: Either a ``module:variable`` reference, a path to a JSON
            file, or ``None`` (returns an empty dict).

    Returns:
        A dict representing the world state.
    """
    if ref is None:
        return {}

    # Try JSON file first.
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(
                f"JSON file {ref!r} must contain an object, got {type(data).__name__}."
            )
        return data

    # Fall back to module:variable reference.
    obj = load_python_variable(ref)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict from {ref!r}, got {type(obj).__name__}.")
    return obj
