"""Core type definitions for LangGOAP."""

from __future__ import annotations

import enum
from typing import Any, Callable, Protocol, runtime_checkable


class ReplanStrategy(str, enum.Enum):
    """When to trigger replanning during execution."""

    ON_DEVIATION = "on_deviation"
    EVERY_ACTION = "every_action"
    NEVER = "never"


class ObjectiveDirection(str, enum.Enum):
    """Direction for numeric optimization objectives."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# Convenience aliases
Minimize = ObjectiveDirection.MINIMIZE
Maximize = ObjectiveDirection.MAXIMIZE


@runtime_checkable
class CostFunction(Protocol):
    """Protocol for dynamic action cost functions.

    Receives the current world state and returns a numeric cost.
    """

    def __call__(self, world_state: dict[str, Any]) -> float: ...
