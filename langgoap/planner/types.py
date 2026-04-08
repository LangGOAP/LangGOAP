"""Types for GOAP planning results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.state import PlanningState


@dataclass(frozen=True)
class PlanMetadata:
    """Metadata about how a plan was generated.

    Attributes:
        nodes_explored: Number of A* nodes expanded during search.
        planning_time_ms: Wall-clock time spent planning in milliseconds.
        actions_pruned: Number of actions removed by optimization passes.
    """

    nodes_explored: int = 0
    planning_time_ms: float = 0.0
    actions_pruned: int = 0


@dataclass(frozen=True)
class Plan:
    """A sequence of actions that achieves a goal from a given start state.

    Attributes:
        actions: Ordered list of ActionSpecs to execute.
        expected_states: The world state expected after each action.
        total_cost: Sum of action costs along the plan.
        metadata: Planning algorithm statistics.
    """

    actions: tuple[ActionSpec, ...]
    expected_states: tuple[PlanningState, ...] = ()
    total_cost: float = 0.0
    metadata: PlanMetadata = field(default_factory=PlanMetadata)

    @property
    def action_names(self) -> list[str]:
        """Return the names of all actions in the plan."""
        return [a.name for a in self.actions]

    def __len__(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        return (
            f"Plan(actions={self.action_names!r}, "
            f"total_cost={self.total_cost:.4g}, "
            f"steps={len(self)})"
        )

    @classmethod
    def empty(cls) -> Plan:
        """Create an empty plan (goal already satisfied)."""
        return cls(actions=(), expected_states=(), total_cost=0.0)
