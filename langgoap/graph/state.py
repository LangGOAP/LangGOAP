"""LangGraph state schema for GOAP execution."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan


@dataclass
class ActionResult:
    """Result of executing a single GOAP action.

    Attributes:
        action_name: Name of the action that was executed.
        success: Whether the action completed successfully.
        state_before: World state before execution.
        state_after: World state after execution.
        error: Error message if the action failed.
    """

    action_name: str
    success: bool
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class GoapState(TypedDict, total=False):
    """LangGraph state schema for the GOAP execution loop.

    Fields:
        world_state: Current world state as a flat dictionary.
        goal: The goal specification to achieve.
        plan: The current action plan (set by planner node).
        current_step: Index of the next action to execute.
        execution_history: Append-only log of action results.
        replan_count: How many times the planner has been invoked.
        replan_reason: Why the last replan was triggered.
        status: Current execution status.
        blacklisted_actions: Action names the planner must skip.
        action_failure_counts: Per-action cumulative failure counts.
    """

    world_state: dict[str, Any]
    goal: GoalSpec
    plan: Plan | None
    current_step: int
    execution_history: Annotated[list[ActionResult], operator.add]
    replan_count: int
    replan_reason: str | None
    status: str
    blacklisted_actions: list[str]
    action_failure_counts: dict[str, int]
