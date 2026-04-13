"""LangGraph state schema for GOAP execution."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgoap.goals import GoalSpec, MultiGoal
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

    Keys
    ----
    ``world_state``
        Current world state as a flat dictionary.
    ``goal``
        The goal specification to achieve. May be a single
        :class:`GoalSpec` or a :class:`MultiGoal` wrapping several
        sub-goals.
    ``plan``
        The current action plan (set by planner node).
    ``current_step``
        Index of the next action to execute.
    ``execution_history``
        Append-only log of action results.
    ``replan_count``
        How many times the planner has been invoked for the current
        (sub-)goal. Reset to zero when a :class:`MultiGoal` advances
        between sequential sub-goals so each sub-goal receives its own
        ``max_replans`` budget.
    ``replan_reason``
        Why the last replan was triggered. One of ``"action_failed"``,
        ``"state_deviation"``, ``"every_action_replan"``,
        ``"plan_exhausted"``, ``"max_replans_exceeded"``, or
        ``"subgoal_achieved"`` (only emitted when a :class:`MultiGoal`
        advances between sequential sub-goals).
    ``status``
        Current execution status.
    ``blacklisted_actions``
        Action names the planner must skip.
    ``action_failure_counts``
        Per-action cumulative failure counts.
    ``current_subgoal_index``
        When ``goal`` is a :class:`MultiGoal` running in ``sequential``
        mode, the 0-based index of the sub-goal being worked on.
        Defaults to 0.
    ``no_plan_explanation``
        Serialised :class:`~langgoap.planner.explain.NoPlanExplanation`
        (via ``.to_dict()``) produced when A* returns ``None``.  ``None``
        when a plan was found.  Use
        :meth:`~langgoap.planner.explain.NoPlanExplanation.from_dict` to
        reconstruct the object.
    ``reflection_context``
        Ordered list of verbal reflection summaries produced by
        :class:`~langgoap.reflexion.ReflexionTracer` after action failures.
        Each entry is a human-readable string of the form
        ``"[action_name] reflection → suggestion"``.  Set by the planner
        node on each planning round so downstream LLM-evaluated conditions
        can read them without coupling directly to the tracer.
    """

    world_state: dict[str, Any]
    goal: GoalSpec | MultiGoal
    plan: Plan | None
    current_step: int
    execution_history: Annotated[list[ActionResult], operator.add]
    replan_count: int
    replan_reason: str | None
    status: str
    blacklisted_actions: list[str]
    action_failure_counts: dict[str, int]
    current_subgoal_index: int
    no_plan_explanation: dict[str, Any] | None
    reflection_context: list[str]
