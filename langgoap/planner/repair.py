"""Plan repair: patch a partially-executed plan instead of full replan.

When state drifts (action failure or unexpected observation), plan repair:
1. Identifies the longest valid prefix of the remaining plan from current_state
2. Simulates the valid prefix to compute the intermediate state
3. Replans from the intermediate state to the goal (using A*)
4. Returns: prefix_actions + new_suffix_actions as a new Plan

If the prefix is empty (no remaining actions are applicable) and the full
replan also fails, returns None.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
from langgoap.state import PlanningState

if TYPE_CHECKING:
    from langgoap.planner.strategy import PlanningStrategy

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepairResult:
    """Outcome of a plan-repair attempt.

    Attributes:
        repaired_plan: The repaired plan, or ``None`` if repair failed
            (both prefix reuse and suffix replanning were unsuccessful).
        repair_applied: ``True`` if a valid prefix was found and reused.
        valid_prefix_length: Number of actions in the reused prefix.
        suffix_replanned: ``True`` if A* was invoked to plan the suffix.
        suffix_length: Number of actions in the newly planned suffix.
    """

    repaired_plan: Plan | None
    repair_applied: bool
    valid_prefix_length: int
    suffix_replanned: bool
    suffix_length: int


# ---------------------------------------------------------------------------
# Core repair function
# ---------------------------------------------------------------------------


def repair_plan(
    plan: Plan,
    current_state: PlanningState,
    goal: GoalSpec,
    actions: list[ActionSpec],
    *,
    current_step: int = 0,
    blacklisted_actions: list[str] | None = None,
) -> RepairResult:
    """Repair a partially-executed plan by reusing its valid prefix.

    Args:
        plan: The original plan.
        current_state: Current world state (may have drifted from expected).
        goal: The goal to achieve.
        actions: Available actions for replanning.
        current_step: First step to consider (steps before this were already
            executed and must not be reconsidered).
        blacklisted_actions: Action names to exclude from suffix replanning.

    Returns:
        A :class:`RepairResult` describing the outcome.  ``repaired_plan``
        is ``None`` only when no suffix plan could be found and the goal
        remains unsatisfied.
    """
    # Fast path: goal already satisfied — no plan needed.
    if current_state.satisfies(goal.conditions):
        return RepairResult(
            repaired_plan=Plan.empty(),
            repair_applied=False,
            valid_prefix_length=0,
            suffix_replanned=False,
            suffix_length=0,
        )

    remaining = plan.actions[current_step:]

    # Walk the remaining actions and collect the longest valid prefix.
    sim_state = current_state
    prefix_actions: list[ActionSpec] = []
    prefix_states: list[PlanningState] = []
    prefix_cost: float = 0.0

    for action in remaining:
        if sim_state.satisfies(action.preconditions):
            sim_dict = sim_state.to_dict()
            cost = action.get_cost(sim_dict)
            prefix_cost += cost
            sim_state = sim_state.apply(action.get_effects(sim_dict))
            prefix_actions.append(action)
            prefix_states.append(sim_state)
        else:
            break  # first inapplicable action ends the valid prefix

    intermediate_state = sim_state
    repair_applied = len(prefix_actions) > 0

    # Replan the suffix from the intermediate state.
    suffix_plan = astar_plan(
        intermediate_state, goal, actions, blacklisted_actions=blacklisted_actions
    )

    if suffix_plan is None:
        return RepairResult(
            repaired_plan=None,
            repair_applied=repair_applied,
            valid_prefix_length=len(prefix_actions),
            suffix_replanned=True,
            suffix_length=0,
        )

    # Merge prefix + suffix into a single Plan.
    all_actions = tuple(prefix_actions) + suffix_plan.actions
    all_states = tuple(prefix_states) + suffix_plan.expected_states
    total_cost = prefix_cost + suffix_plan.total_cost

    repaired = Plan(
        actions=all_actions,
        expected_states=all_states,
        total_cost=total_cost,
        metadata=PlanMetadata(
            nodes_explored=suffix_plan.metadata.nodes_explored,
            planning_time_ms=suffix_plan.metadata.planning_time_ms,
        ),
        score=SimpleScore(scalar=total_cost),
    )

    return RepairResult(
        repaired_plan=repaired,
        repair_applied=repair_applied,
        valid_prefix_length=len(prefix_actions),
        suffix_replanned=True,
        suffix_length=len(suffix_plan.actions),
    )


# ---------------------------------------------------------------------------
# Strategy wrapper
# ---------------------------------------------------------------------------


class RepairStrategy:
    """Plan-repair wrapper around any inner :class:`PlanningStrategy`.

    On the first planning call (``prior_plan=None``), delegates to the
    inner strategy unchanged.  When a ``prior_plan`` is supplied (i.e.
    the planner is replanning after state drift), attempts
    :func:`repair_plan` first.  Falls back to a full inner-strategy replan
    only when repair fails to produce a valid plan.

    Args:
        inner: The base strategy to delegate to.  Defaults to
            :class:`~langgoap.planner.strategy.AStarStrategy` when ``None``.
    """

    def __init__(self, inner: PlanningStrategy | None = None) -> None:
        if inner is None:
            from langgoap.planner.strategy import AStarStrategy

            inner = AStarStrategy()
        self._inner: PlanningStrategy = inner

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
        prior_plan: Plan | None = None,
        current_step: int = 0,
    ) -> Plan | None:
        """Return a plan, attempting repair when ``prior_plan`` is given.

        Args:
            start: Current world state.
            goal: Goal to achieve.
            actions: Available actions.
            blacklisted_actions: Actions to exclude.
            prior_plan: The plan that was active before state drift.  When
                ``None``, behaves exactly like the inner strategy.
            current_step: Step index of the first remaining action in
                ``prior_plan``.  Passed through to :func:`repair_plan`.

        Returns:
            A :class:`~langgoap.planner.types.Plan` or ``None``.
        """
        if prior_plan is None:
            return self._inner.plan(
                start, goal, actions, blacklisted_actions=blacklisted_actions
            )

        result = repair_plan(
            prior_plan,
            start,
            goal,
            actions,
            current_step=current_step,
            blacklisted_actions=blacklisted_actions,
        )

        if result.repair_applied and result.repaired_plan is not None:
            return result.repaired_plan

        # Repair either found no usable prefix or suffix replanning failed;
        # delegate to the inner strategy for a full replan.
        return self._inner.plan(
            start, goal, actions, blacklisted_actions=blacklisted_actions
        )


__all__ = ["RepairResult", "repair_plan", "RepairStrategy"]
