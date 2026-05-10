"""Utility-AI planner: greedy one-step net-value selection.

Mirrors Embabel's contract on
``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
com/embabel/plan/utility/UtilityPlanner.kt``: at each tick the
strategy filters to applicable actions, scores each by
``net_value(state) = action.utility(state) - action.cost(state)``,
and returns the highest-scoring action as a single-step
:class:`~langgoap.planner.types.Plan`.  The GOAP loop's
``ReplanStrategy.EVERY_ACTION`` then invokes the planner again on
the next tick — the utility planner is fundamentally iterative.

Two ways to use it:

1. **Concrete goal** — pair with a normal :class:`GoalSpec` and the
   observer's existing goal-satisfaction check ends the run when the
   agent's actions happen to satisfy it.  When no applicable action
   remains, the loop emits ``no_plan`` (semantically: STUCK).
2. **Nirvana goal** — pass :class:`NirvanaGoal` to drive an always-on
   agent that opportunistically reacts to changing state without a
   completion criterion.  The Nirvana goal is never satisfied by
   :func:`langgoap.graph.nodes._is_goal_satisfied`, so the loop only
   ends when the planner returns ``None`` (no applicable action).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
from langgoap.state import PlanningState

NIRVANA_NAME: str = "Nirvana"
"""Conventional name for the Nirvana goal — matches Embabel's constant."""


class NirvanaGoal(GoalSpec):
    """Marker goal that is never satisfied.

    Useful with :class:`UtilityStrategy` for always-on agents that
    react to state changes opportunistically rather than aiming at a
    fixed completion criterion.  The observer's
    :func:`~langgoap.graph.nodes._is_goal_satisfied` check returns
    ``False`` for any :class:`NirvanaGoal` instance, so the run only
    ends when the utility planner runs out of applicable actions
    (semantically: STUCK without error).

    Subclasses :class:`GoalSpec` so the existing planner / observer
    type signatures continue to accept it; the only behavioural
    difference is the never-satisfied semantics enforced by
    ``_is_goal_satisfied`` and :func:`is_nirvana`.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Carry through any GoalSpec kwargs the user provides
        # (replan_strategy, max_replans, etc.) but ensure conditions
        # is empty — Nirvana never has concrete satisfaction criteria.
        kwargs.setdefault("conditions", {})
        super().__init__(**kwargs)


def is_nirvana(goal: Any) -> bool:
    """Return True iff ``goal`` is a :class:`NirvanaGoal` instance.

    Public helper so the observer / planner / user code can branch on
    "is this an unsatisfiable always-on goal" without import gymnastics.
    """
    return isinstance(goal, NirvanaGoal)


def _resolve_utility(action: ActionSpec, state: dict[str, Any]) -> float:
    """Resolve ``action.utility`` against ``state`` (callable or static).

    ``state`` is a plain ``dict`` rather than a ``Mapping`` so the
    callable form satisfies :class:`~langgoap.types.CostFunction`'s
    signature contract.
    """
    util = action.utility
    if util is None:
        return 0.0
    if callable(util):
        return float(util(state))
    return float(util)


def _net_value(action: ActionSpec, state: PlanningState) -> float:
    """Embabel's ``Action.netValue(state)`` analog: utility − cost."""
    state_dict = state.to_dict()
    return _resolve_utility(action, state_dict) - action.get_cost(state_dict)


class UtilityStrategy:
    """Greedy one-step utility planner.

    On each :meth:`plan` call:

    1. Filter to actions whose preconditions are satisfied by the
       current state and whose names are not in
       ``blacklisted_actions``.
    2. Score each by :func:`_net_value`.
    3. Return the highest-scoring action as a single-step
       :class:`Plan`.  Ties tie-break by list order for stability.
    4. Return ``None`` when no applicable action remains.

    Pair with ``GoalSpec.replan_strategy=ReplanStrategy.EVERY_ACTION``
    so the loop calls the planner again after every action — the
    utility model is iterative, not multi-step.
    """

    name: str = "UtilityStrategy"

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        t0 = time.perf_counter()
        blacklist = set(blacklisted_actions or ())
        applicable = [
            a
            for a in actions
            if a.name not in blacklist and start.satisfies(a.preconditions)
        ]
        if not applicable:
            return None

        # ``max(..., key=...)`` over a stable iteration order picks the
        # first action with the highest net value — matches Embabel's
        # ``sortedByDescending`` then ``firstOrNull`` semantics.
        best = max(applicable, key=lambda a: _net_value(a, start))

        start_dict = start.to_dict()
        end_state = start.apply(best.get_effects(start_dict))
        cost = best.get_cost(start_dict)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return Plan(
            actions=(best,),
            expected_states=(end_state,),
            total_cost=cost,
            metadata=PlanMetadata(
                applicable_count=len(applicable),
                planning_time_ms=elapsed_ms,
                actions_pruned=0,
            ),
            score=SimpleScore(scalar=cost),
        )


__all__ = [
    "NIRVANA_NAME",
    "NirvanaGoal",
    "UtilityStrategy",
    "is_nirvana",
]
