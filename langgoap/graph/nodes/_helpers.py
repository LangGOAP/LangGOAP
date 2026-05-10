"""Shared helpers for the planner / executor / observer node modules.

Module-private utilities that more than one node module needs:

* :func:`_safe_tracer_call` / :func:`_safe_tracer_acall` \u2014 the
  defensive tracer-invocation guards used by every hook site.
* :func:`_planning_keys` \u2014 union of action precondition / effect /
  goal keys used by the planner to slice ``world_state`` down for A*.
* :func:`_is_better_plan` \u2014 feasibility-first comparison for
  ``MultiGoal`` ``any``-mode plan selection.

Lives in its own module so the per-node files do not have to
cross-import each other.
"""

from __future__ import annotations

import logging
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan
from langgoap.score import SimpleScore
from langgoap.tracing import PlanningTracer

logger = logging.getLogger("langgoap.graph.nodes")


def _safe_tracer_call(tracer: PlanningTracer, method: str, *args: Any) -> None:
    """Invoke a sync tracer hook, swallowing any exception.

    Observability must never break the planner \u2014 this is the second
    line of defence beyond :class:`MultiTracer`'s own catch.
    """
    try:
        getattr(tracer, method)(*args)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Tracer raised during %s: %s", method, exc)


async def _safe_tracer_acall(tracer: PlanningTracer, method: str, *args: Any) -> None:
    """Invoke an async tracer hook, swallowing any exception."""
    try:
        await getattr(tracer, method)(*args)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Tracer raised during %s: %s", method, exc)


def _planning_keys(actions: list[ActionSpec], goal: GoalSpec) -> set[str]:
    """Compute the set of world-state keys relevant to A* planning.

    Returns the union of all keys appearing in action preconditions,
    action effects, and goal conditions.  Used to filter a rich
    ``world_state`` dict down to the hashable boolean flags that
    the planner operates on.
    """
    keys: set[str] = set(goal.conditions.keys())
    for action in actions:
        keys.update(action.preconditions.keys())
        keys.update(action.effect_key_set())
    return keys


def _is_better_plan(candidate: Plan, best: Plan) -> bool:
    """Feasibility-first comparison for MultiGoal ``any`` mode.

    Uses three-tier ordering so that CSP soft-score information is never
    silently discarded when sub-goals are compared:

    **Tier 1 \u2014 Feasibility**
        A feasible plan always beats an infeasible one, regardless of cost
        or objective value.

    **Tier 2 \u2014 Native score comparison**
        When both plans share the same concrete :class:`~langgoap.score.Score`
        subtype, the subtype's own comparison is used.  Sign conventions
        differ between subtypes and are handled explicitly:

        * :class:`~langgoap.score.SimpleScore` \u2014 *lower* scalar is better
          (A* minimises path cost; ``scalar`` equals ``total_cost``).
        * :class:`~langgoap.score.HardSoftScore` /
          :class:`~langgoap.score.BendableScore` \u2014 *higher* (less-negative)
          is better (lexicographic convention; ``hard == 0`` is feasible).

        Using native comparison preserves CSP soft-score information: a
        ``HardSoftScore(hard=0, soft=-1)`` plan correctly beats a
        ``HardSoftScore(hard=0, soft=-10)`` plan even when both have the
        same ``total_cost``.

        A :class:`TypeError` raised by cross-subtype comparisons or
        mismatched :class:`~langgoap.score.BendableScore` shapes falls
        through to tier 3.

    **Tier 3 \u2014 total_cost fallback**
        ``total_cost`` (a plain ``float``, always present on every
        :class:`~langgoap.planner.types.Plan`) breaks ties when tier 2
        cannot resolve them.

        Among *infeasible* plans this picks the cheapest rather than the
        *least* infeasible (i.e. the plan whose ``hard`` score is closest
        to zero).  This is acceptable because MultiGoal ``any`` reports
        ``no_plan`` when *every* sub-goal is infeasible regardless of which
        infeasible candidate is selected.  A future improvement could
        compare ``hard`` scores directly for the infeasible-vs-infeasible
        case; for now the simpler ``total_cost`` tiebreaker is documented
        here rather than silently applied.
    """
    c_feasible = candidate.score.is_feasible()
    b_feasible = best.score.is_feasible()
    if c_feasible != b_feasible:
        return c_feasible
    # Tier 2: same-subtype native comparison preserves soft-score information.
    # TypeError fires on cross-subtype or mismatched BendableScore shapes.
    try:
        if isinstance(candidate.score, SimpleScore):
            # SimpleScore: lower scalar is better (cost minimisation).
            return candidate.score < best.score
        # HardSoftScore / BendableScore: higher (less-negative) is better.
        return candidate.score > best.score
    except TypeError:
        pass
    # Tier 3: cross-subtype fallback \u2014 total_cost is a uniform float proxy.
    return candidate.total_cost < best.total_cost


__all__ = [
    "logger",
    "_safe_tracer_call",
    "_safe_tracer_acall",
    "_planning_keys",
    "_is_better_plan",
]
