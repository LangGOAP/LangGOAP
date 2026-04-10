"""Planning strategy hierarchy.

OptaPlanner defines a `Solver` that runs one or more `Phase` instances
(construction heuristic → local search → …).  LangGoap mirrors that
idea with the :class:`PlanningStrategy` Protocol.  A strategy takes a
start state, a goal, and a list of actions, and returns a single
:class:`~langgoap.planner.types.Plan` (or ``None``).

Concrete strategies:

* :class:`AStarStrategy` — forward-chaining A* construction heuristic,
  reused from :mod:`langgoap.planner.astar`.
* :class:`CSPRefinementStrategy` — takes an existing candidate plan and
  runs CSP validation / multi-plan CP-SAT optimization on it.
* :class:`TwoPhasePipelineStrategy` — the default: A* followed by CSP
  refinement.  This is exactly what :mod:`langgoap.planner.pipeline`
  already implements; the strategy class wraps it so user code can pass
  a custom strategy to ``GoapPlanner`` without touching the pipeline
  module.

A strategy is typed as a Protocol so user code can subclass or duck-
type any implementation.  See ``docs/optaplanner_mapping.md`` for a
worked example of a custom strategy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


@runtime_checkable
class PlanningStrategy(Protocol):
    """A pluggable planning strategy.

    Implementations must produce a complete :class:`Plan` — returning
    ``None`` when no plan exists.  Strategies are stateless with respect
    to planning runs; construction-time configuration (heuristics,
    solver knobs, etc.) lives on ``__init__``.
    """

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        """Return a plan from ``start`` to ``goal``."""
        ...


class AStarStrategy:
    """Pure A* construction heuristic.

    Delegates to :func:`langgoap.planner.astar.plan`.  Use this when a
    goal has neither constraints nor objectives and CSP overhead is
    unwanted.
    """

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        return astar_plan(start, goal, actions, blacklisted_actions=blacklisted_actions)


class CSPRefinementStrategy:
    """Run CSP validation / optimization over an existing candidate plan.

    This strategy takes a precomputed candidate plan at construction
    time and refines it through :func:`langgoap.planner.csp.validate_plan`.
    It does **not** run A* itself — compose it with
    :class:`AStarStrategy` via :class:`TwoPhasePipelineStrategy` for
    the common case.
    """

    def __init__(self, candidate: Plan) -> None:
        self._candidate = candidate

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        # Local import to avoid pipeline ↔ strategy import cycles.
        from langgoap.planner.csp import validate_plan
        from langgoap.planner.pipeline import _augment_plan

        csp_meta = validate_plan(self._candidate, goal)
        return _augment_plan(self._candidate, goal, csp_meta)


class TwoPhasePipelineStrategy:
    """A* construction + CSP refinement, composed as a single strategy.

    This is the default strategy: it is equivalent to calling
    :func:`langgoap.planner.pipeline.plan` directly.  The wrapper class
    exists so ``GoapPlanner(strategy=...)`` can accept it via the same
    Protocol-typed kwarg used for custom strategies.
    """

    def __init__(self, *, max_alternatives: int = 5) -> None:
        self._max_alternatives = max_alternatives

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        return pipeline_plan(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            max_alternatives=self._max_alternatives,
        )


__all__ = [
    "PlanningStrategy",
    "AStarStrategy",
    "CSPRefinementStrategy",
    "TwoPhasePipelineStrategy",
]
