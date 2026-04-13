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

import time
from dataclasses import replace
from typing import Protocol, runtime_checkable

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
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

    Args:
        time_budget_ms: Optional wall-clock budget in milliseconds.  When
            set, A* returns the best complete plan found within the budget
            (anytime behaviour).  ``None`` means run until exhaustion.
    """

    def __init__(self, *, time_budget_ms: float | None = None) -> None:
        self._time_budget_ms = time_budget_ms

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        return astar_plan(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            time_budget_ms=self._time_budget_ms,
        )


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


class LazyDecompositionStrategy:
    """ADaPT-style incremental planning.

    Plans only the first ``lookahead`` actions at each planning round.
    After execution, the observer triggers replanning from the updated
    world state.  The overall behaviour emerges from the GOAP loop's
    existing replan machinery.

    Works with any inner strategy (A*, TwoPhase, custom).  When the full
    plan has fewer actions than ``lookahead``, the full plan is returned
    unchanged.

    Args:
        lookahead: Maximum number of actions to keep per planning round.
            ``1`` gives pure ADaPT behaviour (plan one, execute one,
            replan); higher values trade more speculative planning for
            fewer replan cycles.
        inner: Inner planning strategy.  Defaults to
            :class:`AStarStrategy` when ``None``.
    """

    def __init__(
        self,
        *,
        lookahead: int = 1,
        inner: PlanningStrategy | None = None,
    ) -> None:
        if lookahead < 1:
            raise ValueError(f"lookahead must be >= 1, got {lookahead}")
        self._lookahead = lookahead
        self._inner: PlanningStrategy = inner or AStarStrategy()

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        full_plan = self._inner.plan(
            start, goal, actions, blacklisted_actions=blacklisted_actions
        )
        if full_plan is None:
            return None

        # No truncation needed if the plan is already short enough
        if len(full_plan) <= self._lookahead:
            return full_plan

        # Truncate to the first `lookahead` actions
        truncated_actions = full_plan.actions[: self._lookahead]

        # Recompute expected states and cost for the truncated plan
        expected: list[PlanningState] = []
        sim_state = start
        total_cost = 0.0
        for a in truncated_actions:
            total_cost += a.get_cost(sim_state.to_dict())
            sim_state = sim_state.apply(a.effects)
            expected.append(sim_state)

        # Preserve the inner strategy's metadata
        return Plan(
            actions=truncated_actions,
            expected_states=tuple(expected),
            total_cost=total_cost,
            metadata=full_plan.metadata,
            score=SimpleScore(scalar=total_cost),
        )


class AnytimePlanningStrategy:
    """Return the best plan found within a wall-clock time budget.

    Wraps an inner strategy (default: :class:`AStarStrategy`) and applies
    a hard deadline.  When the inner strategy finishes before the budget
    expires, the result is returned immediately.  When it exceeds the
    budget, the best complete plan found so far is returned (which may be
    ``None`` if no complete plan was discovered before the deadline).

    For :class:`AStarStrategy`, anytime behaviour is natively supported via
    the ``time_budget_ms`` constructor parameter.  This strategy is more
    useful when wrapping black-box strategies that do not expose a budget
    knob — it uses :mod:`concurrent.futures` to enforce the deadline.

    Args:
        time_budget_ms: Wall-clock budget in milliseconds.
        inner:          Strategy to wrap.  Defaults to
                        :class:`AStarStrategy` (which natively handles the
                        budget more efficiently).
    """

    def __init__(
        self,
        time_budget_ms: float,
        *,
        inner: PlanningStrategy | None = None,
    ) -> None:
        self._budget_ms = time_budget_ms
        self._inner: PlanningStrategy = inner or AStarStrategy(
            time_budget_ms=time_budget_ms
        )

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._inner.plan,
                start,
                goal,
                actions,
                blacklisted_actions=blacklisted_actions,
            )
            try:
                return future.result(timeout=self._budget_ms / 1000.0)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return None


__all__ = [
    "AStarStrategy",
    "AnytimePlanningStrategy",
    "CSPRefinementStrategy",
    "LazyDecompositionStrategy",
    "PlanningStrategy",
    "TwoPhasePipelineStrategy",
]
