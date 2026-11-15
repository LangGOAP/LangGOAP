"""Planning strategy hierarchy.

LangGOAP supports pluggable planning strategies via the
:class:`PlanningStrategy` Protocol.  A strategy takes a start state,
a goal, and a list of actions, and returns a single
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
type any implementation.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

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
        prior_plan: Plan | None = None,
        current_step: int = 0,
    ) -> Plan | None:
        """Return a plan from ``start`` to ``goal``.

        ``prior_plan`` and ``current_step`` are advisory: most strategies
        ignore them, but repair-style strategies (see
        :class:`~langgoap.planner.repair.RepairStrategy`) use them to
        patch a partially-executed plan instead of replanning from
        scratch.  Strategies that do not consume them must still accept
        the keyword arguments to remain LSP-substitutable.
        """
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
        tracer: Optional :class:`~langgoap.tracing.PlanningTracer` that
            receives the search hooks.  When omitted, ``GoapPlanner`` hands
            the strategy its own tracer, so passing ``strategy=AStarStrategy()``
            explicitly keeps the same tracing as the default (#63).
        record_expansions: Emit ``on_search_expand`` per expanded node.
    """

    def __init__(
        self,
        *,
        time_budget_ms: float | None = None,
        tracer: Any = None,
        record_expansions: bool = False,
    ) -> None:
        self._time_budget_ms = time_budget_ms
        self._tracer = tracer
        self._record_expansions = record_expansions

    def _with_tracer(self, tracer: Any, record_expansions: bool) -> AStarStrategy:
        """A copy reporting to ``tracer``, unless one was set at construction."""
        if self._tracer is not None:
            return self
        return AStarStrategy(
            time_budget_ms=self._time_budget_ms,
            tracer=tracer,
            record_expansions=record_expansions,
        )

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
        del prior_plan, current_step  # A* plans from scratch
        return astar_plan(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            time_budget_ms=self._time_budget_ms,
            tracer=self._tracer,
            record_expansions=self._record_expansions,
        )


def refine_with_csp(plan: Plan, goal: GoalSpec) -> Plan:
    """Validate ``plan`` against ``goal``'s constraints and attach CSP scoring.

    Runs :func:`langgoap.planner.csp.validate_plan` over an existing plan
    and returns the plan re-scored with a
    :class:`~langgoap.score.HardSoftScore` and ``metadata.csp`` filled in.
    It does **not** search: pair it with A* (or use
    :class:`TwoPhasePipelineStrategy`, which does both) when you need a
    plan constructed as well as checked.
    """
    # Local imports avoid pipeline <-> strategy import cycles.
    from langgoap.planner.csp import validate_plan
    from langgoap.planner.pipeline import _augment_plan

    return _augment_plan(plan, goal, validate_plan(plan, goal))


class CSPRefinementStrategy:
    """Deprecated: call :func:`refine_with_csp` instead.

    This class never fit the :class:`PlanningStrategy` contract: its
    ``plan()`` ignored ``start`` and ``actions`` and refined the plan
    given at construction time (#63).  It remains as a thin wrapper until
    0.3.0 and warns on construction.
    """

    def __init__(self, candidate: Plan) -> None:
        warnings.warn(
            "CSPRefinementStrategy is deprecated and will be removed in 0.3.0; "
            "call refine_with_csp(plan, goal) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._candidate = candidate

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
        del start, actions, blacklisted_actions, prior_plan, current_step
        return refine_with_csp(self._candidate, goal)


class TwoPhasePipelineStrategy:
    """A* construction + CSP refinement, composed as a single strategy.

    This is the default strategy: it is equivalent to calling
    :func:`langgoap.planner.pipeline.plan` directly.  The wrapper class
    exists so ``GoapPlanner(strategy=...)`` can accept it via the same
    Protocol-typed kwarg used for custom strategies.
    """

    def __init__(
        self,
        *,
        max_alternatives: int = 8,
        tracer: Any = None,
        record_expansions: bool = False,
    ) -> None:
        self._max_alternatives = max_alternatives
        self._tracer = tracer
        self._record_expansions = record_expansions

    def _with_tracer(
        self, tracer: Any, record_expansions: bool
    ) -> TwoPhasePipelineStrategy:
        """A copy reporting to ``tracer``, unless one was set at construction."""
        if self._tracer is not None:
            return self
        return TwoPhasePipelineStrategy(
            max_alternatives=self._max_alternatives,
            tracer=tracer,
            record_expansions=record_expansions,
        )

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
        del prior_plan, current_step  # pipeline plans from scratch
        return pipeline_plan(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            max_alternatives=self._max_alternatives,
            tracer=self._tracer,
            record_expansions=self._record_expansions,
        )


def rebind_inner_tracer(strategy: Any, tracer: Any, record_expansions: bool) -> Any:
    """Copy ``strategy`` with its ``_inner`` search bound to ``tracer``.

    Shared by the wrapper strategies (lazy, anytime, repair) so a tracer
    given to ``GoapPlanner`` reaches the A* they delegate to (#63).  A
    wrapper whose inner strategy has no ``_with_tracer`` hook is returned
    unchanged.
    """
    binder = getattr(strategy._inner, "_with_tracer", None)
    if binder is None:
        return strategy
    clone = copy.copy(strategy)
    clone._inner = binder(tracer, record_expansions)
    return clone


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

    def _with_tracer(
        self, tracer: Any, record_expansions: bool
    ) -> LazyDecompositionStrategy:
        return rebind_inner_tracer(self, tracer, record_expansions)

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
        # Forward repair kwargs only when meaningful so legacy inner
        # strategies whose signature predates the widened Protocol keep
        # working on initial plans.
        if prior_plan is not None:
            full_plan = self._inner.plan(
                start,
                goal,
                actions,
                blacklisted_actions=blacklisted_actions,
                prior_plan=prior_plan,
                current_step=current_step,
            )
        else:
            full_plan = self._inner.plan(
                start,
                goal,
                actions,
                blacklisted_actions=blacklisted_actions,
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
            sim_dict = sim_state.as_context()
            total_cost += a.get_cost(sim_dict)
            sim_state = sim_state.apply(a.get_effects(sim_dict))
            expected.append(sim_state)

        # Keep the inner strategy's search statistics but never its CSP
        # verdict: resource usage, schedule and status described the *full*
        # plan.  Re-validate the truncated prefix so the score type and the
        # CSP metadata describe what will actually run (issue #45).
        truncated = Plan(
            actions=truncated_actions,
            expected_states=tuple(expected),
            total_cost=total_cost,
            metadata=replace(full_plan.metadata, csp=None),
            score=SimpleScore(scalar=total_cost),
        )
        from langgoap.planner.csp import validate_plan
        from langgoap.planner.pipeline import _augment_plan, needs_csp

        if needs_csp(goal):
            return _augment_plan(truncated, goal, validate_plan(truncated, goal))
        return truncated


class AnytimePlanningStrategy:
    """Return the best plan found within a wall-clock time budget.

    With the default inner strategy (A*) the deadline is enforced natively:
    :func:`langgoap.planner.astar.plan` runs weighted A* with a tightening
    weight schedule, so a complete plan exists early and is improved while
    the budget lasts, and the best complete plan found so far is returned
    on expiry (see :data:`langgoap.planner.astar.ANYTIME_WEIGHTS`).

    Any other ``inner`` strategy is a black box: it runs in a worker thread
    and, when it exceeds the budget, this strategy returns ``None``
    immediately with a :class:`RuntimeWarning`.  Python cannot interrupt a
    running thread, so the inner strategy keeps running in the background
    until it finishes on its own; prefer strategies that accept a budget
    themselves (``AStarStrategy(time_budget_ms=...)``, ``MCTSStrategy``
    iteration limits) when the planning problem is large.

    Args:
        time_budget_ms: Wall-clock budget in milliseconds.
        inner: Strategy to wrap.  ``None`` (default) or an
            :class:`AStarStrategy` uses the native anytime A* path.
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

    def _with_tracer(
        self, tracer: Any, record_expansions: bool
    ) -> AnytimePlanningStrategy:
        return rebind_inner_tracer(self, tracer, record_expansions)

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
        if isinstance(self._inner, AStarStrategy):
            # A* honours the deadline itself and returns the best complete
            # plan found; no thread is needed and nothing is left running.
            return astar_plan(
                start,
                goal,
                actions,
                blacklisted_actions=blacklisted_actions,
                time_budget_ms=self._budget_ms,
            )

        import concurrent.futures
        import warnings

        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="langgoap-anytime"
        )
        try:
            # Forward repair kwargs only when meaningful so legacy inner
            # strategies that predate the widened Protocol keep working.
            if prior_plan is not None:
                future = pool.submit(
                    self._inner.plan,
                    start,
                    goal,
                    actions,
                    blacklisted_actions=blacklisted_actions,
                    prior_plan=prior_plan,
                    current_step=current_step,
                )
            else:
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
                warnings.warn(
                    f"AnytimePlanningStrategy: inner strategy "
                    f"{type(self._inner).__name__} exceeded the "
                    f"{self._budget_ms:g} ms budget; returning None. A "
                    "black-box strategy cannot be interrupted, so it keeps "
                    "running in a background thread until it finishes.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
        finally:
            # Never block the caller on the worker; drop queued work.
            pool.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "AStarStrategy",
    "AnytimePlanningStrategy",
    "CSPRefinementStrategy",
    "LazyDecompositionStrategy",
    "PlanningStrategy",
    "TwoPhasePipelineStrategy",
    "rebind_inner_tracer",
    "refine_with_csp",
]
