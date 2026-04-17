"""Benchmarks for the CSP optimizer (OR-Tools CP-SAT back-end).

Two public APIs are measured:

validate_plan(plan, goal)
    Pure-Python fast path: sums resources, evaluates constraints, returns
    CSPMetadata without invoking the solver.  Should be sub-millisecond.

optimize_plans(plans, goal)
    CP-SAT path: selects the best plan from a pool under hard and soft
    constraints.  Solver time dominates; measures end-to-end latency
    including Python↔solver marshalling.

Skips automatically when OR-Tools is not installed so the benchmark suite
runs on environments without the optional extra.
"""

from __future__ import annotations

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, ObjectiveDirection
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.csp import CSPStatus, optimize_plans, validate_plan
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from tests.benchmarks.conftest import linear_chain

# ---------------------------------------------------------------------------
# Helpers — build plans with resource annotations
# ---------------------------------------------------------------------------


def _resource_chain(n: int, cost_per_step: float = 0.1) -> tuple[Plan, GoalSpec]:
    """Build an n-step linear-chain plan where each action burns cost_per_step."""
    actions_spec = [
        ActionSpec(
            name=f"step_{i}",
            preconditions={f"step_{i - 1}": True} if i > 0 else {},
            effects={f"step_{i}": True},
            cost=1.0,
            resources={"cost_usd": cost_per_step, "tokens": 100},
        )
        for i in range(n)
    ]
    start = PlanningState.from_dict({})
    goal = GoalSpec(conditions={f"step_{n - 1}": True})
    result = astar_plan(start, goal, actions_spec)
    assert result is not None
    return result, goal


def _constrained_goal(n_hard: int, n_soft: int, budget: float = 999.0) -> GoalSpec:
    """GoalSpec with n_hard hard constraints and n_soft soft constraints."""
    from langgoap.goals import ConstraintSpec

    hard = tuple(
        ConstraintSpec(key=f"hard_{i}", max=budget, level="hard") for i in range(n_hard)
    )
    soft = tuple(
        ConstraintSpec(key=f"soft_{i}", max=budget, level="soft", weight=0.5)
        for i in range(n_soft)
    )
    return GoalSpec(conditions={"done": True}, constraints=hard + soft)


# ---------------------------------------------------------------------------
# validate_plan — pure-Python fast path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [3, 10, 20])
def test_bench_validate_plan_no_constraints(benchmark: object, n_steps: int) -> None:
    """validate_plan with no constraints: pure resource aggregation."""
    p, goal = _resource_chain(n_steps)
    benchmark(validate_plan, p, goal)  # type: ignore[call-arg]


@pytest.mark.parametrize("n_hard", [1, 5, 10, 20])
def test_bench_validate_plan_hard_constraints(benchmark: object, n_hard: int) -> None:
    """validate_plan with n_hard hard constraints — all satisfied (no violation)."""
    p, _ = _resource_chain(5)
    goal = _constrained_goal(n_hard=n_hard, n_soft=0, budget=999.0)
    result = benchmark(validate_plan, p, goal)  # type: ignore[call-arg]
    assert result.status != CSPStatus.ERROR


@pytest.mark.parametrize("n_hard,n_soft", [(2, 2), (5, 5), (10, 5)])
def test_bench_validate_plan_mixed_constraints(
    benchmark: object, n_hard: int, n_soft: int
) -> None:
    """validate_plan with mixed hard + soft constraints."""
    p, _ = _resource_chain(5)
    goal = _constrained_goal(n_hard=n_hard, n_soft=n_soft, budget=999.0)
    benchmark(validate_plan, p, goal)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# optimize_plans — CP-SAT solver path
# ---------------------------------------------------------------------------


def _build_plan_pool(n_plans: int) -> list[Plan]:
    """Build n_plans linear-chain plans of increasing cost."""
    pool: list[Plan] = []
    for i in range(n_plans):
        depth = 3 + i  # each plan is slightly longer / more expensive
        p, _ = _resource_chain(depth, cost_per_step=0.1 * (i + 1))
        pool.append(p)
    return pool


@pytest.mark.parametrize("n_plans", [1, 3, 5])
def test_bench_optimize_plans_no_constraints(benchmark: object, n_plans: int) -> None:
    """optimize_plans with n_plans candidates and no constraints.

    Solver selects by lowest cost_usd; validates solver overhead scales
    sub-linearly with pool size.
    """
    pool = _build_plan_pool(n_plans)
    goal = GoalSpec(conditions={"step_2": True})
    result = benchmark(optimize_plans, pool, goal)  # type: ignore[call-arg]
    best, meta = result
    assert meta.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)


@pytest.mark.parametrize("n_plans,n_constraints", [(3, 2), (3, 10), (5, 5)])
def test_bench_optimize_plans_with_constraints(
    benchmark: object, n_plans: int, n_constraints: int
) -> None:
    """optimize_plans under n_constraints hard constraints."""
    pool = _build_plan_pool(n_plans)
    goal = _constrained_goal(n_hard=n_constraints, n_soft=0, budget=999.0)
    result = benchmark(optimize_plans, pool, goal)  # type: ignore[call-arg]
    best, meta = result
    assert meta.status != CSPStatus.ERROR
