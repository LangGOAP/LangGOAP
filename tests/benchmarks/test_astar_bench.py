"""Benchmarks for the A* GOAP planner.

Four axes of scaling are measured:

1. Plan depth  — linear chain of N actions; tests O(N) path expansion.
2. Noise pool  — fixed 3-step solution buried in N irrelevant actions;
                 tests the reachability pre-check and action sorting.
3. Branching   — fan_out^depth alternative paths; tests backtracking cost.
4. Special paths — already-satisfied goal (instant return) and unreachable
                  goal (pre-check fast-fail).

All inputs are fully synthetic — no LLM calls, no I/O.
The `plan()` public API is benchmarked (not the internal `_search`) so the
timings include the reachability pre-check, sorting, and two-pass
optimisation, reflecting real caller behaviour.
"""

from __future__ import annotations

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan
from langgoap.state import PlanningState
from tests.benchmarks.conftest import branching, linear_chain, wide_pool

# ---------------------------------------------------------------------------
# 1. Plan depth scaling — linear chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 10, 20, 50])
def test_bench_astar_linear_depth(benchmark: object, n: int) -> None:
    """A* on a linear n-step chain — single optimal path, no backtracking.

    Scaling should be roughly O(n): one node expanded per step.
    """
    actions, goal, start = linear_chain(n)
    result = benchmark(plan, start, goal, actions)  # type: ignore[call-arg]
    assert result is not None, f"Expected plan for chain depth={n}"
    assert len(result.actions) == n


# ---------------------------------------------------------------------------
# 2. Noise pool — reachability pre-check stress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_noise", [10, 50, 100, 500])
def test_bench_astar_noise_pool(benchmark: object, n_noise: int) -> None:
    """A* with n_noise irrelevant actions — tests pre-check and sort cost.

    The noise actions require impossible preconditions so the planner's
    reachability check eliminates them before search.  The remaining
    3-action path is always found instantly.
    """
    actions, goal, start = wide_pool(n_noise=n_noise, path_depth=3)
    result = benchmark(plan, start, goal, actions)  # type: ignore[call-arg]
    assert result is not None, f"Expected plan with {n_noise} noise actions"
    assert len(result.actions) == 3


# ---------------------------------------------------------------------------
# 3. Branching — alternative-path explosion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth,fan_out", [(3, 2), (4, 2), (3, 3), (2, 4)])
def test_bench_astar_branching(benchmark: object, depth: int, fan_out: int) -> None:
    """A* with fan_out^depth alternative paths.

    Verifies A* still terminates quickly with breadth-first exploration
    of many equally-cheap routes.
    """
    actions, goal, start = branching(depth=depth, fan_out=fan_out)
    result = benchmark(plan, start, goal, actions)  # type: ignore[call-arg]
    assert result is not None, f"Expected plan for depth={depth} fan_out={fan_out}"


# ---------------------------------------------------------------------------
# 4. Special paths
# ---------------------------------------------------------------------------


def test_bench_astar_already_satisfied(benchmark: object) -> None:
    """Goal already satisfied in start state — must return an empty plan instantly."""
    actions, _, _ = linear_chain(10)
    goal = GoalSpec(conditions={"already_done": True})
    start = PlanningState.from_dict({"already_done": True})
    result = benchmark(plan, start, goal, actions)  # type: ignore[call-arg]
    assert result is not None
    assert len(result.actions) == 0


def test_bench_astar_unreachable_goal(benchmark: object) -> None:
    """Goal is unreachable — reachability pre-check must reject without search."""
    actions, _, start = linear_chain(10)
    goal = GoalSpec(conditions={"impossible_key": True})
    result = benchmark(plan, start, goal, actions)  # type: ignore[call-arg]
    assert result is None


def test_bench_astar_blacklist_fallback(benchmark: object) -> None:
    """Blacklist forces the fallback path where all actions are tried again.

    Benchmarks the cost of the two-call pattern inside astar.plan() when
    the blacklist would otherwise block the only solution.
    """
    actions, goal, start = linear_chain(5)
    # Blacklist the first action — fallback re-tries with empty blacklist
    blacklisted = [actions[0].name]

    def run() -> object:
        return plan(start, goal, actions, blacklisted_actions=blacklisted)

    result = benchmark(run)  # type: ignore[call-arg]
    assert result is not None
    assert len(result.actions) == 5
