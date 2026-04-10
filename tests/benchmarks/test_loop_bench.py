"""End-to-end GOAP loop benchmarks.

Measures GoapGraph.invoke() — the complete plan→execute→observe cycle
including LangGraph node dispatch, state merging, and Command routing.

These are integration-level benchmarks: they show the overhead that users
actually experience, not just the planner in isolation.  LangGraph's own
graph compilation and state machinery is included in every measurement.

Scenarios
---------
Short plan (3 steps)   — baseline overhead per loop iteration
Long plan  (10 steps)  — checks executor + observer scaling with plan length
MultiGoal sequential   — 2, 4 sub-goals in order
MultiGoal any          — picks cheapest from 3 alternatives

All actions execute synchronously (no real I/O) by leaving execute=None,
which applies effects directly and returns ActionResult(success=True).
"""

from __future__ import annotations

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_graph(n: int) -> tuple[GoapGraph, GoalSpec, dict]:
    """Return (graph, goal, world_state) for an n-step linear chain."""
    actions = [
        ActionSpec(
            name=f"step_{i}",
            preconditions={f"step_{i - 1}": True} if i > 0 else {},
            effects={f"step_{i}": True},
            cost=1.0,
        )
        for i in range(n)
    ]
    goal = GoalSpec(conditions={f"step_{n - 1}": True})
    return GoapGraph(actions=actions), goal, {}


# ---------------------------------------------------------------------------
# Single GoalSpec — baseline loop overhead
# ---------------------------------------------------------------------------


def test_bench_loop_3_steps(benchmark: object) -> None:
    """Full GoapGraph.invoke() for a 3-step linear plan."""
    graph, goal, ws = _chain_graph(3)
    result = benchmark(graph.invoke, goal=goal, world_state=ws)  # type: ignore[call-arg]
    assert result["status"] == "goal_achieved"


def test_bench_loop_10_steps(benchmark: object) -> None:
    """Full GoapGraph.invoke() for a 10-step linear plan."""
    graph, goal, ws = _chain_graph(10)
    result = benchmark(graph.invoke, goal=goal, world_state=ws)  # type: ignore[call-arg]
    assert result["status"] == "goal_achieved"


def test_bench_loop_20_steps(benchmark: object) -> None:
    """Full GoapGraph.invoke() for a 20-step linear plan."""
    graph, goal, ws = _chain_graph(20)
    result = benchmark(graph.invoke, goal=goal, world_state=ws)  # type: ignore[call-arg]
    assert result["status"] == "goal_achieved"


# ---------------------------------------------------------------------------
# MultiGoal sequential — sub-goal chaining overhead
# ---------------------------------------------------------------------------


def _sequential_multi_goal(n_subgoals: int) -> tuple[GoapGraph, object, dict]:
    """Return (graph, MultiGoal, world_state) for n sequential sub-goals.

    Each sub-goal is a 2-step chain independent of the others.
    """
    from langgoap.goals import MultiGoal

    all_actions: list[ActionSpec] = []
    sub_goals: list[GoalSpec] = []

    for g in range(n_subgoals):
        a0 = ActionSpec(
            name=f"sg{g}_step0",
            preconditions={},
            effects={f"sg{g}_ready": True},
            cost=1.0,
        )
        a1 = ActionSpec(
            name=f"sg{g}_step1",
            preconditions={f"sg{g}_ready": True},
            effects={f"sg{g}_done": True},
            cost=1.0,
        )
        all_actions.extend([a0, a1])
        sub_goals.append(GoalSpec(conditions={f"sg{g}_done": True}))

    goal = MultiGoal(goals=tuple(sub_goals), mode="sequential")
    return GoapGraph(actions=all_actions), goal, {}


@pytest.mark.parametrize("n", [2, 4])
def test_bench_loop_multigoal_sequential(benchmark: object, n: int) -> None:
    """GoapGraph.invoke() for a sequential MultiGoal with n 2-step sub-goals."""
    graph, goal, ws = _sequential_multi_goal(n)
    result = benchmark(graph.invoke, goal=goal, world_state=ws)  # type: ignore[call-arg]
    assert result["status"] == "goal_achieved"


# ---------------------------------------------------------------------------
# MultiGoal any — cheapest-pick selection overhead
# ---------------------------------------------------------------------------


def test_bench_loop_multigoal_any(benchmark: object) -> None:
    """GoapGraph.invoke() for any-mode MultiGoal with 3 alternatives.

    Sub-goal 0 is cheapest (1 step); sub-goals 1 and 2 require 2 steps.
    The planner must enumerate all three and commit to sub-goal 0.
    """
    from langgoap.goals import MultiGoal

    actions = [
        # sub-goal 0 — 1 step, cheapest
        ActionSpec(name="quick", preconditions={}, effects={"quick_done": True}, cost=1.0),
        # sub-goal 1 — 2 steps
        ActionSpec(name="slow_a", preconditions={}, effects={"slow_ready": True}, cost=1.0),
        ActionSpec(name="slow_b", preconditions={"slow_ready": True}, effects={"slow_done": True}, cost=1.0),
        # sub-goal 2 — 2 steps
        ActionSpec(name="alt_a", preconditions={}, effects={"alt_ready": True}, cost=1.0),
        ActionSpec(name="alt_b", preconditions={"alt_ready": True}, effects={"alt_done": True}, cost=1.0),
    ]
    goal = MultiGoal(
        goals=(
            GoalSpec(conditions={"quick_done": True}),
            GoalSpec(conditions={"slow_done": True}),
            GoalSpec(conditions={"alt_done": True}),
        ),
        mode="any",
    )
    graph = GoapGraph(actions=actions)
    result = benchmark(graph.invoke, goal=goal, world_state={})  # type: ignore[call-arg]
    assert result["status"] == "goal_achieved"
    assert result["world_state"].get("quick_done") is True
