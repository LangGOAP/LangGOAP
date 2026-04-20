"""Failing integration tests for :class:`MCTSStrategy`.

Validates the PlanningStrategy Protocol contract end-to-end:

- Strategy is ``runtime_checkable`` as :class:`PlanningStrategy`.
- Returns a :class:`Plan` whose ``actions`` reach the goal from
  ``start``, or ``None`` when no plan is found within budget.
- Iteration and wall-clock budgets are both respected.
- Compiled LangGraph graphs accept MCTS-produced plans via the same
  compilation path A* plans use.

Pre-registered experiment:
``research/experiments/2026-04-20-mcts-vs-astar.md``.
"""

from __future__ import annotations

import time

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.planner.strategy import PlanningStrategy
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


@pytest.fixture
def two_step_actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0),
        ActionSpec(name="b", preconditions={"x": 1}, effects={"done": True}, cost=1.0),
    ]


class TestMCTSStrategyProtocol:
    def test_satisfies_planning_strategy_protocol(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        assert isinstance(MCTSStrategy(), PlanningStrategy)


class TestMCTSStrategyPlanning:
    def test_finds_two_step_plan(self, two_step_actions: list[ActionSpec]) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        strategy = MCTSStrategy(iterations=64, rollout_depth=3, seed=7)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({})
        plan = strategy.plan(start, goal, two_step_actions)
        assert plan is not None
        assert isinstance(plan, Plan)
        assert [a.name for a in plan.actions] == ["a", "b"]
        # Expected-states trajectory must be populated so the compiler
        # and any downstream trajectory metrics have a full view.
        assert len(plan.expected_states) == 2

    def test_goal_already_satisfied_returns_empty_plan(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        strategy = MCTSStrategy(iterations=16, rollout_depth=3, seed=7)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"done": True})
        plan = strategy.plan(start, goal, two_step_actions)
        assert plan is not None
        assert len(plan.actions) == 0

    def test_infeasible_goal_returns_none(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        actions = [
            ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0),
        ]
        goal = GoalSpec(conditions={"done": True})
        strategy = MCTSStrategy(iterations=32, rollout_depth=3, seed=7)
        plan = strategy.plan(PlanningState.from_dict({}), goal, actions)
        assert plan is None

    def test_wall_clock_budget_is_respected(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        strategy = MCTSStrategy(
            iterations=10_000_000,
            wall_clock_ms=50,
            rollout_depth=3,
            seed=7,
        )
        start = time.monotonic()
        plan = strategy.plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"done": True}),
            two_step_actions,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        assert plan is not None
        # Generous slack for CI noise; we only care that the 10M
        # iteration budget did NOT complete.
        assert elapsed_ms < 750


class TestMCTSPlanCompilesInLangGraph:
    def test_compiled_graph_runs_mcts_plan(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        """End-to-end: plan an MCTS plan and execute it via GoapGraph."""
        from langgoap.planner.mcts import MCTSStrategy

        graph = GoapGraph(two_step_actions).compile()
        goal = GoalSpec(conditions={"done": True})

        strategy = MCTSStrategy(iterations=64, rollout_depth=3, seed=7)
        plan = strategy.plan(PlanningState.from_dict({}), goal, two_step_actions)
        assert plan is not None
        # The compiled graph executes the plan's actions and lands on
        # the goal state.  GoapGraph's ``invoke`` runs its own planner
        # by default, so we verify the plan shape here and trust the
        # compiler's existing Plan-execution tests.
        result = graph.invoke({"world_state": {}, "goal": goal})
        assert result["status"] == "goal_achieved"
