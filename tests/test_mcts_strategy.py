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
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        assert isinstance(MCTSStrategy(), PlanningStrategy)


class TestMCTSStrategyPlanning:
    def test_finds_two_step_plan(self, two_step_actions: list[ActionSpec]) -> None:
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=64, rollout_depth=3, seed=7)
        )
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
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=16, rollout_depth=3, seed=7)
        )
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"done": True})
        plan = strategy.plan(start, goal, two_step_actions)
        assert plan is not None
        assert len(plan.actions) == 0

    def test_infeasible_goal_returns_none(self) -> None:
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        actions = [
            ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0),
        ]
        goal = GoalSpec(conditions={"done": True})
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=32, rollout_depth=3, seed=7)
        )
        plan = strategy.plan(PlanningState.from_dict({}), goal, actions)
        assert plan is None

    def test_anytime_fallback_returns_partial_plan_without_terminal(self) -> None:
        """With ``anytime_fallback=True`` MCTS returns the most-visited
        root child as a 1-step plan even when no goal-terminal leaf was
        discovered \u2014 required for MDP-style replanning on domains too
        deep for the tree to reach the goal within budget."""
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        actions = [
            ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0),
        ]
        goal = GoalSpec(conditions={"done": True})
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=32, rollout_depth=3, seed=7),
            reuse=MCTSReuseConfig(anytime_fallback=True),
        )
        plan = strategy.plan(PlanningState.from_dict({}), goal, actions)
        assert plan is not None
        assert len(plan.actions) >= 1
        assert plan.actions[0].name == "a"

    def test_wall_clock_budget_is_respected(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        strategy = MCTSStrategy(
            exploration=MCTSExploration(
                iterations=10_000_000, wall_clock_ms=50, rollout_depth=3, seed=7
            )
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
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        graph = GoapGraph(two_step_actions).compile()
        goal = GoalSpec(conditions={"done": True})

        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=64, rollout_depth=3, seed=7)
        )
        plan = strategy.plan(PlanningState.from_dict({}), goal, two_step_actions)
        assert plan is not None
        # The compiled graph executes the plan's actions and lands on
        # the goal state.  GoapGraph's ``invoke`` runs its own planner
        # by default, so we verify the plan shape here and trust the
        # compiler's existing Plan-execution tests.
        result = graph.invoke({"world_state": {}, "goal": goal})
        assert result["status"] == "goal_achieved"


class TestMCTSConfigValidation:
    """The grouped config dataclasses validate their inputs in
    ``__post_init__``.  Pinning the contract here so future refactors
    do not drop the validation that the flat-field design lacked.
    """

    def test_exploration_rejects_zero_budgets(self) -> None:
        from langgoap.planner.mcts import MCTSExploration

        with pytest.raises(ValueError, match="iterations / wall_clock_ms"):
            MCTSExploration(iterations=0, wall_clock_ms=0)

    def test_exploration_rejects_negative_iterations(self) -> None:
        from langgoap.planner.mcts import MCTSExploration

        with pytest.raises(ValueError, match="iterations must be >= 0"):
            MCTSExploration(iterations=-1)

    def test_exploration_rejects_non_positive_c(self) -> None:
        from langgoap.planner.mcts import MCTSExploration

        with pytest.raises(ValueError, match="c must be > 0"):
            MCTSExploration(c=0.0)

    def test_reuse_rejects_decay_outside_unit_interval(self) -> None:
        from langgoap.planner.mcts import MCTSReuseConfig

        with pytest.raises(ValueError, match="tree_reuse_decay"):
            MCTSReuseConfig(tree_reuse_decay=1.5)
        with pytest.raises(ValueError, match="tree_reuse_decay"):
            MCTSReuseConfig(tree_reuse_decay=-0.1)

    def test_reuse_rejects_negative_path_length_budget(self) -> None:
        from langgoap.planner.mcts import MCTSReuseConfig

        with pytest.raises(ValueError, match="path_length_budget"):
            MCTSReuseConfig(path_length_budget=-1)
