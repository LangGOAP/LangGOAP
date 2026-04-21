"""Failing tests for pluggable ``ScalarHeuristic`` on MCTS.

Exercises the continuous-reward path for MCTS rollouts.  The
default goal-condition-count reward shape quantises to at most
``|goal.conditions| + 1`` distinct values, which starves UCB1 of
discriminative signal on GOAP problems with sparse terminal
goals (e.g. food-clearing).  ``ScalarHeuristic`` lets callers
supply a continuous ``(state, goal) -> float in [-1, +1]`` value
function that is consumed at rollout leaves (and any non-terminal
stop) so UCB1 exploitation terms separate.

Related plan doc: ``research/plans/mcts-scalar-heuristic.md``.
Related experiment:
``research/experiments/2026-04-21-mcts-scalar-heuristic.md``.
"""

from __future__ import annotations

import logging
import random

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.state import PlanningState


@pytest.fixture
def count_actions() -> list[ActionSpec]:
    """Three actions that each decrement a ``remaining`` counter."""

    def _dec(ws: dict) -> dict:
        remaining = int(ws.get("remaining", 0))
        return {"remaining": max(0, remaining - 1), "done": (remaining - 1) <= 0}

    return [
        ActionSpec(
            name=f"step_{i}",
            preconditions={},
            effects=_dec,
            effect_keys=frozenset({"remaining", "done"}),
            cost=1.0,
        )
        for i in range(3)
    ]


@pytest.fixture
def count_goal() -> GoalSpec:
    return GoalSpec(conditions={"done": True})


@pytest.fixture
def count_start() -> PlanningState:
    return PlanningState.from_dict({"remaining": 20, "done": False})


class TestScalarHeuristicProtocol:
    def test_is_runtime_checkable(self) -> None:
        from langgoap.planner.mcts import ScalarHeuristic

        def h(state: PlanningState, goal: GoalSpec) -> float:
            return 0.0

        assert isinstance(h, ScalarHeuristic)


class TestShapeRewardBackwardCompat:
    def test_default_shape_reward_unchanged_when_heuristic_is_none(
        self, count_goal: GoalSpec, count_start: PlanningState
    ) -> None:
        from langgoap.planner.mcts import _shape_reward

        assert _shape_reward(count_start, count_goal) == pytest.approx(-1.0)

    def test_terminal_returns_plus_one_without_heuristic(
        self, count_goal: GoalSpec
    ) -> None:
        from langgoap.planner.mcts import _shape_reward

        satisfied = PlanningState.from_dict({"done": True})
        assert _shape_reward(satisfied, count_goal) == pytest.approx(1.0)


class TestShapeRewardWithHeuristic:
    def test_heuristic_value_propagates_to_reward(
        self, count_goal: GoalSpec
    ) -> None:
        from langgoap.planner.mcts import _shape_reward

        def half_done(state: PlanningState, goal: GoalSpec) -> float:
            remaining = int(state.to_dict().get("remaining", 0))
            return 1.0 - remaining / 20.0  # 0.0 at start, 1.0 empty

        mid = PlanningState.from_dict({"remaining": 10, "done": False})
        # Heuristic yields 0.5; must pass through shape_reward (clamped
        # below 1.0 so we can still distinguish true satisfaction).
        reward = _shape_reward(mid, count_goal, heuristic=half_done)
        assert 0.4 <= reward <= 0.5

    def test_terminal_overrides_heuristic(self, count_goal: GoalSpec) -> None:
        from langgoap.planner.mcts import _shape_reward

        def always_negative(state: PlanningState, goal: GoalSpec) -> float:
            return -1.0

        satisfied = PlanningState.from_dict({"done": True})
        assert _shape_reward(satisfied, count_goal, heuristic=always_negative) == 1.0

    def test_values_above_plus_one_are_clamped_below_one(
        self, count_goal: GoalSpec, count_start: PlanningState
    ) -> None:
        from langgoap.planner.mcts import _shape_reward

        def too_high(state: PlanningState, goal: GoalSpec) -> float:
            return 42.0

        r = _shape_reward(count_start, count_goal, heuristic=too_high)
        assert r < 1.0
        assert r >= 0.99

    def test_values_below_minus_one_are_clamped(
        self, count_goal: GoalSpec, count_start: PlanningState
    ) -> None:
        from langgoap.planner.mcts import _shape_reward

        def too_low(state: PlanningState, goal: GoalSpec) -> float:
            return -42.0

        assert _shape_reward(count_start, count_goal, heuristic=too_low) == -1.0

    def test_heuristic_exception_falls_back_to_default(
        self,
        count_goal: GoalSpec,
        count_start: PlanningState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from langgoap.planner.mcts import _shape_reward

        def broken(state: PlanningState, goal: GoalSpec) -> float:
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger="langgoap.planner.mcts"):
            r = _shape_reward(count_start, count_goal, heuristic=broken)
        assert r == pytest.approx(-1.0)  # default fallback
        assert any("scalar heuristic" in rec.message.lower() for rec in caplog.records)


class TestRolloutPoliciesWithHeuristic:
    def test_heuristic_rollout_uses_scalar_heuristic(
        self,
        count_actions: list[ActionSpec],
        count_goal: GoalSpec,
        count_start: PlanningState,
    ) -> None:
        from langgoap.planner.mcts import HeuristicRollout

        def progress(state: PlanningState, goal: GoalSpec) -> float:
            remaining = int(state.to_dict().get("remaining", 0))
            return 1.0 - remaining / 20.0

        flat = HeuristicRollout(max_depth=3).rollout(
            state=count_start, goal=count_goal, actions=count_actions
        )
        shaped = HeuristicRollout(max_depth=3, scalar_heuristic=progress).rollout(
            state=count_start, goal=count_goal, actions=count_actions
        )
        # Default -> constant -1.0; scalar-heuristic -> ~0.15 after 3 steps.
        assert shaped > flat

    def test_random_rollout_uses_scalar_heuristic(
        self,
        count_actions: list[ActionSpec],
        count_goal: GoalSpec,
        count_start: PlanningState,
    ) -> None:
        from langgoap.planner.mcts import RandomRollout

        def progress(state: PlanningState, goal: GoalSpec) -> float:
            remaining = int(state.to_dict().get("remaining", 0))
            return 1.0 - remaining / 20.0

        rng = random.Random(7)
        shaped = RandomRollout(
            max_depth=3, rng=rng, scalar_heuristic=progress
        ).rollout(state=count_start, goal=count_goal, actions=count_actions)
        assert -1.0 <= shaped <= 1.0
        assert shaped > -1.0  # default without heuristic would be -1.0 exactly


class TestMCTSStrategyWithHeuristic:
    def test_strategy_accepts_scalar_heuristic_kwarg(
        self,
        count_actions: list[ActionSpec],
        count_goal: GoalSpec,
    ) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        def progress(state: PlanningState, goal: GoalSpec) -> float:
            remaining = int(state.to_dict().get("remaining", 0))
            return 1.0 - remaining / 4.0

        start = PlanningState.from_dict({"remaining": 4, "done": False})
        strategy = MCTSStrategy(
            iterations=64,
            rollout_depth=4,
            seed=7,
            scalar_heuristic=progress,
        )
        plan = strategy.plan(start, count_goal, count_actions)
        assert plan is not None
        assert len(plan.actions) >= 1

    def test_heuristic_improves_discrimination_over_default(
        self,
        count_actions: list[ActionSpec],
        count_goal: GoalSpec,
    ) -> None:
        """On a problem where default reward is uniformly -1.0, the
        scalar-heuristic strategy must reach the goal within the same
        budget while the flat-reward one cannot discriminate children."""
        from langgoap.planner.mcts import MCTSStrategy

        def progress(state: PlanningState, goal: GoalSpec) -> float:
            remaining = int(state.to_dict().get("remaining", 0))
            return 1.0 - remaining / 4.0  # start gives 0.0, one step 0.25, ...

        start = PlanningState.from_dict({"remaining": 4, "done": False})

        shaped = MCTSStrategy(
            iterations=128, rollout_depth=2, seed=7, scalar_heuristic=progress
        ).plan(start, count_goal, count_actions)
        assert shaped is not None and len(shaped.actions) == 4

    def test_exports_scalar_heuristic_from_public_api(self) -> None:
        from langgoap.planner.mcts import ScalarHeuristic as FromModule
        from langgoap.planner import ScalarHeuristic as FromPackage

        assert FromModule is FromPackage
