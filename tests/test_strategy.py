"""Tests for the PlanningStrategy hierarchy."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.graph.nodes import GoapPlanner
from langgoap.graph.state import GoapState
from langgoap.planner.strategy import (
    AStarStrategy,
    CSPRefinementStrategy,
    PlanningStrategy,
    TwoPhasePipelineStrategy,
)
from langgoap.planner.types import Plan
from langgoap.score import HardSoftScore, SimpleScore
from langgoap.state import PlanningState
from langgoap.types import ObjectiveDirection


def _simple_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="start",
            preconditions={},
            effects={"ready": True},
            cost=1.0,
        ),
        ActionSpec(
            name="finish",
            preconditions={"ready": True},
            effects={"done": True},
            cost=1.0,
            resources={"cost_usd": 1.0},
        ),
    ]


class TestAStarStrategy:
    def test_plans_without_csp(self) -> None:
        strategy = AStarStrategy()
        goal = GoalSpec(conditions=MappingProxyType({"done": True}))
        start = PlanningState.from_dict({})
        result = strategy.plan(start, goal, _simple_actions())
        assert result is not None
        assert result.action_names == ["start", "finish"]
        # Pure A* → SimpleScore populated from total_cost
        assert isinstance(result.score, SimpleScore)
        assert result.score.scalar == result.total_cost

    def test_runtime_checkable_protocol(self) -> None:
        assert isinstance(AStarStrategy(), PlanningStrategy)


class TestTwoPhasePipelineStrategy:
    def test_plans_with_csp_constraints(self) -> None:
        strategy = TwoPhasePipelineStrategy()
        goal = GoalSpec(
            conditions=MappingProxyType({"done": True}),
            constraints=(ConstraintSpec(key="cost_usd", max=10.0),),
        )
        start = PlanningState.from_dict({})
        result = strategy.plan(start, goal, _simple_actions())
        assert result is not None
        assert result.action_names == ["start", "finish"]
        # Pipeline should replace SimpleScore with HardSoftScore.
        assert isinstance(result.score, HardSoftScore)
        # Feasible plan → hard == 0
        assert result.score.hard == 0.0
        assert result.score.is_feasible()

    def test_soft_constraint_populates_soft_score(self) -> None:
        strategy = TwoPhasePipelineStrategy()
        # Budget of 0.5 but plan costs 1.0 → 0.5 violation
        goal = GoalSpec(
            conditions=MappingProxyType({"done": True}),
            constraints=(
                ConstraintSpec(key="cost_usd", max=0.5, weight=2.0, level="soft"),
            ),
        )
        start = PlanningState.from_dict({})
        result = strategy.plan(start, goal, _simple_actions())
        assert result is not None
        assert isinstance(result.score, HardSoftScore)
        # Hard should be 0 (no hard violations); soft should reflect 0.5 * 2.0 penalty
        assert result.score.hard == 0.0
        assert result.score.soft == -1.0
        assert result.score.is_feasible()

    def test_hard_violation_populates_hard_score(self) -> None:
        strategy = TwoPhasePipelineStrategy()
        # Budget of 0.5 but plan costs 1.0 → hard violation
        goal = GoalSpec(
            conditions=MappingProxyType({"done": True}),
            constraints=(
                ConstraintSpec(key="cost_usd", max=0.5, weight=1.0, level="hard"),
            ),
        )
        start = PlanningState.from_dict({})
        result = strategy.plan(start, goal, _simple_actions())
        assert result is not None
        assert isinstance(result.score, HardSoftScore)
        # Hard should be negative; plan is infeasible
        assert result.score.hard < 0.0
        assert not result.score.is_feasible()

    def test_minimize_objective_subtracts_from_soft(self) -> None:
        strategy = TwoPhasePipelineStrategy()
        goal = GoalSpec(
            conditions=MappingProxyType({"done": True}),
            objectives=MappingProxyType({"cost_usd": ObjectiveDirection.MINIMIZE}),
        )
        start = PlanningState.from_dict({})
        result = strategy.plan(start, goal, _simple_actions())
        assert result is not None
        assert isinstance(result.score, HardSoftScore)
        # cost_usd total is 1.0; minimize → soft -= 1.0
        assert result.score.soft == -1.0

    def test_runtime_checkable_protocol(self) -> None:
        assert isinstance(TwoPhasePipelineStrategy(), PlanningStrategy)


class TestCSPRefinementStrategy:
    def test_refines_candidate_plan(self) -> None:
        # First build a candidate via A*
        astar = AStarStrategy()
        plain_goal = GoalSpec(conditions=MappingProxyType({"done": True}))
        start = PlanningState.from_dict({})
        candidate = astar.plan(start, plain_goal, _simple_actions())
        assert candidate is not None

        # Now refine against a constrained goal
        constrained = GoalSpec(
            conditions=MappingProxyType({"done": True}),
            constraints=(ConstraintSpec(key="cost_usd", max=10.0),),
        )
        refiner = CSPRefinementStrategy(candidate)
        refined = refiner.plan(start, constrained, _simple_actions())
        assert refined is not None
        assert isinstance(refined.score, HardSoftScore)
        assert refined.metadata.csp is not None

    def test_runtime_checkable_protocol(self) -> None:
        # Needs a candidate Plan
        empty = Plan(actions=(), expected_states=(), total_cost=0.0)
        assert isinstance(CSPRefinementStrategy(empty), PlanningStrategy)


class TestGoapPlannerStrategyKwarg:
    def test_custom_strategy_overrides_auto_routing(self) -> None:
        # A sentinel strategy that tracks whether it was called.
        call_count = {"n": 0}

        class SentinelStrategy:
            def plan(
                self,
                start: PlanningState,
                goal: GoalSpec,
                actions: list[ActionSpec],
                *,
                blacklisted_actions: list[str] | None = None,
            ) -> Plan | None:
                call_count["n"] += 1
                return AStarStrategy().plan(
                    start, goal, actions, blacklisted_actions=blacklisted_actions
                )

        actions = _simple_actions()
        planner = GoapPlanner(actions, strategy=SentinelStrategy())
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions=MappingProxyType({"done": True})),
        }
        updates = planner(state)
        assert call_count["n"] == 1
        assert updates["plan"] is not None

    def test_default_still_uses_auto_routing(self) -> None:
        # No strategy kwarg — existing behavior preserved.
        actions = _simple_actions()
        planner = GoapPlanner(actions)
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions=MappingProxyType({"done": True})),
        }
        updates = planner(state)
        assert updates["plan"] is not None
