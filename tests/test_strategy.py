"""Tests for the PlanningStrategy hierarchy."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

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
        with pytest.warns(DeprecationWarning, match="refine_with_csp"):
            refiner = CSPRefinementStrategy(candidate)
        refined = refiner.plan(start, constrained, _simple_actions())
        assert refined is not None
        assert isinstance(refined.score, HardSoftScore)
        assert refined.metadata.csp is not None

    def test_runtime_checkable_protocol(self) -> None:
        # Needs a candidate Plan
        empty = Plan(actions=(), expected_states=(), total_cost=0.0)
        with pytest.warns(DeprecationWarning):
            refiner = CSPRefinementStrategy(empty)
        assert isinstance(refiner, PlanningStrategy)


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


class TestRefineWithCsp:
    """``refine_with_csp`` is the helper that replaces ``CSPRefinementStrategy`` (#63)."""

    def test_refines_candidate_against_constrained_goal(self) -> None:
        from langgoap.planner.strategy import refine_with_csp

        plain_goal = GoalSpec(conditions=MappingProxyType({"done": True}))
        start = PlanningState.from_dict({})
        candidate = AStarStrategy().plan(start, plain_goal, _simple_actions())
        assert candidate is not None

        constrained = GoalSpec(
            conditions=MappingProxyType({"done": True}),
            constraints=(ConstraintSpec(key="cost_usd", max=10.0),),
        )
        refined = refine_with_csp(candidate, constrained)

        assert refined.action_names == candidate.action_names
        assert isinstance(refined.score, HardSoftScore)
        assert refined.metadata.csp is not None

    def test_exported_from_the_package(self) -> None:
        import langgoap

        assert "refine_with_csp" in langgoap.__all__
        assert callable(langgoap.refine_with_csp)


class TestExplicitStrategiesKeepTracing:
    """Passing the default strategies explicitly must not drop search tracing (#63)."""

    @staticmethod
    def _state() -> GoapState:
        return {
            "world_state": {},
            "goal": GoalSpec(conditions=MappingProxyType({"done": True})),
        }

    @staticmethod
    def _expansions(rec: Any) -> list[Any]:
        return [args for name, args in rec.calls if name == "on_search_expand"]

    def test_astar_strategy_uses_the_planner_tracer(self) -> None:
        from tests.test_tracing import RecordingTracer

        rec = RecordingTracer()
        planner = GoapPlanner(
            _simple_actions(),
            strategy=AStarStrategy(),
            tracer=rec,
            record_expansions=True,
        )
        updates = planner(self._state())

        assert updates["plan"] is not None
        assert self._expansions(rec)

    def test_two_phase_strategy_uses_the_planner_tracer(self) -> None:
        from tests.test_tracing import RecordingTracer

        rec = RecordingTracer()
        planner = GoapPlanner(
            _simple_actions(),
            strategy=TwoPhasePipelineStrategy(),
            tracer=rec,
            record_expansions=True,
        )
        updates = planner(self._state())

        assert updates["plan"] is not None
        assert self._expansions(rec)

    def test_strategy_with_its_own_tracer_keeps_it(self) -> None:
        from tests.test_tracing import RecordingTracer

        own, other = RecordingTracer(), RecordingTracer()
        planner = GoapPlanner(
            _simple_actions(),
            strategy=AStarStrategy(tracer=own, record_expansions=True),
            tracer=other,
            record_expansions=True,
        )
        planner(self._state())

        assert self._expansions(own)
        assert not self._expansions(other)

    def test_wrapper_strategies_forward_to_their_inner_search(self) -> None:
        from langgoap.planner.strategy import LazyDecompositionStrategy
        from tests.test_tracing import RecordingTracer

        rec = RecordingTracer()
        planner = GoapPlanner(
            _simple_actions(),
            strategy=LazyDecompositionStrategy(lookahead=2),
            tracer=rec,
            record_expansions=True,
        )
        updates = planner(self._state())

        assert updates["plan"] is not None
        assert self._expansions(rec)
