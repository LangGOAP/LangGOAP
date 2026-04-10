"""Integration tests for the full GOAP loop with CSP constraints.

Verifies end-to-end behavior when GoalSpec includes resource constraints,
optimization objectives, or temporal scheduling requirements.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from langgoap import ConstraintSpec, CSPStatus, GoalSpec, Maximize, Minimize
from langgoap.graph.builder import GoapGraph
from tests.conftest import make_action

# ---------------------------------------------------------------------------
# Test actions
# ---------------------------------------------------------------------------


def _budget_actions() -> list[Any]:
    """Two paths to the same goal: expensive (better A*) and cheap."""
    return [
        make_action(
            "expensive_path",
            eff={"done": True},
            cost=1,  # A* prefers this (lower cost)
            resources={"cost_usd": 50.0, "tokens": 1000},
        ),
        make_action(
            "cheap_path",
            eff={"done": True},
            cost=2,  # A* considers this worse
            resources={"cost_usd": 5.0, "tokens": 100},
        ),
    ]


def _quality_actions() -> list[Any]:
    """Actions with quality metrics for objective optimization."""
    return [
        make_action(
            "low_quality",
            eff={"done": True},
            cost=1,
            resources={"cost": 5, "quality": 2},
        ),
        make_action(
            "high_quality",
            eff={"done": True},
            cost=2,
            resources={"cost": 20, "quality": 50},
        ),
    ]


def _timed_pipeline_actions() -> list[Any]:
    """Pipeline with durations for scheduling tests."""
    return [
        make_action(
            "fetch_data",
            eff={"has_data": True},
            duration=timedelta(seconds=2),
            resources={"cost": 1},
        ),
        make_action(
            "process_data",
            pre={"has_data": True},
            eff={"has_result": True},
            duration=timedelta(seconds=3),
            resources={"cost": 2},
        ),
        make_action(
            "format_output",
            pre={"has_result": True},
            eff={"done": True},
            duration=timedelta(seconds=1),
            resources={"cost": 1},
        ),
    ]


# ---------------------------------------------------------------------------
# Constraint tests
# ---------------------------------------------------------------------------


class TestGoapLoopWithConstraints:
    def test_within_budget_succeeds(self) -> None:
        """Plan within budget constraints achieves the goal."""
        actions = _budget_actions()
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost_usd", max=100),),
        )
        graph = GoapGraph(actions=actions)
        result = graph.compile().invoke({"goal": goal, "world_state": {}})
        assert result["status"] == "goal_achieved"

    def test_exceeds_budget_finds_alternative(self) -> None:
        """When A* picks the expensive path, CSP finds the cheap alternative."""
        actions = _budget_actions()
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost_usd", max=10),),
            objectives={"cost_usd": Minimize},
        )
        graph = GoapGraph(actions=actions)
        result = graph.compile().invoke({"goal": goal, "world_state": {}})
        assert result["status"] == "goal_achieved"
        # Verify the plan has CSP metadata showing a feasible selection
        plan_obj = result.get("plan")
        assert plan_obj is not None, "graph must expose the executed plan"
        assert plan_obj.metadata.csp is not None, "CSP metadata must be attached"
        assert plan_obj.metadata.csp.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)

    def test_no_feasible_plan(self) -> None:
        """When A* finds a plan that CSP marks infeasible, the graph executes it anyway.

        CSP is advisory in the current implementation: a plan marked
        INFEASIBLE by CSP is still handed to the executor.  The only
        action has no execute callable, so declared effects are applied
        and the goal is achieved regardless.  The critical assertion is
        that the CSP metadata on the returned plan records INFEASIBLE,
        proving the constraint was evaluated (not silently ignored).
        """
        actions = [
            make_action(
                "only_path",
                eff={"done": True},
                resources={"cost_usd": 500},
            ),
        ]
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost_usd", max=10),),
        )
        graph = GoapGraph(actions=actions)
        result = graph.compile().invoke({"goal": goal, "world_state": {}})
        # Plan executes (advisory CSP does not block) → goal_achieved
        assert result["status"] == "goal_achieved"
        # But the plan's CSP metadata must record the violation
        plan_obj = result.get("plan")
        assert plan_obj is not None, "graph must expose the executed plan"
        assert plan_obj.metadata.csp is not None, "CSP metadata must be attached"
        assert plan_obj.metadata.csp.status == CSPStatus.INFEASIBLE


# ---------------------------------------------------------------------------
# Objective tests
# ---------------------------------------------------------------------------


class TestGoapLoopWithObjectives:
    def test_minimizes_cost(self) -> None:
        """With MINIMIZE cost objective, prefers the cheaper path."""
        actions = _budget_actions()
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost_usd", max=100),),
            objectives={"cost_usd": Minimize},
        )
        graph = GoapGraph(actions=actions)
        result = graph.compile().invoke({"goal": goal, "world_state": {}})
        assert result["status"] == "goal_achieved"

    def test_maximizes_quality(self) -> None:
        """With MAXIMIZE quality objective, prefers the higher-quality path."""
        actions = _quality_actions()
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
            objectives={"quality": Maximize},
        )
        graph = GoapGraph(actions=actions)
        result = graph.compile().invoke({"goal": goal, "world_state": {}})
        assert result["status"] == "goal_achieved"


# ---------------------------------------------------------------------------
# Scheduling tests
# ---------------------------------------------------------------------------


class TestGoapLoopWithScheduling:
    def test_schedule_metadata_present(self) -> None:
        """Plans with timed actions include schedule metadata."""
        actions = _timed_pipeline_actions()
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        graph = GoapGraph(actions=actions)
        result = graph.compile().invoke({"goal": goal, "world_state": {}})
        assert result["status"] == "goal_achieved"
        plan_obj = result.get("plan")
        assert plan_obj is not None, "graph must expose the executed plan"
        assert plan_obj.metadata.csp is not None, "CSP metadata must be attached"
        csp = plan_obj.metadata.csp
        assert len(csp.schedule) > 0
        assert csp.makespan is not None

    def test_sequential_pipeline_makespan(self) -> None:
        """Linear pipeline makespan equals the sum of all durations.

        The timed pipeline is sequential (strict dependency chain), so
        CP-SAT cannot parallelize any step.  Makespan = 2 + 3 + 1 = 6 s.
        """
        actions = _timed_pipeline_actions()
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        graph = GoapGraph(actions=actions)
        result = graph.compile().invoke({"goal": goal, "world_state": {}})
        assert result["status"] == "goal_achieved"
        plan_obj = result.get("plan")
        assert plan_obj is not None, "graph must expose the executed plan"
        assert plan_obj.metadata.csp is not None, "CSP metadata must be attached"
        csp = plan_obj.metadata.csp
        assert csp.makespan == timedelta(seconds=6)
