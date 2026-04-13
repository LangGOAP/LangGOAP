"""End-to-end integration tests for plan explanation (MUS)."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.planner.csp import CSPStatus
from langgoap.planner.explain import InfeasibilityExplanation
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.state import PlanningState


def test_pipeline_infeasible_plan_has_explanation() -> None:
    """Two-phase pipeline attaches explanation to INFEASIBLE plans."""
    action = ActionSpec(
        name="expensive",
        effects={"done": True},
        resources={"cost_usd": 10.0},
        execute=lambda ws: {"done": True},
    )
    goal = GoalSpec(
        conditions={"done": True},
        constraints=(ConstraintSpec(key="cost_usd", max=5.0, level="hard"),),
    )
    start = PlanningState.from_dict({})
    result = pipeline_plan(start, goal, [action])

    assert result is not None
    assert result.metadata.csp is not None
    assert result.metadata.csp.status == CSPStatus.INFEASIBLE
    assert result.metadata.csp.explanation is not None
    assert isinstance(result.metadata.csp.explanation, InfeasibilityExplanation)
    assert result.metadata.csp.explanation.conflicting_constraints[0].key == "cost_usd"


def test_pipeline_feasible_plan_no_explanation() -> None:
    """Feasible plans have no explanation attached."""
    action = ActionSpec(
        name="cheap",
        effects={"done": True},
        resources={"cost_usd": 3.0},
        execute=lambda ws: {"done": True},
    )
    goal = GoalSpec(
        conditions={"done": True},
        constraints=(ConstraintSpec(key="cost_usd", max=5.0, level="hard"),),
    )
    start = PlanningState.from_dict({})
    result = pipeline_plan(start, goal, [action])

    assert result is not None
    assert result.metadata.csp is not None
    assert result.metadata.csp.status == CSPStatus.FEASIBLE
    assert result.metadata.csp.explanation is None


def test_graph_invoke_with_infeasible_constraints() -> None:
    """Full graph invocation with infeasible constraints still completes.

    The CSP layer is advisory — INFEASIBLE plans are still executed.
    The explanation is attached to the plan metadata.
    """
    action = ActionSpec(
        name="act",
        effects={"done": True},
        resources={"cost": 10.0},
        execute=lambda ws: {"done": True},
    )
    goal = GoalSpec(
        conditions={"done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="hard"),),
    )
    graph = GoapGraph(actions=[action])
    result = graph.invoke(goal=goal, world_state={})

    # The goal is achieved despite the INFEASIBLE CSP status
    assert result["status"] == "goal_achieved"
    plan = result.get("plan")
    assert plan is not None
    if plan.metadata.csp is not None:
        assert plan.metadata.csp.explanation is not None


def test_explanation_serializes_in_ascii_output() -> None:
    """ASCII rendering includes the infeasibility explanation."""
    action = ActionSpec(
        name="act",
        effects={"done": True},
        resources={"cost": 10.0},
    )
    goal = GoalSpec(
        conditions={"done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="hard"),),
    )
    start = PlanningState.from_dict({})
    result = pipeline_plan(start, goal, [action])

    assert result is not None
    ascii_out = result.to_ascii()
    assert "Infeasibility Explanation" in ascii_out
    assert "cost" in ascii_out


def test_explanation_serializes_in_mermaid_output() -> None:
    """Mermaid rendering includes an explanation note."""
    action = ActionSpec(
        name="act",
        effects={"done": True},
        resources={"cost": 10.0},
    )
    goal = GoalSpec(
        conditions={"done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="hard"),),
    )
    start = PlanningState.from_dict({})
    result = pipeline_plan(start, goal, [action])

    assert result is not None
    mermaid_out = result.to_mermaid()
    assert "INFEASIBLE" in mermaid_out
