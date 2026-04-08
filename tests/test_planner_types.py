"""Tests for Plan and PlanMetadata types."""

from __future__ import annotations

import pytest

from langgoap.actions import ActionSpec
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.state import PlanningState


class TestPlanMetadata:
    def test_defaults(self) -> None:
        meta = PlanMetadata()
        assert meta.nodes_explored == 0
        assert meta.planning_time_ms == 0.0
        assert meta.actions_pruned == 0

    def test_custom_values(self) -> None:
        meta = PlanMetadata(nodes_explored=42, planning_time_ms=12.5, actions_pruned=3)
        assert meta.nodes_explored == 42
        assert meta.planning_time_ms == 12.5
        assert meta.actions_pruned == 3


class TestPlan:
    def test_creation(self) -> None:
        a1 = ActionSpec(name="step1", effects={"a": True})
        a2 = ActionSpec(name="step2", effects={"b": True})
        s1 = PlanningState.from_dict({"a": True})
        s2 = PlanningState.from_dict({"a": True, "b": True})

        plan = Plan(
            actions=(a1, a2),
            expected_states=(s1, s2),
            total_cost=5.0,
        )
        assert len(plan) == 2
        assert plan.total_cost == 5.0

    def test_action_names(self) -> None:
        a1 = ActionSpec(name="gather", effects={"data": True})
        a2 = ActionSpec(name="analyze", effects={"result": True})
        a3 = ActionSpec(name="report", effects={"done": True})
        plan = Plan(actions=(a1, a2, a3), total_cost=3.0)
        assert plan.action_names == ["gather", "analyze", "report"]

    def test_empty_plan(self) -> None:
        plan = Plan.empty()
        assert len(plan) == 0
        assert plan.actions == ()
        assert plan.expected_states == ()
        assert plan.total_cost == 0.0
        assert plan.action_names == []

    def test_metadata_default(self) -> None:
        plan = Plan(actions=())
        assert plan.metadata.nodes_explored == 0

    def test_metadata_custom(self) -> None:
        meta = PlanMetadata(nodes_explored=10, planning_time_ms=5.0, actions_pruned=2)
        plan = Plan(actions=(), metadata=meta)
        assert plan.metadata.nodes_explored == 10
        assert plan.metadata.actions_pruned == 2

    def test_frozen(self) -> None:
        plan = Plan(actions=())
        with pytest.raises(AttributeError):
            plan.total_cost = 99.0  # type: ignore[misc]
