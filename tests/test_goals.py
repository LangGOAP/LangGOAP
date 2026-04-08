"""Tests for GoalSpec, ConstraintSpec, and ReplanStrategy."""

from __future__ import annotations

import pytest

from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.types import Maximize, Minimize, ObjectiveDirection, ReplanStrategy


class TestGoalSpecCreation:
    def test_create_with_conditions(self) -> None:
        goal = GoalSpec(conditions={"target_reached": True, "score": 100})
        assert dict(goal.conditions) == {"target_reached": True, "score": 100}

    def test_defaults(self) -> None:
        goal = GoalSpec()
        assert dict(goal.conditions) == {}
        assert goal.replan_strategy == ReplanStrategy.ON_DEVIATION
        assert goal.constraints == ()
        assert goal.objectives is None
        assert goal.priority == 0
        assert goal.max_replans == 10

    def test_frozen(self) -> None:
        goal = GoalSpec(conditions={"a": True})
        with pytest.raises(AttributeError):
            goal.conditions = {}  # type: ignore[misc]

    def test_conditions_are_truly_immutable(self) -> None:
        """conditions MappingProxy must reject mutation after construction."""
        goal = GoalSpec(conditions={"done": True})
        with pytest.raises(TypeError):
            goal.conditions["hacked"] = True  # type: ignore[index]

    def test_goal_alias(self) -> None:
        """Goal is a public alias for GoalSpec."""
        from langgoap.goals import Goal

        assert Goal is GoalSpec
        g = Goal(conditions={"done": True})
        assert isinstance(g, GoalSpec)


class TestReplanStrategy:
    def test_on_deviation(self) -> None:
        assert ReplanStrategy.ON_DEVIATION.value == "on_deviation"

    def test_every_action(self) -> None:
        assert ReplanStrategy.EVERY_ACTION.value == "every_action"

    def test_never(self) -> None:
        assert ReplanStrategy.NEVER.value == "never"

    def test_strategy_in_goal(self) -> None:
        goal = GoalSpec(
            conditions={"done": True},
            replan_strategy=ReplanStrategy.EVERY_ACTION,
        )
        assert goal.replan_strategy == ReplanStrategy.EVERY_ACTION


class TestConstraintSpec:
    def test_max_constraint(self) -> None:
        """ConstraintSpec is a hard resource/budget constraint."""
        c = ConstraintSpec(key="cost_usd", max=10.0)
        assert c.key == "cost_usd"
        assert c.max == 10.0
        assert c.min is None
        assert c.weight == 1.0

    def test_min_constraint(self) -> None:
        c = ConstraintSpec(key="quality", min=0.8)
        assert c.key == "quality"
        assert c.min == 0.8
        assert c.max is None

    def test_range_constraint(self) -> None:
        c = ConstraintSpec(key="temperature", min=0.0, max=1.0, weight=2.0)
        assert c.min == 0.0
        assert c.max == 1.0
        assert c.weight == 2.0

    def test_goal_with_constraints(self) -> None:
        goal = GoalSpec(
            conditions={"done": True},
            constraints=[
                ConstraintSpec(key="total_tokens", max=10000),
                ConstraintSpec(key="cost_usd", max=0.50, weight=3.0),
            ],
        )
        assert len(goal.constraints) == 2
        assert goal.constraints[0].key == "total_tokens"
        assert goal.constraints[1].weight == 3.0

    def test_goal_with_objectives(self) -> None:
        goal = GoalSpec(
            conditions={"done": True},
            objectives={"cost_usd": Minimize, "quality": Maximize},
        )
        assert goal.objectives is not None
        assert goal.objectives["cost_usd"] == Minimize
        assert goal.objectives["quality"] == Maximize


class TestObjectiveDirection:
    def test_aliases(self) -> None:
        assert Minimize is ObjectiveDirection.MINIMIZE
        assert Maximize is ObjectiveDirection.MAXIMIZE

    def test_string_values(self) -> None:
        assert ObjectiveDirection.MINIMIZE.value == "minimize"
        assert ObjectiveDirection.MAXIMIZE.value == "maximize"
