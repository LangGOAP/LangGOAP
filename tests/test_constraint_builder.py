"""Tests for the fluent ConstraintBuilder API.

Covers the chain construction, terminator validation, and the
:meth:`GoalSpec.from_builder` integration.
"""

from __future__ import annotations

import pytest

from langgoap.constraints import (
    BuilderOutput,
    ConstraintBuilder,
)
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.types import ObjectiveDirection


class TestConstraintChain:
    def test_hard_constraint_with_max(self) -> None:
        out = (
            ConstraintBuilder.for_plan()
            .sum_resource("gpu_hours")
            .bounded(max=100.0)
            .penalize(level="hard", weight=1.0)
            .as_constraint("gpu_budget")
        )
        assert out.constraint is not None
        assert out.objective is None
        spec = out.constraint
        assert isinstance(spec, ConstraintSpec)
        assert spec.key == "gpu_hours"
        assert spec.max == 100.0
        assert spec.min is None
        assert spec.level == "hard"
        assert spec.weight == 1.0

    def test_hard_constraint_defaults_to_hard_level(self) -> None:
        out = (
            ConstraintBuilder.for_plan()
            .sum_resource("cost_usd")
            .bounded(max=5.0)
            .as_constraint("budget")
        )
        assert out.constraint is not None
        assert out.constraint.level == "hard"

    def test_soft_constraint_with_weight(self) -> None:
        out = (
            ConstraintBuilder.for_plan()
            .sum_resource("cost_usd")
            .bounded(max=10.0)
            .penalize(level="soft", weight=2.5)
            .as_constraint("soft_budget")
        )
        assert out.constraint is not None
        assert out.constraint.level == "soft"
        assert out.constraint.weight == 2.5

    def test_constraint_with_both_bounds(self) -> None:
        out = (
            ConstraintBuilder.for_plan()
            .sum_resource("temperature")
            .bounded(min=60.0, max=80.0)
            .as_constraint("comfort_range")
        )
        assert out.constraint is not None
        assert out.constraint.min == 60.0
        assert out.constraint.max == 80.0

    def test_as_constraint_requires_bounds(self) -> None:
        with pytest.raises(ValueError, match="no bounds"):
            (
                ConstraintBuilder.for_plan()
                .sum_resource("thing")
                .as_constraint("no_bounds")
            )

    def test_minimize_objective(self) -> None:
        out = (
            ConstraintBuilder.for_plan()
            .sum_resource("cost_usd")
            .minimize()
            .weight(2.0)
            .as_objective("cost")
        )
        assert out.objective is not None
        assert out.constraint is None
        key, direction = out.objective
        assert key == "cost"
        assert direction == ObjectiveDirection.MINIMIZE

    def test_maximize_objective(self) -> None:
        out = (
            ConstraintBuilder.for_plan()
            .sum_resource("quality")
            .maximize()
            .as_objective("quality")
        )
        assert out.objective is not None
        _, direction = out.objective
        assert direction == ObjectiveDirection.MAXIMIZE

    def test_as_objective_requires_direction(self) -> None:
        with pytest.raises(ValueError, match="no direction"):
            (ConstraintBuilder.for_plan().sum_resource("cost").as_objective("cost"))


class TestConstraintBuilderBuild:
    def test_build_empty(self) -> None:
        out = ConstraintBuilder.build()
        assert isinstance(out, BuilderOutput)
        assert out.constraints == ()
        assert dict(out.objectives) == {}

    def test_build_mixes_constraints_and_objectives(self) -> None:
        out = ConstraintBuilder.build(
            ConstraintBuilder.for_plan()
            .sum_resource("gpu_hours")
            .bounded(max=100.0)
            .as_constraint("gpu_budget"),
            ConstraintBuilder.for_plan()
            .sum_resource("cost_usd")
            .minimize()
            .as_objective("cost"),
        )
        assert len(out.constraints) == 1
        assert out.constraints[0].key == "gpu_hours"
        assert "cost" in out.objectives
        assert out.objectives["cost"] == ObjectiveDirection.MINIMIZE

    def test_build_rejects_duplicate_objective_keys(self) -> None:
        with pytest.raises(ValueError, match="Duplicate objective"):
            ConstraintBuilder.build(
                ConstraintBuilder.for_plan()
                .sum_resource("cost")
                .minimize()
                .as_objective("cost"),
                ConstraintBuilder.for_plan()
                .sum_resource("cost")
                .maximize()
                .as_objective("cost"),
            )

    def test_for_each_action_currently_aliases_for_plan(self) -> None:
        # Reserved alias for future per-action filtering.
        out = (
            ConstraintBuilder.for_each_action()
            .sum_resource("tokens")
            .bounded(max=1000.0)
            .as_constraint("token_budget")
        )
        assert out.constraint is not None


class TestGoalSpecFromBuilder:
    def test_from_builder_wires_constraints_and_objectives(self) -> None:
        builder_output = ConstraintBuilder.build(
            ConstraintBuilder.for_plan()
            .sum_resource("gpu_hours")
            .bounded(max=100.0)
            .penalize(level="hard", weight=1.0)
            .as_constraint("gpu_budget"),
            ConstraintBuilder.for_plan()
            .sum_resource("cost_usd")
            .minimize()
            .weight(2.0)
            .as_objective("cost"),
        )
        goal = GoalSpec.from_builder(
            conditions={"done": True}, builder_output=builder_output
        )
        assert dict(goal.conditions) == {"done": True}
        assert len(goal.constraints) == 1
        assert goal.constraints[0].key == "gpu_hours"
        assert goal.objectives is not None
        assert "cost" in goal.objectives

    def test_from_builder_none_means_no_extras(self) -> None:
        goal = GoalSpec.from_builder(conditions={"done": True})
        assert goal.constraints == ()
        assert goal.objectives is None

    def test_from_builder_forwards_kwargs(self) -> None:
        goal = GoalSpec.from_builder(
            conditions={"done": True}, builder_output=None, priority=5, max_replans=3
        )
        assert goal.policy.priority == 5
        assert goal.policy.max_replans == 3
