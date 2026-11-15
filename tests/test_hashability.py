"""Unhashable planning values fail at construction, not inside A* (issue #34)."""

from __future__ import annotations

import pytest

from langgoap import ActionSpec, GoalSpec
from langgoap.goals import SoftGoal


class TestGoalConditionValues:
    def test_list_value_is_rejected_with_the_key_named(self) -> None:
        with pytest.raises(ValueError, match="items"):
            GoalSpec(conditions={"items": [1, 2]})

    def test_dict_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="payload"):
            GoalSpec(conditions={"ok": True, "payload": {"a": 1}})

    def test_soft_goal_values_are_checked(self) -> None:
        with pytest.raises(ValueError, match="items"):
            SoftGoal(conditions={"items": [1]})

    def test_hashable_values_are_accepted(self) -> None:
        goal = GoalSpec(
            conditions={
                "a": True,
                "b": 3,
                "c": "x",
                "d": None,
                "e": (1, 2),
                "f": frozenset({1}),
            }
        )
        assert dict(goal.conditions)["e"] == (1, 2)


class TestActionSpecValues:
    def test_precondition_list_is_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="preconditions.*items|items.*preconditions"
        ):
            ActionSpec(name="a", preconditions={"items": [1, 2]})

    def test_static_effect_dict_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="effects.*payload|payload.*effects"):
            ActionSpec(name="a", effects={"payload": {"k": 1}})

    def test_dynamic_effects_are_not_checked_at_construction(self) -> None:
        spec = ActionSpec(
            name="a", effects=lambda ws: {"k": [1]}, effect_keys=frozenset({"k"})
        )
        assert spec.has_dynamic_effects

    def test_hashable_values_are_accepted(self) -> None:
        spec = ActionSpec(
            name="a", preconditions={"n": 1, "t": (1, 2)}, effects={"s": "x", "z": None}
        )
        assert dict(spec.effects)["s"] == "x"
