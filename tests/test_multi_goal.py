"""Unit tests for the ``MultiGoal`` dataclass.

End-to-end sequential / any-mode execution is covered in
``tests/integration/test_multi_goal_loop.py``; this module focuses on
the data-model contract and validation.
"""

from __future__ import annotations

import pytest

from langgoap.goals import GoalSpec, MultiGoal


class TestMultiGoalConstruction:
    def test_accepts_sequence_of_goal_specs(self) -> None:
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"a": True}),
                GoalSpec(conditions={"b": True}),
            )
        )
        assert len(mg.goals) == 2
        assert mg.mode == "sequential"

    def test_accepts_list_and_coerces_to_tuple(self) -> None:
        mg = MultiGoal(
            goals=[  # type: ignore[arg-type]
                GoalSpec(conditions={"a": True}),
                GoalSpec(conditions={"b": True}),
            ]
        )
        assert isinstance(mg.goals, tuple)
        assert len(mg.goals) == 2

    def test_rejects_empty_goals(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            MultiGoal(goals=())

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            MultiGoal(
                goals=(GoalSpec(conditions={"a": True}),),
                mode="nonsense",  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        mg = MultiGoal(goals=(GoalSpec(conditions={"a": True}),))
        with pytest.raises(AttributeError):
            mg.mode = "any"  # type: ignore[misc]

    def test_rejects_nested_multigoal(self) -> None:
        # H3: HTN-style decomposition is out of scope for v0.1.0.
        # Non-GoalSpec children must raise a clear error at construction
        # time rather than crashing later with AttributeError inside the
        # planner.
        inner = MultiGoal(goals=(GoalSpec(conditions={"a": True}),))
        with pytest.raises(ValueError, match="GoalSpec"):
            MultiGoal(goals=(inner,))  # type: ignore[arg-type]

    def test_rejects_non_goal_children(self) -> None:
        with pytest.raises(ValueError, match="GoalSpec"):
            MultiGoal(goals=({"a": True},))  # type: ignore[arg-type]

    def test_single_element_multigoal_accepted(self) -> None:
        # L1: a single-element MultiGoal is a valid (if degenerate)
        # composite.  It must construct cleanly in both modes.
        mg_seq = MultiGoal(goals=(GoalSpec(conditions={"a": True}),))
        mg_any = MultiGoal(goals=(GoalSpec(conditions={"a": True}),), mode="any")
        assert len(mg_seq.goals) == 1
        assert len(mg_any.goals) == 1
