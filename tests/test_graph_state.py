"""Tests for GoapState TypedDict and ActionResult."""

from __future__ import annotations

from langgoap.goals import GoalSpec
from langgoap.graph.state import ActionResult, GoapState
from langgoap.planner.types import Plan


class TestActionResult:
    def test_success_result(self) -> None:
        result = ActionResult(
            action_name="gather",
            success=True,
            state_before={"a": False},
            state_after={"a": True},
        )
        assert result.action_name == "gather"
        assert result.success is True
        assert result.error is None

    def test_failure_result(self) -> None:
        result = ActionResult(
            action_name="risky",
            success=False,
            error="network timeout",
        )
        assert result.success is False
        assert result.error == "network timeout"

    def test_defaults(self) -> None:
        result = ActionResult(action_name="test", success=True)
        assert result.state_before == {}
        assert result.state_after == {}
        assert result.error is None


class TestGoapState:
    def test_can_create_typed_dict(self) -> None:
        """GoapState is a valid TypedDict that can be instantiated."""
        state: GoapState = {
            "world_state": {"a": True},
            "goal": GoalSpec(conditions={"b": True}),
            "plan": None,
            "current_step": 0,
            "execution_history": [],
            "replan_count": 0,
            "replan_reason": None,
            "status": "idle",
        }
        assert state["world_state"] == {"a": True}
        assert state["status"] == "idle"
        assert state["execution_history"] == []

    def test_partial_state(self) -> None:
        """GoapState with total=False allows partial dicts."""
        state: GoapState = {
            "world_state": {},
            "status": "idle",
        }
        assert state["world_state"] == {}

    def test_execution_history_is_appendable(self) -> None:
        """execution_history uses operator.add for list concatenation."""
        history: list[ActionResult] = []
        new_entry = [ActionResult(action_name="step1", success=True)]
        combined = history + new_entry
        assert len(combined) == 1
        assert combined[0].action_name == "step1"
