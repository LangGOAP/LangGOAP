"""``ExecutionRecord`` reports what actually executed (issue #51).

``actual_cost`` sums the cost of every successful step across replans,
``plan_actions`` lists the executed actions in order, and failed attempts
are kept in ``metadata`` so analytics can separate expected from realised
cost.  Before, both fields were copied from the *final* plan.
"""

from __future__ import annotations

from typing import Any

from langgraph.store.memory import InMemoryStore

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.history import StoreExecutionHistory


def _actions() -> list[ActionSpec]:
    def fail(ws: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("b is down")

    return [
        ActionSpec(
            name="A", effects={"a": True}, cost=1.0, execute=lambda ws: {"a": True}
        ),
        ActionSpec(
            name="B",
            preconditions={"a": True},
            effects={"done": True},
            cost=1.0,
            execute=fail,
        ),
        ActionSpec(
            name="D",
            preconditions={"a": True},
            effects={"done": True},
            cost=4.0,
            execute=lambda ws: {"done": True},
        ),
    ]


class TestRealisedCost:
    def test_record_reflects_executed_actions_not_the_final_plan(self) -> None:
        history = StoreExecutionHistory(InMemoryStore())
        goal = GoalSpec(conditions={"done": True})
        result = GoapGraph(_actions(), history=history).invoke(
            goal=goal, world_state={}
        )
        assert result["status"] == "goal_achieved"
        # First plan [A, B]: A ran, B failed; replan [D] from a=True.
        assert result["plan"].action_names == ["D"]

        records = history.query_by_goal(history.goal_hash_for(goal))
        assert len(records) == 1
        record = records[0]
        assert record.plan_actions == ("A", "D"), "executed successes, in order"
        assert record.actual_cost == 5.0, "A (1.0) + D (4.0)"
        assert record.expected_cost == 4.0, "the final plan's own cost"
        assert record.outcome == "success"
        assert record.replan_count == 1
        failed = record.metadata["failed_attempts"]
        assert [f["action"] for f in failed] == ["B"]
        assert "b is down" in failed[0]["error"]
        assert record.metadata["final_plan"] == ["D"]

    def test_record_without_failures_matches_the_plan(self) -> None:
        history = StoreExecutionHistory(InMemoryStore())
        goal = GoalSpec(conditions={"a": True})
        GoapGraph(_actions(), history=history).invoke(goal=goal, world_state={})
        record = history.query_by_goal(history.goal_hash_for(goal))[0]
        assert record.plan_actions == ("A",)
        assert record.actual_cost == record.expected_cost == 1.0
        assert record.metadata["failed_attempts"] == []
