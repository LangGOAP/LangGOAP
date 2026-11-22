"""A mapping as the goal (issue #100) and the ``GoapResult`` object (issue #101)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from langgoap import ActionSpec, GoalSpec, GoapGraph, GoapResult, MultiGoal


def _boom(world_state: dict[str, Any]) -> None:
    raise RuntimeError("no stock")


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec("order", {}, {"ordered": True}),
        ActionSpec("ship", {"ordered": True}, {"shipped": True}),
    ]


class TestGoalMapping:
    def test_invoke_takes_a_mapping_of_conditions(self) -> None:
        result = GoapGraph(_actions()).invoke({"shipped": True})
        assert result["status"] == "goal_achieved"
        assert isinstance(result["goal"], GoalSpec)

    async def test_ainvoke_takes_a_mapping_of_conditions(self) -> None:
        result = await GoapGraph(_actions()).ainvoke({"shipped": True})
        assert result["status"] == "goal_achieved"

    def test_invoke_falls_back_to_the_graph_goal(self) -> None:
        graph = GoapGraph(_actions(), goal={"shipped": True})
        assert graph.invoke()["status"] == "goal_achieved"
        assert graph.invoke(world_state={"ordered": True})["status"] == "goal_achieved"

    async def test_ainvoke_falls_back_to_the_graph_goal(self) -> None:
        graph = GoapGraph(_actions(), goal={"shipped": True})
        assert (await graph.ainvoke())["status"] == "goal_achieved"

    def test_a_goal_passed_to_invoke_overrides_the_graph_goal(self) -> None:
        graph = GoapGraph(_actions(), goal={"shipped": True})
        result = graph.invoke({"ordered": True})
        assert [r.action_name for r in result["execution_history"]] == ["order"]

    def test_no_goal_anywhere_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="goal"):
            GoapGraph(_actions()).invoke()

    def test_goal_objects_work_as_before(self) -> None:
        graph = GoapGraph(_actions())
        assert (
            graph.invoke(GoalSpec(conditions={"shipped": True}))["status"]
            == "goal_achieved"
        )
        multi = MultiGoal(goals=(GoalSpec(conditions={"ordered": True}),))
        assert graph.invoke(goal=multi, world_state={})["status"] == "goal_achieved"


class TestGoapResult:
    def test_a_finished_run(self) -> None:
        result = GoapGraph(_actions()).invoke({"shipped": True})
        assert isinstance(result, GoapResult)
        assert result.status == "goal_achieved"
        assert result.succeeded is True
        assert result.actions == ["order", "ship"]
        assert result.world_state == {"ordered": True, "shipped": True}
        assert result.interrupt is None

    def test_it_is_still_the_dict_it_always_was(self) -> None:
        result = GoapGraph(_actions()).invoke({"shipped": True})
        assert isinstance(result, dict)
        assert result["status"] == "goal_achieved"
        assert result.get("world_state") == {"ordered": True, "shipped": True}
        assert "execution_history" in result
        assert result == dict(result)
        assert result["execution_history"][0].action_name == "order"

    def test_a_failed_run(self) -> None:
        graph = GoapGraph([ActionSpec("order", {}, {"ordered": True}, execute=_boom)])
        result = graph.invoke({"ordered": True})
        assert result.succeeded is False
        assert result.status != "goal_achieved"
        assert result.actions == []  # only the actions that succeeded

    def test_a_paused_run_reports_its_interrupt(self) -> None:
        graph = GoapGraph(
            [ActionSpec("order", {}, {"ordered": True}, require_human_approval=True)],
            goal={"ordered": True},
        )
        paused = graph.start(thread_id="t")
        assert isinstance(paused, GoapResult)
        assert paused.succeeded is False
        assert paused.interrupt is not None
        assert paused.interrupt["type"] == "goap_action_approval"
        assert paused.interrupt["action"] == "order"
        done = graph.resume(True, thread_id="t")
        assert done is not None and done.succeeded and done.interrupt is None

    def test_summary_is_a_compact_report(self) -> None:
        result = GoapGraph(_actions()).invoke({"shipped": True})
        text = result.summary()
        assert "goal_achieved" in text
        assert "order" in text and "ship" in text

    def test_it_wraps_the_raw_state_of_a_compiled_graph(self) -> None:
        graph = GoapGraph(_actions())
        compiled = graph.compile(checkpointer=MemorySaver())
        raw = compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"shipped": True})),
            {"configurable": {"thread_id": "t"}},
        )
        json.dumps(raw["execution_history"])  # raw: JSON-native history
        result = GoapResult(raw)
        assert result.succeeded is True
        assert result.actions == ["order", "ship"]
        assert graph.rehydrate(raw).actions == ["order", "ship"]
        assert isinstance(graph.rehydrate(raw), GoapResult)

    async def test_async_calls_return_it_too(self) -> None:
        result = await GoapGraph(_actions()).ainvoke({"shipped": True})
        assert isinstance(result, GoapResult) and result.actions == ["order", "ship"]

    def test_a_missing_status_is_not_success(self) -> None:
        assert GoapResult({}).succeeded is False
        assert GoapResult({}).status is None
        assert GoapResult({}).actions == []
        assert GoapResult({}).world_state == {}
