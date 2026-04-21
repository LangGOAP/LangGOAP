"""End-to-end integration tests for ``PlanningTracer`` through the graph.

Verifies that a tracer configured on ``GoapGraph`` receives plan /
action / goal events when the graph is invoked via both ``invoke`` and
``ainvoke`` — the async path only works correctly once NS2 (the
observer/planner RunnableLambda wrapping) is in place.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.tracing import PlanningTracer


class _RecordingTracer:
    """Tracer that appends every hook invocation to a list."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.calls.append("on_plan_start")

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.calls.append("on_plan_complete")

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        self.calls.append("on_plan_failed")

    def on_action_start(self, action: Any, state: Any) -> None:
        self.calls.append("on_action_start")

    def on_action_complete(self, result: Any) -> None:
        self.calls.append("on_action_complete")

    def on_replan(self, reason: str, new_plan: Any) -> None:
        self.calls.append("on_replan")

    def on_goal_achieved(self, final_state: Any) -> None:
        self.calls.append("on_goal_achieved")

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        self.calls.append("on_sensor_complete")

    def on_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.calls.append("on_search_expand")

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        self.calls.append("on_search_dead_end")

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.calls.append("on_search_complete")

    async def aon_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.calls.append("aon_plan_start")

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.calls.append("aon_plan_complete")

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
        self.calls.append("aon_plan_failed")

    async def aon_action_start(self, action: Any, state: Any) -> None:
        self.calls.append("aon_action_start")

    async def aon_action_complete(self, result: Any) -> None:
        self.calls.append("aon_action_complete")

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        self.calls.append("aon_replan")

    async def aon_goal_achieved(self, final_state: Any) -> None:
        self.calls.append("aon_goal_achieved")

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        self.calls.append("aon_sensor_complete")

    async def aon_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.calls.append("aon_search_expand")

    async def aon_search_dead_end(
        self, reason: str, detail: dict[str, Any]
    ) -> None:
        self.calls.append("aon_search_dead_end")

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.calls.append("aon_search_complete")


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="gather", preconditions={}, effects={"data": True}, cost=1.0),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects={"done": True},
            cost=1.0,
        ),
    ]


class TestTracerSyncPath:
    def test_recording_tracer_conforms_to_protocol(self) -> None:
        assert isinstance(_RecordingTracer(), PlanningTracer)

    def test_successful_run_fires_expected_sync_hooks(self) -> None:
        tracer = _RecordingTracer()
        graph = GoapGraph(_actions(), tracer=tracer)
        result = graph.invoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )
        assert result["status"] == "goal_achieved"

        # Plan start/complete must fire once at the beginning.
        assert tracer.calls[0] == "on_plan_start"
        assert "on_plan_complete" in tracer.calls
        # Action hooks fire once per executed action.
        assert tracer.calls.count("on_action_start") == 2
        assert tracer.calls.count("on_action_complete") == 2
        # Goal achievement is the terminal signal.
        assert tracer.calls[-1] == "on_goal_achieved"
        # No async hooks should leak into the sync path.
        assert not any(c.startswith("aon_") for c in tracer.calls)


class TestTracerAsyncPath:
    @pytest.mark.asyncio
    async def test_successful_run_fires_expected_async_hooks(self) -> None:
        tracer = _RecordingTracer()
        graph = GoapGraph(_actions(), tracer=tracer)
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )
        assert result["status"] == "goal_achieved"

        assert tracer.calls[0] == "aon_plan_start"
        assert "aon_plan_complete" in tracer.calls
        assert tracer.calls.count("aon_action_start") == 2
        assert tracer.calls.count("aon_action_complete") == 2
        assert tracer.calls[-1] == "aon_goal_achieved"
        # No sync hooks should leak into the async path — this is the
        # NS2 guarantee: planner + observer are wrapped with
        # RunnableLambda(func, afunc) so LangGraph dispatches to the
        # async variants under ainvoke().
        assert not any(
            c.startswith("on_") and not c.startswith("aon_") for c in tracer.calls
        )


class TestTracerFailurePath:
    def test_unreachable_goal_fires_plan_failed(self) -> None:
        tracer = _RecordingTracer()
        # Goal demands `impossible=True` but no action produces it.
        graph = GoapGraph(_actions(), tracer=tracer)
        result = graph.invoke(
            goal=GoalSpec(conditions={"impossible": True}),
            world_state={},
        )
        assert result["status"] == "no_plan"
        assert "on_plan_failed" in tracer.calls
        assert "on_goal_achieved" not in tracer.calls
