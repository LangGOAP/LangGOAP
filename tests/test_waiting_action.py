"""`GoapGraph.waiting_action(thread_id)` returns the action name a thread is waiting on (issue #105)."""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph


def _sensor(val: bool):
    def sense(ws: dict[str, Any]) -> dict[str, Any]:
        return {"x": val}

    return sense


GOAL = GoalSpec(conditions={"x": True})


class TestWaitingAction:
    def test_returns_action_name_for_a_wait(self) -> None:
        graph = GoapGraph(
            [ActionSpec.wait("hold", {"x": False}, {"x": True})],
            goal=GOAL,
            sensors=[_sensor(False)],
            effects="observed",
        )
        graph.start(world_state={"x": False}, thread_id="t")
        assert graph.waiting_action("t") == "hold"

    def test_returns_none_when_not_waiting(self) -> None:
        graph = GoapGraph(
            [ActionSpec("flip", {"x": False}, {"x": True})],
            goal=GOAL,
        )
        graph.start(world_state={"x": False}, thread_id="t")
        assert graph.waiting_action("t") is None

    def test_returns_none_for_an_approval_gate(self) -> None:
        graph = GoapGraph(
            [ActionSpec("gate", {}, {"x": True}, require_human_approval=True)],
            goal=GOAL,
        )
        graph.start(world_state={}, thread_id="t")
        assert graph.waiting_action("t") is None

    def test_returns_none_for_a_form(self) -> None:
        class F:
            title: str = ""

        graph = GoapGraph(
            [ActionSpec("form", {}, {"x": True}, require_human_approval=F)],
            goal=GOAL,
        )
        graph.start(world_state={}, thread_id="t")
        assert graph.waiting_action("t") is None

    async def test_async_path(self) -> None:
        graph = GoapGraph(
            [ActionSpec.wait("hold", {"x": False}, {"x": True})],
            goal=GOAL,
            sensors=[_sensor(False)],
            effects="observed",
        )
        await graph.astart(world_state={"x": False}, thread_id="t")
        assert graph.waiting_action("t") == "hold"

    def test_no_such_thread_returns_none(self) -> None:
        graph = GoapGraph([ActionSpec("flip", {}, {"x": True})], goal=GOAL)
        graph.start(world_state={"x": False}, thread_id="t")
        assert graph.waiting_action("missing") is None
