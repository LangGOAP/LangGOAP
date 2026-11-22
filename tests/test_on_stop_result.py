"""The `on_stop` callback receives a GoapResult, not a plain dict."""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph, GoapResult

GOAL = GoalSpec(conditions={"done": True})


class TestOnStop:
    def test_on_stop_receives_goap_result(self) -> None:
        received: Any = None

        def capture(thread: str, state: Any) -> None:
            nonlocal received
            received = state

        graph = GoapGraph(
            [ActionSpec("flip", {"done": False}, {"done": True})],
            goal=GOAL,
            on_stop=capture,
        )
        graph.start(world_state={"done": False}, thread_id="t")
        assert isinstance(received, GoapResult)
        assert received.succeeded is True
        assert received.actions == ["flip"]

    async def test_async_on_stop_receives_goap_result(self) -> None:
        received: Any = None

        def capture(thread: str, state: Any) -> None:
            nonlocal received
            received = state

        graph = GoapGraph(
            [ActionSpec("flip", {"done": False}, {"done": True})],
            goal=GOAL,
            on_stop=capture,
        )
        await graph.astart(world_state={"done": False}, thread_id="t")
        assert isinstance(received, GoapResult)
        assert received.succeeded is True

    def test_on_stop_for_a_paused_run(self) -> None:
        received: Any = None

        def capture(thread: str, state: Any) -> None:
            nonlocal received
            received = state

        graph = GoapGraph(
            [ActionSpec("gate", {}, {"done": True}, require_human_approval=True)],
            goal=GOAL,
            on_stop=capture,
        )
        graph.start(world_state={}, thread_id="t")
        assert isinstance(received, GoapResult)
        assert received.succeeded is False
        assert received.interrupt is not None
        assert received.interrupt["action"] == "gate"
