"""Tracer hooks receive thread_id as an explicit keyword argument (issue #104)."""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.tracing import NullTracer


class ThreadCapturingTracer(NullTracer):
    def __init__(self) -> None:
        self.thread_ids: dict[str, str | None] = {}

    def on_plan_start(
        self, goal: Any, state: Any, strategy_name: str, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["plan_start"] = thread_id

    def on_plan_complete(
        self, plan: Any, duration_ms: float, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["plan_complete"] = thread_id

    def on_plan_failed(
        self, reason: str, duration_ms: float, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["plan_failed"] = thread_id

    def on_replan(
        self,
        reason: str,
        new_plan: Any,
        duration_ms: float = 0.0,
        thread_id: str | None = "",
    ) -> None:
        self.thread_ids["replan"] = thread_id

    def on_action_start(
        self, action: Any, state: Any, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["action_start"] = thread_id

    def on_action_complete(self, result: Any, thread_id: str | None = "") -> None:
        self.thread_ids["action_complete"] = thread_id

    def on_goal_achieved(self, final_state: Any, thread_id: str | None = "") -> None:
        self.thread_ids["goal_achieved"] = thread_id

    def on_sensor_complete(
        self, sensor_name: str, updates: Any, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["sensor"] = thread_id

    async def aon_plan_start(
        self, goal: Any, state: Any, strategy_name: str, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["aplan_start"] = thread_id

    async def aon_plan_complete(
        self, plan: Any, duration_ms: float, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["aplan_complete"] = thread_id

    async def aon_action_complete(
        self, result: Any, thread_id: str | None = ""
    ) -> None:
        self.thread_ids["aaction_complete"] = thread_id


class OldStyleTracer(NullTracer):
    """A tracer written before thread_id existed: no parameter, still works."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.calls.append("plan_complete")

    def on_action_complete(self, result: Any) -> None:
        self.calls.append("action_complete")


def _refuse(world_state: dict[str, Any]) -> None:
    raise RuntimeError("no")


ACTIONS = [ActionSpec("flip", {"done": False}, {"done": True})]
GOAL = GoalSpec(conditions={"done": True})


class TestThreadIdInHooks:
    def test_sync_path_passes_thread_id(self) -> None:
        tracer = ThreadCapturingTracer()
        graph = GoapGraph(ACTIONS, goal=GOAL, tracer=tracer)
        graph.start(world_state={"done": False}, thread_id="t-42")
        assert tracer.thread_ids["plan_start"] == "t-42"
        assert tracer.thread_ids["plan_complete"] == "t-42"
        assert tracer.thread_ids["action_start"] == "t-42"
        assert tracer.thread_ids["action_complete"] == "t-42"
        assert tracer.thread_ids["goal_achieved"] == "t-42"

    async def test_async_path_passes_thread_id(self) -> None:
        tracer = ThreadCapturingTracer()
        graph = GoapGraph(ACTIONS, goal=GOAL, tracer=tracer)
        await graph.astart(world_state={"done": False}, thread_id="async-7")
        assert tracer.thread_ids["aplan_start"] == "async-7"
        assert tracer.thread_ids["aplan_complete"] == "async-7"
        assert tracer.thread_ids["aaction_complete"] == "async-7"

    def test_old_style_tracer_still_works(self) -> None:
        tracer = OldStyleTracer()
        graph = GoapGraph(ACTIONS, goal=GOAL, tracer=tracer)
        graph.start(world_state={"done": False}, thread_id="old")
        assert tracer.calls == ["plan_complete", "action_complete"]

    def test_replan_hook_receives_thread_id(self) -> None:
        tracer = ThreadCapturingTracer()
        actions = [
            ActionSpec("cheap", {}, {"done": True}, cost=1.0, execute=_refuse),
            ActionSpec("dear", {}, {"done": True}, cost=5.0),
        ]
        graph = GoapGraph(actions, goal=GOAL, tracer=tracer)
        graph.start(world_state={"done": False}, thread_id="replan-99")
        assert tracer.thread_ids["replan"] == "replan-99"

    def test_an_explicit_thread_id_is_not_overridden(self) -> None:
        from langgoap.tracing import SafeTracerProxy

        tracer = ThreadCapturingTracer()
        SafeTracerProxy(tracer).on_action_complete(None, thread_id="mine")
        assert tracer.thread_ids["action_complete"] == "mine"

    def test_outside_a_run_thread_id_is_none(self) -> None:
        from langgoap.tracing import SafeTracerProxy

        tracer = ThreadCapturingTracer()
        SafeTracerProxy(tracer).on_action_complete(None)
        assert tracer.thread_ids["action_complete"] is None

    def test_sensor_hook_receives_thread_id(self) -> None:
        tracer = ThreadCapturingTracer()

        def sense(ws: dict[str, Any]) -> dict[str, Any]:
            return {"sensed": True}

        graph = GoapGraph(
            [ActionSpec("flip", {"sensed": True}, {"done": True})],
            goal=GOAL,
            sensors=[sense],
            effects="observed",
            tracer=tracer,
        )
        graph.start(world_state={"sensed": True}, thread_id="sensor-1")
        assert tracer.thread_ids["sensor"] == "sensor-1"


def test_every_instance_of_a_tracer_class_gets_its_own_thread_id() -> None:
    """The hook-signature lookup is remembered per function, not per instance."""
    first, second = ThreadCapturingTracer(), ThreadCapturingTracer()
    GoapGraph(ACTIONS, goal=GOAL, tracer=first).start(
        world_state={"done": False}, thread_id="one"
    )
    GoapGraph(ACTIONS, goal=GOAL, tracer=second).start(
        world_state={"done": False}, thread_id="two"
    )
    assert first.thread_ids["action_complete"] == "one"
    assert second.thread_ids["action_complete"] == "two"


def test_a_hook_taking_keyword_arguments_receives_thread_id() -> None:
    class Catching(NullTracer):
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def on_action_complete(self, result: Any, **kwargs: Any) -> None:
            self.kwargs = kwargs

    tracer = Catching()
    GoapGraph(ACTIONS, goal=GOAL, tracer=tracer).start(
        world_state={"done": False}, thread_id="kw"
    )
    assert tracer.kwargs == {"thread_id": "kw"}


def test_multi_tracer_children_receive_thread_id() -> None:
    from langgoap.tracing import MultiTracer

    a, b = ThreadCapturingTracer(), ThreadCapturingTracer()
    graph = GoapGraph(ACTIONS, goal=GOAL, tracer=MultiTracer([a, b]))
    graph.start(world_state={"done": False}, thread_id="m")
    assert a.thread_ids["action_complete"] == "m"
    assert b.thread_ids["plan_complete"] == "m"
