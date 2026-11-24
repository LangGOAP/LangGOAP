"""``ActionSpec.wait``: an action that pauses until the world changes (issue #96).

The wait interrupts the graph; every resume makes it look again through the
sensors, and it completes only when one of its keys differs from the value it
had when the wait began.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgoap import ActionSpec, GoalSpec, GoapGraph


class Kitchen:
    def __init__(self) -> None:
        self.status = "cooking"
        self.resumed = 0
        self.prepared = 0

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        return {"kitchen": self.status, "clock": "tick"}


def _actions(kitchen: Kitchen, **wait_options: Any) -> list[ActionSpec]:
    return [
        ActionSpec.wait(
            "await_kitchen",
            {"kitchen": "cooking"},
            {"kitchen": "ready"},
            **wait_options,
        ),
        ActionSpec("dispatch", {"kitchen": "ready"}, {"delivered": True}),
        ActionSpec(
            "refund",
            {"kitchen": "rejected"},
            {"delivered": True, "refunded": True},
            cost=9.0,
        ),
    ]


def _start(
    kitchen: Kitchen, **wait_options: Any
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    graph = GoapGraph(_actions(kitchen, **wait_options), sensors=[kitchen.sense])
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "order-1"}}
    paused = compiled.invoke(
        graph.initial_state(GoalSpec(conditions={"delivered": True})), config
    )
    return compiled, config, paused


class TestWait:
    def test_it_pauses_with_a_wait_payload(self) -> None:
        _, _, paused = _start(Kitchen())
        assert [i.value for i in paused["__interrupt__"]] == [
            {"type": "goap_action_wait", "action": "await_kitchen"}
        ]
        assert paused["execution_history"] == []

    def test_a_resume_without_a_change_keeps_waiting(self) -> None:
        compiled, config, _ = _start(Kitchen())
        again = compiled.invoke(Command(resume={"kind": "noise"}), config)
        assert [i.value["action"] for i in again["__interrupt__"]] == ["await_kitchen"]
        assert again["execution_history"] == []

    def test_a_change_in_its_keys_ends_the_wait(self) -> None:
        kitchen = Kitchen()
        compiled, config, _ = _start(kitchen)
        kitchen.status = "ready"
        done = compiled.invoke(Command(resume="look again"), config)
        assert done["status"] == "goal_achieved"
        assert [r["action_name"] for r in done["execution_history"]] == [
            "await_kitchen",
            "dispatch",
        ]

    def test_the_truth_is_returned_whatever_the_plan_expected(self) -> None:
        """The kitchen rejects: the wait ends with that, and the planner re-plans."""
        kitchen = Kitchen()
        compiled, config, _ = _start(kitchen)
        kitchen.status = "rejected"
        done = compiled.invoke(Command(resume="look again"), config)
        assert done["status"] == "goal_achieved"
        assert done["world_state"]["refunded"] is True
        assert [r["action_name"] for r in done["execution_history"]] == [
            "await_kitchen",
            "refund",
        ]

    def test_only_its_own_keys_are_watched_unless_told(self) -> None:
        kitchen = Kitchen()
        compiled, config, paused = _start(kitchen)
        assert paused["world_state"]["clock"] == "tick"

        def sense(world_state: dict[str, Any]) -> dict[str, Any]:
            return {"kitchen": "cooking", "clock": "tock"}

        kitchen.sense = sense  # type: ignore[method-assign]
        again = compiled.invoke(Command(resume="look again"), config)
        assert "__interrupt__" in again  # "clock" is not this wait's business

    def test_observe_adds_watched_keys(self) -> None:
        kitchen = Kitchen()
        graph = GoapGraph(
            [
                ActionSpec.wait(
                    "await_kitchen",
                    {"kitchen": "cooking"},
                    {"kitchen": "ready"},
                    observe=("alarm",),
                ),
                ActionSpec("dispatch", {"kitchen": "ready"}, {"delivered": True}),
                ActionSpec("evacuate", {"alarm": True}, {"delivered": True}, cost=0.5),
            ],
            sensors=[
                lambda ws: {"kitchen": "cooking", "alarm": kitchen.status == "fire"}
            ],
        )
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t"}}
        compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"delivered": True})), config
        )
        kitchen.status = "fire"
        done = compiled.invoke(Command(resume="look again"), config)
        assert [r["action_name"] for r in done["execution_history"]] == [
            "await_kitchen",
            "evacuate",
        ]

    def test_on_resume_runs_before_looking_again(self) -> None:
        kitchen = Kitchen()

        def on_resume(world_state: dict[str, Any]) -> None:
            kitchen.resumed += 1
            kitchen.status = "ready"  # e.g. confirm the payment the event announced

        compiled, config, _ = _start(kitchen, on_resume=on_resume)
        assert kitchen.resumed == 0
        done = compiled.invoke(Command(resume="look again"), config)
        assert kitchen.resumed == 1
        assert done["status"] == "goal_achieved"

    def test_execute_prepares_the_wait_before_the_pause(self) -> None:
        kitchen = Kitchen()

        def prepare(world_state: dict[str, Any]) -> None:
            kitchen.prepared += 1

        _start(kitchen, execute=prepare)
        assert kitchen.prepared == 1

    def test_payload_extends_the_interrupt(self) -> None:
        _, _, paused = _start(Kitchen(), payload={"waiting": "the kitchen"})
        assert paused["__interrupt__"][0].value == {
            "type": "goap_action_wait",
            "action": "await_kitchen",
            "waiting": "the kitchen",
        }

    async def test_async_path_waits_too(self) -> None:
        kitchen = Kitchen()
        resumed: list[str] = []

        async def on_resume(world_state: dict[str, Any]) -> None:
            resumed.append("async")

        graph = GoapGraph(
            _actions(kitchen, on_resume=on_resume), sensors=[kitchen.sense]
        )
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "order-1"}}
        paused = await compiled.ainvoke(
            graph.initial_state(GoalSpec(conditions={"delivered": True})), config
        )
        assert paused["__interrupt__"][0].value["type"] == "goap_action_wait"
        kitchen.status = "ready"
        done = await compiled.ainvoke(Command(resume="look again"), config)
        assert done["status"] == "goal_achieved"
        assert resumed == ["async"]

    @pytest.mark.parametrize("executor", ["sequential", "parallel"])
    def test_each_resume_looks_once(self, executor: str) -> None:
        """A notification that changes nothing costs one on_resume and one sensor
        pass, however many came before it: the wait does not replay its history."""
        kitchen = Kitchen()

        def on_resume(world_state: dict[str, Any]) -> None:
            kitchen.resumed += 1

        def prepare(world_state: dict[str, Any]) -> None:
            kitchen.prepared += 1

        sensed: list[int] = []

        def sense(world_state: dict[str, Any]) -> dict[str, Any]:
            sensed.append(1)
            return {"kitchen": kitchen.status}

        graph = GoapGraph(
            _actions(kitchen, on_resume=on_resume, execute=prepare),
            goal={"delivered": True},
            sensors=[sense],
            executor=executor,
        )
        graph.start(thread_id="order-1")
        assert (kitchen.prepared, kitchen.resumed, len(sensed)) == (1, 0, 1)
        for n in range(1, 6):
            before = len(sensed)
            graph.notify("order-1", {"kind": "noise"})
            assert graph.waiting_action("order-1") == "await_kitchen"
            assert kitchen.resumed == n
            assert len(sensed) - before == 1
            # A notification that changes nothing does not run the graph (#116),
            # so the execute that paused is not replayed by it.
            assert kitchen.prepared == 1
        kitchen.status = "ready"
        done = graph.notify("order-1")
        assert done is not None and done.succeeded
        assert done.actions == ["await_kitchen", "dispatch"]
        assert done["prepared_wait"] is None
        # The run that ended the wait replayed the paused task once; the wait
        # was not yet marked prepared, so its execute ran a second time.
        assert kitchen.prepared == 2

    def test_what_execute_returned_reaches_the_final_state(self) -> None:
        """However many idle notifications the wait sat through."""
        kitchen = Kitchen()
        graph = GoapGraph(
            _actions(kitchen, execute=lambda ws: {"ticket": "k-7"}),
            goal={"delivered": True},
            sensors=[kitchen.sense],
        )
        graph.start(thread_id="order-1")
        for _ in range(3):
            graph.notify("order-1")
        kitchen.status = "ready"
        done = graph.notify("order-1")
        assert done is not None and done.succeeded
        assert done.world_state["ticket"] == "k-7"

    def test_a_new_plan_prepares_the_wait_afresh(self) -> None:
        """The marker a paused wait leaves is cleared by the planner, so a
        later plan that waits by the same name runs its execute again."""
        kitchen = Kitchen()
        prepared: list[str] = []
        actions = [
            ActionSpec.wait(
                "await_kitchen",
                {"kitchen": "cooking"},
                {"kitchen": "ready"},
                execute=lambda ws: prepared.append(kitchen.status),
            ),
            ActionSpec("dispatch", {"kitchen": "ready"}, {"delivered": True}),
            ActionSpec(
                "recook",
                {"kitchen": "rejected"},
                {"kitchen": "cooking"},
                execute=lambda ws: setattr(kitchen, "status", "cooking"),
            ),
        ]
        graph = GoapGraph(actions, goal={"delivered": True}, sensors=[kitchen.sense])
        graph.start(thread_id="o")
        graph.notify("o")  # idle: the wait is marked prepared
        first_plan = len(prepared)
        kitchen.status = "rejected"
        graph.notify("o")  # the wait ends with the truth; re-plan: recook, wait again
        assert graph.waiting_action("o") == "await_kitchen"
        assert len(prepared) > first_plan  # prepared again under the new plan
        kitchen.status = "ready"
        done = graph.notify("o")
        assert done is not None and done.succeeded

    @pytest.mark.parametrize("executor", ["sequential", "parallel"])
    async def test_each_resume_looks_once_on_the_async_path(
        self, executor: str
    ) -> None:
        kitchen = Kitchen()
        resumed: list[str] = []

        async def on_resume(world_state: dict[str, Any]) -> None:
            resumed.append("look")

        graph = GoapGraph(
            _actions(kitchen, on_resume=on_resume),
            goal={"delivered": True},
            sensors=[kitchen.sense],
            executor=executor,
        )
        await graph.astart(thread_id="order-1")
        for n in range(1, 6):
            await graph.anotify("order-1")
            assert len(resumed) == n
        kitchen.status = "ready"
        done = await graph.anotify("order-1")
        assert done is not None and done.succeeded

    def test_parallel_executor_runs_a_wait_alone(self) -> None:
        kitchen = Kitchen()
        ran: list[str] = []
        actions = [
            ActionSpec.wait(
                "await_kitchen", {"kitchen": "cooking"}, {"kitchen": "ready"}
            ),
            ActionSpec(
                "print_label",
                {},
                {"label": True},
                execute=lambda ws: ran.append("label"),
            ),
        ]
        graph = GoapGraph(actions, sensors=[kitchen.sense], executor="parallel")
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t"}}
        goal = GoalSpec(conditions={"kitchen": "ready", "label": True})
        compiled.invoke(graph.initial_state(goal), config)
        kitchen.status = "ready"
        done = compiled.invoke(Command(resume="look again"), config)
        assert done["status"] == "goal_achieved"
        assert ran == ["label"]  # never run twice around the pause


class TestWaitDeclaration:
    def test_wait_builds_an_action_spec(self) -> None:
        spec = ActionSpec.wait("w", {"a": 1}, {"a": 2}, cost=3.0, observe=("b",))
        assert isinstance(spec, ActionSpec)
        assert spec.is_wait is True
        assert (spec.name, spec.cost, spec.observe) == ("w", 3.0, ("b",))
        assert ActionSpec("plain", {}, {"a": 1}).is_wait is False

    def test_is_wait_and_payload_round_trip(self) -> None:
        spec = ActionSpec.wait("w", {}, {"a": 2}, payload={"waiting": "x"})
        restored = ActionSpec.from_dict(spec.to_dict())
        assert restored.is_wait is True
        assert dict(restored.wait_payload or {}) == {"waiting": "x"}

    def test_a_wait_without_sensors_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sensors"):
            GoapGraph([ActionSpec.wait("w", {}, {"a": 2})])

    def test_on_resume_outside_a_wait_is_refused(self) -> None:
        with pytest.raises(ValueError, match="wait"):
            ActionSpec("a", {}, {"x": True}, on_resume=lambda ws: None)
