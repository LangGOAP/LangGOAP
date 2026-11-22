"""Start, wake and answer a thread from the graph object (issue #96).

``GoapGraph`` takes the goal, the checkpointer and the limits once; ``start``,
``notify``, ``resume`` and ``waiting`` then work per thread.  Runs are
serialised: a call made while a run is in progress is delivered after it.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from langgoap import ActionSpec, GoalSpec, GoapGraph, GoapResult


class Shop:
    def __init__(self) -> None:
        self.kitchen = "cooking"

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        return {"kitchen": self.kitchen}


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec.wait("await_kitchen", {"kitchen": "cooking"}, {"kitchen": "ready"}),
        ActionSpec(
            "dispatch",
            {"kitchen": "ready"},
            {"delivered": True},
            require_human_approval=True,
        ),
    ]


def _graph(shop: Shop, **options: Any) -> GoapGraph:
    return GoapGraph(
        _actions(), goal={"delivered": True}, sensors=[shop.sense], **options
    )


class TestStartNotifyResume:
    def test_the_whole_life_of_a_thread(self) -> None:
        shop = Shop()
        graph = _graph(shop)

        state = graph.start(thread_id="order-1")
        assert state is not None and state["status"] != "goal_achieved"
        assert graph.waiting("order-1") == {
            "type": "goap_action_wait",
            "action": "await_kitchen",
        }

        shop.kitchen = "ready"
        graph.notify("order-1", {"kind": "kitchen.ready"})
        waiting = graph.waiting("order-1")
        assert waiting is not None and waiting["type"] == "goap_action_approval"

        done = graph.resume(True, thread_id="order-1")
        assert done is not None and done["status"] == "goal_achieved"
        assert graph.waiting("order-1") is None

    def test_start_takes_the_initial_world_and_returns_typed_state(self) -> None:
        graph = GoapGraph(
            [ActionSpec("finish", {"ready": True}, {"done": True})], goal={"done": True}
        )
        state = graph.start({"ready": True}, thread_id="t")
        assert state is not None
        assert state["status"] == "goal_achieved"
        assert isinstance(state["goal"], GoalSpec)
        assert state["execution_history"][0].action_name == "finish"

    def test_threads_are_independent(self) -> None:
        shop = Shop()
        graph = _graph(shop)
        graph.start(thread_id="a")
        graph.start(thread_id="b")
        shop.kitchen = "ready"
        graph.notify("a")
        assert graph.waiting("a")["type"] == "goap_action_approval"  # type: ignore[index]
        assert graph.waiting("b")["type"] == "goap_action_wait"  # type: ignore[index]

    def test_an_event_never_answers_a_question(self) -> None:
        shop = Shop()
        graph = _graph(shop)
        graph.start(thread_id="t")
        shop.kitchen = "ready"
        graph.notify("t")
        assert graph.notify("t", {"approved": True}) is None  # ignored: not a wait
        assert graph.waiting("t")["type"] == "goap_action_approval"  # type: ignore[index]

    def test_an_answer_never_ends_a_wait(self) -> None:
        shop = Shop()
        graph = _graph(shop)
        graph.start(thread_id="t")
        shop.kitchen = "ready"
        assert graph.resume(True, thread_id="t") is None  # ignored: nobody was asked
        assert graph.waiting("t")["type"] == "goap_action_wait"  # type: ignore[index]

    def test_notify_and_resume_on_unknown_or_finished_threads_do_nothing(self) -> None:
        graph = _graph(Shop())
        assert graph.notify("nobody") is None
        assert graph.resume(True, thread_id="nobody") is None
        assert graph.waiting("nobody") is None

    def test_a_typed_form_is_answered_through_resume(self) -> None:
        class Note(BaseModel):
            text: str

        graph = GoapGraph(
            [
                ActionSpec(
                    "sign",
                    {},
                    {"signed": True},
                    require_human_approval=Note,
                    human_input_key="note",
                )
            ],
            goal={"signed": True},
        )
        graph.start(thread_id="t")
        assert graph.waiting("t")["type"] == "goap_action_form"  # type: ignore[index]
        done = graph.resume({"text": "ok"}, thread_id="t")
        assert done is not None and done["world_state"]["note"] == {"text": "ok"}


class TestMailbox:
    def test_a_call_made_during_a_run_is_delivered_after_it(self) -> None:
        """An action publishes an event that wakes another thread."""
        shop = Shop()
        order: list[str] = []
        holder: dict[str, GoapGraph] = {}

        def cook(world_state: dict[str, Any]) -> None:
            order.append("cook:start")
            shop.kitchen = "ready"
            assert (
                holder["g"].notify("waiter") is None
            )  # queued, not run inside this action
            order.append("cook:end")

        graph = GoapGraph(
            [
                ActionSpec.wait(
                    "await_kitchen",
                    {"kitchen": "cooking", "chef": False},
                    {"kitchen": "ready"},
                ),
                ActionSpec("cook", {"chef": True}, {"kitchen": "ready"}, execute=cook),
            ],
            goal={"kitchen": "ready"},
            sensors=[shop.sense],
            on_stop=lambda thread_id, state: order.append(
                f"stop:{thread_id}:{state['status']}"
            ),
        )
        holder["g"] = graph
        graph.start({"chef": False}, thread_id="waiter")
        graph.start({"chef": True}, thread_id="cook")
        assert order == [
            "stop:waiter:executing",
            "cook:start",
            "cook:end",
            "stop:cook:goal_achieved",
            "stop:waiter:goal_achieved",
        ]

    def test_calls_made_from_on_stop_are_delivered_after_it(self) -> None:
        """on_stop reacts to each stop (an application that cancels an order
        publishes an event its own listener turns into a notify): the calls
        are queued and run once the current one is over, in order."""
        shop = Shop()
        seen: list[str] = []
        holder: dict[str, GoapGraph] = {}

        def after(thread_id: str, state: GoapResult) -> None:
            paused = state.interrupt
            seen.append(f"stop:{paused['type'] if paused else state.status}")
            if paused is None:
                return
            if paused["type"] == "goap_action_wait":
                shop.kitchen = "ready"
                assert holder["g"].notify(thread_id) is None  # queued, not run here
            else:
                assert holder["g"].resume(True, thread_id=thread_id) is None

        graph = _graph(shop, on_stop=after)
        holder["g"] = graph
        result = graph.start(thread_id="t")
        assert seen == [
            "stop:goap_action_wait",
            "stop:goap_action_approval",
            "stop:goal_achieved",
        ]
        # The caller gets the answer to its own call, the first pause.
        assert result is not None and result.interrupt == {
            "type": "goap_action_wait",
            "action": "await_kitchen",
        }
        assert graph.waiting("t") is None

    def test_on_stop_sees_the_interrupt(self) -> None:
        stops: list[Any] = []
        graph = _graph(Shop(), on_stop=lambda thread_id, state: stops.append(state))
        graph.start(thread_id="t")
        assert stops[0]["__interrupt__"][0].value["action"] == "await_kitchen"


class TestConstructorShorthands:
    def test_goal_may_be_a_goal_spec(self) -> None:
        graph = GoapGraph(
            [ActionSpec("finish", {}, {"done": True})],
            goal=GoalSpec(conditions={"done": True}),
        )
        assert graph.start(thread_id="t")["status"] == "goal_achieved"  # type: ignore[index]

    def test_start_can_override_the_goal(self) -> None:
        graph = GoapGraph(
            [
                ActionSpec("a", {}, {"a": True}),
                ActionSpec("b", {}, {"b": True}),
            ],
            goal={"a": True},
        )
        state = graph.start(thread_id="t", goal={"b": True})
        assert [r.action_name for r in state["execution_history"]] == ["b"]  # type: ignore[index]

    def test_start_without_any_goal_is_an_error(self) -> None:
        graph = GoapGraph([ActionSpec("a", {}, {"a": True})])
        with pytest.raises(ValueError, match="goal"):
            graph.start(thread_id="t")

    def test_max_actions_stops_a_run(self) -> None:
        graph = GoapGraph(
            [
                ActionSpec("one", {}, {"one": True}),
                ActionSpec("two", {"one": True}, {"two": True}),
                ActionSpec("three", {"two": True}, {"three": True}),
            ],
            goal={"three": True},
            max_actions=2,
        )
        state = graph.start(thread_id="t")
        assert state is not None
        assert state["status"] == "terminated"
        assert len(state["execution_history"]) == 2

    def test_the_given_checkpointer_is_used(self) -> None:
        saver = MemorySaver()
        graph = _graph(Shop(), checkpointer=saver)
        graph.start(thread_id="t")
        assert saver.get_tuple({"configurable": {"thread_id": "t"}}) is not None

    def test_invoke_still_works_without_any_of_it(self) -> None:
        graph = GoapGraph([ActionSpec("finish", {}, {"done": True})])
        assert (
            graph.invoke(GoalSpec(conditions={"done": True}))["status"]
            == "goal_achieved"
        )


class TestAsyncThreadApi:
    async def test_the_whole_life_of_a_thread(self) -> None:
        shop = Shop()
        graph = _graph(shop)
        state = await graph.astart(thread_id="order-1")
        assert state is not None
        assert graph.waiting("order-1")["action"] == "await_kitchen"  # type: ignore[index]
        shop.kitchen = "ready"
        await graph.anotify("order-1")
        done = await graph.aresume(True, thread_id="order-1")
        assert done is not None and done["status"] == "goal_achieved"


class TestAsyncWaiting:
    async def test_awaiting_reports_the_interrupt(self) -> None:
        shop = Shop()
        graph = _graph(shop)
        assert await graph.awaiting("order-1") is None
        await graph.astart(thread_id="order-1")
        paused = await graph.awaiting("order-1")
        assert paused == {"type": "goap_action_wait", "action": "await_kitchen"}


class TestThreadSafety:
    def test_calls_from_many_os_threads_never_overlap_and_none_is_lost(self) -> None:
        import threading
        import time

        active = 0
        peak = 0
        guard = threading.Lock()
        stopped: list[str] = []

        def work(world_state: dict[str, Any]) -> None:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.005)
            with guard:
                active -= 1

        graph = GoapGraph(
            [ActionSpec("work", {}, {"done": True}, execute=work)],
            goal={"done": True},
            on_stop=lambda thread_id, state: stopped.append(thread_id),
        )
        names = [f"t-{n}" for n in range(24)]
        workers = [
            threading.Thread(target=graph.start, kwargs={"thread_id": name})
            for name in names
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        assert peak == 1  # runs are serialised across OS threads
        assert sorted(stopped) == sorted(names)  # every call was delivered

    def test_a_failed_run_does_not_block_the_next_call(self) -> None:
        def explode(thread_id: str, state: Any) -> None:
            if thread_id == "bad":
                raise RuntimeError("on_stop failed")

        graph = GoapGraph(
            [ActionSpec("finish", {}, {"done": True})],
            goal={"done": True},
            on_stop=explode,
        )
        with pytest.raises(RuntimeError, match="on_stop failed"):
            graph.start(thread_id="bad")
        state = graph.start(thread_id="good")
        assert state is not None and state["status"] == "goal_achieved"
