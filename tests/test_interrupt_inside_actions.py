"""An action may pause the graph with ``interrupt()``; the executor must let it through.

LangGOAP's promise is that the runtime is plain LangGraph, so ``interrupt()``
works. Inside an action's ``execute`` it raises ``GraphInterrupt``, which the
executor's failure handling must never mistake for a failed action.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from langgoap import ActionSpec, GoalSpec, GoapGraph


def _wait_for_kitchen(world_state: dict) -> dict:
    event = interrupt({"waiting": "kitchen"})
    return {"kitchen": event["kitchen"]}


def _dispatch(world_state: dict) -> dict:
    return {"delivered": True}


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            "await_kitchen",
            {"kitchen": "pending"},
            {"kitchen": "ready"},
            cost=1.0,
            execute=_wait_for_kitchen,
        ),
        ActionSpec(
            "dispatch",
            {"kitchen": "ready"},
            {"delivered": True},
            cost=1.0,
            execute=_dispatch,
        ),
    ]


def test_interrupt_inside_execute_pauses_and_resumes() -> None:
    graph = GoapGraph(_actions())
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "order-1"}}
    paused = compiled.invoke(
        graph.initial_state(
            GoalSpec(conditions={"delivered": True}), {"kitchen": "pending"}
        ),
        config,
    )
    assert [i.value for i in paused["__interrupt__"]] == [{"waiting": "kitchen"}]
    assert (
        paused["execution_history"] == []
    )  # nothing failed, nothing ran to completion
    assert paused["blacklisted_actions"] == []
    done = compiled.invoke(Command(resume={"kitchen": "ready"}), config)
    assert done["status"] == "goal_achieved"
    assert done["world_state"]["delivered"] is True
    assert [(r["action_name"], r["success"]) for r in done["execution_history"]] == [
        ("await_kitchen", True),
        ("dispatch", True),
    ]


def test_an_unexpected_outcome_after_the_pause_makes_the_planner_replan() -> None:
    """The wait returns what really happened; a precondition fails; a new plan follows."""

    def replace(world_state: dict) -> dict:
        return {"kitchen": "pending", "replaced": True}

    actions = _actions() + [
        ActionSpec(
            "replace_order",
            {"kitchen": "rejected"},
            {"kitchen": "pending", "replaced": True},
            cost=4.0,
            execute=replace,
        )
    ]
    graph = GoapGraph(actions)
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "order-2"}}
    compiled.invoke(
        graph.initial_state(
            GoalSpec(conditions={"delivered": True}), {"kitchen": "pending"}
        ),
        config,
    )
    rejected = compiled.invoke(Command(resume={"kitchen": "rejected"}), config)
    assert rejected["replan_count"] == 1
    assert [i.value for i in rejected["__interrupt__"]] == [
        {"waiting": "kitchen"}
    ]  # waiting again
    assert [
        (r["action_name"], r["success"]) for r in rejected["execution_history"]
    ] == [
        ("await_kitchen", True),
        ("replace_order", True),
    ]
    done = compiled.invoke(Command(resume={"kitchen": "ready"}), config)
    assert done["status"] == "goal_achieved" and done["world_state"]["replaced"] is True


@pytest.mark.asyncio
async def test_interrupt_inside_aexecute_pauses_and_resumes() -> None:
    async def wait(world_state: dict) -> dict:
        event = interrupt({"waiting": "kitchen"})
        return {"kitchen": event["kitchen"]}

    async def dispatch(world_state: dict) -> dict:
        return {"delivered": True}

    actions = [
        ActionSpec(
            "await_kitchen", {"kitchen": "pending"}, {"kitchen": "ready"}, aexecute=wait
        ),
        ActionSpec(
            "dispatch", {"kitchen": "ready"}, {"delivered": True}, aexecute=dispatch
        ),
    ]
    graph = GoapGraph(actions)
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "order-3"}}
    paused = await compiled.ainvoke(
        graph.initial_state(
            GoalSpec(conditions={"delivered": True}), {"kitchen": "pending"}
        ),
        config,
    )
    assert [i.value for i in paused["__interrupt__"]] == [{"waiting": "kitchen"}]
    done = await compiled.ainvoke(Command(resume={"kitchen": "ready"}), config)
    assert (
        done["status"] == "goal_achieved" and done["world_state"]["delivered"] is True
    )


def test_interrupt_inside_execute_with_the_parallel_executor() -> None:
    graph = GoapGraph(_actions(), executor="parallel")
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "order-4"}}
    paused = compiled.invoke(
        graph.initial_state(
            GoalSpec(conditions={"delivered": True}), {"kitchen": "pending"}
        ),
        config,
    )
    assert [i.value for i in paused["__interrupt__"]] == [{"waiting": "kitchen"}]
    done = compiled.invoke(Command(resume={"kitchen": "ready"}), config)
    assert (
        done["status"] == "goal_achieved" and done["world_state"]["delivered"] is True
    )
