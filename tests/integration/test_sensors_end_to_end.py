"""End-to-end integration tests for the sensor system."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.sensors import FunctionalSensor
from langgoap.types import ReplanStrategy


def test_sensor_e2e_sensor_discovers_state_enabling_plan() -> None:
    """Full graph: sensor sets a precondition, enabling A* to find a plan."""
    sensor = FunctionalSensor("discover", lambda ws: {"data_available": True})

    action = ActionSpec(
        name="process",
        preconditions={"data_available": True},
        effects={"processed": True},
        execute=lambda ws: {"processed": True},
    )

    goal = GoalSpec(conditions={"processed": True})
    graph = GoapGraph(actions=[action], sensors=[sensor])
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    assert result["world_state"]["data_available"] is True
    assert result["world_state"]["processed"] is True


def test_sensor_e2e_sensor_updates_on_every_replan() -> None:
    """Sensor is called on every planning round (including replans).

    Setup: action always fails the first time. Sensor increments a counter.
    With EVERY_ACTION replan strategy, the sensor should be called on replans.
    """
    call_count = 0
    fail_once = {"failed": False}

    def counting_sensor(ws: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"sensor_count": call_count}

    def flaky_execute(ws: dict[str, Any]) -> dict[str, Any]:
        if not fail_once["failed"]:
            fail_once["failed"] = True
            raise RuntimeError("transient failure")
        return {"done": True}

    action = ActionSpec(
        name="flaky",
        effects={"done": True},
        execute=flaky_execute,
        max_retries=1,
    )

    goal = GoalSpec(
        conditions={"done": True},
        replan_strategy=ReplanStrategy.ON_DEVIATION,
        max_replans=5,
    )

    sensor = FunctionalSensor("counter", counting_sensor)
    graph = GoapGraph(actions=[action], sensors=[sensor])
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    # Sensor should have been called at least twice (initial plan + replan)
    assert call_count >= 2


@pytest.mark.asyncio
async def test_sensor_e2e_async_sensor_in_graph() -> None:
    """Async sensors work via ainvoke."""

    class MyAsyncSensor:
        @property
        def name(self) -> str:
            return "async_check"

        async def asense(self, ws: dict[str, Any]) -> dict[str, Any]:
            return {"checked": True}

    action = ActionSpec(
        name="act",
        preconditions={"checked": True},
        effects={"done": True},
        execute=lambda ws: {"done": True},
    )

    goal = GoalSpec(conditions={"done": True})
    sensor = MyAsyncSensor()
    graph = GoapGraph(actions=[action], sensors=[sensor])  # type: ignore[arg-type]
    result = await graph.ainvoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    assert result["world_state"]["checked"] is True


def test_sensor_e2e_multiple_sensors_ordered() -> None:
    """Multiple sensors run in order, each seeing previous sensors' updates."""
    s1 = FunctionalSensor("s1", lambda ws: {"step1": True})
    s2 = FunctionalSensor("s2", lambda ws: {"step2": ws.get("step1", False)})

    action = ActionSpec(
        name="act",
        preconditions={"step2": True},
        effects={"done": True},
        execute=lambda ws: {"done": True},
    )

    goal = GoalSpec(conditions={"done": True})
    graph = GoapGraph(actions=[action], sensors=[s1, s2])
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    assert result["world_state"]["step1"] is True
    assert result["world_state"]["step2"] is True


def test_sensor_e2e_failing_sensor_does_not_block() -> None:
    """A crashing sensor doesn't prevent planning."""

    def boom(ws: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("sensor crash")

    sensor_bad = FunctionalSensor("bad", boom)
    sensor_good = FunctionalSensor("good", lambda ws: {"ready": True})

    action = ActionSpec(
        name="act",
        preconditions={"ready": True},
        effects={"done": True},
        execute=lambda ws: {"done": True},
    )

    goal = GoalSpec(conditions={"done": True})
    graph = GoapGraph(actions=[action], sensors=[sensor_bad, sensor_good])
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
