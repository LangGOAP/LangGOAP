"""Tests for the sensor system (langgoap.sensors)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from langgoap.sensors import (
    AsyncSensor,
    FunctionalSensor,
    Sensor,
    run_sensors_async,
    run_sensors_sync,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class MySensor:
    """Duck-typed sync sensor."""

    @property
    def name(self) -> str:
        return "my_sensor"

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        return {"sensed": True}


class MyAsyncSensor:
    """Duck-typed async sensor."""

    @property
    def name(self) -> str:
        return "my_async_sensor"

    async def asense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        return {"async_sensed": True}


def test_sync_sensor_protocol_conformance() -> None:
    s = MySensor()
    assert isinstance(s, Sensor)


def test_async_sensor_protocol_conformance() -> None:
    s = MyAsyncSensor()
    assert isinstance(s, AsyncSensor)


# ---------------------------------------------------------------------------
# FunctionalSensor
# ---------------------------------------------------------------------------


def test_functional_sensor_construction_and_sense() -> None:
    fn = lambda ws: {"api_up": True}
    sensor = FunctionalSensor("check_api", fn)
    assert sensor.name == "check_api"
    result = sensor.sense({"existing": 1})
    assert result == {"api_up": True}


def test_functional_sensor_is_sensor() -> None:
    sensor = FunctionalSensor("s", lambda ws: {})
    assert isinstance(sensor, Sensor)


def test_functional_sensor_empty_return() -> None:
    sensor = FunctionalSensor("noop", lambda ws: {})
    result = sensor.sense({"a": 1})
    assert result == {}


# ---------------------------------------------------------------------------
# run_sensors_sync
# ---------------------------------------------------------------------------


def test_run_sensors_sync_merges_into_world_state() -> None:
    ws: dict[str, Any] = {"initial": True}
    s1 = FunctionalSensor("s1", lambda ws: {"from_s1": True})
    s2 = FunctionalSensor("s2", lambda ws: {"from_s2": ws.get("from_s1", False)})
    results = run_sensors_sync([s1, s2], ws)

    assert len(results) == 2
    assert results[0] == ("s1", {"from_s1": True})
    # s2 should see s1's update because world_state is mutated in order
    assert results[1] == ("s2", {"from_s2": True})
    assert ws == {"initial": True, "from_s1": True, "from_s2": True}


def test_run_sensors_sync_failing_sensor_logged_not_crash() -> None:
    def bad_sensor(ws: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("sensor boom")

    ws: dict[str, Any] = {}
    s1 = FunctionalSensor("bad", bad_sensor)
    s2 = FunctionalSensor("good", lambda ws: {"ok": True})
    results = run_sensors_sync([s1, s2], ws)

    # bad sensor skipped, good sensor ran
    assert len(results) == 1
    assert results[0] == ("good", {"ok": True})
    assert ws == {"ok": True}


def test_run_sensors_sync_empty_list() -> None:
    ws: dict[str, Any] = {"a": 1}
    results = run_sensors_sync([], ws)
    assert results == []
    assert ws == {"a": 1}


def test_run_sensors_sync_async_only_skipped() -> None:
    """Async-only sensors are skipped in the sync path with a warning."""
    s = MyAsyncSensor()
    ws: dict[str, Any] = {}
    results = run_sensors_sync([s], ws)  # type: ignore[arg-type]
    assert len(results) == 0


# ---------------------------------------------------------------------------
# run_sensors_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sensors_async_basic() -> None:
    s1 = FunctionalSensor("s1", lambda ws: {"from_s1": True})
    ws: dict[str, Any] = {}
    results = await run_sensors_async([s1], ws)
    assert results == [("s1", {"from_s1": True})]
    assert ws == {"from_s1": True}


@pytest.mark.asyncio
async def test_run_sensors_async_with_async_sensor() -> None:
    s = MyAsyncSensor()
    ws: dict[str, Any] = {}
    results = await run_sensors_async([s], ws)  # type: ignore[arg-type]
    assert results == [("my_async_sensor", {"async_sensed": True})]
    assert ws == {"async_sensed": True}


@pytest.mark.asyncio
async def test_run_sensors_async_failing_sensor() -> None:
    class FailingSensor:
        @property
        def name(self) -> str:
            return "fail"

        async def asense(self, ws: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("async boom")

    ws: dict[str, Any] = {}
    s_fail = FailingSensor()
    s_ok = FunctionalSensor("ok", lambda ws: {"ok": True})
    results = await run_sensors_async([s_fail, s_ok], ws)  # type: ignore[arg-type]
    assert len(results) == 1
    assert results[0] == ("ok", {"ok": True})


@pytest.mark.asyncio
async def test_run_sensors_async_merges_in_order() -> None:
    s1 = FunctionalSensor("s1", lambda ws: {"a": 1})
    s2 = FunctionalSensor("s2", lambda ws: {"b": ws.get("a", 0) + 1})
    ws: dict[str, Any] = {}
    results = await run_sensors_async([s1, s2], ws)
    assert results[1] == ("s2", {"b": 2})
    assert ws == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Sensor with GoapPlanner integration
# ---------------------------------------------------------------------------


def test_sensor_updates_world_state_before_planning() -> None:
    """Sensor-driven state feeds into A* planning."""
    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.graph.nodes import GoapPlanner

    # Action requires "api_up" to be True
    action = ActionSpec(
        name="call_api",
        preconditions={"api_up": True},
        effects={"result_ready": True},
    )
    goal = GoalSpec(conditions={"result_ready": True})

    # Without sensor: "api_up" is False → no plan
    planner_no_sensor = GoapPlanner([action])
    state_no_sensor: dict[str, Any] = {
        "goal": goal,
        "world_state": {"api_up": False},
    }
    result_no = planner_no_sensor(state_no_sensor)
    assert result_no.get("status") == "no_plan"

    # With sensor: sets "api_up" to True → plan found
    sensor = FunctionalSensor("check_api", lambda ws: {"api_up": True})
    planner_with_sensor = GoapPlanner([action], sensors=[sensor])
    state_with_sensor: dict[str, Any] = {
        "goal": goal,
        "world_state": {"api_up": False},
    }
    result_yes = planner_with_sensor(state_with_sensor)
    assert result_yes.get("plan") is not None
    assert result_yes.get("status") == "executing"


def test_sensor_called_on_every_planning_round() -> None:
    """Track that sensors are invoked each time the planner is called."""
    call_count = 0

    def counting_sensor(ws: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"count": call_count}

    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.graph.nodes import GoapPlanner

    action = ActionSpec(name="a", effects={"done": True})
    goal = GoalSpec(conditions={"done": True})
    planner = GoapPlanner(
        [action], sensors=[FunctionalSensor("counter", counting_sensor)]
    )

    state: dict[str, Any] = {"goal": goal, "world_state": {}}
    planner(state)
    assert call_count == 1
    # Simulate replan call
    planner(state)
    assert call_count == 2


def test_sensor_tracer_hook_fires() -> None:
    """on_sensor_complete tracer hook fires for each sensor."""
    events: list[tuple[str, Any]] = []

    class CapturingTracer:
        def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
            events.append((sensor_name, updates))

        def on_plan_start(self, *a: Any) -> None:
            pass

        def on_plan_complete(self, *a: Any) -> None:
            pass

        def on_plan_failed(self, *a: Any) -> None:
            pass

        def on_action_start(self, *a: Any) -> None:
            pass

        def on_action_complete(self, *a: Any) -> None:
            pass

        def on_replan(self, *a: Any) -> None:
            pass

        def on_goal_achieved(self, *a: Any) -> None:
            pass

    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.graph.nodes import GoapPlanner

    sensor = FunctionalSensor("my_sensor", lambda ws: {"x": 42})
    action = ActionSpec(name="a", effects={"done": True})
    goal = GoalSpec(conditions={"done": True})
    tracer = CapturingTracer()
    planner = GoapPlanner(
        [action],
        tracer=tracer,  # type: ignore[arg-type]
        sensors=[sensor],
    )
    planner({"goal": goal, "world_state": {}})

    assert len(events) == 1
    assert events[0] == ("my_sensor", {"x": 42})


@pytest.mark.asyncio
async def test_async_sensor_tracer_hook_fires_in_graph() -> None:
    """aon_sensor_complete fires when GoapGraph.ainvoke runs sensors."""
    async_events: list[tuple[str, Any]] = []

    class AsyncCapturingTracer:
        """Minimal tracer that records async sensor hook calls."""

        def on_plan_start(self, *a: Any) -> None: ...
        def on_plan_complete(self, *a: Any) -> None: ...
        def on_plan_failed(self, *a: Any) -> None: ...
        def on_action_start(self, *a: Any) -> None: ...
        def on_action_complete(self, *a: Any) -> None: ...
        def on_replan(self, *a: Any) -> None: ...
        def on_goal_achieved(self, *a: Any) -> None: ...
        def on_sensor_complete(self, *a: Any) -> None: ...

        async def aon_plan_start(self, *a: Any) -> None: ...
        async def aon_plan_complete(self, *a: Any) -> None: ...
        async def aon_plan_failed(self, *a: Any) -> None: ...
        async def aon_action_start(self, *a: Any) -> None: ...
        async def aon_action_complete(self, *a: Any) -> None: ...
        async def aon_replan(self, *a: Any) -> None: ...
        async def aon_goal_achieved(self, *a: Any) -> None: ...

        async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
            async_events.append((sensor_name, updates))

    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.graph.builder import GoapGraph

    sensor = FunctionalSensor("async_hook_sensor", lambda ws: {"hooked": True})
    action = ActionSpec(name="a", effects={"done": True})
    goal = GoalSpec(conditions={"done": True})
    tracer = AsyncCapturingTracer()

    result = await GoapGraph(
        actions=[action],
        sensors=[sensor],
        tracer=tracer,  # type: ignore[arg-type]
    ).ainvoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    # The async sensor hook must have fired at least once during the async path.
    assert len(async_events) >= 1
    assert async_events[0] == ("async_hook_sensor", {"hooked": True})
