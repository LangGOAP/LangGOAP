"""End-to-end integration tests for Reflexion self-critique."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalPolicy, GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.reflexion import ReflexionTracer
from langgoap.tracing import MultiTracer, NullTracer
from langgoap.types import ReplanStrategy


def test_reflexion_e2e_action_fails_then_succeeds() -> None:
    """Action fails once, reflection generated, replan succeeds."""
    fail_count = {"n": 0}

    def flaky_execute(ws: dict[str, Any]) -> dict[str, Any]:
        fail_count["n"] += 1
        if fail_count["n"] == 1:
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
        policy=GoalPolicy(replan_strategy=ReplanStrategy.ON_DEVIATION, max_replans=5),
    )

    tracer = ReflexionTracer()
    graph = GoapGraph(actions=[action], tracer=tracer)
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    # At least one reflection from the transient failure
    assert len(tracer.reflections) >= 1
    assert tracer.reflections[0].action_name == "flaky"
    assert "transient failure" in tracer.reflections[0].error


def test_reflexion_e2e_no_failures_no_reflections() -> None:
    """When all actions succeed, no reflections are generated."""
    action = ActionSpec(
        name="reliable",
        effects={"done": True},
        execute=lambda ws: {"done": True},
    )
    goal = GoalSpec(conditions={"done": True})

    tracer = ReflexionTracer()
    graph = GoapGraph(actions=[action], tracer=tracer)
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    assert len(tracer.reflections) == 0


@pytest.mark.asyncio
async def test_reflexion_e2e_async() -> None:
    """Reflexion tracer works in async graph invocation."""
    fail_once = {"done": False}

    def flaky(ws: dict[str, Any]) -> dict[str, Any]:
        if not fail_once["done"]:
            fail_once["done"] = True
            raise ValueError("async failure")
        return {"done": True}

    action = ActionSpec(
        name="async_flaky",
        effects={"done": True},
        execute=flaky,
        max_retries=1,
    )
    goal = GoalSpec(conditions={"done": True}, policy=GoalPolicy(max_replans=5))

    tracer = ReflexionTracer()
    graph = GoapGraph(actions=[action], tracer=tracer)
    result = await graph.ainvoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    assert len(tracer.reflections) >= 1


def test_reflexion_e2e_composable_with_multi_tracer() -> None:
    """ReflexionTracer works inside a MultiTracer."""
    fail_count = {"n": 0}

    def flaky(ws: dict[str, Any]) -> dict[str, Any]:
        fail_count["n"] += 1
        if fail_count["n"] == 1:
            raise RuntimeError("boom")
        return {"done": True}

    action = ActionSpec(
        name="act",
        effects={"done": True},
        execute=flaky,
        max_retries=1,
    )
    goal = GoalSpec(conditions={"done": True}, policy=GoalPolicy(max_replans=5))

    reflexion = ReflexionTracer()
    multi = MultiTracer([reflexion, NullTracer()])
    graph = GoapGraph(actions=[action], tracer=multi)
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    assert len(reflexion.reflections) >= 1


def test_reflexion_e2e_multiple_failures() -> None:
    """Multiple failures produce multiple reflections."""
    call_count = {"n": 0}

    def very_flaky(ws: dict[str, Any]) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError(f"failure_{call_count['n']}")
        return {"done": True}

    action = ActionSpec(
        name="very_flaky",
        effects={"done": True},
        execute=very_flaky,
        max_retries=2,
    )
    goal = GoalSpec(conditions={"done": True}, policy=GoalPolicy(max_replans=10))

    tracer = ReflexionTracer(max_reflections=10)
    graph = GoapGraph(actions=[action], tracer=tracer)
    result = graph.invoke(goal=goal, world_state={})

    assert result["status"] == "goal_achieved"
    assert len(tracer.reflections) >= 2
