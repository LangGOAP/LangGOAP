"""Tracers that implement only part of the protocol never break execution
(issue #36).

``SafeTracerProxy`` used to resolve the attribute outside its exception
guard, so a tracer missing a newer hook (``on_action_retry``,
``on_strategy_chosen``, the ``on_search_*`` family) raised ``AttributeError``
at the call site: an ``ActionQos`` retry turned into an action failure and
a replan.  ``ReflexionTracer`` itself lacked seven hooks.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.qos import ActionQos
from langgoap.reflexion import ReflexionTracer
from langgoap.tracing import PlanningTracer
from langgoap.tracing._safe_proxy import SafeTracerProxy


class _OnlyPlanComplete:
    """A legacy tracer that predates most of the protocol."""

    def __init__(self) -> None:
        self.plans: list[Any] = []

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.plans.append(plan)


def _flaky_actions(calls: dict[str, int]) -> list[ActionSpec]:
    def flaky(ws: dict[str, Any]) -> dict[str, Any]:
        calls["n"] = calls.get("n", 0) + 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return {"done": True}

    return [
        ActionSpec(
            name="flaky",
            effects={"done": True},
            execute=flaky,
            qos=ActionQos(max_attempts=5, backoff_initial_ms=0.0, backoff_max_ms=0.0),
        )
    ]


def _protocol_hooks() -> list[str]:
    return [
        name
        for name, member in inspect.getmembers(PlanningTracer)
        if callable(member) and (name.startswith("on_") or name.startswith("aon_"))
    ]


class TestSafeTracerProxyMissingHooks:
    def test_missing_sync_hook_is_a_no_op(self) -> None:
        proxy = SafeTracerProxy(_OnlyPlanComplete())
        assert proxy.on_action_retry(None, 1, RuntimeError("x"), 0.0) is None
        assert proxy.on_strategy_chosen("astar") is None
        assert proxy.on_search_complete(1, 0.0, True) is None

    @pytest.mark.asyncio
    async def test_missing_async_hook_is_an_awaitable_no_op(self) -> None:
        proxy = SafeTracerProxy(_OnlyPlanComplete())
        assert await proxy.aon_action_retry(None, 1, RuntimeError("x"), 0.0) is None
        assert await proxy.aon_strategy_chosen("astar") is None

    def test_non_hook_attributes_still_raise(self) -> None:
        proxy = SafeTracerProxy(_OnlyPlanComplete())
        with pytest.raises(AttributeError):
            proxy.not_a_hook_at_all

    def test_present_hooks_are_still_forwarded(self) -> None:
        inner = _OnlyPlanComplete()
        proxy = SafeTracerProxy(inner)
        proxy.on_plan_complete("plan", 1.0)
        assert inner.plans == ["plan"]


class TestPartialTracerDoesNotChangeOutcomes:
    def test_qos_retries_in_place_with_a_partial_tracer(self) -> None:
        calls: dict[str, int] = {}
        tracer = _OnlyPlanComplete()
        result = GoapGraph(_flaky_actions(calls), tracer=tracer).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 0, "retries happen inside the executor"
        assert calls["n"] == 3
        assert all(h.success for h in result["execution_history"])
        assert len(tracer.plans) == 1

    @pytest.mark.asyncio
    async def test_async_path_with_a_partial_tracer(self) -> None:
        calls: dict[str, int] = {}
        result = await GoapGraph(
            _flaky_actions(calls), tracer=_OnlyPlanComplete()
        ).ainvoke(goal=GoalSpec(conditions={"done": True}), world_state={})
        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 0

    def test_reflexion_tracer_retries_in_place(self) -> None:
        calls: dict[str, int] = {}
        result = GoapGraph(_flaky_actions(calls), tracer=ReflexionTracer()).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 0
        assert not [h for h in result["execution_history"] if not h.success]


class TestReflexionTracerImplementsTheWholeProtocol:
    def test_every_protocol_hook_exists(self) -> None:
        missing = [
            name for name in _protocol_hooks() if not hasattr(ReflexionTracer, name)
        ]
        assert missing == []

    def test_satisfies_the_runtime_protocol(self) -> None:
        assert isinstance(ReflexionTracer(), PlanningTracer)
