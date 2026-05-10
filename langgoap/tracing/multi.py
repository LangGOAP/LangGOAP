"""Fan-out :class:`~langgoap.tracing.PlanningTracer` composer."""

from __future__ import annotations

import logging
from typing import Any

from langgoap.tracing._protocol import PlanningTracer

logger = logging.getLogger("langgoap.tracing")


class MultiTracer:
    """Fan a single event out to multiple tracers.

    Each inner tracer is invoked in sequence; exceptions raised by an
    individual tracer are caught and logged so one broken tracer
    cannot break the planner or stop other tracers from receiving
    the event.
    """

    def __init__(self, tracers: list[PlanningTracer]) -> None:
        self._tracers = list(tracers)

    @property
    def reflections(self) -> list[Any]:
        """Aggregate reflections from every inner tracer that exposes them.

        Enables :class:`~langgoap.graph.nodes.GoapPlanner` to surface
        reflection context into :class:`~langgoap.graph.state.GoapState`
        transparently when a :class:`~langgoap.reflexion.ReflexionTracer`
        is composed inside a ``MultiTracer``.
        """
        result: list[Any] = []
        for t in self._tracers:
            result.extend(getattr(t, "reflections", []))
        return result

    def _fan_sync(self, method: str, *args: Any) -> None:
        for t in self._tracers:
            try:
                getattr(t, method)(*args)
            except Exception as exc:
                logger.warning(
                    "Tracer %r raised during %s: %s",
                    type(t).__name__,
                    method,
                    exc,
                )

    async def _fan_async(self, method: str, *args: Any) -> None:
        for t in self._tracers:
            try:
                await getattr(t, method)(*args)
            except Exception as exc:
                logger.warning(
                    "Tracer %r raised during %s: %s",
                    type(t).__name__,
                    method,
                    exc,
                )

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self._fan_sync("on_plan_start", goal, state, strategy_name)

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self._fan_sync("on_plan_complete", plan, duration_ms)

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        self._fan_sync("on_plan_failed", reason, duration_ms)

    def on_action_start(self, action: Any, state: Any) -> None:
        self._fan_sync("on_action_start", action, state)

    def on_action_complete(self, result: Any) -> None:
        self._fan_sync("on_action_complete", result)

    def on_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        self._fan_sync(
            "on_action_retry", action, attempt, exception, backoff_ms
        )

    def on_strategy_chosen(self, strategy_name: str) -> None:
        self._fan_sync("on_strategy_chosen", strategy_name)

    def on_replan(self, reason: str, new_plan: Any) -> None:
        self._fan_sync("on_replan", reason, new_plan)

    def on_goal_achieved(self, final_state: Any) -> None:
        self._fan_sync("on_goal_achieved", final_state)

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        self._fan_sync("on_sensor_complete", sensor_name, updates)

    def on_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self._fan_sync(
            "on_search_expand", node_id, state, g, h, f, parent_id, action_name
        )

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        self._fan_sync("on_search_dead_end", reason, detail)

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self._fan_sync("on_search_complete", nodes_explored, duration_ms, found)

    async def aon_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        await self._fan_async("aon_plan_start", goal, state, strategy_name)

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
        await self._fan_async("aon_plan_complete", plan, duration_ms)

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
        await self._fan_async("aon_plan_failed", reason, duration_ms)

    async def aon_action_start(self, action: Any, state: Any) -> None:
        await self._fan_async("aon_action_start", action, state)

    async def aon_action_complete(self, result: Any) -> None:
        await self._fan_async("aon_action_complete", result)

    async def aon_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        await self._fan_async(
            "aon_action_retry", action, attempt, exception, backoff_ms
        )

    async def aon_strategy_chosen(self, strategy_name: str) -> None:
        await self._fan_async("aon_strategy_chosen", strategy_name)

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        await self._fan_async("aon_replan", reason, new_plan)

    async def aon_goal_achieved(self, final_state: Any) -> None:
        await self._fan_async("aon_goal_achieved", final_state)

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        await self._fan_async("aon_sensor_complete", sensor_name, updates)

    async def aon_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        await self._fan_async(
            "aon_search_expand", node_id, state, g, h, f, parent_id, action_name
        )

    async def aon_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        await self._fan_async("aon_search_dead_end", reason, detail)

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        await self._fan_async("aon_search_complete", nodes_explored, duration_ms, found)


__all__ = ["MultiTracer"]
