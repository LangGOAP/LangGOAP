"""No-op :class:`~langgoap.tracing.PlanningTracer` implementation."""

from __future__ import annotations

from typing import Any


class NullTracer:
    """No-op tracer \u2014 the default when no tracer is configured.

    Every hook is a silent pass.  Zero overhead.
    """

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        pass

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        pass

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        pass

    def on_action_start(self, action: Any, state: Any) -> None:
        pass

    def on_action_complete(self, result: Any) -> None:
        pass

    def on_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        pass

    def on_strategy_chosen(self, strategy_name: str) -> None:
        pass

    def on_replan(self, reason: str, new_plan: Any) -> None:
        pass

    def on_goal_achieved(self, final_state: Any) -> None:
        pass

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        pass

    async def aon_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        pass

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
        pass

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
        pass

    async def aon_action_start(self, action: Any, state: Any) -> None:
        pass

    async def aon_action_complete(self, result: Any) -> None:
        pass

    async def aon_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        pass

    async def aon_strategy_chosen(self, strategy_name: str) -> None:
        pass

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        pass

    async def aon_goal_achieved(self, final_state: Any) -> None:
        pass

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        pass

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
        pass

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        pass

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        pass

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
        pass

    async def aon_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        pass

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        pass


__all__ = ["NullTracer"]
