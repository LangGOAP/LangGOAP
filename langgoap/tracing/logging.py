"""Stdlib-``logging``-backed :class:`~langgoap.tracing.PlanningTracer`."""

from __future__ import annotations

# ``import logging`` resolves to the stdlib module via Python 3's absolute
# import semantics; this file's name shadows nothing at the top level.
import logging
from typing import Any

logger = logging.getLogger("langgoap.tracing")


class LoggingTracer:
    """Tracer that routes every event through stdlib ``logging``.

    Useful for quick local debugging without setting up OTel or
    LangSmith.  Each hook logs at ``INFO`` level to the
    ``langgoap.tracing`` logger.
    """

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        logger.info("plan_start strategy=%s goal=%r", strategy_name, goal)

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        logger.info("plan_complete duration_ms=%.2f plan=%r", duration_ms, plan)

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        logger.info("plan_failed reason=%s duration_ms=%.2f", reason, duration_ms)

    def on_action_start(self, action: Any, state: Any) -> None:
        name = getattr(action, "name", repr(action))
        logger.info("action_start name=%s", name)

    def on_action_complete(self, result: Any) -> None:
        logger.info("action_complete result=%r", result)

    def on_replan(self, reason: str, new_plan: Any) -> None:
        logger.info("replan reason=%s new_plan=%r", reason, new_plan)

    def on_goal_achieved(self, final_state: Any) -> None:
        logger.info("goal_achieved final_state=%r", final_state)

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        logger.info("sensor_complete name=%s updates=%r", sensor_name, updates)

    async def aon_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.on_plan_start(goal, state, strategy_name)

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.on_plan_complete(plan, duration_ms)

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
        self.on_plan_failed(reason, duration_ms)

    async def aon_action_start(self, action: Any, state: Any) -> None:
        self.on_action_start(action, state)

    async def aon_action_complete(self, result: Any) -> None:
        self.on_action_complete(result)

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        self.on_replan(reason, new_plan)

    async def aon_goal_achieved(self, final_state: Any) -> None:
        self.on_goal_achieved(final_state)

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        self.on_sensor_complete(sensor_name, updates)

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
        logger.info(
            "search_expand node=%d parent=%s g=%.3f h=%.3f f=%.3f action=%s",
            node_id,
            parent_id,
            g,
            h,
            f,
            action_name,
        )

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        logger.info("search_dead_end reason=%s detail=%r", reason, detail)

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        logger.info(
            "search_complete nodes_explored=%d duration_ms=%.2f found=%s",
            nodes_explored,
            duration_ms,
            found,
        )

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
        self.on_search_expand(node_id, state, g, h, f, parent_id, action_name)

    async def aon_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        self.on_search_dead_end(reason, detail)

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.on_search_complete(nodes_explored, duration_ms, found)


__all__ = ["LoggingTracer"]
