"""The :class:`PlanningTracer` Protocol.

Lives in its own module so concrete tracer implementations can import
just the Protocol without dragging in the LangSmith adapter and its
optional dependencies.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PlanningTracer(Protocol):
    """Sync + async observability hooks for the GOAP planning loop.

    Every hook is fire-and-forget.  Implementations must **never**
    raise \u2014 :class:`~langgoap.tracing.multi.MultiTracer` catches and
    logs exceptions, but user-facing nodes also wrap each call in a
    try/except as a second line of defence.
    """

    # ------------------------------------------------------------------
    # Sync hooks
    # ------------------------------------------------------------------
    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None: ...

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None: ...

    def on_plan_failed(self, reason: str, duration_ms: float) -> None: ...

    def on_action_start(self, action: Any, state: Any) -> None: ...

    def on_action_complete(self, result: Any) -> None:
        """Called immediately after an action node returns.

        ``result`` is the :class:`~langgoap.graph.state.GoapState` update
        dict produced by :class:`~langgoap.graph.nodes.GoapExecutor`.  Key
        fields to inspect:

        * ``result.get("status")`` \u2014 ``"executing"`` on success,
          ``"action_failed"`` on failure.
        * ``result.get("execution_history", [])`` \u2014 list of
          :class:`~langgoap.graph.state.ActionResult` objects for the
          current planning round.
        * ``result.get("world_state")`` \u2014 updated world state after the
          action ran.

        Implementations must **never** raise.
        """
        ...

    def on_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        """Called between retry attempts inside the executor.

        Fires when an action's ``execute`` / ``aexecute`` raised, the
        action's :class:`~langgoap.qos.ActionQos` policy approved a
        retry, and the executor is about to sleep ``backoff_ms``
        before re-invoking.  ``attempt`` is 1-indexed and refers to
        the attempt that just failed.  Implementations must **never**
        raise.
        """
        ...

    def on_strategy_chosen(self, strategy_name: str) -> None:
        """Called when :class:`~langgoap.planner.router.StrategyRouter`
        picks a planning strategy.  ``strategy_name`` is the simple
        class name.
        """
        ...

    def on_replan(self, reason: str, new_plan: Any) -> None: ...

    def on_goal_achieved(self, final_state: Any) -> None: ...

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None: ...

    # ------------------------------------------------------------------
    # A* search-tree hooks (gated by ``record_expansions`` on the graph /
    # planner \u2014 silent by default because per-expansion firehose is
    # high-volume and expensive to persist).  Aligned with the
    # OpenTelemetry GenAI convention of capturing high-volume content
    # as span *events* rather than nested child spans; LangSmith maps
    # OTel events to run events natively.
    # ------------------------------------------------------------------
    def on_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None: ...

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None: ...

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None: ...

    # ------------------------------------------------------------------
    # Async hooks (parity with CLAUDE.md dual-implementation pattern)
    # ------------------------------------------------------------------
    async def aon_plan_start(
        self, goal: Any, state: Any, strategy_name: str
    ) -> None: ...

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None: ...

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None: ...

    async def aon_action_start(self, action: Any, state: Any) -> None: ...

    async def aon_action_complete(self, result: Any) -> None: ...

    async def aon_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None: ...

    async def aon_strategy_chosen(self, strategy_name: str) -> None: ...

    async def aon_replan(self, reason: str, new_plan: Any) -> None: ...

    async def aon_goal_achieved(self, final_state: Any) -> None: ...

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None: ...

    async def aon_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None: ...

    async def aon_search_dead_end(
        self, reason: str, detail: dict[str, Any]
    ) -> None: ...

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None: ...


__all__ = ["PlanningTracer"]
