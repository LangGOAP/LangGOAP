"""Observability hooks for the GOAP planning loop.

The :class:`PlanningTracer` Protocol is the extension point for
metrics, logging, LangSmith custom events, OpenTelemetry spans, and
any other cross-cutting observability concern.  Every hook has a
synchronous variant (called from :class:`GoapPlanner` / ``GoapObserver``
/ ``GoapExecutor`` in the sync path) and an asynchronous variant
(called from their ``acall`` counterparts under
:meth:`GoapGraph.ainvoke`).

A tracer is construction-time configuration on the graph — it does
**not** live in the ``GoapState`` dict because tracers are not
serialisable and checkpointers would choke on them.

Three built-in tracers are provided:

* :class:`NullTracer` — the no-op default, zero overhead.
* :class:`LoggingTracer` — routes every event to ``logging.getLogger
  ("langgoap.tracing")``.
* :class:`MultiTracer` — fans out to a list of tracers.  Exceptions
  raised by individual tracers are caught and logged so observability
  cannot break the planner (this is a hard invariant).

OpenTelemetry and LangSmith adapters are **not** shipped in the core
package — the v0.1.0 contract is zero extra pip dependencies.  See
``examples/basics/tracing_and_history.ipynb`` for ~30-line worked
examples of writing an ``OTelTracer`` or integrating with LangSmith.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("langgoap.tracing")


@runtime_checkable
class PlanningTracer(Protocol):
    """Sync + async observability hooks for the GOAP planning loop.

    Every hook is fire-and-forget.  Implementations must **never**
    raise — :class:`MultiTracer` catches and logs exceptions, but
    user-facing nodes also wrap each call in a try/except as a
    second line of defence.
    """

    # ------------------------------------------------------------------
    # Sync hooks
    # ------------------------------------------------------------------
    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None: ...

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None: ...

    def on_plan_failed(self, reason: str, duration_ms: float) -> None: ...

    def on_action_start(self, action: Any, state: Any) -> None: ...

    def on_action_complete(self, result: Any) -> None: ...

    def on_replan(self, reason: str, new_plan: Any) -> None: ...

    def on_goal_achieved(self, final_state: Any) -> None: ...

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

    async def aon_replan(self, reason: str, new_plan: Any) -> None: ...

    async def aon_goal_achieved(self, final_state: Any) -> None: ...


class NullTracer:
    """No-op tracer — the default when no tracer is configured.

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

    def on_replan(self, reason: str, new_plan: Any) -> None:
        pass

    def on_goal_achieved(self, final_state: Any) -> None:
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

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        pass

    async def aon_goal_achieved(self, final_state: Any) -> None:
        pass


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


class MultiTracer:
    """Fan a single event out to multiple tracers.

    Each inner tracer is invoked in sequence; exceptions raised by an
    individual tracer are caught and logged so one broken tracer
    cannot break the planner or stop other tracers from receiving
    the event.
    """

    def __init__(self, tracers: list[PlanningTracer]) -> None:
        self._tracers = list(tracers)

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

    def on_replan(self, reason: str, new_plan: Any) -> None:
        self._fan_sync("on_replan", reason, new_plan)

    def on_goal_achieved(self, final_state: Any) -> None:
        self._fan_sync("on_goal_achieved", final_state)

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

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        await self._fan_async("aon_replan", reason, new_plan)

    async def aon_goal_achieved(self, final_state: Any) -> None:
        await self._fan_async("aon_goal_achieved", final_state)


__all__ = [
    "PlanningTracer",
    "NullTracer",
    "LoggingTracer",
    "MultiTracer",
]
