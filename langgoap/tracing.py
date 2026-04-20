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

Four built-in tracers are provided:

* :class:`NullTracer` — the no-op default, zero overhead.
* :class:`LoggingTracer` — routes every event to ``logging.getLogger
  ("langgoap.tracing")``.
* :class:`MultiTracer` — fans out to a list of tracers.  Exceptions
  raised by individual tracers are caught and logged so observability
  cannot break the planner (this is a hard invariant).
* :class:`LangSmithTracer` — emits each planning cycle and action
  execution as LangSmith runs.  ``langsmith`` is already a transitive
  dependency via ``langchain-core``, so this adapter ships in-tree
  with no new pip requirements.  Falls back to no-op behaviour when
  no ``LANGCHAIN_API_KEY`` / ``LANGSMITH_API_KEY`` is configured.

An OpenTelemetry adapter is still out of scope for v0.1.0; write it
as ~30 lines of user code following the :class:`LangSmithTracer`
pattern if needed.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
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

    def on_action_complete(self, result: Any) -> None:
        """Called immediately after an action node returns.

        ``result`` is the :class:`~langgoap.graph.state.GoapState` update
        dict produced by :class:`~langgoap.graph.nodes.GoapExecutor`.  Key
        fields to inspect:

        * ``result.get("status")`` — ``"executing"`` on success,
          ``"action_failed"`` on failure.
        * ``result.get("execution_history", [])`` — list of
          :class:`~langgoap.graph.state.ActionResult` objects for the
          current planning round.
        * ``result.get("world_state")`` — updated world state after the
          action ran.

        Implementations must **never** raise.
        """
        ...

    def on_replan(self, reason: str, new_plan: Any) -> None: ...

    def on_goal_achieved(self, final_state: Any) -> None: ...

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None: ...

    # ------------------------------------------------------------------
    # A* search-tree hooks (gated by ``record_expansions`` on the graph /
    # planner — silent by default because per-expansion firehose is
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
            "on_search_expand",
            node_id,
            state,
            g,
            h,
            f,
            parent_id,
            action_name,
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
            "aon_search_expand",
            node_id,
            state,
            g,
            h,
            f,
            parent_id,
            action_name,
        )

    async def aon_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        await self._fan_async("aon_search_dead_end", reason, detail)

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        await self._fan_async("aon_search_complete", nodes_explored, duration_ms, found)


def _utc_now() -> datetime:
    """Timezone-aware ``datetime.utcnow`` replacement.

    LangSmith's HTTP wire format expects ISO-8601 timestamps with
    explicit UTC offsets; naive datetimes trigger a server-side
    warning that is easy to miss.  Every timestamp the tracer passes
    to ``client.create_run`` / ``client.update_run`` goes through
    this helper.
    """
    return datetime.now(timezone.utc)


def _describe_goal(goal: Any) -> Any:
    """Render a goal for LangSmith inputs.

    :class:`~langgoap.goals.GoalSpec` and
    :class:`~langgoap.goals.MultiGoal` are frozen dataclasses; falling
    back to ``repr`` always produces *something* readable even for
    bare dicts or strings.
    """
    if goal is None:
        return None
    for attr in ("conditions", "goals"):
        value = getattr(goal, attr, None)
        if value is not None:
            return {attr: _jsonable(value)}
    return _jsonable(goal)


def _describe_plan(plan: Any) -> Any:
    """Render a :class:`~langgoap.planner.types.Plan` as a plain dict."""
    if plan is None:
        return None
    action_names = getattr(plan, "action_names", None)
    total_cost = getattr(plan, "total_cost", None)
    if action_names is None and total_cost is None:
        return _jsonable(plan)
    return {
        "action_names": list(action_names) if action_names is not None else [],
        "total_cost": total_cost,
    }


def _describe_action(action: Any) -> Any:
    """Render an :class:`~langgoap.actions.ActionSpec` as a plain dict."""
    if action is None:
        return None
    name = getattr(action, "name", None)
    if name is None:
        return _jsonable(action)
    return {
        "name": name,
        "preconditions": _jsonable(getattr(action, "preconditions", None)),
        "effects": _jsonable(getattr(action, "effects", None)),
        "cost": getattr(action, "cost", None),
    }


def _describe_result(result: Any) -> Any:
    """Render an ``ActionResult`` as a plain dict."""
    if result is None:
        return None
    fields = ("action_name", "success", "state_after", "error")
    payload = {name: _jsonable(getattr(result, name, None)) for name in fields}
    if all(v is None for v in payload.values()):
        return _jsonable(result)
    return payload


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary values to JSON-friendly types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    # Frozen dataclasses, enums, etc. -- repr is always a safe
    # last resort.
    return repr(value)


class LangSmithTracer:
    """Emit GOAP planning events to LangSmith as a structured run tree.

    Each full planner → executor → observer cycle becomes one root run
    of kind ``"chain"`` named ``"goap_plan"``; each executed action
    becomes a child run of kind ``"tool"`` underneath.  Replans inside
    a single graph invocation are modelled as sibling root runs whose
    ``parent_run_id`` points back to the previous root, producing a
    nested chain in the LangSmith UI so the replan history is visible
    at a glance.

    This tracer is complementary to LangGraph's automatic LangSmith
    tracing (``LANGCHAIN_TRACING_V2=true``).  LangGraph's node-level
    spans record the graph structure; this tracer records the GOAP
    domain events (plan starts, plan completes, replans, goal
    achievement) that the node spans do not see.  Running both
    produces two complementary trace trees in the same project.

    Args:
        client: Optional pre-configured :class:`langsmith.Client`.
            When omitted, a default ``Client()`` is constructed.
            Injectable for tests via a mock.
        project_name: Project name override.  When omitted, LangSmith
            uses ``LANGCHAIN_PROJECT`` or its built-in default.
        max_search_events: Cap on ``search_expand`` events retained for
            a single root run (default ``1000``).  Beyond the cap a
            single ``search_truncated`` marker is appended and further
            expansions are dropped for that run; ``search_dead_end``
            events bypass the cap since they are always low-cardinality
            and diagnostically valuable.
        flush_every: Number of appended search events that must
            accumulate before a batched ``update_run`` upload fires
            (default ``1`` — send every event, preserving pre-existing
            behaviour).  Enterprise deployments typically set this to
            ``50`` or higher to amortise the quadratic upload cost of
            resending the full event list on every update.  Terminal
            lifecycle events (``on_plan_complete`` / ``on_plan_failed``
            / ``on_goal_achieved``) always force a final flush so no
            pending events are lost.

    The tracer degrades to no-op behaviour and emits a single
    ``logger.warning`` when no API key is configured, so misconfigured
    environments never break the planner.  Every ``client`` call is
    wrapped in a ``try``/``except`` that swallows and logs exceptions —
    observability must never propagate failures into the GOAP loop.

    Concurrency note: a single :class:`LangSmithTracer` instance holds
    mutable state for the currently-active root run.  Construct one
    tracer per graph invocation (or per thread) when running multiple
    graphs concurrently.

    Example::

        from langgoap import LangSmithTracer, GoapGraph
        tracer = LangSmithTracer(project_name="my-agent")
        graph = GoapGraph(actions, tracer=tracer)
        result = graph.invoke(goal=goal, world_state={})
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        project_name: str | None = None,
        max_search_events: int = 1000,
        flush_every: int = 1,
    ) -> None:
        if max_search_events < 1:
            raise ValueError("max_search_events must be >= 1")
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        self._project_name = project_name or os.environ.get("LANGCHAIN_PROJECT")
        self._max_search_events = max_search_events
        self._flush_every = flush_every
        self._client: Any | None = None
        self._disabled = False

        if client is not None:
            self._client = client
        elif not (
            os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
        ):
            logger.warning(
                "LangSmithTracer: neither LANGCHAIN_API_KEY nor "
                "LANGSMITH_API_KEY is set; tracer will no-op. "
                "Set an API key to enable LangSmith tracing."
            )
            self._disabled = True
        else:
            try:
                from langsmith import Client

                self._client = Client()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "LangSmithTracer: failed to construct langsmith.Client: "
                    "%s; tracer will no-op.",
                    exc,
                )
                self._disabled = True

        # Mutable per-invocation state.
        self._root_run_id: uuid.UUID | None = None
        self._root_start: datetime | None = None
        self._action_run_id: uuid.UUID | None = None
        self._action_start: datetime | None = None
        # Accumulated A* search events for the currently-open root run.
        # We send the full list on every flush because LangSmith's
        # ``extra.metadata`` merge semantics replace list-valued keys
        # rather than concatenating them; the ``max_search_events`` cap
        # and ``flush_every`` batching amortise the quadratic cost.
        self._search_events: list[dict[str, Any]] = []
        self._events_since_flush = 0
        self._expand_count = 0
        self._dropped_expansions = 0
        self._truncation_recorded = False
        self._truncation_index: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _safe(self, op_name: str, func: Any, *args: Any, **kwargs: Any) -> None:
        if self._disabled or self._client is None:
            return
        try:
            func(*args, **kwargs)
        except Exception as exc:
            logger.warning("LangSmithTracer %s raised: %s", op_name, exc)

    def _start_root(
        self,
        goal: Any,
        state: Any,
        strategy_name: str,
        parent_run_id: uuid.UUID | None,
    ) -> None:
        if self._disabled or self._client is None:
            return
        run_id = uuid.uuid4()
        now = _utc_now()
        self._root_run_id = run_id
        self._root_start = now
        self._search_events = []
        self._events_since_flush = 0
        self._expand_count = 0
        self._dropped_expansions = 0
        self._truncation_recorded = False
        self._truncation_index = None
        create_kwargs: dict[str, Any] = {
            "name": "goap_plan",
            "run_type": "chain",
            "inputs": {
                "goal": _describe_goal(goal),
                "state": _jsonable(state),
                "strategy": strategy_name,
            },
            "id": run_id,
            "start_time": now,
            "extra": {"metadata": {"goap_strategy": strategy_name}},
        }
        if self._project_name is not None:
            create_kwargs["project_name"] = self._project_name
        if parent_run_id is not None:
            create_kwargs["parent_run_id"] = parent_run_id
        self._safe("create_run", self._client.create_run, **create_kwargs)

    def _finalize_root(
        self,
        *,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        end_time = _utc_now()
        update_kwargs: dict[str, Any] = {"end_time": end_time}
        if outputs is not None:
            update_kwargs["outputs"] = outputs
        if error is not None:
            update_kwargs["error"] = error
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            **update_kwargs,
        )
        self._root_run_id = None
        self._root_start = None

    # ------------------------------------------------------------------
    # Sync hooks
    # ------------------------------------------------------------------
    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        # If a previous root is still open we are entering a replan:
        # close the previous one as "superseded" and chain the new
        # root via parent_run_id so the replan is nested in the UI.
        previous_root_id = self._root_run_id
        if previous_root_id is not None:
            self._finalize_root(outputs={"status": "superseded_by_replan"})
        self._start_root(goal, state, strategy_name, parent_run_id=previous_root_id)

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        # Planning finished successfully but action execution has not
        # started — we leave the root run open and record the plan in
        # its metadata so it appears immediately in the LangSmith UI
        # instead of only on goal_achieved.
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        # Terminal lifecycle boundary: flush any pending search events
        # so consumers never lose the tail of the search tree when
        # ``flush_every`` is set above 1.
        self._flush_search_events()
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={
                "metadata": {
                    "planning_duration_ms": duration_ms,
                    "plan": _describe_plan(plan),
                }
            },
        )

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        self._flush_search_events()
        self._finalize_root(
            outputs={"reason": reason, "duration_ms": duration_ms},
            error=f"plan_failed: {reason}",
        )

    def on_action_start(self, action: Any, state: Any) -> None:
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        run_id = uuid.uuid4()
        now = _utc_now()
        self._action_run_id = run_id
        self._action_start = now
        self._safe(
            "create_run",
            self._client.create_run,
            name=f"goap_action:{getattr(action, 'name', 'unknown')}",
            run_type="tool",
            inputs={
                "action": _describe_action(action),
                "state": _jsonable(state),
            },
            id=run_id,
            start_time=now,
            parent_run_id=self._root_run_id,
            project_name=self._project_name,
        )

    def on_action_complete(self, result: Any) -> None:
        if self._disabled or self._client is None or self._action_run_id is None:
            return
        success = bool(getattr(result, "success", True))
        outputs = {"result": _describe_result(result)}
        error: str | None = None
        if not success:
            error = str(getattr(result, "error", None) or "action_failed")
        self._safe(
            "update_run",
            self._client.update_run,
            self._action_run_id,
            outputs=outputs,
            error=error,
            end_time=_utc_now(),
        )
        self._action_run_id = None
        self._action_start = None

    def on_replan(self, reason: str, new_plan: Any) -> None:
        # The planner has fired on_plan_start already (which opened a
        # new root); this hook only updates that root's metadata with
        # the replan reason and the refreshed plan.  No new run is
        # created here.
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={
                "metadata": {
                    "replan_reason": reason,
                    "plan": _describe_plan(new_plan),
                }
            },
        )

    def on_goal_achieved(self, final_state: Any) -> None:
        self._flush_search_events()
        self._finalize_root(
            outputs={"status": "goal_achieved", "final_state": _jsonable(final_state)}
        )

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        """Record a sensor completion event as metadata on the root LangSmith run."""
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={"metadata": {f"sensor_{sensor_name}": _jsonable(updates)}},
        )

    # ------------------------------------------------------------------
    # A* search hooks — attached as OTel-style events on the root
    # ``goap_plan`` run rather than as nested child runs.  Per-expansion
    # firehose data is high-volume; LangSmith maps OTel events to run
    # events natively.
    #
    # Two enterprise-hardening guarantees apply here:
    #
    #   1. ``max_search_events`` caps the number of ``search_expand``
    #      events retained for a single root.  ``search_dead_end``
    #      events bypass the cap because they are always low-cardinality
    #      and carry diagnostic value disproportionate to their volume.
    #   2. ``flush_every`` batches ``update_run`` uploads.  Terminal
    #      lifecycle events (``on_plan_complete`` / ``on_plan_failed``
    #      / ``on_goal_achieved``) force a final flush so no pending
    #      events are lost at the end of a run.
    # ------------------------------------------------------------------
    def _flush_search_events(self) -> None:
        """Send the accumulated search-event list to LangSmith.

        No-op when disabled, no root is open, or nothing is pending.
        Called at every ``flush_every`` boundary and forced at
        terminal lifecycle events.
        """
        if self._disabled or self._client is None or self._root_run_id is None:
            self._events_since_flush = 0
            return
        if not self._search_events:
            self._events_since_flush = 0
            return
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={"metadata": {"search_events": list(self._search_events)}},
        )
        self._events_since_flush = 0

    def _record_expand_event(self, event: dict[str, Any]) -> None:
        """Capped append for ``search_expand`` events.

        Once the cap fires, a single ``search_truncated`` marker is
        appended (or its ``dropped`` counter is updated in place) and
        further expansions are not retained — only counted.  The
        expansion counter is maintained incrementally so this path is
        O(1) per call and does not re-scan ``_search_events``.
        """
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        if self._expand_count >= self._max_search_events:
            self._dropped_expansions += 1
            if not self._truncation_recorded:
                self._truncation_index = len(self._search_events)
                self._search_events.append(
                    {
                        "kind": "search_truncated",
                        "dropped": self._dropped_expansions,
                    }
                )
                self._truncation_recorded = True
            elif self._truncation_index is not None:
                self._search_events[self._truncation_index][
                    "dropped"
                ] = self._dropped_expansions
        else:
            self._search_events.append(event)
            self._expand_count += 1
        self._events_since_flush += 1
        if self._events_since_flush >= self._flush_every:
            self._flush_search_events()

    def _record_uncapped_event(self, event: dict[str, Any]) -> None:
        """Uncapped append for low-cardinality events (``search_dead_end``)."""
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        self._search_events.append(event)
        self._events_since_flush += 1
        if self._events_since_flush >= self._flush_every:
            self._flush_search_events()

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
        self._record_expand_event(
            {
                "kind": "search_expand",
                "node_id": node_id,
                "parent_id": parent_id,
                "action_name": action_name,
                "g": g,
                "h": h,
                "f": f,
                "state": _jsonable(state),
            }
        )

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        self._record_uncapped_event(
            {
                "kind": "search_dead_end",
                "reason": reason,
                "detail": _jsonable(detail),
            }
        )

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={
                "metadata": {
                    "search_summary": {
                        "nodes_explored": nodes_explored,
                        "duration_ms": duration_ms,
                        "found": found,
                    }
                }
            },
        )

    # ------------------------------------------------------------------
    # Async hooks — langsmith.Client is thread-safe and uses its own
    # background sender thread, so delegating to the sync hooks is
    # correct and does not block the event loop beyond a cheap
    # in-memory queue append.
    # ------------------------------------------------------------------
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


__all__ = [
    "PlanningTracer",
    "NullTracer",
    "LoggingTracer",
    "MultiTracer",
    "LangSmithTracer",
]
