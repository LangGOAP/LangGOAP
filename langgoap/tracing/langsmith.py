"""LangSmith adapter for the :class:`~langgoap.tracing.PlanningTracer` Protocol.

Each full planner \u2192 executor \u2192 observer cycle becomes one root run of
kind ``"chain"`` named ``"goap_plan"``; each executed action becomes a
child run of kind ``"tool"`` underneath.  Replans inside a single graph
invocation are modelled as sibling root runs whose ``parent_run_id``
points back to the previous root.

The search-event accumulator (cap, truncation, batched flush) lives in
:class:`~langgoap.tracing._search_buffer.SearchEventBuffer` so this
module focuses on LangSmith-API mechanics.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any

from langgoap.tracing._helpers import (
    describe_action,
    describe_goal,
    describe_plan,
    describe_result,
    jsonable,
    utc_now,
)
from langgoap.tracing._search_buffer import SearchEventBuffer

logger = logging.getLogger("langgoap.tracing")


class LangSmithTracer:
    """Emit GOAP planning events to LangSmith as a structured run tree.

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
            events bypass the cap.
        flush_every: Number of appended search events that must
            accumulate before a batched ``update_run`` upload fires
            (default ``1``).  Terminal lifecycle events
            (``on_plan_complete`` / ``on_plan_failed`` /
            ``on_goal_achieved``) always force a final flush.

    The tracer degrades to no-op behaviour and emits a single
    ``logger.warning`` when no API key is configured, so misconfigured
    environments never break the planner.  Every ``client`` call is
    wrapped in a ``try``/``except`` that swallows and logs exceptions \u2014
    observability must never propagate failures into the GOAP loop.

    Concurrency note: a single instance holds mutable per-invocation
    state (the open root run, the search-event buffer, and the in-flight
    action-run map).  All mutations are serialised by an internal
    :class:`threading.Lock` so the tracer is safe to share across the
    threads that
    :class:`~langgoap.graph.nodes.ParallelGoapExecutor` and the asyncio
    ``aon_*`` hooks fan out across \u2014 concurrent ``on_action_start`` /
    ``on_action_complete`` pairs are correlated by ``action_name``.
    Two distinct top-level graph invocations sharing one tracer is still
    unsupported: build one tracer per top-level graph call.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        project_name: str | None = None,
        max_search_events: int = 1000,
        flush_every: int = 1,
    ) -> None:
        self._project_name = project_name or os.environ.get("LANGCHAIN_PROJECT")
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

        self._lock = threading.Lock()
        self._root_run_id: uuid.UUID | None = None
        self._root_start: datetime | None = None
        self._action_runs: dict[str, tuple[uuid.UUID, datetime]] = {}
        self._search_buffer = SearchEventBuffer(
            max_events=max_search_events, flush_every=flush_every
        )

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

    def _drain_and_upload(self, root_id: uuid.UUID) -> None:
        """Drain pending search events and fire one ``update_run``.

        Intended for use *outside* ``self._lock`` after the caller
        observed a flush threshold trip; the drain itself takes the
        lock briefly to snapshot state.
        """
        if self._client is None:
            return
        with self._lock:
            snapshot = self._search_buffer.drain()
        if snapshot is None:
            return
        self._safe(
            "update_run",
            self._client.update_run,
            root_id,
            extra={"metadata": {"search_events": snapshot}},
        )

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
        now = utc_now()
        with self._lock:
            self._root_run_id = run_id
            self._root_start = now
            self._action_runs.clear()
            self._search_buffer.reset()
        create_kwargs: dict[str, Any] = {
            "name": "goap_plan",
            "run_type": "chain",
            "inputs": {
                "goal": describe_goal(goal),
                "state": jsonable(state),
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
        with self._lock:
            if self._disabled or self._client is None or self._root_run_id is None:
                return
            root_id = self._root_run_id
            self._root_run_id = None
            self._root_start = None
        end_time = utc_now()
        update_kwargs: dict[str, Any] = {"end_time": end_time}
        if outputs is not None:
            update_kwargs["outputs"] = outputs
        if error is not None:
            update_kwargs["error"] = error
        self._safe(
            "update_run",
            self._client.update_run,
            root_id,
            **update_kwargs,
        )

    def _flush_search_events(self) -> None:
        """Force-drain the search-event buffer to LangSmith.

        No-op when disabled, no root is open, or nothing is pending.
        Called at every ``flush_every`` boundary and forced at terminal
        lifecycle events.
        """
        with self._lock:
            if self._disabled or self._client is None or self._root_run_id is None:
                self._search_buffer.drain()  # reset counter regardless
                return
            root_id = self._root_run_id
            snapshot = self._search_buffer.drain()
        if snapshot is None:
            return
        self._safe(
            "update_run",
            self._client.update_run,
            root_id,
            extra={"metadata": {"search_events": snapshot}},
        )

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
        # started \u2014 we leave the root run open and record the plan in
        # its metadata so it appears immediately in the LangSmith UI
        # instead of only on goal_achieved.
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        self._flush_search_events()
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={
                "metadata": {
                    "planning_duration_ms": duration_ms,
                    "plan": describe_plan(plan),
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
        action_name = str(getattr(action, "name", "unknown"))
        run_id = uuid.uuid4()
        now = utc_now()
        with self._lock:
            if self._disabled or self._client is None or self._root_run_id is None:
                return
            # Concurrent waves may legitimately start the same action
            # name across distinct planning cycles; within a single wave
            # names are unique by construction.  Last-writer-wins on a
            # collision keeps the latest run live and lets the older one
            # leak a child run rather than corrupting the active mapping.
            self._action_runs[action_name] = (run_id, now)
            parent_id = self._root_run_id
        self._safe(
            "create_run",
            self._client.create_run,
            name=f"goap_action:{action_name}",
            run_type="tool",
            inputs={
                "action": describe_action(action),
                "state": jsonable(state),
            },
            id=run_id,
            start_time=now,
            parent_run_id=parent_id,
            project_name=self._project_name,
        )

    def on_action_complete(self, result: Any) -> None:
        # The executor calls this hook with the state-update dict it
        # returns to LangGraph (``{"execution_history": [ActionResult],
        # "world_state": {...}, ...}``); other call sites (notably the
        # parallel-executor and the unit tests for parallel-action
        # correlation) pass a raw :class:`ActionResult`.  Support both
        # so the run is closed against its real ``action_name`` either
        # way.
        result_obj: Any = result
        action_name = ""
        success = True
        if isinstance(result, dict):
            history = result.get("execution_history") or []
            if history:
                result_obj = history[-1]
                action_name = str(getattr(result_obj, "action_name", "") or "")
                success = bool(getattr(result_obj, "success", True))
        else:
            action_name = str(getattr(result, "action_name", "") or "")
            success = bool(getattr(result, "success", True))
        with self._lock:
            if self._disabled or self._client is None:
                return
            entry = self._action_runs.pop(action_name, None)
            if entry is None:
                return
            run_id, _start = entry
        outputs = {"result": describe_result(result_obj)}
        error: str | None = None
        if not success:
            error = str(getattr(result_obj, "error", None) or "action_failed")
        self._safe(
            "update_run",
            self._client.update_run,
            run_id,
            outputs=outputs,
            error=error,
            end_time=utc_now(),
        )

    def on_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        """Record an in-executor retry as metadata on the action's run."""
        if self._disabled or self._client is None:
            return
        action_name = str(getattr(action, "name", "") or "")
        with self._lock:
            entry = self._action_runs.get(action_name)
            if entry is None:
                return
            run_id, _start = entry
        retry_event = {
            f"retry_attempt_{attempt}": {
                "exc_type": type(exception).__name__,
                "exc_message": str(exception),
                "backoff_ms": backoff_ms,
            }
        }
        self._safe(
            "update_run",
            self._client.update_run,
            run_id,
            extra={"metadata": retry_event},
        )

    def on_strategy_chosen(self, strategy_name: str) -> None:
        """Record the routed strategy as metadata on the active root run."""
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={"metadata": {"strategy_chosen": strategy_name}},
        )

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
                    "plan": describe_plan(new_plan),
                }
            },
        )

    def on_goal_achieved(self, final_state: Any) -> None:
        self._flush_search_events()
        self._finalize_root(
            outputs={"status": "goal_achieved", "final_state": jsonable(final_state)}
        )

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        """Record a sensor completion event as metadata on the root LangSmith run."""
        if self._disabled or self._client is None or self._root_run_id is None:
            return
        self._safe(
            "update_run",
            self._client.update_run,
            self._root_run_id,
            extra={"metadata": {f"sensor_{sensor_name}": jsonable(updates)}},
        )

    # ------------------------------------------------------------------
    # A* search hooks \u2014 attached as OTel-style events on the root run.
    # ``max_search_events`` caps ``search_expand`` events; ``flush_every``
    # batches uploads.  See :class:`SearchEventBuffer` for details.
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
    ) -> None:
        event = {
            "kind": "search_expand",
            "node_id": node_id,
            "parent_id": parent_id,
            "action_name": action_name,
            "g": g,
            "h": h,
            "f": f,
            "state": jsonable(state),
        }
        snapshot: list[dict[str, Any]] | None = None
        root_id: uuid.UUID | None = None
        with self._lock:
            if self._disabled or self._client is None or self._root_run_id is None:
                return
            should_flush = self._search_buffer.add_expand(event)
            if should_flush:
                root_id = self._root_run_id
                snapshot = self._search_buffer.drain()
        if snapshot is not None and root_id is not None:
            self._safe(
                "update_run",
                self._client.update_run,
                root_id,
                extra={"metadata": {"search_events": snapshot}},
            )

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        event = {
            "kind": "search_dead_end",
            "reason": reason,
            "detail": jsonable(detail),
        }
        snapshot: list[dict[str, Any]] | None = None
        root_id: uuid.UUID | None = None
        with self._lock:
            if self._disabled or self._client is None or self._root_run_id is None:
                return
            should_flush = self._search_buffer.add_uncapped(event)
            if should_flush:
                root_id = self._root_run_id
                snapshot = self._search_buffer.drain()
        if snapshot is not None and root_id is not None:
            self._safe(
                "update_run",
                self._client.update_run,
                root_id,
                extra={"metadata": {"search_events": snapshot}},
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
    # Async hooks \u2014 langsmith.Client is thread-safe and uses its own
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

    async def aon_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        self.on_action_retry(action, attempt, exception, backoff_ms)

    async def aon_strategy_chosen(self, strategy_name: str) -> None:
        self.on_strategy_chosen(strategy_name)

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


__all__ = ["LangSmithTracer"]
