"""Parallel-wave LangGraph executor node.

When a plan contains a sequence of actions whose preconditions /
effects do not interact, they form a *parallel wave* that can be
executed concurrently in the async path or sequentially-in-one-tick
in the sync path.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.graph.nodes._execution import (
    _apply_result,
    _build_failure,
    _build_guard_block,
    _build_success,
    _prepare_execution,
)
from langgoap.graph.nodes._helpers import logger
from langgoap.graph.nodes.executor import async_execute_action
from langgoap.graph.state import ActionResult, GoapState
from langgoap.guards import (
    ActionGuard,
    AsyncActionGuard,
    has_blocking_failure,
    run_guards_async,
    run_guards_sync,
)
from langgoap.planner.transitions import TransitionModel
from langgoap.tracing import NullTracer, PlanningTracer, SafeTracerProxy


def _find_parallel_group(
    plan_actions: tuple[ActionSpec, ...],
    current_step: int,
    deps: dict[int, list[int]] | None = None,
) -> list[int]:
    """Return the maximal set of plan indices that can run in parallel from *current_step*.

    Uses the CSP dependency graph: action j is added to the group as long as it
    does not depend (transitively or directly) on any action already in the
    group.  The group always contains *current_step* as its first element.

    Args:
        plan_actions: All actions in the plan (full tuple, not a slice).
        current_step: Index of the first action to include.
        deps: Pre-computed dependency graph (output of
            :func:`~langgoap.planner.csp.build_dependency_graph`).  When
            ``None``, computed on-the-fly.  Pass this to avoid redundant
            graph construction across waves of the same plan.

    Returns:
        A list of action indices (all >= current_step) that can execute in
        parallel.  A single-element list means no parallelism is available at
        this step.
    """
    if current_step >= len(plan_actions):
        return []

    if deps is None:
        from langgoap.planner.csp import build_dependency_graph

        deps = build_dependency_graph(plan_actions)

    group: list[int] = [current_step]
    group_set: set[int] = {current_step}

    for j in range(current_step + 1, len(plan_actions)):
        # Stop if j depends on any action already in the group.
        if any(d in group_set for d in deps[j]):
            break
        group.append(j)
        group_set.add(j)

    return group


class ParallelGoapExecutor:
    """LangGraph node that executes independent plan actions in parallel.

    See :class:`~langgoap.graph.nodes.executor.GoapExecutor` for
    sequential semantics; this class differs only in detecting the
    maximal *parallel wave* starting at ``current_step`` and running
    its members concurrently in the async path.

    Args:
        tracer: Optional :class:`~langgoap.tracing.PlanningTracer` invoked for
            each action in the wave.
        guards: Optional list of :class:`~langgoap.guards.ActionGuard` /
            :class:`~langgoap.guards.AsyncActionGuard` evaluated before each
            action.  A ``BLOCK`` failure aborts that action and the whole wave.
        transition_model: Optional
            :class:`~langgoap.planner.transitions.TransitionModel` consumed
            when a wave action's ``execute`` callable returns ``None`` (or
            is absent).  Mirrors :class:`GoapExecutor` so direct users of
            this class get identical stochastic runtime semantics.
        rng: Optional ``random.Random`` forwarded to
            ``transition_model.sample``.  When omitted, a single
            ``random.Random()`` is created at construction time and
            reused across waves.
    """

    def __init__(
        self,
        *,
        tracer: PlanningTracer | None = None,
        guards: list[ActionGuard | AsyncActionGuard] | None = None,
        transition_model: TransitionModel | None = None,
        rng: random.Random | None = None,
    ) -> None:
        # See ``GoapExecutor.__init__`` for the SafeTracerProxy rationale.
        self._tracer: PlanningTracer = SafeTracerProxy(tracer or NullTracer())
        self._guards: list[ActionGuard | AsyncActionGuard] = guards or []
        self._transition_model: TransitionModel | None = transition_model
        self._rng: random.Random = rng if rng is not None else random.Random()
        # Cache the dependency graph by the plan's action-name signature
        # so it is computed once per plan rather than once per wave.  A
        # name tuple is used (not ``id(plan_actions)``) because Python
        # may reuse object addresses after garbage collection, which
        # would let a stale cache leak across distinct plans.
        self._dep_cache: tuple[tuple[str, ...], dict[int, list[int]]] | None = None

    def _get_deps(self, plan_actions: tuple[ActionSpec, ...]) -> dict[int, list[int]]:
        """Return the dependency graph, caching by action-name signature."""
        key = tuple(a.name for a in plan_actions)
        if self._dep_cache is not None and self._dep_cache[0] == key:
            return self._dep_cache[1]
        from langgoap.planner.csp import build_dependency_graph

        deps = build_dependency_graph(plan_actions)
        self._dep_cache = (key, deps)
        return deps

    # ------------------------------------------------------------------
    # Sync path
    # ------------------------------------------------------------------

    def __call__(self, state: GoapState) -> dict[str, Any]:
        prep = _prepare_execution(state)
        if isinstance(prep, dict):
            return prep
        world_state, plan_obj, current_step, _first_action = prep

        deps = self._get_deps(plan_obj.actions)
        group = _find_parallel_group(plan_obj.actions, current_step, deps)
        merged_world: dict[str, Any] = dict(world_state)
        all_results: list[ActionResult] = []

        for idx in group:
            action = plan_obj.actions[idx]
            logger.info(
                "ParallelExecutor: running action %r (wave idx %d/%d)",
                action.name,
                group.index(idx) + 1,
                len(group),
            )
            self._tracer.on_action_start(action, merged_world)
            state_before = dict(merged_world)

            if self._guards:
                guard_results = run_guards_sync(self._guards, action, merged_world)
                if has_blocking_failure(guard_results):
                    result = _build_guard_block(
                        action, guard_results, state_before, merged_world
                    )
                    self._tracer.on_action_complete(result)
                    return result

            try:
                if action.aexecute is not None and action.execute is None:
                    raise RuntimeError(
                        f"Action {action.name!r} is async-only. Use ainvoke()."
                    )
                raw = (
                    action.execute(merged_world) if action.execute is not None else None
                )
                _apply_result(
                    raw,
                    action,
                    merged_world,
                    transition_model=self._transition_model,
                    rng=self._rng,
                )
                action_result = _build_success(action, idx, state_before, merged_world)
            except Exception as exc:
                action_result = _build_failure(action, exc, state_before, merged_world)
                self._tracer.on_action_complete(action_result)
                return action_result

            self._tracer.on_action_complete(action_result)
            if "execution_history" in action_result:
                all_results.extend(action_result["execution_history"])

        return {
            "world_state": merged_world,
            "current_step": max(group) + 1,
            "execution_history": all_results,
        }

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def acall(self, state: GoapState) -> dict[str, Any]:
        """Async variant \u2014 executes the wave actions concurrently via asyncio.gather."""
        prep = _prepare_execution(state)
        if isinstance(prep, dict):
            return prep
        world_state, plan_obj, current_step, _first_action = prep

        deps = self._get_deps(plan_obj.actions)
        group = _find_parallel_group(plan_obj.actions, current_step, deps)

        async def _run_one(idx: int) -> dict[str, Any]:
            action = plan_obj.actions[idx]
            # Each parallel action gets a snapshot of the world-state as it
            # was at the start of this wave (independent actions don't see
            # each other's in-progress effects).
            ws_snapshot = dict(world_state)

            await self._tracer.aon_action_start(action, ws_snapshot)
            state_before = dict(ws_snapshot)

            if self._guards:
                guard_results = await run_guards_async(
                    self._guards, action, ws_snapshot
                )
                if has_blocking_failure(guard_results):
                    result = _build_guard_block(
                        action, guard_results, state_before, ws_snapshot
                    )
                    await self._tracer.aon_action_complete(result)
                    return result

            try:
                raw = await async_execute_action(action, ws_snapshot)
                _apply_result(
                    raw,
                    action,
                    ws_snapshot,
                    transition_model=self._transition_model,
                    rng=self._rng,
                )
                result = _build_success(action, idx, state_before, ws_snapshot)
            except Exception as exc:
                result = _build_failure(action, exc, state_before, ws_snapshot)

            await self._tracer.aon_action_complete(result)
            return result

        raw_results: list[Any] = list(
            await asyncio.gather(
                *(_run_one(idx) for idx in group), return_exceptions=False
            )
        )

        # Check for any failures
        for r in raw_results:
            if isinstance(r, dict) and r.get("status") == "action_failed":
                return r  # abort wave, let observer handle replanning

        # Merge all effects into a single world_state update
        merged_world = dict(world_state)
        all_results: list[ActionResult] = []
        for r in raw_results:
            if "world_state" in r:
                merged_world.update(r["world_state"])
            if "execution_history" in r:
                all_results.extend(r["execution_history"])

        return {
            "world_state": merged_world,
            "current_step": max(group) + 1,
            "execution_history": all_results,
        }


__all__ = ["ParallelGoapExecutor", "_find_parallel_group"]
