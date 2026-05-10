"""LangGraph executor node \u2014 runs the current action in the plan."""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.graph.nodes._execution import (
    _apply_result,
    _build_failure,
    _build_guard_block,
    _build_success,
    _check_human_approval,
    _prepare_execution,
)
from langgoap.graph.nodes._helpers import logger
from langgoap.graph.state import GoapState
from langgoap.guards import (
    ActionGuard,
    AsyncActionGuard,
    has_blocking_failure,
    run_guards_async,
    run_guards_sync,
)
from langgoap.planner.transitions import TransitionModel
from langgoap.tracing import NullTracer, PlanningTracer, SafeTracerProxy


class GoapExecutor:
    """LangGraph node that executes the current action in the plan.

    Runs ``plan.actions[current_step]``, updates world_state with the
    action's effects, and appends to execution_history.

    Both sync (:meth:`__call__`) and async (:meth:`acall`) paths share the
    same pre-condition checks, result building, and failure handling via the
    module-level helpers :func:`_prepare_execution`, :func:`_apply_result`,
    :func:`_build_success`, and :func:`_build_failure`.

    Args:
        tracer: Optional :class:`PlanningTracer` invoked around each
            action execution.
        guards: Optional list of :class:`~langgoap.guards.ActionGuard` or
            :class:`~langgoap.guards.AsyncActionGuard` objects evaluated
            *before* each action executes.  A ``WARN``-severity failure is
            logged but execution continues.  A ``BLOCK``-severity failure
            aborts the action and immediately blacklists it (equivalent to
            ``max_retries + 1`` failures), triggering replanning via the
            observer.
        transition_model: Optional
            :class:`~langgoap.planner.transitions.TransitionModel` that
            drives runtime effects when an action's ``execute`` callable
            returns ``None`` (or is absent).  Dict returns from ``execute``
            remain authoritative.  Pair with a seeded ``rng`` for
            reproducible stochastic rollouts.
        rng: Optional ``random.Random`` forwarded to
            ``transition_model.sample``.  When omitted, a single
            ``random.Random()`` is created at construction time and
            reused for every action \u2014 callers who want true
            reproducibility should pass a seeded instance.
    """

    def __init__(
        self,
        *,
        tracer: PlanningTracer | None = None,
        guards: list[ActionGuard | AsyncActionGuard] | None = None,
        transition_model: TransitionModel | None = None,
        rng: random.Random | None = None,
        actions: list[ActionSpec] | None = None,
    ) -> None:
        # Wrap the tracer in SafeTracerProxy at construction so call
        # sites can invoke hooks directly (``self._tracer.on_X(...)``)
        # without per-call try/except scaffolding.  Idempotent if the
        # caller already passed a SafeTracerProxy.
        self._tracer: PlanningTracer = SafeTracerProxy(tracer or NullTracer())
        self._guards: list[ActionGuard | AsyncActionGuard] = guards or []
        self._transition_model: TransitionModel | None = transition_model
        self._rng: random.Random = rng if rng is not None else random.Random()
        # Optional source-of-truth action map for rebinding fields the
        # serializer cannot round-trip (callable execute, Pydantic
        # type references on require_human_approval, human_input_key
        # strings, qos).  Restored by :meth:`_rebind` before each
        # action runs.
        self._actions: dict[str, ActionSpec] = (
            {a.name: a for a in actions} if actions else {}
        )

    def _rebind(self, action: ActionSpec) -> ActionSpec:
        """Restore the live ``ActionSpec`` for ``action.name`` if known.

        Falls back to the supplied ``action`` when the executor was
        constructed without an action list (the legacy case).  Mirrors
        the planner-side rebind for fields the serializer drops on the
        way through msgpack.
        """
        return self._actions.get(action.name, action)

    def __call__(self, state: GoapState) -> dict[str, Any]:
        prep = _prepare_execution(state)
        if isinstance(prep, dict):
            return prep
        world_state, plan_obj, current_step, action = prep
        action = self._rebind(action)

        logger.info(
            "Executing action %r (step %d/%d)",
            action.name,
            current_step + 1,
            len(plan_obj),
        )

        self._tracer.on_action_start(action, world_state)
        state_before = dict(world_state)

        # GuardRails: run all registered guards before executing.
        if self._guards:
            guard_results = run_guards_sync(self._guards, action, world_state)
            if has_blocking_failure(guard_results):
                result = _build_guard_block(
                    action, guard_results, state_before, world_state
                )
                self._tracer.on_action_complete(result)
                return result

        # Human-in-the-loop gate: interrupt() raises GraphInterrupt on
        # first pass (checkpointer persists state); returns the resume
        # payload on the second pass.
        denial = _check_human_approval(action, world_state, state_before)
        if denial is not None:
            self._tracer.on_action_complete(denial)
            return denial

        try:
            raw = _execute_with_retries_sync(
                action,
                world_state,
                tracer=self._tracer,
                rng=self._rng,
            )
            _apply_result(
                raw,
                action,
                world_state,
                transition_model=self._transition_model,
                rng=self._rng,
            )
            result = _build_success(action, current_step, state_before, world_state)
        except Exception as e:
            result = _build_failure(action, e, state_before, world_state)
        self._tracer.on_action_complete(result)
        return result

    async def acall(self, state: GoapState) -> dict[str, Any]:
        """Async variant of the executor node.

        Uses :func:`async_execute_action` to resolve the best callable:
        1. ``action.aexecute`` (explicit async callable)
        2. ``action.execute`` if it is a coroutine function
        3. ``action.execute`` via ``loop.run_in_executor`` (sync in thread)
        4. No callable \u2014 declared effects applied
        """
        prep = _prepare_execution(state)
        if isinstance(prep, dict):
            return prep
        world_state, plan_obj, current_step, action = prep
        action = self._rebind(action)

        logger.info(
            "Executing action %r async (step %d/%d)",
            action.name,
            current_step + 1,
            len(plan_obj),
        )

        await self._tracer.aon_action_start(action, world_state)
        state_before = dict(world_state)

        if self._guards:
            guard_results = await run_guards_async(self._guards, action, world_state)
            if has_blocking_failure(guard_results):
                result = _build_guard_block(
                    action, guard_results, state_before, world_state
                )
                await self._tracer.aon_action_complete(result)
                return result

        denial = _check_human_approval(action, world_state, state_before)
        if denial is not None:
            await self._tracer.aon_action_complete(denial)
            return denial

        try:
            raw = await _execute_with_retries_async(
                action,
                world_state,
                tracer=self._tracer,
                rng=self._rng,
            )
            _apply_result(
                raw,
                action,
                world_state,
                transition_model=self._transition_model,
                rng=self._rng,
            )
            result = _build_success(action, current_step, state_before, world_state)
        except Exception as e:
            result = _build_failure(action, e, state_before, world_state)
        await self._tracer.aon_action_complete(result)
        return result


def _execute_with_retries_sync(
    action: ActionSpec,
    world_state: dict[str, Any],
    *,
    tracer: PlanningTracer,
    rng: random.Random | None,
) -> Any:
    """Run ``action`` with the sync retry loop dictated by ``action.qos``.

    When ``action.qos`` is ``None`` this is a single attempt — preserving
    the legacy non-retry semantics.  Otherwise the executor re-invokes
    the action up to ``qos.max_attempts`` times, sleeping
    ``qos.compute_backoff_ms(attempt)`` between attempts and emitting
    ``on_action_retry`` before each sleep.  Returns the raw result of
    the successful invocation, or re-raises the last exception when
    every attempt is exhausted.
    """
    if action.aexecute is not None and action.execute is None:
        raise RuntimeError(
            f"Action {action.name!r} is async-only (aexecute is set, "
            "execute is None). Use GoapGraph.ainvoke() or "
            "compiled.ainvoke() to run async actions."
        )
    if action.execute is not None and inspect.iscoroutinefunction(action.execute):
        raise RuntimeError(
            f"Action {action.name!r} has an async execute callable. "
            "Use GoapGraph.ainvoke() or compiled.ainvoke() instead."
        )

    qos = action.qos
    max_attempts = qos.max_attempts if qos is not None else 1
    backoff_rng = rng if rng is not None else random.Random()

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if action.execute is not None:
                return action.execute(world_state)
            return None
        except Exception as exc:
            last_exc = exc
            if qos is None or not qos.should_retry(exc, attempt):
                raise
            backoff_ms = qos.compute_backoff_ms(attempt, rng=backoff_rng)
            tracer.on_action_retry(action, attempt, exc, backoff_ms)
            time.sleep(backoff_ms / 1000.0)
    assert last_exc is not None  # pragma: no cover - guarded by ActionQos validator
    raise last_exc


async def _execute_with_retries_async(
    action: ActionSpec,
    world_state: dict[str, Any],
    *,
    tracer: PlanningTracer,
    rng: random.Random | None,
) -> Any:
    """Async sibling of :func:`_execute_with_retries_sync`."""
    qos = action.qos
    max_attempts = qos.max_attempts if qos is not None else 1
    backoff_rng = rng if rng is not None else random.Random()

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await async_execute_action(action, world_state)
        except Exception as exc:
            last_exc = exc
            if qos is None or not qos.should_retry(exc, attempt):
                raise
            backoff_ms = qos.compute_backoff_ms(attempt, rng=backoff_rng)
            await tracer.aon_action_retry(action, attempt, exc, backoff_ms)
            await asyncio.sleep(backoff_ms / 1000.0)
    assert last_exc is not None  # pragma: no cover - guarded by ActionQos validator
    raise last_exc


async def async_execute_action(action: ActionSpec, world_state: dict[str, Any]) -> Any:
    """Resolve and await the best async callable for *action*.

    Resolution priority:
    1. ``action.aexecute`` \u2014 explicit async callable (highest priority)
    2. ``action.execute`` if it is a coroutine function
    3. ``action.execute`` run in the default thread-pool executor (sync \u2192 async)
    4. No callable \u2014 returns ``None``; caller applies declared effects

    This is a public helper so that advanced callers (e.g. custom executors,
    test utilities) can reuse the resolution logic without duplicating it.
    """
    if action.aexecute is not None:
        return await action.aexecute(world_state)
    if action.execute is not None:
        if inspect.iscoroutinefunction(action.execute):
            return await action.execute(world_state)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, action.execute, world_state)
    return None


__all__ = ["GoapExecutor", "async_execute_action"]
