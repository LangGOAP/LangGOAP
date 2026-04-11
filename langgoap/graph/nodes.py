"""LangGraph nodes for GOAP planning, execution, and observation.

These nodes form the core GOAP loop:
  planner → executor → observer → (executor | planner | END)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langgraph.graph import END
from langgraph.types import Command

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.state import ActionResult, GoapState
from langgoap.history import (
    ExecutionRecord,
    StoreExecutionHistory,
    compute_goal_hash,
)
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from langgoap.tracing import NullTracer, PlanningTracer
from langgoap.types import ReplanStrategy

if TYPE_CHECKING:
    from langgoap.planner.strategy import PlanningStrategy

logger = logging.getLogger(__name__)


def _safe_tracer_call(tracer: PlanningTracer, method: str, *args: Any) -> None:
    """Invoke a sync tracer hook, swallowing any exception.

    Observability must never break the planner — this is the second
    line of defence beyond :class:`MultiTracer`'s own catch.
    """
    try:
        getattr(tracer, method)(*args)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Tracer raised during %s: %s", method, exc)


async def _safe_tracer_acall(tracer: PlanningTracer, method: str, *args: Any) -> None:
    """Invoke an async tracer hook, swallowing any exception."""
    try:
        await getattr(tracer, method)(*args)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Tracer raised during %s: %s", method, exc)


def _planning_keys(actions: list[ActionSpec], goal: GoalSpec) -> set[str]:
    """Compute the set of world-state keys relevant to A* planning.

    Returns the union of all keys appearing in action preconditions,
    action effects, and goal conditions.  Used to filter a rich
    ``world_state`` dict down to the hashable boolean flags that
    the planner operates on.
    """
    keys: set[str] = set(goal.conditions.keys())
    for action in actions:
        keys.update(action.preconditions.keys())
        keys.update(action.effects.keys())
    return keys


class GoapPlanner:
    """LangGraph node that invokes the planner.

    Reads world_state and goal from GoapState, runs the configured
    :class:`~langgoap.planner.strategy.PlanningStrategy` (default: pure
    A* when the goal has no constraints/objectives, or the two-phase
    A*→CSP pipeline otherwise), and writes the resulting plan back.

    Args:
        actions: Available actions for planning.
        strategy: Optional custom :class:`PlanningStrategy`.  When
            ``None`` (default) the planner auto-selects between pure A*
            and the two-phase pipeline based on the goal — identical to
            the pre-strategy behavior.  Pass a concrete strategy (or a
            user-defined one implementing the Protocol) to override
            this routing.
    """

    def __init__(
        self,
        actions: list[ActionSpec],
        *,
        strategy: PlanningStrategy | None = None,
        tracer: PlanningTracer | None = None,
    ) -> None:
        self.actions = actions
        self._strategy = strategy
        self._tracer: PlanningTracer = tracer or NullTracer()

    def _strategy_name(self, goal: GoalSpec | MultiGoal | None) -> str:
        """Human-readable label passed to ``on_plan_start``."""
        if self._strategy is not None:
            return type(self._strategy).__name__
        if isinstance(goal, GoalSpec) and (
            goal.constraints or goal.objectives is not None
        ):
            return "TwoPhasePipeline"
        return "AStar"

    def _effective_goal_for_tracer(
        self, state: GoapState
    ) -> GoalSpec | MultiGoal | None:
        """Resolve the ``MultiGoal`` wrapper for tracer visibility.

        For ``"sequential"`` mode, tracers should see the plain
        :class:`GoalSpec` currently being planned so hooks like
        ``on_plan_start`` receive a goal with a readable
        ``conditions`` attribute — otherwise every sub-goal advance
        would hand the tracer a ``MultiGoal`` without the expected
        interface, silently crashing any recorder that inspects the
        current goal.  For ``"any"`` mode the planner has not yet
        picked a sub-goal, so the raw ``MultiGoal`` is the right
        thing to expose: tracers can either branch on
        ``isinstance(goal, MultiGoal)`` or read ``goal.goals``
        directly.
        """
        goal = state.get("goal")
        if isinstance(goal, MultiGoal) and goal.mode == "sequential":
            idx = state.get("current_subgoal_index", 0)
            return goal.goals[idx]
        return goal

    def _plan_single(
        self,
        start: PlanningState,
        goal: GoalSpec,
        blacklisted: list[str],
    ) -> Plan | None:
        """Run the configured strategy (or automatic routing) once.

        Extracted so both the single-goal path and the ``MultiGoal``
        dispatch can share planning logic without duplication.
        """
        if self._strategy is not None:
            try:
                return self._strategy.plan(
                    start,
                    goal,
                    self.actions,
                    blacklisted_actions=blacklisted,
                )
            except ImportError:
                logger.warning(
                    "Configured strategy requires ortools which is not "
                    "installed. Falling back to pure A*."
                )
                return astar_plan(
                    start, goal, self.actions, blacklisted_actions=blacklisted
                )
        if goal.constraints or goal.objectives is not None:
            try:
                from langgoap.planner.pipeline import plan as pipeline_plan

                return pipeline_plan(
                    start, goal, self.actions, blacklisted_actions=blacklisted
                )
            except ImportError:
                logger.warning(
                    "CSP constraints/objectives specified but ortools not "
                    "installed. Falling back to pure A*."
                )
                return astar_plan(
                    start, goal, self.actions, blacklisted_actions=blacklisted
                )
        return astar_plan(start, goal, self.actions, blacklisted_actions=blacklisted)

    def _plan_core(self, state: GoapState) -> tuple[dict[str, Any], bool]:
        """Shared planning logic for sync and async entry points.

        Returns:
            ``(updates, was_replan)`` — ``updates`` is the GoapState
            delta to return from the node, and ``was_replan`` is
            ``True`` when a prior plan was already present so the
            caller can fire ``on_replan`` instead of
            ``on_plan_complete``.
        """
        world_state = state.get("world_state", {})
        raw_goal = state.get("goal")
        if raw_goal is None:
            logger.error("GoapPlanner called with no goal in state")
            return (
                {"plan": None, "status": "error", "current_step": 0},
                False,
            )

        # Only increment replan_count when a prior plan already existed.
        # The initial planning call is not a replan.
        existing_plan = state.get("plan")
        was_replan = existing_plan is not None
        replan_count = state.get("replan_count", 0)
        if was_replan:
            replan_count += 1
            logger.info(
                "Replanning (replan #%d), reason=%s, world_state=%s",
                replan_count,
                state.get("replan_reason"),
                world_state,
            )

        blacklisted = list(state.get("blacklisted_actions", []))

        # Resolve ``MultiGoal`` to an effective ``GoalSpec``.  The A*
        # planner and CSP pipeline never see ``MultiGoal`` directly —
        # that abstraction lives at the planner/observer dispatch layer.
        extra_updates: dict[str, Any] = {}
        precomputed_result: Plan | None = None
        effective_goal: GoalSpec
        if isinstance(raw_goal, MultiGoal):
            if raw_goal.mode == "sequential":
                idx = state.get("current_subgoal_index", 0)
                effective_goal = raw_goal.goals[idx]
                if not was_replan:
                    logger.info(
                        "Planning for MultiGoal (sequential, subgoal %d/%d): %s",
                        idx + 1,
                        len(raw_goal.goals),
                        dict(effective_goal.conditions),
                    )
            else:  # "any"
                # Enumerate sub-goals, plan each, pick the cheapest feasible.
                # TODO: replace ``total_cost`` with ``plan.score`` comparison
                # so that mixed-feasibility alternatives (hard violations)
                # lose to feasible ones regardless of path cost.  Blocked on
                # cross-subclass Score comparison (``SimpleScore`` vs
                # ``HardSoftScore``) which currently raises ``TypeError``.
                best_idx = -1
                best_plan: Plan | None = None
                for i, sg in enumerate(raw_goal.goals):
                    sub_pkeys = _planning_keys(self.actions, sg)
                    sub_start = PlanningState.from_dict(world_state, keys=sub_pkeys)
                    candidate = self._plan_single(sub_start, sg, blacklisted)
                    if candidate is None:
                        continue
                    if best_plan is None or candidate.total_cost < best_plan.total_cost:
                        best_idx = i
                        best_plan = candidate
                if best_plan is None:
                    logger.warning("MultiGoal 'any' mode: no sub-goal is reachable")
                    return (
                        {
                            "plan": None,
                            "status": "no_plan",
                            "current_step": 0,
                            "replan_count": replan_count,
                        },
                        was_replan,
                    )
                logger.info(
                    "MultiGoal 'any' mode: picked sub-goal %d with cost %.2f",
                    best_idx,
                    best_plan.total_cost,
                )
                effective_goal = raw_goal.goals[best_idx]
                precomputed_result = best_plan
                extra_updates["current_subgoal_index"] = best_idx
        else:
            effective_goal = raw_goal
            if not was_replan:
                logger.info("Planning for goal %s", dict(effective_goal.conditions))

        if precomputed_result is not None:
            result: Plan | None = precomputed_result
        else:
            pkeys = _planning_keys(self.actions, effective_goal)
            start = PlanningState.from_dict(world_state, keys=pkeys)
            result = self._plan_single(start, effective_goal, blacklisted)

        # Detect blacklist fallback: if the plan uses a blacklisted action,
        # the A* fallback in ``astar_plan`` fired because the filtered
        # action set could not reach the goal. Clear the blacklist so the
        # same fallback is not re-triggered on every subsequent replan.
        used_blacklisted = False
        if result is not None and blacklisted:
            plan_names = set(result.action_names)
            if plan_names & set(blacklisted):
                used_blacklisted = True
                logger.info(
                    "Blacklist fallback: cleared blacklist, plan uses %s",
                    plan_names & set(blacklisted),
                )

        if result is None:
            logger.warning(
                "A* found no plan for goal %s", dict(effective_goal.conditions)
            )
            return (
                {
                    "plan": None,
                    "status": "no_plan",
                    "current_step": 0,
                    "replan_count": replan_count,
                    **extra_updates,
                },
                was_replan,
            )

        logger.info(
            "Plan found: %s (cost=%.2f, nodes_explored=%d)",
            result.action_names,
            result.total_cost,
            result.metadata.nodes_explored,
        )
        updates: dict[str, Any] = {
            "plan": result,
            "status": "executing",
            "current_step": 0,
            "replan_count": replan_count,
            **extra_updates,
        }
        if used_blacklisted:
            # Clear blacklist and failure counts so the retried action
            # gets a fresh start instead of being immediately re-blacklisted.
            updates["blacklisted_actions"] = []
            updates["action_failure_counts"] = {}
        return updates, was_replan

    def __call__(self, state: GoapState) -> dict[str, Any]:
        goal = state.get("goal")
        world_state = state.get("world_state", {})
        strategy_name = self._strategy_name(goal)
        tracer_goal = self._effective_goal_for_tracer(state)
        _safe_tracer_call(
            self._tracer, "on_plan_start", tracer_goal, world_state, strategy_name
        )
        started = time.perf_counter()
        updates, was_replan = self._plan_core(state)
        duration_ms = (time.perf_counter() - started) * 1000
        plan = updates.get("plan")
        if plan is None:
            _safe_tracer_call(
                self._tracer,
                "on_plan_failed",
                updates.get("status", "no_plan"),
                duration_ms,
            )
        elif was_replan:
            reason = state.get("replan_reason", "unknown")
            _safe_tracer_call(self._tracer, "on_replan", reason, plan)
        else:
            _safe_tracer_call(self._tracer, "on_plan_complete", plan, duration_ms)
        return updates

    async def acall(self, state: GoapState) -> dict[str, Any]:
        """Async entry point — fires async tracer hooks."""
        goal = state.get("goal")
        world_state = state.get("world_state", {})
        strategy_name = self._strategy_name(goal)
        tracer_goal = self._effective_goal_for_tracer(state)
        await _safe_tracer_acall(
            self._tracer, "aon_plan_start", tracer_goal, world_state, strategy_name
        )
        started = time.perf_counter()
        updates, was_replan = self._plan_core(state)
        duration_ms = (time.perf_counter() - started) * 1000
        plan = updates.get("plan")
        if plan is None:
            await _safe_tracer_acall(
                self._tracer,
                "aon_plan_failed",
                updates.get("status", "no_plan"),
                duration_ms,
            )
        elif was_replan:
            reason = state.get("replan_reason", "unknown")
            await _safe_tracer_acall(self._tracer, "aon_replan", reason, plan)
        else:
            await _safe_tracer_acall(
                self._tracer, "aon_plan_complete", plan, duration_ms
            )
        return updates


def _apply_result(
    raw_result: Any,
    action: ActionSpec,
    world_state: dict[str, Any],
) -> None:
    """Merge an action's return value into world_state in place.

    If the callable returned a dict, its contents overwrite matching keys.
    Otherwise (None or any other type), the action's declared effects are
    applied — this covers the "no execute callable" path as well as callables
    that return non-dict sentinels.
    """
    if isinstance(raw_result, dict):
        world_state.update(raw_result)
    else:
        world_state.update(action.effects)


def _build_success(
    action: ActionSpec,
    current_step: int,
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the success return dict for the executor.

    When ``action.effect_validator`` rejects the post-state, the executor
    rolls the world state back to the snapshot taken before the action
    ran.  This is critical: a validator's purpose is to detect actions
    that did not actually accomplish what they claimed, and a rejection
    means the effects are *not trustworthy*.  Leaking the mutated
    world_state through would let the rejected action satisfy the
    planner's goal predicate (because the effect keys are already set),
    short-circuiting the blacklist + replan dance the validator was
    designed to trigger.  The mutated post-state is still exposed via
    ``ActionResult.state_after`` for diagnostics.
    """
    state_after = dict(world_state)
    if action.effect_validator is not None and not action.validate_effects(
        state_before, state_after
    ):
        logger.warning("Action %r effect validation failed", action.name)
        return {
            "status": "action_failed",
            "world_state": state_before,
            "execution_history": [
                ActionResult(
                    action_name=action.name,
                    success=False,
                    state_before=state_before,
                    state_after=state_after,
                    error="Effect validation failed",
                )
            ],
        }
    logger.debug("Action %r succeeded, world_state=%s", action.name, world_state)
    return {
        "world_state": world_state,
        "current_step": current_step + 1,
        "execution_history": [
            ActionResult(
                action_name=action.name,
                success=True,
                state_before=state_before,
                state_after=state_after,
            )
        ],
    }


def _build_failure(
    action: ActionSpec,
    exc: Exception,
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the failure return dict for the executor."""
    logger.warning("Action %r failed: %s", action.name, exc)
    return {
        "status": "action_failed",
        "execution_history": [
            ActionResult(
                action_name=action.name,
                success=False,
                state_before=state_before,
                state_after=dict(world_state),
                error=str(exc),
            )
        ],
    }


def _prepare_execution(
    state: GoapState,
) -> tuple[dict[str, Any], Plan, int, ActionSpec] | dict[str, Any]:
    """Validate executor pre-conditions and extract execution context.

    Returns either a ``(world_state, plan, step, action)`` tuple for the
    normal execution path, or a short-circuit result dict when the executor
    should return early without running any action.
    """
    status: str = state.get("status", "")
    plan_obj: Plan | None = state.get("plan")
    current_step: int = state.get("current_step", 0)
    world_state: dict[str, Any] = dict(state.get("world_state", {}))

    if status == "no_plan":
        return {}

    # Empty plan with no remaining steps — the planner returned a
    # zero-action Plan because the (sub-)goal was already satisfied
    # in the current world state.  Return a no-op update so the
    # observer can detect satisfaction and route accordingly;
    # emitting a ``"<none>"`` failure record here would clutter
    # execution history for legitimate pre-satisfied goals (notably
    # ``MultiGoal`` sub-goals that begin already met).
    if plan_obj is not None and len(plan_obj) == 0:
        return {}

    if plan_obj is None or current_step >= len(plan_obj):
        return {
            "status": "error",
            "execution_history": [
                ActionResult(
                    action_name="<none>",
                    success=False,
                    error="No action to execute",
                )
            ],
        }

    action = plan_obj.actions[current_step]
    return world_state, plan_obj, current_step, action


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
    """

    def __init__(self, *, tracer: PlanningTracer | None = None) -> None:
        self._tracer: PlanningTracer = tracer or NullTracer()

    def __call__(self, state: GoapState) -> dict[str, Any]:
        prep = _prepare_execution(state)
        if isinstance(prep, dict):
            return prep
        world_state, plan_obj, current_step, action = prep

        logger.info(
            "Executing action %r (step %d/%d)",
            action.name,
            current_step + 1,
            len(plan_obj),
        )

        _safe_tracer_call(self._tracer, "on_action_start", action, world_state)
        state_before = dict(world_state)
        try:
            if action.aexecute is not None and action.execute is None:
                # Action is async-only — fail fast with an actionable message
                # instead of silently applying declared effects.
                raise RuntimeError(
                    f"Action {action.name!r} is async-only (aexecute is set, "
                    "execute is None). Use GoapGraph.ainvoke() or "
                    "compiled.ainvoke() to run async actions."
                )
            if action.execute is not None:
                if inspect.iscoroutinefunction(action.execute):
                    # Defensive guard for manually-constructed ActionSpec where
                    # execute was set to a coroutine function by the caller.
                    raise RuntimeError(
                        f"Action {action.name!r} has an async execute callable. "
                        "Use GoapGraph.ainvoke() or compiled.ainvoke() instead."
                    )
                raw = action.execute(world_state)
            else:
                raw = None  # no callable — _apply_result will use declared effects
            _apply_result(raw, action, world_state)
            result = _build_success(action, current_step, state_before, world_state)
        except Exception as e:
            result = _build_failure(action, e, state_before, world_state)
        _safe_tracer_call(self._tracer, "on_action_complete", result)
        return result

    async def acall(self, state: GoapState) -> dict[str, Any]:
        """Async variant of the executor node.

        Uses :func:`_async_execute_action` to resolve the best callable:
        1. ``action.aexecute`` (explicit async callable)
        2. ``action.execute`` if it is a coroutine function
        3. ``action.execute`` via ``loop.run_in_executor`` (sync in thread)
        4. No callable — declared effects applied
        """
        prep = _prepare_execution(state)
        if isinstance(prep, dict):
            return prep
        world_state, plan_obj, current_step, action = prep

        logger.info(
            "Executing action %r async (step %d/%d)",
            action.name,
            current_step + 1,
            len(plan_obj),
        )

        await _safe_tracer_acall(self._tracer, "aon_action_start", action, world_state)
        state_before = dict(world_state)
        try:
            raw = await async_execute_action(action, world_state)
            _apply_result(raw, action, world_state)
            result = _build_success(action, current_step, state_before, world_state)
        except Exception as e:
            result = _build_failure(action, e, state_before, world_state)
        await _safe_tracer_acall(self._tracer, "aon_action_complete", result)
        return result


async def async_execute_action(action: ActionSpec, world_state: dict[str, Any]) -> Any:
    """Resolve and await the best async callable for *action*.

    Resolution priority:
    1. ``action.aexecute`` — explicit async callable (highest priority)
    2. ``action.execute`` if it is a coroutine function
    3. ``action.execute`` run in the default thread-pool executor (sync → async)
    4. No callable — returns ``None``; caller applies declared effects

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


def _get_last_failed_action(state: GoapState) -> str | None:
    """Extract the action name from the last failed ActionResult in history."""
    history: list[ActionResult] = state.get("execution_history", [])
    for result in reversed(history):
        if not result.success:
            return result.action_name
    return None


def _get_max_retries(action_name: str, actions: list[ActionSpec]) -> int:
    """Look up the max_retries value for *action_name*, defaulting to 0."""
    for action in actions:
        if action.name == action_name:
            return action.max_retries
    return 0


class GoapObserver:
    """LangGraph node that decides the next step via Command routing.

    After each action execution, the observer checks:
    1. Goal achieved → END
    2. Action failed → replan
    3. EVERY_ACTION replan strategy → replan
    4. State deviation from expected → replan
    5. More actions remain → continue executing
    6. Plan exhausted but goal not met → replan

    Args:
        actions: The available action specs.  When provided, deviation
            detection compares only planning-relevant keys (those
            appearing in action preconditions/effects and goal
            conditions) rather than the entire world state.  This
            prevents non-planning data (document lists, LLM responses)
            from triggering spurious replanning.
        tracer: Optional :class:`PlanningTracer` invoked on terminal
            states (``on_goal_achieved`` / ``on_plan_failed``).
            **Known limitation (v0.1.0):** mid-run ``MultiGoal``
            sub-goal completions do **not** emit tracer events — the
            tracer stays silent until the whole ``MultiGoal``
            terminates.  A future ``on_subgoal_achieved`` hook is out
            of scope for v0.1.0.
        history: Optional :class:`StoreExecutionHistory` that records
            one :class:`ExecutionRecord` per terminal state so downstream
            analytics can query past runs by goal or by failing action.
            For ``MultiGoal`` runs the record captures the sub-goal
            that was active at termination.
    """

    def __init__(
        self,
        actions: list[ActionSpec] | None = None,
        *,
        tracer: PlanningTracer | None = None,
        history: StoreExecutionHistory | None = None,
    ) -> None:
        self._actions = actions or []
        self._tracer: PlanningTracer = tracer or NullTracer()
        self._history = history

    def __call__(self, state: GoapState) -> Command[str]:
        cmd = self._route(state)
        self._fire_sync_tracer(state, cmd)
        self._maybe_record_sync(state, cmd)
        return cmd

    async def acall(self, state: GoapState) -> Command[str]:
        """Async variant — mirrors :meth:`__call__` but fires async hooks."""
        cmd = self._route(state)
        await self._fire_async_tracer(state, cmd)
        await self._maybe_record_async(state, cmd)
        return cmd

    def _fire_sync_tracer(self, state: GoapState, cmd: Command[str]) -> None:
        update = cmd.update or {}
        if cmd.goto == END:
            status = update.get("status")
            if status == "goal_achieved":
                _safe_tracer_call(
                    self._tracer,
                    "on_goal_achieved",
                    state.get("world_state", {}),
                )
            elif status in {"failed", "no_plan", "error"}:
                reason = update.get("replan_reason") or status
                _safe_tracer_call(self._tracer, "on_plan_failed", reason, 0.0)

    async def _fire_async_tracer(self, state: GoapState, cmd: Command[str]) -> None:
        update = cmd.update or {}
        if cmd.goto == END:
            status = update.get("status")
            if status == "goal_achieved":
                await _safe_tracer_acall(
                    self._tracer,
                    "aon_goal_achieved",
                    state.get("world_state", {}),
                )
            elif status in {"failed", "no_plan", "error"}:
                reason = update.get("replan_reason") or status
                await _safe_tracer_acall(self._tracer, "aon_plan_failed", reason, 0.0)

    def _build_record(
        self, state: GoapState, cmd: Command[str]
    ) -> ExecutionRecord | None:
        """Return an ``ExecutionRecord`` for a terminal command, or ``None``.

        Records are only built when the command routes to ``END`` with a
        non-trivial status — no record is emitted for intermediate
        observer decisions.  For ``MultiGoal`` runs, the record captures
        the sub-goal that was active at termination so downstream
        analytics can attribute success/failure correctly.
        """
        if cmd.goto != END:
            return None
        raw_goal = state.get("goal")
        if raw_goal is None:
            return None
        if isinstance(raw_goal, MultiGoal):
            idx = state.get("current_subgoal_index", 0)
            if idx >= len(raw_goal.goals):
                idx = len(raw_goal.goals) - 1
            goal: GoalSpec = raw_goal.goals[idx]
        else:
            goal = raw_goal
        update = cmd.update or {}
        status = update.get("status", "unknown")
        if status == "goal_achieved":
            outcome = "success"
        elif status in {"failed", "no_plan", "error"}:
            outcome = "failed"
        else:
            return None
        plan_obj: Plan | None = state.get("plan")
        return ExecutionRecord(
            goal_hash=compute_goal_hash(goal),
            goal_conditions=dict(goal.conditions),
            plan_actions=tuple(plan_obj.action_names) if plan_obj else (),
            expected_cost=plan_obj.total_cost if plan_obj else 0.0,
            actual_cost=plan_obj.total_cost if plan_obj else 0.0,
            outcome=outcome,
            replan_count=state.get("replan_count", 0),
            timestamp=datetime.now(timezone.utc),
        )

    def _maybe_record_sync(self, state: GoapState, cmd: Command[str]) -> None:
        if self._history is None:
            return
        record = self._build_record(state, cmd)
        if record is None:
            return
        try:
            self._history.record(record)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to record execution history")

    async def _maybe_record_async(self, state: GoapState, cmd: Command[str]) -> None:
        if self._history is None:
            return
        record = self._build_record(state, cmd)
        if record is None:
            return
        try:
            await self._history.arecord(record)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to record execution history")

    def _route(self, state: GoapState) -> Command[str]:
        raw_goal: GoalSpec | MultiGoal | None = state.get("goal")
        world_state: dict[str, Any] = state.get("world_state", {})
        plan_obj: Plan | None = state.get("plan")
        current_step: int = state.get("current_step", 0)
        status: str = state.get("status", "")

        if raw_goal is None:
            return Command(
                goto=END,
                update={"status": "error", "replan_reason": "no goal specified"},
            )

        # ``MultiGoal`` dispatch: resolve to the current sub-goal before
        # the existing single-goal logic runs.  In ``sequential`` mode,
        # advance ``current_subgoal_index`` when a sub-goal is satisfied.
        # In ``any`` mode, the planner already committed to a specific
        # sub-goal via ``current_subgoal_index`` — satisfying it ends
        # the run (we do not chase the remaining alternatives).
        goal: GoalSpec
        if isinstance(raw_goal, MultiGoal):
            idx = state.get("current_subgoal_index", 0)
            if idx >= len(raw_goal.goals):
                return Command(goto=END, update={"status": "goal_achieved"})
            current_sub = raw_goal.goals[idx]
            sub_satisfied = all(
                k in world_state and world_state[k] == v
                for k, v in current_sub.conditions.items()
            )
            if sub_satisfied:
                if raw_goal.mode == "sequential" and idx + 1 < len(raw_goal.goals):
                    # Reset per-sub-goal accounting so the next
                    # sub-goal gets a fresh replan budget, blacklist,
                    # and failure counts.  ``max_replans`` is declared
                    # per ``GoalSpec`` so its budget must also apply
                    # per sub-goal; similarly, an action blacklisted
                    # while working on sub-goal i should not carry
                    # over to sub-goal i+1 which may need it.
                    return Command(
                        goto="planner",
                        update={
                            "current_subgoal_index": idx + 1,
                            "plan": None,
                            "current_step": 0,
                            "replan_reason": "subgoal_achieved",
                            "replan_count": 0,
                            "blacklisted_actions": [],
                            "action_failure_counts": {},
                        },
                    )
                return Command(goto=END, update={"status": "goal_achieved"})
            # Not yet satisfied — fall through with the effective sub-goal.
            goal = current_sub
        else:
            goal = raw_goal

        # Goal achieved — check directly on the world_state dict to
        # avoid creating a PlanningState (which would choke on
        # non-hashable values like document lists).
        # NOTE: must check `k in world_state` first; dict.get(k) returns None
        # for missing keys, which would incorrectly satisfy a None-valued goal
        # condition when the key is simply absent.
        if all(
            k in world_state and world_state[k] == v for k, v in goal.conditions.items()
        ):
            return Command(
                goto=END,
                update={"status": "goal_achieved"},
            )

        # Replan limit exceeded → give up to avoid infinite loops
        replan_count: int = state.get("replan_count", 0)
        if goal.max_replans > 0 and replan_count >= goal.max_replans:
            return Command(
                goto=END,
                update={
                    "status": "failed",
                    "replan_reason": f"max_replans_exceeded ({goal.max_replans})",
                },
            )

        # No plan possible → stop
        if status == "no_plan":
            return Command(
                goto=END,
                update={"status": "no_plan"},
            )

        never = goal.replan_strategy == ReplanStrategy.NEVER

        # Action failed → track failure, blacklist if threshold exceeded, replan
        if status == "action_failed":
            if never:
                # With NEVER, treat a failed action as a hard stop.
                return Command(
                    goto=END,
                    update={"status": "failed", "replan_reason": "action_failed"},
                )

            failed_name = _get_last_failed_action(state)
            if failed_name is not None:
                counts = dict(state.get("action_failure_counts", {}))
                counts[failed_name] = counts.get(failed_name, 0) + 1
                max_retries = _get_max_retries(failed_name, self._actions)
                blacklist = list(state.get("blacklisted_actions", []))
                if counts[failed_name] > max_retries:
                    if failed_name not in blacklist:
                        blacklist.append(failed_name)
                return Command(
                    goto="planner",
                    update={
                        "replan_reason": "action_failed",
                        "action_failure_counts": counts,
                        "blacklisted_actions": blacklist,
                    },
                )
            return Command(
                goto="planner",
                update={"replan_reason": "action_failed"},
            )

        # EVERY_ACTION replan strategy (not applicable if NEVER)
        if goal.replan_strategy == ReplanStrategy.EVERY_ACTION:
            return Command(
                goto="planner",
                update={"replan_reason": "every_action_replan"},
            )

        # Check for state deviation from expected plan (ON_DEVIATION only).
        # Compare only planning-relevant keys so that changes to execution
        # context (documents, generations, etc.) don't trigger replanning.
        if (
            not never
            and plan_obj is not None
            and current_step > 0
            and current_step <= len(plan_obj.expected_states)
            and goal.replan_strategy == ReplanStrategy.ON_DEVIATION
        ):
            pkeys = _planning_keys(self._actions, goal)
            planning_state = PlanningState.from_dict(world_state, keys=pkeys)
            expected_raw = plan_obj.expected_states[current_step - 1]
            # Filter expected state to the same planning keys so
            # manually-constructed plans (e.g. in tests) don't cause
            # false deviations from non-planning keys.
            expected = PlanningState.from_dict(expected_raw.to_dict(), keys=pkeys)
            if planning_state != expected:
                return Command(
                    goto="planner",
                    update={"replan_reason": "state_deviation"},
                )

        # More actions in the plan → continue
        if plan_obj is not None and current_step < len(plan_obj):
            return Command(goto="executor")

        # Plan exhausted but goal not met → replan (unless NEVER)
        if never:
            return Command(
                goto=END,
                update={"status": "failed", "replan_reason": "plan_exhausted"},
            )
        return Command(
            goto="planner",
            update={"replan_reason": "plan_exhausted"},
        )
