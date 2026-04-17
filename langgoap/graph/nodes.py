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
from langgraph.types import Command, interrupt

from langgoap.actions import ActionSpec
from langgoap.conditions import (
    AsyncConditionResolver,
    ConditionResolver,
    aresolve_conditions,
    resolve_conditions,
)
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.state import ActionResult, GoapState
from langgoap.guards import (
    ActionGuard,
    AsyncActionGuard,
    GuardResult,
    has_blocking_failure,
    run_guards_async,
    run_guards_sync,
)
from langgoap.history import (
    ExecutionRecord,
    StoreExecutionHistory,
    compute_goal_hash,
)
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.explain import explain_no_plan
from langgoap.planner.types import Plan
from langgoap.score import SimpleScore
from langgoap.sensors import (
    AsyncSensor,
    Sensor,
    run_sensors_async,
    run_sensors_sync,
)
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


def _is_better_plan(candidate: Plan, best: Plan) -> bool:
    """Feasibility-first comparison for MultiGoal ``any`` mode.

    Uses three-tier ordering so that CSP soft-score information is never
    silently discarded when sub-goals are compared:

    **Tier 1 — Feasibility**
        A feasible plan always beats an infeasible one, regardless of cost
        or objective value.

    **Tier 2 — Native score comparison**
        When both plans share the same concrete :class:`~langgoap.score.Score`
        subtype, the subtype's own comparison is used.  Sign conventions
        differ between subtypes and are handled explicitly:

        * :class:`~langgoap.score.SimpleScore` — *lower* scalar is better
          (A* minimises path cost; ``scalar`` equals ``total_cost``).
        * :class:`~langgoap.score.HardSoftScore` /
          :class:`~langgoap.score.BendableScore` — *higher* (less-negative)
          is better (lexicographic convention; ``hard == 0`` is feasible).

        Using native comparison preserves CSP soft-score information: a
        ``HardSoftScore(hard=0, soft=-1)`` plan correctly beats a
        ``HardSoftScore(hard=0, soft=-10)`` plan even when both have the
        same ``total_cost``.

        A :class:`TypeError` raised by cross-subtype comparisons or
        mismatched :class:`~langgoap.score.BendableScore` shapes falls
        through to tier 3.

    **Tier 3 — total_cost fallback**
        ``total_cost`` (a plain ``float``, always present on every
        :class:`~langgoap.planner.types.Plan`) breaks ties when tier 2
        cannot resolve them.

        Among *infeasible* plans this picks the cheapest rather than the
        *least* infeasible (i.e. the plan whose ``hard`` score is closest
        to zero).  This is acceptable because MultiGoal ``any`` reports
        ``no_plan`` when *every* sub-goal is infeasible regardless of which
        infeasible candidate is selected.  A future improvement could
        compare ``hard`` scores directly for the infeasible-vs-infeasible
        case; for now the simpler ``total_cost`` tiebreaker is documented
        here rather than silently applied.
    """
    c_feasible = candidate.score.is_feasible()
    b_feasible = best.score.is_feasible()
    if c_feasible != b_feasible:
        return c_feasible
    # Tier 2: same-subtype native comparison preserves soft-score information.
    # TypeError fires on cross-subtype or mismatched BendableScore shapes.
    try:
        if isinstance(candidate.score, SimpleScore):
            # SimpleScore: lower scalar is better (cost minimisation).
            return candidate.score < best.score
        # HardSoftScore / BendableScore: higher (less-negative) is better.
        # mypy sees Score (base class) here; __gt__ is defined on every
        # concrete subclass but not on Score itself — hence the ignore.
        return candidate.score > best.score  # type: ignore[operator]
    except TypeError:
        pass
    # Tier 3: cross-subtype fallback — total_cost is a uniform float proxy.
    return candidate.total_cost < best.total_cost


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
        sensors: list[Sensor | AsyncSensor] | None = None,
        resolvers: list[ConditionResolver | AsyncConditionResolver] | None = None,
    ) -> None:
        self.actions = actions
        self._strategy = strategy
        self._tracer: PlanningTracer = tracer or NullTracer()
        self._sensors: list[Sensor | AsyncSensor] = list(sensors) if sensors else []
        self._resolvers: list[ConditionResolver | AsyncConditionResolver] = (
            list(resolvers) if resolvers else []
        )

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
            return self._strategy.plan(
                start,
                goal,
                self.actions,
                blacklisted_actions=blacklisted,
            )
        if goal.constraints or goal.objectives is not None:
            from langgoap.planner.pipeline import plan as pipeline_plan

            return pipeline_plan(
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
                # Enumerate sub-goals, plan each, pick the best feasible.
                # Feasibility-first: a feasible plan always beats an
                # infeasible one regardless of cost.  Among plans with the
                # same feasibility status, lower total_cost wins.  Using
                # total_cost (a plain float) avoids the cross-subclass
                # TypeError that Score.__lt__ raises when comparing e.g.
                # SimpleScore (from pure A*) against HardSoftScore (from
                # the CSP pipeline).
                best_idx = -1
                best_plan: Plan | None = None
                for i, sg in enumerate(raw_goal.goals):
                    sub_pkeys = _planning_keys(self.actions, sg)
                    sub_start = PlanningState.from_dict(world_state, keys=sub_pkeys)
                    candidate = self._plan_single(sub_start, sg, blacklisted)
                    if candidate is None:
                        continue
                    if best_plan is None or _is_better_plan(candidate, best_plan):
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
            no_plan_expl = explain_no_plan(start, effective_goal, self.actions)
            logger.warning(
                "A* found no plan for goal %s — %s",
                dict(effective_goal.conditions),
                no_plan_expl.suggestion,
            )
            return (
                {
                    "plan": None,
                    "status": "no_plan",
                    "current_step": 0,
                    "replan_count": replan_count,
                    "no_plan_explanation": no_plan_expl.to_dict(),
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
            # Clear any stale no-plan explanation from a previous failed attempt.
            "no_plan_explanation": None,
            **extra_updates,
        }
        if used_blacklisted:
            # Clear blacklist and failure counts so the retried action
            # gets a fresh start instead of being immediately re-blacklisted.
            updates["blacklisted_actions"] = []
            updates["action_failure_counts"] = {}
        # Surface reflections from a ReflexionTracer (or MultiTracer containing
        # one) into GoapState so downstream components (e.g. PromptConditions)
        # can read them without coupling to the tracer directly.
        reflections = getattr(self._tracer, "reflections", None)
        if reflections:
            updates["reflection_context"] = [
                f"[{r.action_name}] {r.reflection} → {r.suggestion}"
                for r in reflections
            ]
        return updates, was_replan

    def _run_sensors_sync(self, state: GoapState) -> dict[str, Any] | None:
        """Run sensors and return world_state update, or None if no sensors."""
        if not self._sensors:
            return None
        world_state = dict(state.get("world_state", {}))
        sensor_results = run_sensors_sync(self._sensors, world_state)
        for sensor_name, updates in sensor_results:
            _safe_tracer_call(self._tracer, "on_sensor_complete", sensor_name, updates)
        return world_state

    async def _run_sensors_async(self, state: GoapState) -> dict[str, Any] | None:
        """Run sensors async and return world_state update, or None if no sensors."""
        if not self._sensors:
            return None
        world_state = dict(state.get("world_state", {}))
        sensor_results = await run_sensors_async(self._sensors, world_state)
        for sensor_name, updates in sensor_results:
            await _safe_tracer_acall(
                self._tracer, "aon_sensor_complete", sensor_name, updates
            )
        return world_state

    def __call__(self, state: GoapState) -> dict[str, Any]:
        # Run sensors before planning
        sensor_ws = self._run_sensors_sync(state)
        if sensor_ws is not None:
            state = {**state, "world_state": sensor_ws}

        # Run condition resolvers to fill in UNKNOWN/missing keys
        if self._resolvers:
            ws = dict(state.get("world_state", {}))
            # Collect all action precondition/effect keys (goal-independent)
            resolver_keys: list[str] = []
            for a in self.actions:
                resolver_keys.extend(a.preconditions.keys())
                resolver_keys.extend(a.effects.keys())
            ws = resolve_conditions(self._resolvers, list(set(resolver_keys)), ws)
            state = {**state, "world_state": ws}

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
        # If sensors updated world_state, propagate it to the output
        if sensor_ws is not None and "world_state" not in updates:
            updates["world_state"] = sensor_ws
        return updates

    async def acall(self, state: GoapState) -> dict[str, Any]:
        """Async entry point — fires async tracer hooks."""
        # Run sensors before planning
        sensor_ws = await self._run_sensors_async(state)
        if sensor_ws is not None:
            state = {**state, "world_state": sensor_ws}

        # Run condition resolvers (async path)
        if self._resolvers:
            ws = dict(state.get("world_state", {}))
            resolver_keys: list[str] = []
            for a in self.actions:
                resolver_keys.extend(a.preconditions.keys())
                resolver_keys.extend(a.effects.keys())
            ws = await aresolve_conditions(
                self._resolvers, list(set(resolver_keys)), ws
            )
            state = {**state, "world_state": ws}

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
        # If sensors updated world_state, propagate it to the output
        if sensor_ws is not None and "world_state" not in updates:
            updates["world_state"] = sensor_ws
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


def _build_guard_block(
    action: ActionSpec,
    guard_results: list[GuardResult],
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure result dict for a BLOCK-severity guard failure.

    Immediately blacklists the action (failure_count = max_retries + 1) so
    the observer's blacklist logic trips on the first guard block regardless
    of ``action.max_retries``.  A guard block is a policy decision, not a
    transient flake, so retries would be wrong.
    """
    blocking = [r for r in guard_results if not r.passed]
    reason = "; ".join(r.message for r in blocking) if blocking else "guard_blocked"
    result = _build_failure(
        action,
        RuntimeError(f"guard_blocked: {reason}"),
        state_before,
        world_state,
    )
    result["action_failure_counts"] = {action.name: action.max_retries + 1}
    return result


def _is_approved(resume_value: Any) -> bool:
    """Interpret a ``Command(resume=...)`` value as an approval decision.

    The resume value sent by the human client is interpreted generously:

    - ``None`` → approved (implicit: resume with no payload = continue)
    - ``True`` → approved
    - ``False`` → denied
    - ``{"approved": True}`` → approved
    - ``{"approved": False}`` → denied
    - Any other truthy value → approved
    """
    if resume_value is None or resume_value is True:
        return True
    if resume_value is False:
        return False
    if isinstance(resume_value, dict):
        return bool(resume_value.get("approved", True))
    return bool(resume_value)


def _check_human_approval(
    action: ActionSpec,
    world_state: dict[str, Any],
    state_before: dict[str, Any],
) -> dict[str, Any] | None:
    """Gate execution behind a human approval interrupt when required.

    Calls :func:`~langgraph.types.interrupt` to pause the graph.  On
    first encounter LangGraph raises ``GraphInterrupt`` (persisted by
    the checkpointer).  On resume, ``interrupt()`` returns the
    ``Command(resume=...)`` payload.

    Returns:
        ``None`` if the action is approved (or does not require approval).
        A ``GoapState`` delta dict if the action was denied — the caller
        should return this immediately and skip execution.
    """
    if not action.require_human_approval:
        return None

    resume_value = interrupt(
        {
            "type": "goap_action_approval",
            "action": action.name,
            "preconditions": dict(action.preconditions),
            "effects": dict(action.effects),
            "world_state": dict(world_state),
        }
    )

    if _is_approved(resume_value):
        return None

    # Denied — build a failure result.  Pre-set the failure count to
    # max_retries + 1 so the observer's blacklist threshold trips on the
    # first denial regardless of max_retries (a human denial is a
    # deliberate decision, not a transient flake).
    reason = "denied by operator"
    if isinstance(resume_value, dict):
        reason = resume_value.get("reason", reason)
    result = _build_failure(
        action,
        RuntimeError(f"human_approval_denied: {reason}"),
        state_before,
        world_state,
    )
    result["action_failure_counts"] = {action.name: action.max_retries + 1}
    return result


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
        guards: Optional list of :class:`~langgoap.guards.ActionGuard` or
            :class:`~langgoap.guards.AsyncActionGuard` objects evaluated
            *before* each action executes.  A ``WARN``-severity failure is
            logged but execution continues.  A ``BLOCK``-severity failure
            aborts the action and immediately blacklists it (equivalent to
            ``max_retries + 1`` failures), triggering replanning via the
            observer.
    """

    def __init__(
        self,
        *,
        tracer: PlanningTracer | None = None,
        guards: list[ActionGuard | AsyncActionGuard] | None = None,
    ) -> None:
        self._tracer: PlanningTracer = tracer or NullTracer()
        self._guards: list[ActionGuard | AsyncActionGuard] = guards or []

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

        # GuardRails: run all registered guards before executing.
        if self._guards:
            guard_results = run_guards_sync(self._guards, action, world_state)
            if has_blocking_failure(guard_results):
                result = _build_guard_block(
                    action, guard_results, state_before, world_state
                )
                _safe_tracer_call(self._tracer, "on_action_complete", result)
                return result

        # Human-in-the-loop gate: interrupt() raises GraphInterrupt on
        # first pass (checkpointer persists state); returns the resume
        # payload on the second pass.
        denial = _check_human_approval(action, world_state, state_before)
        if denial is not None:
            _safe_tracer_call(self._tracer, "on_action_complete", denial)
            return denial

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

        # GuardRails: run all registered guards before executing (async path).
        if self._guards:
            guard_results = await run_guards_async(self._guards, action, world_state)
            if has_blocking_failure(guard_results):
                result = _build_guard_block(
                    action, guard_results, state_before, world_state
                )
                await _safe_tracer_acall(self._tracer, "aon_action_complete", result)
                return result

        denial = _check_human_approval(action, world_state, state_before)
        if denial is not None:
            await _safe_tracer_acall(self._tracer, "aon_action_complete", denial)
            return denial

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

    When a plan contains a sequence of actions that share no precondition/effect
    dependencies, they form a *parallel wave*.  This executor detects the maximal
    wave starting at ``current_step`` and runs all wave members concurrently:

    * **Async path** (:meth:`acall`): uses :func:`asyncio.gather` for true
      concurrency.  Mixed sync/async callables are handled via
      :func:`async_execute_action` (sync callables are offloaded to
      :func:`asyncio.loop.run_in_executor`).
    * **Sync path** (:meth:`__call__`): executes wave actions sequentially in the
      same thread (Python's GIL limits true parallelism here).  Users gain fewer
      observer round-trips — the full wave completes in a single node invocation.

    On success the merged world-state update advances ``current_step`` to the
    start of the next wave.  On failure (any action raises) the wave aborts, no
    effects are applied, and the first failing action is blacklisted (triggering
    replanning via the observer).

    Args:
        tracer: Optional :class:`~langgoap.tracing.PlanningTracer` invoked for
            each action in the wave.
        guards: Optional list of :class:`~langgoap.guards.ActionGuard` /
            :class:`~langgoap.guards.AsyncActionGuard` evaluated before each
            action.  A ``BLOCK`` failure aborts that action and the whole wave.
    """

    def __init__(
        self,
        *,
        tracer: PlanningTracer | None = None,
        guards: list[ActionGuard | AsyncActionGuard] | None = None,
    ) -> None:
        self._tracer: PlanningTracer = tracer or NullTracer()
        self._guards: list[ActionGuard | AsyncActionGuard] = guards or []
        # Cache the dependency graph by plan-actions identity so it is
        # computed once per plan rather than once per wave.
        self._dep_cache: tuple[int, dict[int, list[int]]] | None = None

    def _get_deps(self, plan_actions: tuple[ActionSpec, ...]) -> dict[int, list[int]]:
        """Return the dependency graph, caching by object identity."""
        key = id(plan_actions)
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
            _safe_tracer_call(self._tracer, "on_action_start", action, merged_world)
            state_before = dict(merged_world)

            # Guards
            if self._guards:
                guard_results = run_guards_sync(self._guards, action, merged_world)
                if has_blocking_failure(guard_results):
                    result = _build_guard_block(
                        action, guard_results, state_before, merged_world
                    )
                    _safe_tracer_call(self._tracer, "on_action_complete", result)
                    return result

            try:
                if action.aexecute is not None and action.execute is None:
                    raise RuntimeError(
                        f"Action {action.name!r} is async-only. Use ainvoke()."
                    )
                raw = (
                    action.execute(merged_world) if action.execute is not None else None
                )
                _apply_result(raw, action, merged_world)
                action_result = _build_success(action, idx, state_before, merged_world)
            except Exception as exc:
                action_result = _build_failure(action, exc, state_before, merged_world)
                _safe_tracer_call(self._tracer, "on_action_complete", action_result)
                return action_result

            _safe_tracer_call(self._tracer, "on_action_complete", action_result)
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
        """Async variant — executes the wave actions concurrently via asyncio.gather."""
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

            await _safe_tracer_acall(
                self._tracer, "aon_action_start", action, ws_snapshot
            )
            state_before = dict(ws_snapshot)

            if self._guards:
                guard_results = await run_guards_async(
                    self._guards, action, ws_snapshot
                )
                if has_blocking_failure(guard_results):
                    result = _build_guard_block(
                        action, guard_results, state_before, ws_snapshot
                    )
                    await _safe_tracer_acall(
                        self._tracer, "aon_action_complete", result
                    )
                    return result

            try:
                raw = await async_execute_action(action, ws_snapshot)
                _apply_result(raw, action, ws_snapshot)
                result = _build_success(action, idx, state_before, ws_snapshot)
            except Exception as exc:
                result = _build_failure(action, exc, state_before, ws_snapshot)

            await _safe_tracer_acall(self._tracer, "aon_action_complete", result)
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
