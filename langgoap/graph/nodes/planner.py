"""LangGraph planner node \u2014 invokes the configured planning strategy."""

from __future__ import annotations

import asyncio
import contextvars
import time
from typing import TYPE_CHECKING, Any

from langgoap.actions import ActionSpec, ensure_unique_action_names
from langgoap.conditions import (
    AsyncConditionResolver,
    ConditionResolver,
    aresolve_conditions,
    resolve_conditions,
)
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.nodes._helpers import (
    _is_better_plan,
    _planning_keys,
    logger,
)
from langgoap.graph.nodes.observer import _is_goal_satisfied
from langgoap.graph.state import (
    GoapState,
    rehydrate_state,
    serialize_updates,
    state_defaults,
)
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.csp import CSPStatus
from langgoap.planner.explain import explain_no_plan
from langgoap.planner.types import Plan
from langgoap.sensors import (
    AsyncSensor,
    Sensor,
    run_sensors_async,
    run_sensors_sync,
)
from langgoap.state import PlanningState
from langgoap.tracing import NullTracer, PlanningTracer, SafeTracerProxy

if TYPE_CHECKING:
    from langgoap.planner.strategy import PlanningStrategy
    from langgoap.stuck import StuckHandler


def _any_mode_no_plan_dict(
    goal: MultiGoal, rejections: list[dict[str, Any]], blacklisted: list[str]
) -> dict[str, Any]:
    """Build the ``no_plan_explanation`` for an any/best_value ``MultiGoal``.

    Each sub-goal's own explanation is kept under ``subgoals`` so a stuck
    handler can see *why* every alternative failed instead of ``None`` (#64).
    """
    lines = [
        f"[{i}] {rej.get('suggestion', 'no plan')}" for i, rej in enumerate(rejections)
    ]
    payload: dict[str, Any] = {
        "mode": goal.mode,
        "subgoals": rejections,
        "suggestion": (
            f"None of the {len(rejections)} sub-goal(s) of this MultiGoal "
            f"({goal.mode} mode) can be planned. " + " ".join(lines)
        ),
    }
    if blacklisted:
        payload["blacklisted_actions"] = list(blacklisted)
    return payload


_DENIAL_PREFIX = "human_approval_denied"


def _denial_reasons(state: GoapState, blacklisted: list[str]) -> dict[str, str]:
    """Map each blacklisted action an operator denied to the stated reason.

    A denial (boolean gate or typed form) is recorded as a failed
    :class:`ActionResult` whose error starts with ``human_approval_denied``.
    The most recent such record per action wins.
    """
    reasons: dict[str, str] = {}
    for record in state.get("execution_history", []):
        if record.success or record.action_name not in blacklisted:
            continue
        error = str(record.error or "")
        if error.startswith(_DENIAL_PREFIX):
            reasons[record.action_name] = error[len(_DENIAL_PREFIX) :].lstrip(": ")
    return reasons


def _blacklist_suggestion(
    failed: list[str], spent: list[str], denied: dict[str, str] | None = None
) -> str:
    """Explain a goal that is reachable only through blacklisted actions.

    ``failed`` actions were blacklisted after exhausting ``max_retries``;
    ``spent`` ones are ``can_rerun=False`` actions that already ran once;
    ``denied`` maps actions an operator declined to the stated reason.
    The three need different advice: retries and QoS do nothing for an
    action that succeeded and simply cannot run again (a ``NirvanaGoal``
    run ends this way by design), and nothing for a deliberate human
    decision either.
    """
    parts: list[str] = []
    if denied:
        described = ", ".join(
            f"{name} ({reason})" if reason else name for name, reason in denied.items()
        )
        parts.append(
            "Every remaining path to the goal needs an action the operator "
            f"denied: {described}. Retries and ActionQos do not apply to a "
            "denial: add an alternative action, attach a stuck handler that "
            "relaxes the goal, or run again once the decision changes."
        )
    if failed:
        parts.append(
            "Every remaining path to the goal needs a blacklisted action "
            f"({', '.join(failed)}). Raise ActionSpec.max_retries or attach an "
            "ActionQos for transient failures, add an alternative action, or set "
            "GoalPolicy(blacklist_fallback=True) to re-admit blacklisted actions."
        )
    if spent:
        parts.append(
            "Every remaining path to the goal needs an action that already ran "
            f"and cannot run again (can_rerun=False): {', '.join(spent)}. Add an "
            "alternative action, or set can_rerun=True if repeating it is "
            "acceptable. For a NirvanaGoal this is the normal end of the run."
        )
    return " ".join(parts)


def _infeasible_no_plan_dict(plan: Plan, goal: GoalSpec) -> dict[str, Any]:
    """Build the ``no_plan_explanation`` payload for a rejected infeasible plan.

    Shaped like :meth:`~langgoap.planner.explain.NoPlanExplanation.to_dict`
    (so ``NoPlanExplanation.from_dict`` still works) plus two extra keys:
    ``infeasible_plan`` (the action names of the rejected plan) and
    ``resource_shortfalls`` (one dict per violated hard constraint).
    """
    csp = plan.metadata.csp
    shortfalls: list[dict[str, Any]] = []
    suggestion = (
        "The best plan found violates the goal's hard constraints (CSP status "
        "infeasible). Relax a constraint, add cheaper actions, or set "
        "GoalPolicy(on_infeasible='execute') to run it anyway."
    )
    explanation = csp.explanation if csp is not None else None
    if explanation is not None:
        shortfalls = [
            {
                "key": s.key,
                "required": s.required,
                "available_max": s.available_max,
                "available_min": s.available_min,
                "overrun": s.overrun,
            }
            for s in explanation.resource_shortfalls
        ]
        if explanation.suggestion:
            suggestion = explanation.suggestion
    return {
        "unreachable_conditions": [],
        "missing_preconditions": [],
        "suggestion": suggestion,
        "infeasible_plan": list(plan.action_names),
        "resource_shortfalls": shortfalls,
    }


# Per-task overrides consulted by the planning core.  ``acall`` sets them
# around a worker-thread run of the (synchronous) planning core so a
# strategy's ``aplan`` and coroutine stuck handlers execute on the event
# loop while the core stays single-sourced (issue #53).
_STRATEGY_OVERRIDE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "langgoap_strategy_override", default=None
)
_STUCK_OVERRIDE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "langgoap_stuck_override", default=None
)


class _AsyncStrategyBridge:
    """Expose a strategy's ``aplan`` as a blocking ``plan`` for the core thread."""

    def __init__(self, inner: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._inner = inner
        self._loop = loop

    def plan(self, start: Any, goal: Any, actions: Any, **kwargs: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self._inner.aplan(start, goal, actions, **kwargs), self._loop
        )
        return future.result()


class _AsyncStuckBridge:
    """Expose ``MulticastStuckHandler.ahandle_stuck`` as a blocking call."""

    def __init__(self, inner: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._inner = inner
        self._loop = loop

    def handle_stuck(self, state: Any, reason: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self._inner.ahandle_stuck(state, reason), self._loop
        )
        return future.result()


class GoapPlanner:
    """LangGraph node that invokes the planner.

    Reads world_state and goal from GoapState, runs the configured
    :class:`~langgoap.planner.strategy.PlanningStrategy` (default: pure
    A* when the goal has no constraints/objectives, or the two-phase
    A*\u2192CSP pipeline otherwise), and writes the resulting plan back.

    Args:
        actions: Available actions for planning.
        strategy: Optional custom :class:`PlanningStrategy`.  When
            ``None`` (default) the planner auto-selects between pure A*
            and the two-phase pipeline based on the goal \u2014 identical to
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
        record_expansions: bool = False,
        stuck_handlers: "list[StuckHandler] | None" = None,
        max_stuck_iterations: int = 3,
    ) -> None:
        from langgoap.stuck import MulticastStuckHandler

        ensure_unique_action_names(actions)
        self.actions = actions

        self._actions_by_name = {a.name: a for a in self.actions}
        # Resolvers answer questions the actions cannot: precondition keys
        # that no action produces.  Effect keys and goal conditions are
        # the planner's business, and asking an (LLM) resolver about them
        # lets it declare the goal already achieved (issue #32).
        produced: set[str] = set()
        required: set[str] = set()
        for a in self.actions:
            produced.update(a.effect_key_set())
            required.update(a.preconditions.keys())
        self._resolver_keys: list[str] = sorted(required - produced)
        self._strategy = strategy
        # See ``GoapExecutor.__init__`` for the SafeTracerProxy rationale.
        self._tracer: PlanningTracer = SafeTracerProxy(tracer or NullTracer())
        self._has_tracer = tracer is not None
        self._record_expansions = record_expansions
        self._sensors: list[Sensor | AsyncSensor] = list(sensors) if sensors else []
        self._resolvers: list[ConditionResolver | AsyncConditionResolver] = (
            list(resolvers) if resolvers else []
        )
        # Wrap user handlers in a multicast composite so the planner
        # only consults a single object.  Empty list short-circuits to
        # NO_RESOLUTION on each consultation.
        self._stuck_handler: MulticastStuckHandler | None = (
            MulticastStuckHandler(list(stuck_handlers)) if stuck_handlers else None
        )
        if max_stuck_iterations < 1:
            raise ValueError(
                f"max_stuck_iterations must be >= 1, got {max_stuck_iterations}"
            )
        self._max_stuck_iterations = max_stuck_iterations

    def _strategy_name(self, goal: GoalSpec | MultiGoal | None) -> str:
        """Human-readable label passed to ``on_plan_start``."""
        if self._strategy is not None:
            return type(self._strategy).__name__
        from langgoap.planner.pipeline import needs_csp

        if isinstance(goal, GoalSpec) and needs_csp(goal):
            return "TwoPhasePipeline"
        return "AStar"

    def _effective_goal_for_tracer(
        self, state: GoapState
    ) -> GoalSpec | MultiGoal | None:
        """Resolve the ``MultiGoal`` wrapper for tracer visibility.

        For ``"sequential"`` mode, tracers should see the plain
        :class:`GoalSpec` currently being planned so hooks like
        ``on_plan_start`` receive a goal with a readable
        ``conditions`` attribute \u2014 otherwise every sub-goal advance
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
        *,
        prior_plan: Plan | None = None,
        current_step: int = 0,
    ) -> Plan | None:
        """Run the configured strategy (or automatic routing) once.

        Extracted so both the single-goal path and the ``MultiGoal``
        dispatch can share planning logic without duplication.

        ``prior_plan`` and ``current_step`` are forwarded only to the
        configured strategy.  The default A*/pipeline routing always
        plans from scratch, so the kwargs are inert there.  Repair-
        style strategies (e.g.
        :class:`~langgoap.planner.repair.RepairStrategy`) consume them
        to patch a partially-executed plan instead of replanning.
        """
        strategy = _STRATEGY_OVERRIDE.get() or self._strategy
        if strategy is not None:
            strategy = self._bind_tracer(strategy)
            # Forward repair kwargs only when meaningful so that legacy
            # user strategies whose ``plan`` signature predates the
            # widened Protocol (no ``prior_plan``/``current_step`` kwargs)
            # keep working on initial plans.  Strategies that opt into the
            # new Protocol shape will accept them either way.
            extras: dict[str, Any] = {}
            if prior_plan is not None:
                extras["prior_plan"] = prior_plan
                extras["current_step"] = current_step
            return strategy.plan(
                start,
                goal,
                self.actions,
                blacklisted_actions=blacklisted,
                **extras,
            )
        from langgoap.planner.pipeline import needs_csp

        if needs_csp(goal):
            from langgoap.planner.pipeline import plan as pipeline_plan

            return pipeline_plan(
                start,
                goal,
                self.actions,
                blacklisted_actions=blacklisted,
                tracer=self._tracer,
                record_expansions=self._record_expansions,
            )
        return astar_plan(
            start,
            goal,
            self.actions,
            blacklisted_actions=blacklisted,
            tracer=self._tracer,
            record_expansions=self._record_expansions,
        )

    def _bind_tracer(self, strategy: Any) -> Any:
        """Give a searching strategy this planner's tracer when it has none.

        ``AStarStrategy()`` and ``TwoPhasePipelineStrategy()`` passed
        explicitly used to lose search tracing that the default path kept
        (#63).  Strategies expose an optional ``_with_tracer`` hook; those
        constructed with their own tracer keep it.
        """
        if not self._has_tracer:
            return strategy
        binder = getattr(strategy, "_with_tracer", None)
        if binder is None:
            return strategy
        return binder(self._tracer, self._record_expansions)

    def _plan_single_checked(
        self,
        start: PlanningState,
        goal: GoalSpec,
        blacklisted: list[str],
        *,
        prior_plan: Plan | None = None,
        current_step: int = 0,
    ) -> tuple[Plan | None, dict[str, Any] | None]:
        """Run :meth:`_plan_single` and apply ``GoalPolicy.on_infeasible``.

        Returns ``(plan, rejection)``.  When the strategy's best plan
        carries ``CSPStatus.INFEASIBLE`` and the goal's policy is
        ``"fail"`` (the default), the plan is rejected: ``plan`` is
        ``None`` and ``rejection`` is the ``no_plan_explanation`` payload
        describing the violated hard constraints.  Otherwise ``rejection``
        is ``None`` and ``plan`` is whatever the strategy returned
        (possibly an infeasible plan under ``on_infeasible="execute"``).
        """
        result = self._plan_single(
            start,
            goal,
            blacklisted,
            prior_plan=prior_plan,
            current_step=current_step,
        )
        if result is None:
            return None, None
        csp = result.metadata.csp
        if (
            csp is not None
            and csp.status == CSPStatus.INFEASIBLE
            and goal.policy.on_infeasible == "fail"
        ):
            logger.warning(
                "Rejecting plan %s: it violates a hard constraint "
                "(CSP infeasible) and GoalPolicy.on_infeasible='fail'",
                result.action_names,
            )
            return None, _infeasible_no_plan_dict(result, goal)
        return result, None

    def _plan_sequential(
        self,
        raw_goal: MultiGoal,
        start_idx: int,
        world_state: dict[str, Any],
        state: GoapState,
        was_replan: bool,
        blacklisted: list[str],
    ) -> tuple[Plan | None, GoalSpec, int, list[int]]:
        """Sequential sub-goal dispatch.

        Starting from ``start_idx`` the planner passes over sub-goals that
        are already satisfied in the sensed world state and plans the first
        one that is not.  A sub-goal that cannot be planned ends the
        invocation (``MultiGoal.on_infeasible="fail"``, the default: the
        docstring promises ``goals[0]`` to completion before ``goals[1]``).
        With ``on_infeasible="skip"`` the planner records the index and
        moves on, so forward progress on later stages survives an
        unreachable earlier one (the per-entity Pac-Man use case); the
        observer then finishes the run as ``partial`` (issue #35).

        Returns ``(plan, effective_goal, landed_idx, skipped)``.  ``plan``
        is ``None`` when the dispatch stopped without a plan; then
        ``effective_goal`` and ``landed_idx`` refer to the sub-goal that
        failed so the caller's ``no_plan`` reporting carries meaningful
        conditions.
        """
        total = len(raw_goal.goals)
        last_goal = raw_goal.goals[start_idx]
        last_idx = start_idx
        skipped: list[int] = []
        for idx in range(start_idx, total):
            candidate = raw_goal.goals[idx]
            last_goal = candidate
            last_idx = idx
            if _is_goal_satisfied(candidate, world_state):
                if not was_replan:
                    logger.info(
                        "MultiGoal (sequential) sub-goal %d/%d already satisfied, "
                        "preflight advance: %s",
                        idx + 1,
                        total,
                        dict(candidate.conditions),
                    )
                continue
            if not was_replan:
                logger.info(
                    "Planning for MultiGoal (sequential, subgoal %d/%d): %s",
                    idx + 1,
                    total,
                    dict(candidate.conditions),
                )
            pkeys = _planning_keys(self.actions, candidate)
            start = PlanningState.from_dict(world_state, keys=pkeys)
            plan, _rejected = self._plan_single_checked(start, candidate, blacklisted)
            if plan is not None:
                return plan, candidate, idx, skipped
            if raw_goal.on_infeasible == "fail":
                return None, candidate, idx, skipped
            skipped.append(idx)
            if idx + 1 < total:
                logger.info(
                    "MultiGoal (sequential, on_infeasible='skip') sub-goal %d/%d "
                    "infeasible, skipping to %d/%d",
                    idx + 1,
                    total,
                    idx + 2,
                    total,
                )
        # Every sub-goal from ``start_idx`` onward was infeasible or
        # already satisfied without an actionable successor.  Return
        # ``None`` so the caller emits ``no_plan`` using the last
        # attempted goal's conditions for diagnostics.
        return None, last_goal, last_idx, skipped

    def _resolve_any_subgoal(
        self,
        raw_goal: MultiGoal,
        world_state: dict[str, Any],
        blacklisted: list[str],
    ) -> tuple[tuple[int, Plan] | None, list[dict[str, Any]]]:
        """Enumerate sub-goals of an ``any``- or ``best_value``-mode
        ``MultiGoal`` and return the best plan.

        ``"any"``: feasibility-first; among feasible plans the lowest
        ``total_cost`` wins.  Uses ``total_cost`` (plain float) to avoid
        the cross-subclass ``TypeError`` ``Score.__lt__`` raises when
        comparing e.g. ``SimpleScore`` (from pure A*) against
        ``HardSoftScore`` (from the CSP pipeline).

        ``"best_value"``: selection key is ``plan.net_value(goal,
        world_state) = goal.value - plan.total_cost``.  Ties on net
        value tie-break to lower cost via ``_is_better_plan``.

        Also returns one ``no_plan_explanation`` payload per sub-goal that
        could not be planned, so an all-infeasible result can explain
        itself to stuck handlers (#64).
        """
        rejections: list[dict[str, Any]] = []
        best_idx = -1
        best_plan: Plan | None = None
        best_goal: GoalSpec | None = None
        is_best_value = raw_goal.mode == "best_value"
        for i, sg in enumerate(raw_goal.goals):
            sub_pkeys = _planning_keys(self.actions, sg)
            sub_start = PlanningState.from_dict(world_state, keys=sub_pkeys)
            candidate, rejected = self._plan_single_checked(sub_start, sg, blacklisted)
            if candidate is None:
                rejections.append(
                    rejected or explain_no_plan(sub_start, sg, self.actions).to_dict()
                )
                continue
            if best_plan is None:
                best_idx, best_plan, best_goal = i, candidate, sg
                continue
            if is_best_value:
                cand_nv = candidate.net_value(sg, world_state)
                best_nv = best_plan.net_value(best_goal, world_state)
                if cand_nv > best_nv or (
                    cand_nv == best_nv and _is_better_plan(candidate, best_plan)
                ):
                    best_idx, best_plan, best_goal = i, candidate, sg
            else:
                if _is_better_plan(candidate, best_plan):
                    best_idx, best_plan, best_goal = i, candidate, sg
        if best_plan is None:
            return None, rejections
        return (best_idx, best_plan), rejections

    def _plan_with_stuck_recovery(
        self, state: GoapState
    ) -> tuple[dict[str, Any], bool]:
        """Wrap ``_plan_core`` with the configured stuck-handler loop.

        When the inner planning attempt produces a ``no_plan`` status
        and a ``StuckHandler`` is configured, the handler is consulted.
        ``REPLAN`` results carrying ``state_updates`` / ``new_goal`` are
        merged into the planning state, the inner attempt is repeated
        (bounded by ``max_stuck_iterations``), and on success the
        recovered ``world_state`` / ``goal`` are folded into the returned
        update dict so they flow downstream through the graph \u2014 otherwise
        the executor would re-encounter the unfixed state and the
        observer would replan, looping forever.
        """
        from langgoap.stuck import StuckHandlingResultCode

        updates, was_replan = self._plan_core(state)
        stuck_handler = _STUCK_OVERRIDE.get() or self._stuck_handler
        if stuck_handler is None:
            return updates, was_replan

        recovered_world_state: dict[str, Any] | None = None
        recovered_goal: Any = None
        for iteration in range(self._max_stuck_iterations):
            if updates.get("status") != "no_plan":
                if recovered_world_state is not None:
                    updates["world_state"] = recovered_world_state
                if recovered_goal is not None:
                    updates["goal"] = recovered_goal
                return updates, was_replan
            reason_dict = updates.get("no_plan_explanation")
            handler_result = stuck_handler.handle_stuck(state, reason_dict)
            if handler_result.code is not StuckHandlingResultCode.REPLAN:
                logger.info(
                    "Stuck handlers exhausted on iteration %d: %s",
                    iteration + 1,
                    handler_result.message,
                )
                return updates, was_replan
            logger.info(
                "Stuck handler %r resolved planning failure on iteration %d: %s",
                handler_result.handler_name,
                iteration + 1,
                handler_result.message,
            )
            new_world_state = dict(state.get("world_state", {}))
            new_world_state.update(dict(handler_result.state_updates))
            recovered_world_state = new_world_state
            recovered_state: GoapState = {
                **state,
                "world_state": new_world_state,
            }
            if handler_result.new_goal is not None:
                recovered_state["goal"] = handler_result.new_goal
                recovered_goal = handler_result.new_goal
            state = recovered_state
            updates, was_replan = self._plan_core(state)
        return updates, was_replan

    async def _aplan_with_stuck_recovery(
        self, state: GoapState
    ) -> tuple[dict[str, Any], bool]:
        """Async entry to the planning core (issue #53).

        When the strategy exposes ``aplan`` or a stuck handler is coroutine
        based, the synchronous core runs in a worker thread with bridges
        that hand those coroutines back to this event loop; otherwise the
        core runs inline exactly as under ``invoke``.
        """
        strategy = self._strategy
        wants_aplan = strategy is not None and hasattr(strategy, "aplan")
        wants_async_stuck = (
            self._stuck_handler is not None and self._stuck_handler.has_async_handlers
        )
        if not (wants_aplan or wants_async_stuck):
            return self._plan_with_stuck_recovery(state)
        loop = asyncio.get_running_loop()
        tokens: list[contextvars.Token[Any]] = []
        if wants_aplan:
            tokens.append(_STRATEGY_OVERRIDE.set(_AsyncStrategyBridge(strategy, loop)))
        if self._stuck_handler is not None:
            tokens.append(
                _STUCK_OVERRIDE.set(_AsyncStuckBridge(self._stuck_handler, loop))
            )
        try:
            # ``to_thread`` copies the current context, so the overrides are
            # visible to the core while the loop stays free to run the
            # bridged coroutines.
            return await asyncio.to_thread(self._plan_with_stuck_recovery, state)
        finally:
            for token in reversed(tokens):
                if token.var is _STRATEGY_OVERRIDE:
                    _STRATEGY_OVERRIDE.reset(token)
                else:
                    _STUCK_OVERRIDE.reset(token)

    def _plan_core(self, state: GoapState) -> tuple[dict[str, Any], bool]:
        """Shared planning logic for sync and async entry points.

        Returns:
            ``(updates, was_replan)`` \u2014 ``updates`` is the GoapState
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

        # Auto-blacklist single-use actions (``can_rerun=False``) that
        # already executed successfully.  Without this guard, an
        # iterative strategy like UtilityStrategy running under
        # ReplanStrategy.EVERY_ACTION would pick the same applicable
        # action every tick because its preconditions remain satisfied,
        # eventually exhausting GoalSpec max_replans.  A* applies
        # can_rerun within a single plan path; this is the cross-replan
        # equivalent.
        executed_once: set[str] = set()
        for record in state.get("execution_history", []):
            if not record.success:
                continue
            for a in self.actions:
                if a.name == record.action_name and not a.can_rerun:
                    executed_once.add(a.name)
                    break
        for name in executed_once:
            if name not in blacklisted:
                blacklisted.append(name)

        # Resolve ``MultiGoal`` to an effective ``GoalSpec``.  The A*
        # planner and CSP pipeline never see ``MultiGoal`` directly \u2014
        # that abstraction lives at the planner/observer dispatch layer.
        extra_updates: dict[str, Any] = {}
        precomputed_result: Plan | None = None
        infeasible_rejection: dict[str, Any] | None = None
        effective_goal: GoalSpec
        start: PlanningState
        if isinstance(raw_goal, MultiGoal):
            if raw_goal.mode == "sequential":
                start_idx = state.get("current_subgoal_index", 0)
                result, effective_goal, landed_idx, skipped = self._plan_sequential(
                    raw_goal, start_idx, world_state, state, was_replan, blacklisted
                )
                if landed_idx != start_idx:
                    extra_updates["current_subgoal_index"] = landed_idx
                if skipped:
                    already = set(state.get("skipped_subgoals", []) or [])
                    extra_updates["skipped_subgoals"] = sorted(already | set(skipped))
                pkeys = _planning_keys(self.actions, effective_goal)
                start = PlanningState.from_dict(world_state, keys=pkeys)
            else:  # "any"
                resolved, rejections = self._resolve_any_subgoal(
                    raw_goal, world_state, blacklisted
                )
                if resolved is None:
                    no_plan_dict = _any_mode_no_plan_dict(
                        raw_goal, rejections, blacklisted
                    )
                    logger.warning(
                        "MultiGoal %r mode: no sub-goal is reachable. %s",
                        raw_goal.mode,
                        no_plan_dict["suggestion"],
                    )
                    return (
                        {
                            "plan": None,
                            "status": "no_plan",
                            "current_step": 0,
                            "replan_count": replan_count,
                            "no_plan_explanation": no_plan_dict,
                        },
                        was_replan,
                    )
                best_idx, best_plan = resolved
                logger.info(
                    "MultiGoal 'any' mode: picked sub-goal %d with cost %.2f",
                    best_idx,
                    best_plan.total_cost,
                )
                effective_goal = raw_goal.goals[best_idx]
                precomputed_result = best_plan
                extra_updates["current_subgoal_index"] = best_idx
                pkeys = _planning_keys(self.actions, effective_goal)
                start = PlanningState.from_dict(world_state, keys=pkeys)
                result = precomputed_result
        else:
            effective_goal = raw_goal
            if not was_replan:
                logger.info("Planning for goal %s", dict(effective_goal.conditions))
            pkeys = _planning_keys(self.actions, effective_goal)
            start = PlanningState.from_dict(world_state, keys=pkeys)
            # Pass prior_plan/current_step on replans so repair-style
            # strategies can patch the in-flight plan; non-repair
            # strategies ignore them.
            result, infeasible_rejection = self._plan_single_checked(
                start,
                effective_goal,
                blacklisted,
                prior_plan=existing_plan if was_replan else None,
                current_step=state.get("current_step", 0) if was_replan else 0,
            )

        # Detect the opt-in blacklist fallback (``GoalPolicy.blacklist_fallback``):
        # if the plan uses a blacklisted action, ``astar_plan`` re-admitted it
        # because the filtered action set could not reach the goal.
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
            if infeasible_rejection is not None:
                no_plan_dict = infeasible_rejection
                logger.warning(
                    "No feasible plan for goal %s \u2014 %s",
                    dict(effective_goal.conditions),
                    no_plan_dict["suggestion"],
                )
            else:
                no_plan_expl = explain_no_plan(start, effective_goal, self.actions)
                no_plan_dict = no_plan_expl.to_dict()
                if blacklisted:
                    no_plan_dict["blacklisted_actions"] = list(blacklisted)
                    if self._blacklist_is_the_cause(start, effective_goal):
                        counts = state.get("action_failure_counts", {}) or {}
                        denied = _denial_reasons(state, blacklisted)
                        failed = [
                            a for a in blacklisted if counts.get(a) and a not in denied
                        ]
                        spent = [a for a in blacklisted if not counts.get(a)]
                        if denied:
                            no_plan_dict["denied_actions"] = list(denied)
                        if spent:
                            no_plan_dict["spent_actions"] = spent
                        no_plan_dict["suggestion"] = _blacklist_suggestion(
                            failed, spent, denied
                        )
                logger.warning(
                    "No plan for goal %s: %s",
                    dict(effective_goal.conditions),
                    no_plan_dict["suggestion"],
                )
            return (
                {
                    "plan": None,
                    "status": "no_plan",
                    "current_step": 0,
                    "replan_count": replan_count,
                    "no_plan_explanation": no_plan_dict,
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
            # Clear the blacklist so the opt-in fallback does not fire on
            # every subsequent replan, but keep ``action_failure_counts``:
            # the history of what failed must survive (issue #15).
            updates["blacklisted_actions"] = []
        # Surface reflections from a ReflexionTracer (or MultiTracer containing
        # one) into GoapState so downstream components (e.g. PromptConditions)
        # can read them without coupling to the tracer directly.
        reflections = getattr(self._tracer, "reflections", None)
        if reflections:
            updates["reflection_context"] = [
                f"[{r.action_name}] {r.reflection} \u2192 {r.suggestion}"
                for r in reflections
            ]
        return updates, was_replan

    def _blacklist_is_the_cause(self, start: PlanningState, goal: GoalSpec) -> bool:
        """True when a plan exists with every action admitted.

        Diagnostic only, run on the (rare) no-plan path so the explanation
        can distinguish "unreachable" from "reachable only through a
        blacklisted action".
        """
        try:
            return astar_plan(start, goal, self.actions) is not None
        except Exception:  # pragma: no cover - diagnostics must not raise
            return False

    def _run_sensors_sync(self, state: GoapState) -> dict[str, Any] | None:
        """Run sensors and return world_state update, or None if no sensors."""
        if not self._sensors:
            return None
        world_state = dict(state.get("world_state", {}))
        sensor_results = run_sensors_sync(self._sensors, world_state)
        for sensor_name, updates in sensor_results:
            self._tracer.on_sensor_complete(sensor_name, updates)
        return world_state

    async def _run_sensors_async(self, state: GoapState) -> dict[str, Any] | None:
        """Run sensors async and return world_state update, or None if no sensors."""
        if not self._sensors:
            return None
        world_state = dict(state.get("world_state", {}))
        sensor_results = await run_sensors_async(self._sensors, world_state)
        for sensor_name, updates in sensor_results:
            await self._tracer.aon_sensor_complete(sensor_name, updates)
        return world_state

    def __call__(self, state: GoapState) -> dict[str, Any]:
        # Channel values are JSON-native: work with typed objects inside
        # the node and serialise the update on the way out.
        raw_keys = set(state)
        state = rehydrate_state(state, self._actions_by_name)
        # Anchor the wall-clock budget on the first planner invocation
        # so MaxWallClockPolicy has a stable start time.  Epoch seconds, not
        # ``time.monotonic()``: the anchor is checkpointed and may be read
        # by another process after a resume (issue #50).  The newly set
        # anchor is also added to the returned update dict so LangGraph
        # persists it for downstream nodes.
        new_wall_clock_anchor: float | None = None
        if state.get("wall_clock_started_at") is None:
            new_wall_clock_anchor = time.time()
            state = {**state, "wall_clock_started_at": new_wall_clock_anchor}

        # Run sensors before planning
        sensor_ws = self._run_sensors_sync(state)
        if sensor_ws is not None:
            state = {**state, "world_state": sensor_ws}

        # Run condition resolvers to fill in UNKNOWN/missing keys.  The
        # resolved values are persisted below so the executed world state
        # matches the planned one (issue #32).
        resolved_ws: dict[str, Any] | None = None
        if self._resolvers:
            ws = dict(state.get("world_state", {}))
            ws = resolve_conditions(self._resolvers, self._resolver_keys, ws)
            resolved_ws = ws
            state = {**state, "world_state": ws}

        goal = state.get("goal")
        world_state = state.get("world_state", {})
        strategy_name = self._strategy_name(goal)
        tracer_goal = self._effective_goal_for_tracer(state)
        self._tracer.on_plan_start(tracer_goal, world_state, strategy_name)
        started = time.perf_counter()
        updates, was_replan = self._plan_with_stuck_recovery(state)
        duration_ms = (time.perf_counter() - started) * 1000
        chosen = getattr(self._strategy, "last_chosen", None)
        if chosen:
            # A StrategyRouter records the strategy it dispatched to; relay
            # it to the tracer's dedicated hook (issue #46).
            self._tracer.on_strategy_chosen(chosen)
        plan = updates.get("plan")
        if plan is None:
            self._tracer.on_plan_failed(updates.get("status", "no_plan"), duration_ms)
        elif was_replan:
            reason = state.get("replan_reason") or "unknown"
            self._tracer.on_replan(reason, plan, duration_ms=duration_ms)
        else:
            self._tracer.on_plan_complete(plan, duration_ms)
        observed_ws = resolved_ws if resolved_ws is not None else sensor_ws
        if observed_ws is not None and "world_state" not in updates:
            updates["world_state"] = observed_ws
        if new_wall_clock_anchor is not None:
            updates.setdefault("wall_clock_started_at", new_wall_clock_anchor)
        _fill_missing_defaults(raw_keys, updates)
        return serialize_updates(updates)

    async def acall(self, state: GoapState) -> dict[str, Any]:
        """Async entry point \u2014 fires async tracer hooks."""
        raw_keys = set(state)
        state = rehydrate_state(state, self._actions_by_name)
        new_wall_clock_anchor: float | None = None
        if state.get("wall_clock_started_at") is None:
            new_wall_clock_anchor = time.time()
            state = {**state, "wall_clock_started_at": new_wall_clock_anchor}

        sensor_ws = await self._run_sensors_async(state)
        if sensor_ws is not None:
            state = {**state, "world_state": sensor_ws}

        resolved_ws: dict[str, Any] | None = None
        if self._resolvers:
            ws = dict(state.get("world_state", {}))
            ws = await aresolve_conditions(self._resolvers, self._resolver_keys, ws)
            resolved_ws = ws
            state = {**state, "world_state": ws}

        goal = state.get("goal")
        world_state = state.get("world_state", {})
        strategy_name = self._strategy_name(goal)
        tracer_goal = self._effective_goal_for_tracer(state)
        await self._tracer.aon_plan_start(tracer_goal, world_state, strategy_name)
        started = time.perf_counter()
        updates, was_replan = await self._aplan_with_stuck_recovery(state)
        duration_ms = (time.perf_counter() - started) * 1000
        chosen = getattr(self._strategy, "last_chosen", None)
        if chosen:
            await self._tracer.aon_strategy_chosen(chosen)
        plan = updates.get("plan")
        if plan is None:
            await self._tracer.aon_plan_failed(
                updates.get("status", "no_plan"), duration_ms
            )
        elif was_replan:
            reason = state.get("replan_reason") or "unknown"
            await self._tracer.aon_replan(reason, plan, duration_ms=duration_ms)
        else:
            await self._tracer.aon_plan_complete(plan, duration_ms)
        observed_ws = resolved_ws if resolved_ws is not None else sensor_ws
        if observed_ws is not None and "world_state" not in updates:
            updates["world_state"] = observed_ws
        if new_wall_clock_anchor is not None:
            updates.setdefault("wall_clock_started_at", new_wall_clock_anchor)
        _fill_missing_defaults(raw_keys, updates)
        return serialize_updates(updates)


def _fill_missing_defaults(raw_keys: set[str], updates: dict[str, Any]) -> None:
    """Write defaults for bookkeeping channels the input never set.

    ``GoapGraph.initial_state`` seeds them; a caller who builds the input
    dict by hand (``{"goal": ..., "world_state": ...}``) does not.  LangGraph
    returns only written channels, so without this the result of such a run
    had no ``blacklisted_actions`` key when nothing had been blacklisted.
    """
    for key, default in state_defaults().items():
        if key not in raw_keys:
            updates.setdefault(key, default)


__all__ = ["GoapPlanner"]
