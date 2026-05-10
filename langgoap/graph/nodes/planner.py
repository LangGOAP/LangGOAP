"""LangGraph planner node \u2014 invokes the configured planning strategy."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langgoap.actions import ActionSpec
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
from langgoap.graph.state import GoapState
from langgoap.planner.astar import plan as astar_plan
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

        self.actions = actions
        self._strategy = strategy
        # See ``GoapExecutor.__init__`` for the SafeTracerProxy rationale.
        self._tracer: PlanningTracer = SafeTracerProxy(tracer or NullTracer())
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
        if self._strategy is not None:
            # Forward repair kwargs only when meaningful so that legacy
            # user strategies whose ``plan`` signature predates the
            # widened Protocol (no ``prior_plan``/``current_step`` kwargs)
            # keep working on initial plans.  Strategies that opt into the
            # new Protocol shape will accept them either way.
            extras: dict[str, Any] = {}
            if prior_plan is not None:
                extras["prior_plan"] = prior_plan
                extras["current_step"] = current_step
            return self._strategy.plan(
                start,
                goal,
                self.actions,
                blacklisted_actions=blacklisted,
                **extras,
            )
        if goal.constraints or goal.objectives is not None or goal.metrics:
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

    def _resolve_sequential_subgoal(
        self, raw_goal: MultiGoal, state: GoapState, was_replan: bool
    ) -> GoalSpec:
        """Pick the current sub-goal from a sequential ``MultiGoal``."""
        idx = state.get("current_subgoal_index", 0)
        effective_goal = raw_goal.goals[idx]
        if not was_replan:
            logger.info(
                "Planning for MultiGoal (sequential, subgoal %d/%d): %s",
                idx + 1,
                len(raw_goal.goals),
                dict(effective_goal.conditions),
            )
        return effective_goal

    def _plan_sequential_best_effort(
        self,
        raw_goal: MultiGoal,
        start_idx: int,
        world_state: dict[str, Any],
        state: GoapState,
        was_replan: bool,
        blacklisted: list[str],
    ) -> tuple[Plan | None, GoalSpec, int]:
        """Best-effort sequential sub-goal dispatch.

        Starting from ``start_idx`` the planner tries each sub-goal in
        order: the first one that (a) is already satisfied in the
        sensed world state or (b) yields an A* plan becomes the
        effective goal.  Infeasible sub-goals along the way are
        skipped rather than failing the whole invocation.  The
        previous strict-sequential behaviour would get stuck on the
        first unreachable sub-goal (e.g. a per-ghost escape whose
        single flee step is not enough) and collapse Pac-Man to
        ``STOP``; the best-effort variant preserves forward progress
        on later sub-goals even when an earlier one cannot be
        established this tick.

        Returns ``(plan, effective_goal, landed_idx)``.  ``plan`` may
        be ``None`` iff every sub-goal from ``start_idx`` onward
        failed; in that case ``effective_goal`` and ``landed_idx``
        refer to the final attempt so the caller's ``no_plan``
        reporting still carries meaningful conditions.
        """
        total = len(raw_goal.goals)
        last_goal = raw_goal.goals[start_idx]
        last_idx = start_idx
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
            plan = self._plan_single(start, candidate, blacklisted)
            if plan is not None:
                return plan, candidate, idx
            if idx + 1 < total:
                logger.info(
                    "MultiGoal (sequential) sub-goal %d/%d infeasible, "
                    "best-effort fallback to %d/%d",
                    idx + 1,
                    total,
                    idx + 2,
                    total,
                )
        # Every sub-goal from ``start_idx`` onward was infeasible or
        # already satisfied without an actionable successor.  Return
        # ``None`` so the caller emits ``no_plan`` using the last
        # attempted goal's conditions for diagnostics.
        return None, last_goal, last_idx

    def _resolve_any_subgoal(
        self,
        raw_goal: MultiGoal,
        world_state: dict[str, Any],
        blacklisted: list[str],
    ) -> tuple[int, Plan] | None:
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
        """
        best_idx = -1
        best_plan: Plan | None = None
        best_goal: GoalSpec | None = None
        is_best_value = raw_goal.mode == "best_value"
        for i, sg in enumerate(raw_goal.goals):
            sub_pkeys = _planning_keys(self.actions, sg)
            sub_start = PlanningState.from_dict(world_state, keys=sub_pkeys)
            candidate = self._plan_single(sub_start, sg, blacklisted)
            if candidate is None:
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
            return None
        return best_idx, best_plan

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
        if self._stuck_handler is None:
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
            handler_result = self._stuck_handler.handle_stuck(state, reason_dict)
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
        effective_goal: GoalSpec
        start: PlanningState
        if isinstance(raw_goal, MultiGoal):
            if raw_goal.mode == "sequential":
                start_idx = state.get("current_subgoal_index", 0)
                result, effective_goal, landed_idx = self._plan_sequential_best_effort(
                    raw_goal, start_idx, world_state, state, was_replan, blacklisted
                )
                if landed_idx != start_idx:
                    extra_updates["current_subgoal_index"] = landed_idx
                pkeys = _planning_keys(self.actions, effective_goal)
                start = PlanningState.from_dict(world_state, keys=pkeys)
            else:  # "any"
                resolved = self._resolve_any_subgoal(raw_goal, world_state, blacklisted)
                if resolved is None:
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
            result = self._plan_single(
                start,
                effective_goal,
                blacklisted,
                prior_plan=existing_plan if was_replan else None,
                current_step=state.get("current_step", 0) if was_replan else 0,
            )

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
                "A* found no plan for goal %s \u2014 %s",
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
            updates["blacklisted_actions"] = []
            updates["action_failure_counts"] = {}
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
        # Anchor the wall-clock budget on the first planner invocation
        # so MaxWallClockPolicy has a stable start time.  The newly set
        # anchor is also added to the returned update dict so LangGraph
        # persists it for downstream nodes.
        new_wall_clock_anchor: float | None = None
        if state.get("wall_clock_started_at") is None:
            new_wall_clock_anchor = time.monotonic()
            state = {**state, "wall_clock_started_at": new_wall_clock_anchor}

        # Run sensors before planning
        sensor_ws = self._run_sensors_sync(state)
        if sensor_ws is not None:
            state = {**state, "world_state": sensor_ws}

        # Run condition resolvers to fill in UNKNOWN/missing keys
        if self._resolvers:
            ws = dict(state.get("world_state", {}))
            resolver_keys: list[str] = []
            for a in self.actions:
                resolver_keys.extend(a.preconditions.keys())
                resolver_keys.extend(a.effect_key_set())
            ws = resolve_conditions(self._resolvers, list(set(resolver_keys)), ws)
            state = {**state, "world_state": ws}

        goal = state.get("goal")
        world_state = state.get("world_state", {})
        strategy_name = self._strategy_name(goal)
        tracer_goal = self._effective_goal_for_tracer(state)
        self._tracer.on_plan_start(tracer_goal, world_state, strategy_name)
        started = time.perf_counter()
        updates, was_replan = self._plan_with_stuck_recovery(state)
        duration_ms = (time.perf_counter() - started) * 1000
        plan = updates.get("plan")
        if plan is None:
            self._tracer.on_plan_failed(updates.get("status", "no_plan"), duration_ms)
        elif was_replan:
            reason = state.get("replan_reason") or "unknown"
            self._tracer.on_replan(reason, plan)
        else:
            self._tracer.on_plan_complete(plan, duration_ms)
        if sensor_ws is not None and "world_state" not in updates:
            updates["world_state"] = sensor_ws
        if new_wall_clock_anchor is not None:
            updates.setdefault("wall_clock_started_at", new_wall_clock_anchor)
        return updates

    async def acall(self, state: GoapState) -> dict[str, Any]:
        """Async entry point \u2014 fires async tracer hooks."""
        new_wall_clock_anchor: float | None = None
        if state.get("wall_clock_started_at") is None:
            new_wall_clock_anchor = time.monotonic()
            state = {**state, "wall_clock_started_at": new_wall_clock_anchor}

        sensor_ws = await self._run_sensors_async(state)
        if sensor_ws is not None:
            state = {**state, "world_state": sensor_ws}

        if self._resolvers:
            ws = dict(state.get("world_state", {}))
            resolver_keys: list[str] = []
            for a in self.actions:
                resolver_keys.extend(a.preconditions.keys())
                resolver_keys.extend(a.effect_key_set())
            ws = await aresolve_conditions(
                self._resolvers, list(set(resolver_keys)), ws
            )
            state = {**state, "world_state": ws}

        goal = state.get("goal")
        world_state = state.get("world_state", {})
        strategy_name = self._strategy_name(goal)
        tracer_goal = self._effective_goal_for_tracer(state)
        await self._tracer.aon_plan_start(tracer_goal, world_state, strategy_name)
        started = time.perf_counter()
        updates, was_replan = self._plan_with_stuck_recovery(state)
        duration_ms = (time.perf_counter() - started) * 1000
        plan = updates.get("plan")
        if plan is None:
            await self._tracer.aon_plan_failed(
                updates.get("status", "no_plan"), duration_ms
            )
        elif was_replan:
            reason = state.get("replan_reason") or "unknown"
            await self._tracer.aon_replan(reason, plan)
        else:
            await self._tracer.aon_plan_complete(plan, duration_ms)
        if sensor_ws is not None and "world_state" not in updates:
            updates["world_state"] = sensor_ws
        if new_wall_clock_anchor is not None:
            updates.setdefault("wall_clock_started_at", new_wall_clock_anchor)
        return updates


__all__ = ["GoapPlanner"]
