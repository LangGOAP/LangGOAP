"""LangGraph nodes for GOAP planning, execution, and observation.

These nodes form the core GOAP loop:
  planner → executor → observer → (executor | planner | END)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from langgraph.graph import END
from langgraph.types import Command

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.state import ActionResult, GoapState
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from langgoap.types import ReplanStrategy

logger = logging.getLogger(__name__)


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
    """LangGraph node that invokes the A* planner.

    Reads world_state and goal from GoapState, runs A* search,
    and writes the resulting plan back.
    """

    def __init__(self, actions: list[ActionSpec]) -> None:
        self.actions = actions

    def __call__(self, state: GoapState) -> dict[str, Any]:
        world_state = state.get("world_state", {})
        goal = state.get("goal")
        if goal is None:
            logger.error("GoapPlanner called with no goal in state")
            return {
                "plan": None,
                "status": "error",
                "current_step": 0,
            }

        # Only increment replan_count when a prior plan already existed.
        # The initial planning call is not a replan.
        existing_plan = state.get("plan")
        replan_count = state.get("replan_count", 0)
        if existing_plan is not None:
            replan_count += 1
            logger.info(
                "Replanning (replan #%d), reason=%s, world_state=%s",
                replan_count,
                state.get("replan_reason"),
                world_state,
            )
        else:
            logger.info("Planning for goal %s", dict(goal.conditions))

        pkeys = _planning_keys(self.actions, goal)
        start = PlanningState.from_dict(world_state, keys=pkeys)
        blacklisted = list(state.get("blacklisted_actions", []))

        # First try with blacklist; if that fails, try without (Embabel fallback).
        result = astar_plan(start, goal, self.actions, blacklisted_actions=blacklisted)

        # Detect Embabel fallback: if the plan uses a blacklisted action,
        # the fallback fired — clear the blacklist to avoid repeated fallbacks.
        used_blacklisted = False
        if result is not None and blacklisted:
            plan_names = set(result.action_names)
            if plan_names & set(blacklisted):
                used_blacklisted = True
                logger.info(
                    "Embabel fallback: blacklist cleared, plan uses %s",
                    plan_names & set(blacklisted),
                )

        if result is None:
            logger.warning("A* found no plan for goal %s", dict(goal.conditions))
            return {
                "plan": None,
                "status": "no_plan",
                "current_step": 0,
                "replan_count": replan_count,
            }

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
        }
        if used_blacklisted:
            # Clear blacklist and failure counts so the retried action
            # gets a fresh start instead of being immediately re-blacklisted.
            updates["blacklisted_actions"] = []
            updates["action_failure_counts"] = {}
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
    """Build the success return dict for the executor."""
    state_after = dict(world_state)
    if action.effect_validator is not None and not action.validate_effects(
        state_before, state_after
    ):
        logger.warning("Action %r effect validation failed", action.name)
        return {
            "status": "action_failed",
            "world_state": world_state,
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
    """

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
            return _build_success(action, current_step, state_before, world_state)
        except Exception as e:
            return _build_failure(action, e, state_before, world_state)

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

        state_before = dict(world_state)
        try:
            raw = await async_execute_action(action, world_state)
            _apply_result(raw, action, world_state)
            return _build_success(action, current_step, state_before, world_state)
        except Exception as e:
            return _build_failure(action, e, state_before, world_state)


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


# Back-compat alias; prefer async_execute_action in new code.
_async_execute_action = async_execute_action


def _get_last_failed_action(state: GoapState) -> str | None:
    """Extract the action name from the last failed ActionResult in history."""
    history: list[Any] = state.get("execution_history", [])
    for result in reversed(history):
        if not result.success:
            return result.action_name  # type: ignore[no-any-return]
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
    """

    def __init__(self, actions: list[ActionSpec] | None = None) -> None:
        self._actions = actions or []

    def __call__(self, state: GoapState) -> Command[str]:
        goal: GoalSpec | None = state.get("goal")
        world_state: dict[str, Any] = state.get("world_state", {})
        plan_obj: Plan | None = state.get("plan")
        current_step: int = state.get("current_step", 0)
        status: str = state.get("status", "")

        if goal is None:
            return Command(
                goto=END,
                update={"status": "error", "replan_reason": "no goal specified"},
            )

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
