"""LangGraph nodes for GOAP planning, execution, and observation.

These nodes form the core GOAP loop:
  planner → executor → observer → (executor | planner | END)
"""

from __future__ import annotations

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
        result = astar_plan(start, goal, self.actions)

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
        return {
            "plan": result,
            "status": "executing",
            "current_step": 0,
            "replan_count": replan_count,
        }


class GoapExecutor:
    """LangGraph node that executes the current action in the plan.

    Runs ``plan.actions[current_step]``, updates world_state with
    the action's effects, and appends to execution_history.
    """

    def __call__(self, state: GoapState) -> dict[str, Any]:
        status: str = state.get("status", "")
        plan_obj: Plan | None = state.get("plan")
        current_step: int = state.get("current_step", 0)
        world_state: dict[str, Any] = dict(state.get("world_state", {}))

        # Short-circuit: planner found no plan, pass through to observer
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
        state_before = dict(world_state)

        logger.info(
            "Executing action %r (step %d/%d)",
            action.name,
            current_step + 1,
            len(plan_obj),
        )

        try:
            if action.execute is not None:
                result = action.execute(world_state)
                if isinstance(result, dict):
                    world_state.update(result)
                else:
                    # If execute doesn't return a dict, apply declared effects
                    world_state.update(action.effects)
            else:
                # No execute function: apply declared effects
                world_state.update(action.effects)

            state_after = dict(world_state)

            # Runtime postcondition check — only when a custom validator
            # was explicitly provided (the default None means "trust the
            # declared effects").
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

            logger.debug(
                "Action %r succeeded, world_state=%s", action.name, world_state
            )
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
        except Exception as e:
            logger.warning("Action %r failed: %s", action.name, e)
            return {
                "status": "action_failed",
                "execution_history": [
                    ActionResult(
                        action_name=action.name,
                        success=False,
                        state_before=state_before,
                        state_after=dict(world_state),
                        error=str(e),
                    )
                ],
            }


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

        # Action failed → replan (unless NEVER strategy forbids it)
        if status == "action_failed":
            if never:
                # With NEVER, treat a failed action as a hard stop.
                return Command(
                    goto=END,
                    update={"status": "failed", "replan_reason": "action_failed"},
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
