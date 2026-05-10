"""LangGraph observer node \u2014 routes the graph after each action.

After each action execution the observer decides whether to:
  - end (goal achieved, replan budget exhausted, no plan, or hard fail),
  - re-enter the planner (replan), or
  - continue executing (more actions remain in the current plan).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgoap.termination import TerminationPolicy

from langgraph.graph import END
from langgraph.types import Command

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.nodes._helpers import _planning_keys, logger
from langgoap.graph.state import ActionResult, GoapState
from langgoap.history import (
    ExecutionRecord,
    StoreExecutionHistory,
    compute_goal_hash,
)
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from langgoap.tracing import NullTracer, PlanningTracer, SafeTracerProxy
from langgoap.types import ReplanStrategy


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


def _resolve_active_goal(state: GoapState) -> GoalSpec | None:
    """Resolve the goal that was active at termination (handles ``MultiGoal``)."""
    raw_goal = state.get("goal")
    if raw_goal is None:
        return None
    if isinstance(raw_goal, MultiGoal):
        idx = state.get("current_subgoal_index", 0)
        if idx >= len(raw_goal.goals):
            idx = len(raw_goal.goals) - 1
        return raw_goal.goals[idx]
    return raw_goal


def _outcome_from_status(status: str) -> str | None:
    """Map a terminal status string to an ``ExecutionRecord`` outcome, or ``None``."""
    if status == "goal_achieved":
        return "success"
    if status in {"failed", "no_plan", "error"}:
        return "failed"
    return None


def _is_goal_satisfied(goal: GoalSpec, world_state: dict[str, Any]) -> bool:
    """Return True when every goal condition is present and equal in ``world_state``.

    ``k in world_state`` must be checked first because ``dict.get(k)`` returns
    ``None`` for missing keys, which would incorrectly satisfy a ``None``-valued
    goal condition when the key is simply absent.

    :class:`~langgoap.planner.utility.NirvanaGoal` instances are
    explicitly never satisfied — utility-AI agents drive themselves
    forward without a completion criterion and the loop ends only when
    the planner runs out of applicable actions.
    """
    from langgoap.planner.utility import is_nirvana

    if is_nirvana(goal):
        return False
    return all(
        k in world_state and world_state[k] == v for k, v in goal.conditions.items()
    )


def _route_multi_goal(
    raw_goal: MultiGoal, state: GoapState, world_state: dict[str, Any]
) -> Command[str] | None:
    """Advance ``MultiGoal`` bookkeeping; ``None`` means fall through to single-goal logic.

    In ``sequential`` mode, a satisfied sub-goal advances the index and
    resets per-sub-goal accounting (replan budget, blacklist, failure counts).
    In ``any`` mode, the planner already committed to a specific sub-goal via
    ``current_subgoal_index`` \u2014 satisfying it ends the run; we do not chase
    the remaining alternatives.
    """
    idx = state.get("current_subgoal_index", 0)
    if idx >= len(raw_goal.goals):
        return Command(goto=END, update={"status": "goal_achieved"})
    current_sub = raw_goal.goals[idx]
    if not _is_goal_satisfied(current_sub, world_state):
        return None
    if raw_goal.mode == "sequential" and idx + 1 < len(raw_goal.goals):
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


def _route_replan_limit(goal: GoalSpec, state: GoapState) -> Command[str] | None:
    """Emit a terminal ``failed`` command when the replan budget is exhausted."""
    replan_count: int = state.get("replan_count", 0)
    if goal.policy.max_replans > 0 and replan_count >= goal.policy.max_replans:
        return Command(
            goto=END,
            update={
                "status": "failed",
                "replan_reason": f"max_replans_exceeded ({goal.policy.max_replans})",
            },
        )
    return None


class GoapObserver:
    """LangGraph node that decides the next step via Command routing.

    After each action execution, the observer checks:
    1. Goal achieved \u2192 END
    2. Action failed \u2192 replan
    3. EVERY_ACTION replan strategy \u2192 replan
    4. State deviation from expected \u2192 replan
    5. More actions remain \u2192 continue executing
    6. Plan exhausted but goal not met \u2192 replan

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
            sub-goal completions do **not** emit tracer events \u2014 the
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
        termination_policies: "list[TerminationPolicy] | None" = None,
    ) -> None:
        self._actions = actions or []
        # See ``GoapExecutor.__init__`` for the SafeTracerProxy rationale.
        self._tracer: PlanningTracer = SafeTracerProxy(tracer or NullTracer())
        self._history = history
        self._termination_policies: list[TerminationPolicy] = (
            list(termination_policies) if termination_policies else []
        )

    def _consult_termination(self, state: GoapState) -> Command[str] | None:
        """Return a terminal Command if any policy fires, else ``None``.

        Each policy call is wrapped in try/except so a misbehaving
        user policy never crashes the observer — termination is a
        routing decision, not a place to leak exceptions.
        """
        for policy in self._termination_policies:
            try:
                decision = policy.should_terminate(state)
            except Exception as exc:
                logger.warning(
                    "Termination policy %r raised: %s — treating as no-op",
                    getattr(policy, "name", type(policy).__name__),
                    exc,
                )
                continue
            if decision is not None:
                logger.info(
                    "Early termination by %s: %s",
                    decision.policy_name,
                    decision.reason,
                )
                return Command(
                    goto=END,
                    update={
                        "status": "terminated",
                        "replan_reason": (f"{decision.policy_name}: {decision.reason}"),
                    },
                )
        return None

    def __call__(self, state: GoapState) -> Command[str]:
        cmd = self._route(state)
        self._fire_sync_tracer(state, cmd)
        self._maybe_record_sync(state, cmd)
        return cmd

    async def acall(self, state: GoapState) -> Command[str]:
        """Async variant \u2014 mirrors :meth:`__call__` but fires async hooks."""
        cmd = self._route(state)
        await self._fire_async_tracer(state, cmd)
        await self._maybe_record_async(state, cmd)
        return cmd

    def _fire_sync_tracer(self, state: GoapState, cmd: Command[str]) -> None:
        update = cmd.update or {}
        if cmd.goto == END:
            status = update.get("status")
            if status == "goal_achieved":
                self._tracer.on_goal_achieved(state.get("world_state", {}))
            elif status in {"failed", "no_plan", "error", "terminated"}:
                reason = update.get("replan_reason") or status
                self._tracer.on_plan_failed(reason, 0.0)

    async def _fire_async_tracer(self, state: GoapState, cmd: Command[str]) -> None:
        update = cmd.update or {}
        if cmd.goto == END:
            status = update.get("status")
            if status == "goal_achieved":
                await self._tracer.aon_goal_achieved(state.get("world_state", {}))
            elif status in {"failed", "no_plan", "error", "terminated"}:
                reason = update.get("replan_reason") or status
                await self._tracer.aon_plan_failed(reason, 0.0)

    def _build_record(
        self, state: GoapState, cmd: Command[str]
    ) -> ExecutionRecord | None:
        """Return an ``ExecutionRecord`` for a terminal command, or ``None``.

        Records are only built when the command routes to ``END`` with a
        non-trivial status \u2014 no record is emitted for intermediate
        observer decisions.  For ``MultiGoal`` runs, the record captures
        the sub-goal that was active at termination so downstream
        analytics can attribute success/failure correctly.
        """
        if cmd.goto != END:
            return None
        goal = _resolve_active_goal(state)
        if goal is None:
            return None
        outcome = _outcome_from_status((cmd.update or {}).get("status", "unknown"))
        if outcome is None:
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

        # Termination policies short-circuit everything else: a hit
        # cost ceiling or wall-clock budget must beat the goal-satisfied
        # check below so a run that overshoots its budget while
        # incidentally finishing still surfaces the termination reason.
        terminate_cmd = self._consult_termination(state)
        if terminate_cmd is not None:
            return terminate_cmd

        if raw_goal is None:
            return Command(
                goto=END,
                update={"status": "error", "replan_reason": "no goal specified"},
            )

        # ``MultiGoal`` dispatch: advance sub-goals or short-circuit to
        # the single-goal logic for the remainder.
        if isinstance(raw_goal, MultiGoal):
            multi_cmd = _route_multi_goal(raw_goal, state, world_state)
            if multi_cmd is not None:
                return multi_cmd
            goal = raw_goal.goals[state.get("current_subgoal_index", 0)]
        else:
            goal = raw_goal

        if _is_goal_satisfied(goal, world_state):
            return Command(goto=END, update={"status": "goal_achieved"})

        limit_cmd = _route_replan_limit(goal, state)
        if limit_cmd is not None:
            return limit_cmd

        if status == "no_plan":
            return Command(goto=END, update={"status": "no_plan"})

        never = goal.policy.replan_strategy == ReplanStrategy.NEVER

        if status == "action_failed":
            return self._route_action_failed(state, never)

        if goal.policy.replan_strategy == ReplanStrategy.EVERY_ACTION:
            return Command(
                goto="planner",
                update={"replan_reason": "every_action_replan"},
            )

        if (
            not never
            and plan_obj is not None
            and current_step > 0
            and current_step <= len(plan_obj.expected_states)
            and goal.policy.replan_strategy == ReplanStrategy.ON_DEVIATION
        ):
            deviation_cmd = self._route_deviation_check(
                goal, plan_obj, world_state, current_step
            )
            if deviation_cmd is not None:
                return deviation_cmd

        if plan_obj is not None and current_step < len(plan_obj):
            return Command(goto="executor")

        if never:
            return Command(
                goto=END,
                update={"status": "failed", "replan_reason": "plan_exhausted"},
            )
        return Command(
            goto="planner",
            update={"replan_reason": "plan_exhausted"},
        )

    def _route_action_failed(self, state: GoapState, never: bool) -> Command[str]:
        """Handle the ``action_failed`` status: blacklist + replan, or stop."""
        if never:
            # With NEVER, treat a failed action as a hard stop.
            return Command(
                goto=END,
                update={"status": "failed", "replan_reason": "action_failed"},
            )
        failed_name = _get_last_failed_action(state)
        if failed_name is None:
            return Command(
                goto="planner",
                update={"replan_reason": "action_failed"},
            )
        counts = dict(state.get("action_failure_counts", {}))
        counts[failed_name] = counts.get(failed_name, 0) + 1
        blacklist = list(state.get("blacklisted_actions", []))
        max_retries = _get_max_retries(failed_name, self._actions)
        if counts[failed_name] > max_retries and failed_name not in blacklist:
            blacklist.append(failed_name)
        return Command(
            goto="planner",
            update={
                "replan_reason": "action_failed",
                "action_failure_counts": counts,
                "blacklisted_actions": blacklist,
            },
        )

    def _route_deviation_check(
        self,
        goal: GoalSpec,
        plan_obj: Plan,
        world_state: dict[str, Any],
        current_step: int,
    ) -> Command[str] | None:
        """Return a replan command when planning-relevant state has deviated."""
        # Compare only planning-relevant keys so that changes to execution
        # context (documents, generations, etc.) don't trigger replanning.
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
        return None


__all__ = ["GoapObserver"]
