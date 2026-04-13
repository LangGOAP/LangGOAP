"""Tests for GoapPlanner, GoapExecutor, and GoapObserver nodes."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.nodes import (
    GoapExecutor,
    GoapObserver,
    GoapPlanner,
    ParallelGoapExecutor,
    _find_parallel_group,
    _is_approved,
    _is_better_plan,
)
from langgoap.graph.state import ActionResult, GoapState
from langgoap.guards import FunctionalGuard, GuardResult, GuardSeverity
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import BendableScore, HardSoftScore, SimpleScore
from langgoap.state import PlanningState
from langgoap.types import ReplanStrategy
from tests.conftest import make_action as _action
from tests.conftest import make_plan as _make_plan

# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------


class TestGoapPlanner:
    def test_produces_plan(self) -> None:
        actions = [
            _action("step1", eff={"a": True}),
            _action("step2", pre={"a": True}, eff={"b": True}),
        ]
        planner = GoapPlanner(actions)

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"b": True}),
        }
        result = planner(state)

        assert result["status"] == "executing"
        assert result["plan"] is not None
        assert result["current_step"] == 0
        assert len(result["plan"]) == 2

    def test_handles_no_plan(self) -> None:
        actions = [_action("useless", eff={"x": True})]
        planner = GoapPlanner(actions)

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"unreachable": True}),
        }
        result = planner(state)

        assert result["status"] == "no_plan"
        assert result["plan"] is None

    def test_handles_no_goal(self) -> None:
        planner = GoapPlanner([])
        state: GoapState = {"world_state": {}}
        result = planner(state)

        assert result["status"] == "error"
        assert result["plan"] is None

    def test_initial_plan_does_not_increment_replan_count(self) -> None:
        """replan_count stays 0 on the very first planning call (no prior plan)."""
        actions = [_action("act", eff={"done": True})]
        planner = GoapPlanner(actions)
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"done": True}),
        }
        result = planner(state)
        assert result["replan_count"] == 0

    def test_increments_replan_count(self) -> None:
        """replan_count increments when a prior plan exists in state."""
        actions = [_action("act", eff={"done": True})]
        planner = GoapPlanner(actions)
        prior_plan = _make_plan(_action("act", eff={"done": True}))
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"done": True}),
            "plan": prior_plan,  # existing plan → genuine replan
            "replan_count": 2,
        }
        result = planner(state)
        assert result["replan_count"] == 3


# ---------------------------------------------------------------------------
# Executor node
# ---------------------------------------------------------------------------


class TestGoapExecutor:
    def test_executes_action(self) -> None:
        action = _action("set_a", eff={"a": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = executor(state)

        assert result["world_state"]["a"] is True
        assert result["current_step"] == 1
        assert len(result["execution_history"]) == 1
        assert result["execution_history"][0].success is True

    def test_executes_custom_function(self) -> None:
        def custom_fn(ws: dict[str, Any]) -> dict[str, Any]:
            return {"computed": ws.get("input", 0) * 2}

        action = _action("compute", eff={"computed": 0}, execute=custom_fn)
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {"input": 5},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = executor(state)

        assert result["world_state"]["computed"] == 10

    def test_handles_action_failure(self) -> None:
        def failing_fn(ws: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

        action = _action("fail", eff={"x": True}, execute=failing_fn)
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = executor(state)

        assert result["status"] == "action_failed"
        assert result["execution_history"][0].success is False
        assert "boom" in (result["execution_history"][0].error or "")

    def test_handles_no_plan(self) -> None:
        executor = GoapExecutor()
        state: GoapState = {"world_state": {}, "plan": None, "current_step": 0}
        result = executor(state)

        assert result["status"] == "error"
        assert result["execution_history"][0].success is False

    def test_short_circuits_on_no_plan_status(self) -> None:
        """Executor passes through when planner found no plan."""
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {},
            "plan": None,
            "current_step": 0,
            "status": "no_plan",
        }
        result = executor(state)
        assert result == {}


# ---------------------------------------------------------------------------
# Observer node
# ---------------------------------------------------------------------------


class TestGoapObserver:
    def test_goal_achieved(self) -> None:
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {"done": True},
            "goal": GoalSpec(conditions={"done": True}),
            "plan": _make_plan(_action("act", eff={"done": True})),
            "current_step": 1,
            "status": "executing",
        }
        cmd = observer(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "goal_achieved"

    def test_continue_executing(self) -> None:
        a1 = _action("step1", eff={"a": True})
        a2 = _action("step2", pre={"a": True}, eff={"b": True})
        plan_obj = _make_plan(a1, a2)

        observer = GoapObserver()
        state: GoapState = {
            "world_state": {"a": True},
            "goal": GoalSpec(conditions={"b": True}),
            "plan": plan_obj,
            "current_step": 1,
            "status": "executing",
        }
        cmd = observer(state)

        assert cmd.goto == "executor"

    def test_deviation_replan(self) -> None:
        a1 = _action("step1", eff={"a": True, "b": True})
        a2 = _action("step2", pre={"a": True, "b": True}, eff={"c": True})
        plan_obj = Plan(
            actions=(a1, a2),
            expected_states=(
                PlanningState.from_dict({"a": True, "b": True}),
                PlanningState.from_dict({"a": True, "b": True, "c": True}),
            ),
            total_cost=2.0,
        )

        observer = GoapObserver(actions=[a1, a2])
        state: GoapState = {
            "world_state": {"a": True, "b": False},  # b deviated!
            "goal": GoalSpec(
                conditions={"c": True},
                replan_strategy=ReplanStrategy.ON_DEVIATION,
            ),
            "plan": plan_obj,
            "current_step": 1,
            "status": "executing",
        }
        cmd = observer(state)

        assert cmd.goto == "planner"
        assert cmd.update["replan_reason"] == "state_deviation"

    def test_every_action_replan(self) -> None:
        a1 = _action("step1", eff={"a": True})
        plan_obj = _make_plan(a1)

        observer = GoapObserver()
        state: GoapState = {
            "world_state": {"a": True},
            "goal": GoalSpec(
                conditions={"b": True},
                replan_strategy=ReplanStrategy.EVERY_ACTION,
            ),
            "plan": plan_obj,
            "current_step": 1,
            "status": "executing",
        }
        cmd = observer(state)

        assert cmd.goto == "planner"
        assert cmd.update["replan_reason"] == "every_action_replan"

    def test_plan_exhausted_replans(self) -> None:
        a1 = _action("step1", eff={"a": True})
        plan_obj = _make_plan(a1)

        observer = GoapObserver()
        state: GoapState = {
            "world_state": {"a": True},
            "goal": GoalSpec(conditions={"b": True}),  # Not achieved
            "plan": plan_obj,
            "current_step": 1,  # Past end of 1-action plan
            "status": "executing",
        }
        cmd = observer(state)

        assert cmd.goto == "planner"
        assert cmd.update["replan_reason"] == "plan_exhausted"

    def test_action_failed_replans(self) -> None:
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"done": True}),
            "plan": _make_plan(_action("act", eff={"done": True})),
            "current_step": 0,
            "status": "action_failed",
        }
        cmd = observer(state)

        assert cmd.goto == "planner"
        assert cmd.update["replan_reason"] == "action_failed"

    def test_no_plan_routes_to_end(self) -> None:
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"done": True}),
            "plan": None,
            "current_step": 0,
            "status": "no_plan",
        }
        cmd = observer(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "no_plan"

    def test_no_goal_routes_to_end(self) -> None:
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {},
            "status": "executing",
        }
        cmd = observer(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "error"

    def test_never_replan_skips_deviation_check(self) -> None:
        """With NEVER strategy, deviation does NOT trigger replan."""
        a1 = _action("step1", eff={"a": True, "b": True})
        a2 = _action("step2", pre={"a": True}, eff={"c": True})
        plan_obj = Plan(
            actions=(a1, a2),
            expected_states=(
                PlanningState.from_dict({"a": True, "b": True}),
                PlanningState.from_dict({"a": True, "b": True, "c": True}),
            ),
            total_cost=2.0,
        )

        observer = GoapObserver()
        state: GoapState = {
            "world_state": {"a": True, "b": False},  # deviated!
            "goal": GoalSpec(
                conditions={"c": True},
                replan_strategy=ReplanStrategy.NEVER,
            ),
            "plan": plan_obj,
            "current_step": 1,
            "status": "executing",
        }
        cmd = observer(state)

        # With NEVER, should continue executing despite deviation
        assert cmd.goto == "executor"

    def test_never_strategy_stops_on_action_failure(self) -> None:
        """With NEVER strategy, action failure routes to END (not planner)."""
        a1 = _action("step1", eff={"a": True})
        plan_obj = _make_plan(a1)
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(
                conditions={"a": True},
                replan_strategy=ReplanStrategy.NEVER,
            ),
            "plan": plan_obj,
            "current_step": 0,
            "status": "action_failed",
        }
        cmd = observer(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "failed"
        assert cmd.update["replan_reason"] == "action_failed"

    def test_max_replans_stops_loop(self) -> None:
        """When replan_count reaches max_replans, observer gives up."""
        a1 = _action("step1", eff={"a": True})
        plan_obj = _make_plan(a1)
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"a": True}, max_replans=3),
            "plan": plan_obj,
            "current_step": 0,
            "status": "action_failed",
            "replan_count": 3,  # already at limit
        }
        cmd = observer(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "failed"
        assert "max_replans_exceeded" in cmd.update["replan_reason"]

    def test_max_replans_zero_disables_limit(self) -> None:
        """Setting max_replans=0 disables the guard."""
        a1 = _action("step1", eff={"a": True})
        plan_obj = _make_plan(a1)
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"a": True}, max_replans=0),
            "plan": plan_obj,
            "current_step": 0,
            "status": "action_failed",
            "replan_count": 999,  # high count — should NOT stop
        }
        cmd = observer(state)

        # Should still try to replan, not stop
        assert cmd.goto == "planner"


# ---------------------------------------------------------------------------
# Executor: validate_effects
# ---------------------------------------------------------------------------


class TestGoapExecutorValidation:
    def test_effect_validation_called_when_validator_provided(self) -> None:
        """Executor calls validate_effects when effect_validator is set."""

        def always_fail(pre: dict[str, Any], post: dict[str, Any]) -> bool:
            return False

        action = ActionSpec(
            name="validated",
            effects={"done": True},
            execute=lambda ws: {"done": True},
            effect_validator=always_fail,
        )
        plan_obj = _make_plan(action)
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = executor(state)

        assert result["status"] == "action_failed"
        assert result["execution_history"][0].success is False
        assert "validation failed" in (result["execution_history"][0].error or "")

    def test_validator_rejection_rolls_back_world_state(self) -> None:
        """Failed validation must restore the world state to its pre-action snapshot.

        The validator's purpose is to detect actions that did not actually
        accomplish what they claimed.  If the rejected effects were left
        in world state, the planner's goal predicate could be satisfied
        by an unverified action and the blacklist + replan dance the
        validator was designed to trigger would be short-circuited.
        """

        def always_fail(pre: dict[str, Any], post: dict[str, Any]) -> bool:
            return False

        action = ActionSpec(
            name="validated",
            effects={"done": True},
            execute=lambda ws: {"done": True, "side_effect": "leaked"},
            effect_validator=always_fail,
        )
        plan_obj = _make_plan(action)
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {"prior": "value"},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = executor(state)

        # World state must NOT contain the rejected effects.
        assert result["world_state"] == {"prior": "value"}
        assert "done" not in result["world_state"]
        assert "side_effect" not in result["world_state"]
        # The diagnostic record still shows what the action *tried* to do.
        history_entry = result["execution_history"][0]
        assert history_entry.state_after.get("done") is True
        assert history_entry.state_after.get("side_effect") == "leaked"
        assert history_entry.state_before == {"prior": "value"}

    def test_no_validation_when_no_validator(self) -> None:
        """Without effect_validator, executor skips validation entirely."""

        def custom_fn(ws: dict[str, Any]) -> dict[str, Any]:
            return {"computed": ws.get("input", 0) * 2}

        # effects say computed=0 but execute returns computed=10
        # This mismatch is fine because no effect_validator is set.
        action = _action("compute", eff={"computed": 0}, execute=custom_fn)
        plan_obj = _make_plan(action)
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {"input": 5},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = executor(state)

        assert result["current_step"] == 1  # success, not action_failed
        assert result["world_state"]["computed"] == 10

    def test_validation_passes_with_correct_effects(self) -> None:
        """When validator passes, execution proceeds normally."""

        def check_done(pre: dict[str, Any], post: dict[str, Any]) -> bool:
            return post.get("done") is True

        action = ActionSpec(
            name="good",
            effects={"done": True},
            execute=lambda ws: {"done": True},
            effect_validator=check_done,
        )
        plan_obj = _make_plan(action)
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = executor(state)

        assert result["current_step"] == 1
        assert result["execution_history"][0].success is True


# ---------------------------------------------------------------------------
# Planner: rich world_state
# ---------------------------------------------------------------------------


class TestGoapPlannerRichState:
    def test_planner_handles_unhashable_world_state(self) -> None:
        """Planner works when world_state contains lists and dicts."""
        actions = [_action("process", pre={"has_data": True}, eff={"done": True})]
        planner = GoapPlanner(actions)

        state: GoapState = {
            "world_state": {
                "has_data": True,
                "documents": [{"id": 1}, {"id": 2}],
                "question": "what is GOAP?",
            },
            "goal": GoalSpec(conditions={"done": True}),
        }
        result = planner(state)

        assert result["status"] == "executing"
        assert result["plan"] is not None
        assert len(result["plan"]) == 1


# ---------------------------------------------------------------------------
# Observer: rich world_state
# ---------------------------------------------------------------------------


class TestGoapObserverRichState:
    def test_goal_not_satisfied_when_key_absent_and_value_is_none(self) -> None:
        """Regression: world_state.get(k)==None must NOT satisfy goal when key
        is absent. dict.get returns None for missing keys, which would
        incorrectly match a goal condition whose value happens to be None."""
        observer = GoapObserver()
        a1 = _action("act", eff={"result": None})
        state: GoapState = {
            # "result" is absent — goal should NOT be satisfied
            "world_state": {},
            "goal": GoalSpec(conditions={"result": None}),
            "plan": _make_plan(a1),
            "current_step": 0,
            "status": "executing",
        }
        cmd = observer(state)

        # Goal not satisfied (key absent) → should continue to executor
        assert cmd.goto == "executor"
        # Must NOT terminate as goal_achieved
        update_status = (cmd.update or {}).get("status")
        assert update_status != "goal_achieved"

    def test_goal_check_with_unhashable_world_state(self) -> None:
        """Observer checks goal directly on dict, not via PlanningState."""
        observer = GoapObserver()
        state: GoapState = {
            "world_state": {
                "done": True,
                "documents": [1, 2, 3],  # unhashable
            },
            "goal": GoalSpec(conditions={"done": True}),
            "plan": _make_plan(_action("act", eff={"done": True})),
            "current_step": 1,
            "status": "executing",
        }
        cmd = observer(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "goal_achieved"

    def test_deviation_ignores_non_planning_keys(self) -> None:
        """Deviation detection only compares planning-relevant keys."""
        a1 = _action("step1", eff={"a": True})
        a2 = _action("step2", pre={"a": True}, eff={"b": True})
        plan_obj = Plan(
            actions=(a1, a2),
            expected_states=(
                PlanningState.from_dict({"a": True}),
                PlanningState.from_dict({"a": True, "b": True}),
            ),
            total_cost=2.0,
        )

        # Observer with actions → knows planning keys = {"a", "b"}
        observer = GoapObserver(actions=[a1, a2])
        state: GoapState = {
            # Planning key "a" matches expected. Extra key "extra" is
            # non-planning and should not trigger deviation.
            "world_state": {"a": True, "extra": "irrelevant_data"},
            "goal": GoalSpec(
                conditions={"b": True},
                replan_strategy=ReplanStrategy.ON_DEVIATION,
            ),
            "plan": plan_obj,
            "current_step": 1,  # after step1, before step2
            "status": "executing",
        }
        cmd = observer(state)

        # Should continue executing, NOT trigger deviation replan
        assert cmd.goto == "executor"


# ---------------------------------------------------------------------------
# _is_approved helper
# ---------------------------------------------------------------------------


class TestIsApproved:
    """Unit tests for the ``_is_approved`` resume-value interpreter."""

    def test_none_is_approved(self) -> None:
        assert _is_approved(None) is True

    def test_true_is_approved(self) -> None:
        assert _is_approved(True) is True

    def test_false_is_denied(self) -> None:
        assert _is_approved(False) is False

    def test_dict_approved_true(self) -> None:
        assert _is_approved({"approved": True}) is True

    def test_dict_approved_false(self) -> None:
        assert _is_approved({"approved": False}) is False

    def test_dict_missing_approved_defaults_true(self) -> None:
        """A dict without 'approved' key defaults to approval."""
        assert _is_approved({"reason": "let's go"}) is True

    def test_truthy_string_is_approved(self) -> None:
        assert _is_approved("yes") is True

    def test_empty_string_is_denied(self) -> None:
        assert _is_approved("") is False

    def test_zero_is_denied(self) -> None:
        assert _is_approved(0) is False

    def test_positive_int_is_approved(self) -> None:
        assert _is_approved(1) is True


# ---------------------------------------------------------------------------
# GuardRails integration in GoapExecutor
# ---------------------------------------------------------------------------


def _passing_guard(name: str = "pass") -> FunctionalGuard:
    return FunctionalGuard(name, lambda a, ws: GuardResult(passed=True, message="ok"))


def _warn_guard(name: str = "warn") -> FunctionalGuard:
    return FunctionalGuard(
        name,
        lambda a, ws: GuardResult(
            passed=False, message="warn msg", severity=GuardSeverity.WARN
        ),
    )


def _block_guard(name: str = "block") -> FunctionalGuard:
    return FunctionalGuard(
        name,
        lambda a, ws: GuardResult(
            passed=False, message="blocked!", severity=GuardSeverity.BLOCK
        ),
    )


class TestGoapExecutorGuards:
    def test_passing_guard_allows_execution(self) -> None:
        action = _action("set_x", eff={"x": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[_passing_guard()])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert result["world_state"]["x"] is True
        assert result["execution_history"][0].success is True

    def test_warn_guard_allows_execution(self) -> None:
        """WARN-severity guard failure is logged but execution continues."""
        action = _action("set_x", eff={"x": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[_warn_guard()])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        # Execution should proceed despite the WARN guard failure
        assert result["world_state"]["x"] is True
        assert result["execution_history"][0].success is True

    def test_block_guard_aborts_execution(self) -> None:
        """BLOCK-severity guard failure aborts action and sets action_failed status."""
        executed: list[bool] = []

        def track_fn(ws: dict[str, Any]) -> dict[str, Any]:
            executed.append(True)
            return {}

        action = _action("set_x", eff={"x": True}, execute=track_fn)
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[_block_guard()])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)

        assert result.get("status") == "action_failed"
        assert not executed, "action callable must NOT run when a guard blocks"
        assert result["execution_history"][0].success is False
        assert "blocked!" in (result["execution_history"][0].error or "")

    def test_block_guard_blacklists_action(self) -> None:
        """Guard block sets failure_count = max_retries+1 to trigger blacklisting."""
        action = ActionSpec(
            name="risky", preconditions={}, effects={"done": True}, max_retries=2
        )
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[_block_guard()])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        counts = result.get("action_failure_counts", {})
        assert counts.get("risky", 0) == action.max_retries + 1

    def test_warn_then_block_guard_aborts(self) -> None:
        """A mix of WARN and BLOCK guards aborts on the BLOCK guard."""
        action = _action("set_x", eff={"x": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[_warn_guard("w"), _block_guard("b")])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert result.get("status") == "action_failed"

    def test_no_guards_no_overhead(self) -> None:
        """GoapExecutor with no guards behaves identically to the default."""
        action = _action("set_y", eff={"y": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert result["world_state"]["y"] is True

    @pytest.mark.asyncio
    async def test_async_block_guard_aborts_execution(self) -> None:
        """Async executor path also aborts on a BLOCK guard."""
        action = _action("async_act", eff={"done": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[_block_guard()])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = await executor.acall(state)
        assert result.get("status") == "action_failed"
        assert result["execution_history"][0].success is False

    @pytest.mark.asyncio
    async def test_async_warn_guard_continues(self) -> None:
        """Async executor path continues after a WARN guard."""
        action = _action("async_act", eff={"done": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor(guards=[_warn_guard()])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = await executor.acall(state)
        assert result["world_state"]["done"] is True

    def test_success_omits_status_key(self) -> None:
        """Success path must NOT include 'status' — the planner-set 'executing'
        persists in GoapState via LangGraph's key-merge semantics."""
        action = _action("set_z", eff={"z": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor()
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert "status" not in result
        assert result["world_state"]["z"] is True


# ---------------------------------------------------------------------------
# _find_parallel_group helper
# ---------------------------------------------------------------------------


class TestFindParallelGroup:
    def test_single_action_group(self) -> None:
        a = _action("a", eff={"x": True})
        plan_obj = _make_plan(a)
        group = _find_parallel_group(plan_obj.actions, 0)
        assert group == [0]

    def test_two_independent_actions_same_wave(self) -> None:
        a = _action("a", eff={"x": True})
        b = _action("b", eff={"y": True})
        plan_obj = _make_plan(a, b)
        group = _find_parallel_group(plan_obj.actions, 0)
        assert set(group) == {0, 1}

    def test_dependent_action_not_in_same_wave(self) -> None:
        a = _action("a", eff={"x": True})
        b = _action("b", pre={"x": True}, eff={"y": True})
        plan_obj = _make_plan(a, b)
        # wave from step 0: only a (b depends on a's effect)
        group = _find_parallel_group(plan_obj.actions, 0)
        assert group == [0]

    def test_second_wave_starts_after_first(self) -> None:
        a = _action("a", eff={"x": True})
        b = _action("b", pre={"x": True}, eff={"y": True})
        plan_obj = _make_plan(a, b)
        # wave from step 1: only b
        group = _find_parallel_group(plan_obj.actions, 1)
        assert group == [1]

    def test_past_end_returns_empty(self) -> None:
        a = _action("a", eff={"x": True})
        plan_obj = _make_plan(a)
        group = _find_parallel_group(plan_obj.actions, 5)
        assert group == []


# ---------------------------------------------------------------------------
# ParallelGoapExecutor
# ---------------------------------------------------------------------------


class TestParallelGoapExecutor:
    def test_executes_independent_actions_together(self) -> None:
        a = _action("a", eff={"x": True})
        b = _action("b", eff={"y": True})
        plan_obj = _make_plan(a, b)
        executor = ParallelGoapExecutor()
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert result["world_state"]["x"] is True
        assert result["world_state"]["y"] is True
        assert result["current_step"] == 2  # both actions consumed
        assert len(result["execution_history"]) == 2

    def test_dependent_action_executes_only_first_wave(self) -> None:
        a = _action("a", eff={"x": True})
        b = _action("b", pre={"x": True}, eff={"y": True})
        plan_obj = _make_plan(a, b)
        executor = ParallelGoapExecutor()
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        # Only a should have run (b depends on a)
        assert result["world_state"]["x"] is True
        assert "y" not in result["world_state"]
        assert result["current_step"] == 1

    def test_failure_in_wave_returns_action_failed(self) -> None:
        def boom(ws: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("exploded")

        a = _action("a", eff={"x": True}, execute=boom)
        b = _action("b", eff={"y": True})
        plan_obj = _make_plan(a, b)
        executor = ParallelGoapExecutor()
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert result.get("status") == "action_failed"

    def test_block_guard_aborts_wave(self) -> None:
        a = _action("a", eff={"x": True})
        b = _action("b", eff={"y": True})
        plan_obj = _make_plan(a, b)
        executor = ParallelGoapExecutor(guards=[_block_guard()])
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert result.get("status") == "action_failed"

    def test_short_circuits_on_no_plan_status(self) -> None:
        executor = ParallelGoapExecutor()
        state: GoapState = {
            "world_state": {},
            "plan": None,
            "current_step": 0,
            "status": "no_plan",
        }
        result = executor(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_async_executes_independent_actions_concurrently(self) -> None:
        """Two independent async actions run in the same wave via asyncio.gather."""
        order: list[str] = []

        async def slow_a(ws: dict[str, Any]) -> dict[str, Any]:
            import asyncio as _aio

            await _aio.sleep(0.05)
            order.append("a")
            return {"x": True}

        async def fast_b(ws: dict[str, Any]) -> dict[str, Any]:
            import asyncio as _aio

            await _aio.sleep(0.01)
            order.append("b")
            return {"y": True}

        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, aexecute=slow_a)
        b = ActionSpec(name="b", preconditions={}, effects={"y": True}, aexecute=fast_b)
        plan_obj = _make_plan(a, b)
        executor = ParallelGoapExecutor()
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = await executor.acall(state)
        # fast_b finishes before slow_a → concurrent execution confirmed
        assert order == ["b", "a"]
        assert result["world_state"]["x"] is True
        assert result["world_state"]["y"] is True
        assert result["current_step"] == 2

    @pytest.mark.asyncio
    async def test_async_block_guard_aborts_wave(self) -> None:
        a = _action("a", eff={"x": True})
        executor = ParallelGoapExecutor(guards=[_block_guard()])
        plan_obj = _make_plan(a)
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = await executor.acall(state)
        assert result.get("status") == "action_failed"

    def test_success_omits_status_key(self) -> None:
        """Success path must NOT set 'status' — the planner-set 'executing'
        persists in GoapState via LangGraph's key-merge semantics.  Only
        failure paths set 'status' to 'action_failed'."""
        a = _action("a", eff={"x": True})
        plan_obj = _make_plan(a)
        executor = ParallelGoapExecutor()
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result = executor(state)
        assert "status" not in result
        assert result["world_state"]["x"] is True

    def test_dep_cache_reused_across_waves(self) -> None:
        """Dependency graph is computed once per plan, not per wave."""
        a = _action("a", eff={"x": True})
        b = _action("b", pre={"x": True}, eff={"y": True})
        plan_obj = _make_plan(a, b)
        executor = ParallelGoapExecutor()

        # First wave: only a runs (b depends on a)
        state: GoapState = {"world_state": {}, "plan": plan_obj, "current_step": 0}
        result1 = executor(state)
        assert result1["current_step"] == 1
        cache_after_wave1 = executor._dep_cache
        assert cache_after_wave1 is not None

        # Second wave: b runs.  Same plan tuple → cache hit.
        state2: GoapState = {
            "world_state": result1["world_state"],
            "plan": plan_obj,
            "current_step": 1,
        }
        result2 = executor(state2)
        assert result2["current_step"] == 2
        assert executor._dep_cache is cache_after_wave1  # same cache object


# ---------------------------------------------------------------------------
# _is_better_plan — feasibility-first comparison
# ---------------------------------------------------------------------------


def _plan_with_score(cost: float, score: Any) -> Plan:
    """Build a minimal Plan with explicit total_cost and score."""
    return Plan(actions=(), total_cost=cost, score=score)


class TestIsBetterPlan:
    """Verify feasibility-first comparison across Score subclasses."""

    def test_feasible_beats_infeasible_even_with_higher_cost(self) -> None:
        feasible = _plan_with_score(10.0, SimpleScore(scalar=10.0))
        infeasible = _plan_with_score(1.0, HardSoftScore(hard=-5.0, soft=0.0))
        assert _is_better_plan(feasible, infeasible) is True
        assert _is_better_plan(infeasible, feasible) is False

    def test_both_feasible_lower_cost_wins(self) -> None:
        cheap = _plan_with_score(2.0, SimpleScore(scalar=2.0))
        expensive = _plan_with_score(5.0, HardSoftScore(hard=0.0, soft=-1.0))
        assert _is_better_plan(cheap, expensive) is True
        assert _is_better_plan(expensive, cheap) is False

    def test_both_infeasible_lower_cost_wins(self) -> None:
        less_bad = _plan_with_score(3.0, HardSoftScore(hard=-1.0, soft=0.0))
        worse = _plan_with_score(8.0, HardSoftScore(hard=-10.0, soft=0.0))
        assert _is_better_plan(less_bad, worse) is True
        assert _is_better_plan(worse, less_bad) is False

    def test_equal_feasibility_and_cost_not_better(self) -> None:
        a = _plan_with_score(5.0, SimpleScore(scalar=5.0))
        b = _plan_with_score(5.0, HardSoftScore(hard=0.0, soft=-2.0))
        assert _is_better_plan(a, b) is False
        assert _is_better_plan(b, a) is False

    def test_bendable_score_infeasible(self) -> None:
        feasible = _plan_with_score(7.0, SimpleScore(scalar=7.0))
        infeasible = _plan_with_score(1.0, BendableScore(hard_levels=(-3.0,)))
        assert _is_better_plan(feasible, infeasible) is True
        assert _is_better_plan(infeasible, feasible) is False

    def test_same_subclass_feasible(self) -> None:
        cheap = _plan_with_score(1.0, SimpleScore(scalar=1.0))
        expensive = _plan_with_score(9.0, SimpleScore(scalar=9.0))
        assert _is_better_plan(cheap, expensive) is True
        assert _is_better_plan(expensive, cheap) is False

    # ------------------------------------------------------------------
    # F1 fix: soft-score preserved for same-type feasible plans
    # ------------------------------------------------------------------

    def test_hard_soft_score_soft_breaks_tie_when_costs_equal(self) -> None:
        """F1: two feasible HardSoftScore plans with equal total_cost but
        different soft scores — the one with the better (higher) soft score
        must win.  Before the fix both were treated as equivalent because
        total_cost was compared instead of the native score."""
        better_soft = _plan_with_score(5.0, HardSoftScore(hard=0.0, soft=-1.0))
        worse_soft  = _plan_with_score(5.0, HardSoftScore(hard=0.0, soft=-10.0))
        assert _is_better_plan(better_soft, worse_soft) is True
        assert _is_better_plan(worse_soft, better_soft) is False

    def test_hard_soft_score_soft_wins_even_when_candidate_costs_more(self) -> None:
        """F1: a CSP-optimised plan with a better soft score beats a cheaper
        plan whose soft score is significantly worse.  This is the information
        loss that total_cost-only comparison caused."""
        high_quality = _plan_with_score(7.0, HardSoftScore(hard=0.0, soft=-2.0))
        low_quality  = _plan_with_score(4.0, HardSoftScore(hard=0.0, soft=-50.0))
        assert _is_better_plan(high_quality, low_quality) is True
        assert _is_better_plan(low_quality, high_quality) is False

    def test_hard_soft_score_equal_soft_not_better(self) -> None:
        """Two feasible HardSoftScore plans with identical hard and soft —
        neither is better than the other."""
        a = _plan_with_score(3.0, HardSoftScore(hard=0.0, soft=-5.0))
        b = _plan_with_score(9.0, HardSoftScore(hard=0.0, soft=-5.0))
        # costs differ but scores are identical → not better in either direction
        assert _is_better_plan(a, b) is False
        assert _is_better_plan(b, a) is False

    def test_cross_subtype_feasible_falls_back_to_total_cost(self) -> None:
        """F1/F3: cross-subtype comparison (SimpleScore vs HardSoftScore) must
        not raise TypeError and must fall back to total_cost."""
        cheap_simple   = _plan_with_score(2.0, SimpleScore(scalar=2.0))
        pricey_hss     = _plan_with_score(8.0, HardSoftScore(hard=0.0, soft=-1.0))
        assert _is_better_plan(cheap_simple, pricey_hss) is True
        assert _is_better_plan(pricey_hss, cheap_simple) is False

    def test_bendable_score_soft_breaks_tie(self) -> None:
        """F1: same-type feasible BendableScore plans use native comparison."""
        better = _plan_with_score(4.0, BendableScore(hard_levels=(0.0,), soft_levels=(-1.0,)))
        worse  = _plan_with_score(4.0, BendableScore(hard_levels=(0.0,), soft_levels=(-9.0,)))
        assert _is_better_plan(better, worse) is True
        assert _is_better_plan(worse, better) is False

    def test_bendable_score_mismatched_shape_falls_back_to_cost(self) -> None:
        """F3: BendableScore shape mismatch raises TypeError → total_cost fallback."""
        two_hard  = _plan_with_score(3.0, BendableScore(hard_levels=(0.0, 0.0)))
        one_hard  = _plan_with_score(7.0, BendableScore(hard_levels=(0.0,)))
        # Both feasible, shape mismatch → cost tiebreaker; 3.0 < 7.0 so two_hard wins
        assert _is_better_plan(two_hard, one_hard) is True
        assert _is_better_plan(one_hard, two_hard) is False
