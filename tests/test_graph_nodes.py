"""Tests for GoapPlanner, GoapExecutor, and GoapObserver nodes."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.nodes import GoapExecutor, GoapObserver, GoapPlanner
from langgoap.graph.state import ActionResult, GoapState
from langgoap.planner.types import Plan, PlanMetadata
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
