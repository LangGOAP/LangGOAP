"""Unit tests for error recovery / blacklisting.

Tests cover:
- max_retries field on ActionSpec, goap_action, GoapAction
- _get_last_failed_action / _get_max_retries helpers
- Observer blacklisting logic in the action_failed branch
- Planner blacklist filtering and blacklist fallback
- Backward compatibility of new GoapState fields
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from langgoap.actions import ActionSpec, GoapAction, goap_action
from langgoap.goals import GoalPolicy, GoalSpec
from langgoap.graph.nodes import (
    GoapObserver,
    GoapPlanner,
    _get_last_failed_action,
    _get_max_retries,
)
from langgoap.graph.state import ActionResult, GoapState
from langgoap.planner.astar import plan as astar_plan
from langgoap.state import PlanningState
from langgoap.types import ReplanStrategy
from tests.conftest import make_action as _action
from tests.conftest import make_plan as _make_plan

# ---------------------------------------------------------------------------
# TestMaxRetriesField
# ---------------------------------------------------------------------------


class TestMaxRetriesField:
    def test_default_is_zero(self) -> None:
        action = _action("a", eff={"x": True})
        assert action.max_retries == 0

    def test_custom_value(self) -> None:
        action = _action("a", eff={"x": True}, max_retries=3)
        assert action.max_retries == 3

    def test_frozen(self) -> None:
        action = _action("a", eff={"x": True}, max_retries=2)
        with pytest.raises(AttributeError):
            action.max_retries = 5  # type: ignore[misc]

    def test_decorator_passes_through(self) -> None:
        @goap_action(effects={"x": True}, max_retries=4)
        def my_action(state: dict[str, Any]) -> dict[str, Any]:
            return {"x": True}

        assert my_action.max_retries == 4

    def test_decorator_default_zero(self) -> None:
        @goap_action(effects={"x": True})
        def my_action(state: dict[str, Any]) -> dict[str, Any]:
            return {"x": True}

        assert my_action.max_retries == 0

    def test_goap_action_class_default(self) -> None:
        class MyAction(GoapAction):
            effects = {"x": True}

        spec = MyAction().to_spec()
        assert spec.max_retries == 0

    def test_goap_action_class_custom(self) -> None:
        class MyAction(GoapAction):
            effects = {"x": True}
            max_retries = 5

        spec = MyAction().to_spec()
        assert spec.max_retries == 5

    def test_repr_shows_max_retries_when_nonzero(self) -> None:
        action = _action("a", eff={"x": True}, max_retries=3)
        assert "max_retries=3" in repr(action)

    def test_repr_hides_max_retries_when_zero(self) -> None:
        action = _action("a", eff={"x": True})
        assert "max_retries" not in repr(action)


# ---------------------------------------------------------------------------
# TestBlacklistHelpers
# ---------------------------------------------------------------------------


class TestBlacklistHelpers:
    def test_get_last_failed_empty_history(self) -> None:
        state: GoapState = {"execution_history": []}
        assert _get_last_failed_action(state) is None

    def test_get_last_failed_no_history_key(self) -> None:
        state: GoapState = {}
        assert _get_last_failed_action(state) is None

    def test_get_last_failed_all_success(self) -> None:
        state: GoapState = {
            "execution_history": [
                ActionResult(action_name="a", success=True),
                ActionResult(action_name="b", success=True),
            ]
        }
        assert _get_last_failed_action(state) is None

    def test_get_last_failed_single_failure(self) -> None:
        state: GoapState = {
            "execution_history": [
                ActionResult(action_name="a", success=True),
                ActionResult(action_name="b", success=False, error="boom"),
            ]
        }
        assert _get_last_failed_action(state) == "b"

    def test_get_last_failed_multiple_failures_returns_last(self) -> None:
        state: GoapState = {
            "execution_history": [
                ActionResult(action_name="a", success=False, error="err1"),
                ActionResult(action_name="b", success=True),
                ActionResult(action_name="c", success=False, error="err2"),
            ]
        }
        assert _get_last_failed_action(state) == "c"

    def test_get_max_retries_found(self) -> None:
        actions = [
            _action("a", eff={"x": True}, max_retries=3),
            _action("b", eff={"y": True}, max_retries=1),
        ]
        assert _get_max_retries("a", actions) == 3
        assert _get_max_retries("b", actions) == 1

    def test_get_max_retries_not_found(self) -> None:
        actions = [_action("a", eff={"x": True}, max_retries=2)]
        assert _get_max_retries("missing", actions) == 0

    def test_get_max_retries_default(self) -> None:
        actions = [_action("a", eff={"x": True})]
        assert _get_max_retries("a", actions) == 0


# ---------------------------------------------------------------------------
# TestObserverBlacklisting
# ---------------------------------------------------------------------------


class TestObserverBlacklisting:
    def test_blacklisted_on_first_failure_default(self) -> None:
        """max_retries=0: action blacklisted on first failure."""
        actions = [
            _action("flaky", eff={"x": True}, max_retries=0),
            _action("backup", eff={"x": True}),
        ]
        observer = GoapObserver(actions=actions)

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": _make_plan(actions[0]),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="flaky", success=False, error="boom"),
            ],
        }
        cmd = observer(state)

        assert cmd.goto == "planner"
        assert cmd.update["replan_reason"] == "action_failed"
        assert cmd.update["blacklisted_actions"] == ["flaky"]
        assert cmd.update["action_failure_counts"] == {"flaky": 1}

    def test_not_blacklisted_until_threshold(self) -> None:
        """max_retries=2: action NOT blacklisted on first or second failure."""
        actions = [_action("flaky", eff={"x": True}, max_retries=2)]
        observer = GoapObserver(actions=actions)

        # First failure
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": _make_plan(actions[0]),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="flaky", success=False, error="err"),
            ],
        }
        cmd = observer(state)
        assert cmd.update["blacklisted_actions"] == []
        assert cmd.update["action_failure_counts"] == {"flaky": 1}

        # Second failure (count already 1)
        state2: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": _make_plan(actions[0]),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="flaky", success=False, error="err"),
            ],
            "action_failure_counts": {"flaky": 1},
        }
        cmd2 = observer(state2)
        assert cmd2.update["blacklisted_actions"] == []
        assert cmd2.update["action_failure_counts"] == {"flaky": 2}

    def test_blacklisted_after_threshold_exceeded(self) -> None:
        """max_retries=2: blacklisted on third failure (count > max_retries)."""
        actions = [_action("flaky", eff={"x": True}, max_retries=2)]
        observer = GoapObserver(actions=actions)

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": _make_plan(actions[0]),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="flaky", success=False, error="err"),
            ],
            "action_failure_counts": {"flaky": 2},
        }
        cmd = observer(state)
        assert "flaky" in cmd.update["blacklisted_actions"]
        assert cmd.update["action_failure_counts"] == {"flaky": 3}

    def test_never_strategy_ignores_blacklisting(self) -> None:
        """NEVER strategy goes to END, does not blacklist."""
        actions = [_action("flaky", eff={"x": True})]
        observer = GoapObserver(actions=actions)

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(
                conditions={"x": True},
                policy=GoalPolicy(replan_strategy=ReplanStrategy.NEVER),
            ),
            "plan": _make_plan(actions[0]),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="flaky", success=False, error="boom"),
            ],
        }
        cmd = observer(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "failed"
        assert cmd.update["replan_reason"] == "action_failed"
        # No blacklist fields in update — NEVER bypasses blacklisting entirely
        assert "blacklisted_actions" not in cmd.update
        assert "action_failure_counts" not in cmd.update

    def test_blacklist_accumulates_different_actions(self) -> None:
        """Multiple different actions blacklisted across failures."""
        actions = [
            _action("a", eff={"x": True}),
            _action("b", eff={"x": True}),
        ]
        observer = GoapObserver(actions=actions)

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": _make_plan(actions[1]),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="b", success=False, error="err"),
            ],
            "action_failure_counts": {"a": 1},
            "blacklisted_actions": ["a"],
        }
        cmd = observer(state)
        assert sorted(cmd.update["blacklisted_actions"]) == ["a", "b"]
        assert cmd.update["action_failure_counts"] == {"a": 1, "b": 1}

    def test_observer_without_actions_defaults_max_retries_zero(self) -> None:
        """Observer with no action specs defaults to max_retries=0 for all actions."""
        observer = GoapObserver()  # no actions provided

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": _make_plan(_action("unknown", eff={"x": True})),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="unknown", success=False, error="err"),
            ],
        }
        cmd = observer(state)
        # Without action specs, _get_max_retries returns 0 → blacklisted on first failure
        assert cmd.update["blacklisted_actions"] == ["unknown"]
        assert cmd.update["action_failure_counts"] == {"unknown": 1}

    def test_already_blacklisted_not_readded(self) -> None:
        """An action already in the blacklist is not duplicated."""
        actions = [_action("a", eff={"x": True})]
        observer = GoapObserver(actions=actions)

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": _make_plan(actions[0]),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="a", success=False, error="err"),
            ],
            "action_failure_counts": {"a": 1},
            "blacklisted_actions": ["a"],
        }
        cmd = observer(state)
        assert cmd.update["blacklisted_actions"] == ["a"]  # no duplicate
        assert cmd.update["action_failure_counts"] == {"a": 2}


# ---------------------------------------------------------------------------
# TestPlannerBlacklist
# ---------------------------------------------------------------------------


class TestPlannerBlacklist:
    def test_blacklisted_action_filtered(self) -> None:
        """Blacklisted action is skipped; alternative is used."""
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"x": True})
        a = _action("cheap", eff={"x": True}, cost=1.0)
        b = _action("expensive", eff={"x": True}, cost=5.0)
        result = astar_plan(start, goal, [a, b], blacklisted_actions=["cheap"])
        assert result is not None
        assert result.action_names == ["expensive"]

    def test_alternative_path_found(self) -> None:
        """Blacklisting forces planner to find a different route."""
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"c": True})
        # Direct path: a → c (cheap)
        direct = _action("direct", eff={"c": True}, cost=1.0)
        # Indirect path: step1 → b, step2 → c (via b)
        step1 = _action("step1", eff={"b": True}, cost=2.0)
        step2 = _action("step2", pre={"b": True}, eff={"c": True}, cost=2.0)

        result = astar_plan(
            start, goal, [direct, step1, step2], blacklisted_actions=["direct"]
        )
        assert result is not None
        assert "direct" not in result.action_names
        assert result.action_names == ["step1", "step2"]

    def test_blacklist_fallback_clears_blacklist(self) -> None:
        """When blacklist makes goal unreachable, retry with all actions."""
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"x": True})
        only_action = _action("only", eff={"x": True})

        # only_action is the only way; blacklisting it should trigger fallback
        result = astar_plan(start, goal, [only_action], blacklisted_actions=["only"])
        assert result is not None
        assert result.action_names == ["only"]

    def test_empty_blacklist_no_change(self) -> None:
        """Empty blacklist has no effect on planning."""
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"x": True})
        a = _action("a", eff={"x": True})

        result_normal = astar_plan(start, goal, [a])
        result_empty = astar_plan(start, goal, [a], blacklisted_actions=[])
        assert result_normal is not None
        assert result_empty is not None
        assert result_normal.action_names == result_empty.action_names

    def test_all_actions_blacklisted_fallback(self) -> None:
        """All actions blacklisted → blacklist fallback retries with all."""
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"x": True})
        a = _action("a", eff={"x": True})

        result = astar_plan(start, goal, [a], blacklisted_actions=["a"])
        # Blacklist fallback means it should still find a plan
        assert result is not None
        assert result.action_names == ["a"]

    def test_truly_unreachable_returns_none(self) -> None:
        """When goal is unreachable even without blacklist, returns None."""
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"impossible": True})
        a = _action("a", eff={"x": True})

        result = astar_plan(start, goal, [a], blacklisted_actions=[])
        assert result is None

    def test_none_blacklist_same_as_empty(self) -> None:
        """blacklisted_actions=None behaves same as empty list."""
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"x": True})
        a = _action("a", eff={"x": True})

        result = astar_plan(start, goal, [a], blacklisted_actions=None)
        assert result is not None
        assert result.action_names == ["a"]


# ---------------------------------------------------------------------------
# TestPlannerNodeBlacklist
# ---------------------------------------------------------------------------


class TestPlannerNodeBlacklist:
    def test_planner_passes_blacklist_to_astar(self) -> None:
        """GoapPlanner reads blacklisted_actions from state and passes to A*."""
        cheap = _action("cheap", eff={"x": True}, cost=1.0)
        expensive = _action("expensive", eff={"x": True}, cost=5.0)
        planner = GoapPlanner(actions=[cheap, expensive])

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "blacklisted_actions": ["cheap"],
        }
        result = planner(state)

        assert result["status"] == "executing"
        assert result["plan"] is not None
        assert result["plan"].action_names == ["expensive"]

    def test_planner_no_blacklist_uses_cheapest(self) -> None:
        """Without blacklist, planner picks the cheapest action."""
        cheap = _action("cheap", eff={"x": True}, cost=1.0)
        expensive = _action("expensive", eff={"x": True}, cost=5.0)
        planner = GoapPlanner(actions=[cheap, expensive])

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
        }
        result = planner(state)

        assert result["plan"].action_names == ["cheap"]

    def test_planner_blacklist_fallback_via_state(self) -> None:
        """Planner triggers blacklist fallback when blacklist blocks the only path."""
        only = _action("only", eff={"x": True})
        planner = GoapPlanner(actions=[only])

        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "blacklisted_actions": ["only"],
            "action_failure_counts": {"only": 1},
        }
        result = planner(state)

        # Blacklist fallback should find the plan
        assert result["status"] == "executing"
        assert result["plan"].action_names == ["only"]
        # Fallback clears blacklist and failure counts to give a fresh start
        assert result["blacklisted_actions"] == []
        assert result["action_failure_counts"] == {}


# ---------------------------------------------------------------------------
# TestBlacklistState
# ---------------------------------------------------------------------------


class TestBlacklistState:
    def test_backward_compat_missing_fields(self) -> None:
        """State without blacklist fields works (defaults to empty)."""
        state: GoapState = {
            "world_state": {"x": True},
            "goal": GoalSpec(conditions={"x": True}),
            "status": "executing",
        }
        # Observer should work fine without blacklist fields
        observer = GoapObserver()
        cmd = observer(state)
        assert cmd.update["status"] == "goal_achieved"

    def test_blacklist_fields_round_trip(self) -> None:
        """Blacklist fields can be set and read from GoapState."""
        state: GoapState = {
            "blacklisted_actions": ["a", "b"],
            "action_failure_counts": {"a": 2, "b": 1},
        }
        assert state["blacklisted_actions"] == ["a", "b"]
        assert state["action_failure_counts"] == {"a": 2, "b": 1}
