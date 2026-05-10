"""Integration tests for ``MultiGoal(mode="best_value")`` and ``GoalSpec.value``.

Mirrors Embabel's "best value plan to any goal" semantics on
``research/repos/embabel-agent/embabel-agent-api/src/test/kotlin/
com/embabel/plan/goap/OptimizingGoapPlannerTest.kt``: when several
goals are simultaneously achievable, the planner selects the goal
whose **net value** (``value - plan.total_cost``) is highest, not the
goal whose plan has the lowest cost.

LangGOAP additions:

* ``GoalSpec.value: float | Callable[[PlanningState], float] = 1.0``
  — utility weight per goal.  Callable form lets the value depend on
  the current world state (e.g. "this goal is worth more during peak
  hours").
* ``MultiGoal.mode = "best_value"`` — third mode peer to
  ``"sequential"`` and ``"any"``.
* ``Plan.net_value(goal) -> float`` — convenience accessor.
"""

from __future__ import annotations

from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.builder import GoapGraph
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from tests.conftest import make_action as _action


def _three_goal_actions() -> list[ActionSpec]:
    """Three independent goals, each reachable from start with one action."""
    return [
        _action("achieve_a", eff={"goal_a": True}, cost=8.0),
        _action("achieve_b", eff={"goal_b": True}, cost=2.0),
        _action("achieve_c", eff={"goal_c": True}, cost=0.5),
    ]


# ---------------------------------------------------------------------------
# Plan.net_value helper
# ---------------------------------------------------------------------------


class TestPlanNetValue:
    def test_net_value_subtracts_cost_from_goal_value(self) -> None:
        action = _action("a", eff={"x": True}, cost=3.0)
        plan = Plan(
            actions=(action,),
            expected_states=(PlanningState.from_dict({"x": True}),),
            total_cost=3.0,
        )
        goal = GoalSpec(conditions={"x": True}, value=10.0)
        assert plan.net_value(goal) == 7.0

    def test_callable_value_resolved_against_state(self) -> None:
        action = _action("a", eff={"x": True}, cost=4.0)
        plan = Plan(
            actions=(action,),
            expected_states=(PlanningState.from_dict({"x": True}),),
            total_cost=4.0,
        )
        # Value depends on whether 'is_peak_hour' is set on the start
        # state.  We pass the plan's expected end state for resolution
        # because that's what the planner already has access to.
        goal = GoalSpec(
            conditions={"x": True},
            value=lambda state: 100.0 if state.get("is_peak_hour") else 5.0,
        )
        assert plan.net_value(goal) == 1.0  # 5.0 - 4.0


# ---------------------------------------------------------------------------
# MultiGoal mode="best_value"
# ---------------------------------------------------------------------------


class TestBestValueMode:
    def test_picks_goal_with_highest_net_value(self) -> None:
        """Three goals: values (10, 5, 2), costs (8, 2, 0.5).
        Net values: 2, 3, 1.5. Goal B wins (highest net value 3.0)."""
        goals = MultiGoal(
            goals=(
                GoalSpec(conditions={"goal_a": True}, value=10.0),
                GoalSpec(conditions={"goal_b": True}, value=5.0),
                GoalSpec(conditions={"goal_c": True}, value=2.0),
            ),
            mode="best_value",
        )
        graph = GoapGraph(_three_goal_actions())

        result = graph.invoke(goal=goals, world_state={})

        assert result["status"] == "goal_achieved"
        # Net values: A=10-8=2, B=5-2=3, C=2-0.5=1.5 → B wins.
        assert result["world_state"].get("goal_b") is True
        assert result["world_state"].get("goal_a") is not True
        assert result["world_state"].get("goal_c") is not True

    def test_picks_high_value_goal_even_when_others_are_cheaper(self) -> None:
        """Best-value diverges from any-mode (cheapest-cost) selection."""
        goals = MultiGoal(
            goals=(
                # Expensive but very valuable.
                GoalSpec(conditions={"goal_a": True}, value=100.0),
                # Cheap and worthless — would win in any-mode.
                GoalSpec(conditions={"goal_c": True}, value=1.0),
            ),
            mode="best_value",
        )
        graph = GoapGraph(_three_goal_actions())

        result = graph.invoke(goal=goals, world_state={})

        assert result["status"] == "goal_achieved"
        # A: 100-8=92. C: 1-0.5=0.5. A wins.
        assert result["world_state"].get("goal_a") is True

    def test_falls_back_to_lowest_cost_on_value_tie(self) -> None:
        """When net values are equal the existing _is_better_plan
        tiebreak (lower cost) wins for stability."""
        goals = MultiGoal(
            goals=(
                # Net value: 10-8=2.
                GoalSpec(conditions={"goal_a": True}, value=10.0),
                # Net value: 4-2=2 — tied with A.
                GoalSpec(conditions={"goal_b": True}, value=4.0),
            ),
            mode="best_value",
        )
        graph = GoapGraph(_three_goal_actions())

        result = graph.invoke(goal=goals, world_state={})

        assert result["status"] == "goal_achieved"
        # Tied on net value; lower cost wins → B.
        assert result["world_state"].get("goal_b") is True

    def test_callable_value_resolved_at_planning_time(self) -> None:
        """Callable values are resolved against the plan's expected end
        state so context-aware utilities (peak hours, deadlines, etc.)
        work end-to-end."""
        goals = MultiGoal(
            goals=(
                GoalSpec(
                    conditions={"goal_a": True},
                    # Worth a lot only when is_peak_hour is True.
                    value=lambda state: 50.0 if state.get("is_peak_hour") else 1.0,
                ),
                GoalSpec(conditions={"goal_c": True}, value=3.0),
            ),
            mode="best_value",
        )
        graph = GoapGraph(_three_goal_actions())

        # Off-peak: A's net value = 1 - 8 = -7; C's = 3 - 0.5 = 2.5 → C wins.
        result = graph.invoke(goal=goals, world_state={})
        assert result["world_state"].get("goal_c") is True

        # Peak: A's net value = 50 - 8 = 42; C's = 2.5 → A wins.
        result = graph.invoke(goal=goals, world_state={"is_peak_hour": True})
        assert result["world_state"].get("goal_a") is True


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_default_value_is_1_0(self) -> None:
        goal = GoalSpec(conditions={"x": True})
        assert goal.value == 1.0

    def test_any_mode_unchanged_picks_lowest_cost(self) -> None:
        """Existing 'any' mode must still select lowest cost regardless
        of value — only 'best_value' uses net-value selection."""
        goals = MultiGoal(
            goals=(
                # High value but expensive — would win in best_value.
                GoalSpec(conditions={"goal_a": True}, value=100.0),
                # Low value but cheapest — wins in any mode.
                GoalSpec(conditions={"goal_c": True}, value=1.0),
            ),
            mode="any",
        )
        graph = GoapGraph(_three_goal_actions())

        result = graph.invoke(goal=goals, world_state={})

        assert result["status"] == "goal_achieved"
        # any-mode picks cheapest plan (C, cost 0.5) regardless of value.
        assert result["world_state"].get("goal_c") is True
        assert result["world_state"].get("goal_a") is not True


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class TestModeValidator:
    def test_invalid_mode_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="best_value"):
            MultiGoal(
                goals=(GoalSpec(conditions={"x": True}),),
                mode="banana",  # type: ignore[arg-type]
            )
