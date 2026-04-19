"""Tests for the A* GOAP planner."""

from __future__ import annotations

import time
from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan
from langgoap.state import PlanningState
from tests.conftest import make_action as _action

# ---------------------------------------------------------------------------
# Basic planning scenarios
# ---------------------------------------------------------------------------


class TestBasicPlanning:
    def test_linear_chain(self) -> None:
        """A→B→C: three actions forming a linear dependency chain."""
        actions = [
            _action("a_to_b", pre={"a": True}, eff={"b": True}),
            _action("b_to_c", pre={"b": True}, eff={"c": True}),
        ]
        start = PlanningState.from_dict({"a": True})
        goal = GoalSpec(conditions={"c": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["a_to_b", "b_to_c"]

    def test_already_satisfied_goal(self) -> None:
        """Goal already met → empty plan."""
        actions = [_action("noop", eff={"x": True})]
        start = PlanningState.from_dict({"done": True})
        goal = GoalSpec(conditions={"done": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert len(result) == 0
        assert result.total_cost == 0.0

    def test_no_plan_possible(self) -> None:
        """No action produces the required effect → None."""
        actions = [_action("useless", eff={"x": True})]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"unreachable": True})

        result = plan(start, goal, actions)

        assert result is None


class TestCostOptimization:
    def test_lowest_cost_path_wins(self) -> None:
        """When two paths reach the goal, the cheaper one is selected."""
        actions = [
            _action("expensive", eff={"goal": True}, cost=10.0),
            _action("cheap", eff={"goal": True}, cost=1.0),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["cheap"]
        assert result.total_cost == 1.0

    def test_multi_step_cost_comparison(self) -> None:
        """Two-step cheap path beats one-step expensive path."""
        actions = [
            _action("direct", eff={"goal": True}, cost=10.0),
            _action("step1", eff={"mid": True}, cost=2.0),
            _action("step2", pre={"mid": True}, eff={"goal": True}, cost=2.0),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.total_cost == 4.0
        assert result.action_names == ["step1", "step2"]


class TestSpecificityTieBreaking:
    def test_more_specific_action_preferred(self) -> None:
        """When costs are equal, prefer the action with more preconditions."""
        actions = [
            _action("generic", pre={}, eff={"goal": True}, cost=1.0),
            _action(
                "specific", pre={"a": True, "b": True}, eff={"goal": True}, cost=1.0
            ),
        ]
        start = PlanningState.from_dict({"a": True, "b": True})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["specific"]

    def test_cost_overrides_specificity(self) -> None:
        """A cheaper generic action beats a more expensive specific one."""
        actions = [
            _action("cheap_generic", pre={}, eff={"goal": True}, cost=1.0),
            _action(
                "expensive_specific", pre={"a": True}, eff={"goal": True}, cost=5.0
            ),
        ]
        start = PlanningState.from_dict({"a": True})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["cheap_generic"]


class TestMultiStepBranching:
    def test_crime_scenario(self) -> None:
        """Multi-branch scenario with optimal path selection.

        Scenario: An agent wants to get_rich. It can:
        - Rob a bank (needs weapon) → get rich
        - Get a job → get rich (but costs more)
        - Steal weapon → has weapon
        - Buy weapon (needs money) → has weapon
        """
        actions = [
            _action("get_job", eff={"rich": True}, cost=5.0),
            _action("rob_bank", pre={"has_weapon": True}, eff={"rich": True}, cost=1.0),
            _action("steal_weapon", eff={"has_weapon": True}, cost=2.0),
            _action(
                "buy_weapon",
                pre={"has_money": True},
                eff={"has_weapon": True},
                cost=1.0,
            ),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"rich": True})

        result = plan(start, goal, actions)

        assert result is not None
        # steal_weapon (2) + rob_bank (1) = 3 < get_job (5)
        assert result.total_cost == 3.0
        assert result.action_names == ["steal_weapon", "rob_bank"]

    def test_crime_scenario_with_money(self) -> None:
        """With money, buy weapon + rob bank is cheapest."""
        actions = [
            _action("get_job", eff={"rich": True}, cost=5.0),
            _action("rob_bank", pre={"has_weapon": True}, eff={"rich": True}, cost=1.0),
            _action("steal_weapon", eff={"has_weapon": True}, cost=2.0),
            _action(
                "buy_weapon",
                pre={"has_money": True},
                eff={"has_weapon": True},
                cost=1.0,
            ),
        ]
        start = PlanningState.from_dict({"has_money": True})
        goal = GoalSpec(conditions={"rich": True})

        result = plan(start, goal, actions)

        assert result is not None
        # buy_weapon (1) + rob_bank (1) = 2
        assert result.total_cost == 2.0
        assert result.action_names == ["buy_weapon", "rob_bank"]


class TestOptimization:
    def test_removes_redundant_actions(self) -> None:
        """Forward pass prunes actions whose effects are already satisfied."""
        # set_a is needed, but the plan should NOT use both set_a variants
        actions = [
            _action("set_a", eff={"a": True}),
            _action("set_b", pre={"a": True}, eff={"b": True}),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"b": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["set_a", "set_b"]

    def test_forward_pass_skips_already_true_effects(self) -> None:
        """Forward optimization skips actions with pre-satisfied effects.

        If A* produces [set_a, set_b] and 'a' is already in start state,
        forward pass should prune set_a.
        """
        actions = [
            _action("set_a", eff={"a": True}),
            _action("set_b", pre={"a": True}, eff={"b": True}),
        ]
        start = PlanningState.from_dict({"a": True})
        goal = GoalSpec(conditions={"b": True})

        result = plan(start, goal, actions)

        assert result is not None
        # set_a is unnecessary since a is already True
        assert result.action_names == ["set_b"]

    def test_removes_actions_not_needed_for_goal(self) -> None:
        """Backward pass removes actions that don't contribute to goal."""
        # If the planner finds a path that includes unnecessary detours,
        # optimization should prune them
        actions = [
            _action("set_a", eff={"a": True}),
            _action("set_goal", pre={"a": True}, eff={"goal": True}),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["set_a", "set_goal"]


class TestDynamicCost:
    def test_dynamic_cost_affects_planning(self) -> None:
        """Actions with callable costs are evaluated during planning."""

        def high_cost(ws: dict[str, Any]) -> float:
            return 100.0

        actions = [
            ActionSpec(
                name="dynamic_expensive", effects={"goal": True}, cost=high_cost
            ),
            _action("static_cheap", eff={"goal": True}, cost=2.0),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["static_cheap"]


class TestPlanMetadata:
    def test_expected_states_populated(self) -> None:
        """Plan should include expected intermediate states."""
        actions = [
            _action("step1", eff={"a": True}),
            _action("step2", pre={"a": True}, eff={"b": True}),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"b": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert len(result.expected_states) == len(result.actions)
        # After step1, state should have a=True
        assert result.expected_states[0].satisfies({"a": True})
        # After step2, state should have b=True
        assert result.expected_states[1].satisfies({"b": True})

    def test_metadata_populated(self) -> None:
        """Planning metadata should have non-zero values."""
        actions = [_action("act", eff={"done": True})]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.metadata.nodes_explored >= 1
        assert result.metadata.planning_time_ms >= 0.0

    def test_simple_score_matches_total_cost(self) -> None:
        """A*-only plans must carry ``SimpleScore(value=total_cost)``.

        Regression for audit finding NC1: without explicit ``score=`` at every
        A* ``Plan(...)`` construction site the score would silently default to
        ``SimpleScore(0.0)`` regardless of the actual path cost.
        """
        from langgoap.score import SimpleScore

        actions = [
            _action("cheap", eff={"a": True}, cost=1.5),
            _action("medium", pre={"a": True}, eff={"b": True}, cost=2.5),
            _action("expensive", pre={"b": True}, eff={"done": True}, cost=4.0),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["cheap", "medium", "expensive"]
        assert isinstance(result.score, SimpleScore)
        assert result.score.scalar == result.total_cost
        assert result.total_cost == pytest.approx(8.0)

    def test_empty_plan_score_is_zero(self) -> None:
        """Already-satisfied goal yields an empty plan with ``SimpleScore(0.0)``."""
        from langgoap.score import SimpleScore

        actions = [_action("noop", eff={"done": True})]
        start = PlanningState.from_dict({"done": True})
        goal = GoalSpec(conditions={"done": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert len(result) == 0
        assert isinstance(result.score, SimpleScore)
        assert result.score.scalar == 0.0


class TestScalability:
    def test_many_actions_completes_quickly(self) -> None:
        """300+ actions should plan in under 2 seconds."""
        # Create a chain: action_0 → action_1 → ... → action_9 (goal)
        # Plus 290 irrelevant actions that don't lead to goal
        chain_actions = []
        for i in range(10):
            pre = {f"state_{i}": True} if i > 0 else {}
            eff = {f"state_{i + 1}": True}
            chain_actions.append(_action(f"chain_{i}", pre=pre, eff=eff))

        noise_actions = []
        for i in range(290):
            noise_actions.append(
                _action(
                    f"noise_{i}",
                    pre={f"noise_pre_{i}": True},
                    eff={f"noise_eff_{i}": True},
                )
            )

        all_actions = chain_actions + noise_actions
        start = PlanningState.from_dict({"state_0": True})
        goal = GoalSpec(conditions={"state_10": True})

        t0 = time.monotonic()
        result = plan(start, goal, all_actions)
        elapsed = time.monotonic() - t0

        assert result is not None
        assert len(result) == 10
        assert elapsed < 2.0, f"Planning took {elapsed:.2f}s, expected < 2s"


class TestEdgeCases:
    def test_empty_actions_list(self) -> None:
        """No actions available → no plan unless goal already satisfied."""
        start = PlanningState.from_dict({"a": True})
        goal = GoalSpec(conditions={"b": True})
        result = plan(start, goal, [])
        assert result is None

    def test_empty_actions_already_satisfied(self) -> None:
        """No actions but goal already satisfied → empty plan."""
        start = PlanningState.from_dict({"done": True})
        goal = GoalSpec(conditions={"done": True})
        result = plan(start, goal, [])
        assert result is not None
        assert len(result) == 0

    def test_action_with_no_preconditions(self) -> None:
        """Action with no preconditions can fire from any state."""
        actions = [_action("universal", eff={"goal": True})]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["universal"]

    def test_plan_accepts_dict_start(self) -> None:
        """plan() accepts a plain dict and coerces it to PlanningState."""
        actions = [_action("act", eff={"done": True})]
        result = plan({"a": True}, GoalSpec(conditions={"done": True}), actions)
        assert result is not None
        assert result.action_names == ["act"]

    def test_multi_condition_goal(self) -> None:
        """Goal requiring multiple conditions to be satisfied."""
        actions = [
            _action("set_a", eff={"a": True}),
            _action("set_b", eff={"b": True}),
            _action("set_c", pre={"a": True, "b": True}, eff={"c": True}),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"a": True, "b": True, "c": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert "set_a" in result.action_names
        assert "set_b" in result.action_names
        assert "set_c" in result.action_names
