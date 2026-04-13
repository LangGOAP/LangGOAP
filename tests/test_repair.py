"""Tests for plan repair (langgoap.planner.repair)."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.repair import RepairResult, RepairStrategy, repair_plan
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
from langgoap.state import PlanningState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_action(
    name: str,
    pre: dict[str, Any] | None = None,
    eff: dict[str, Any] | None = None,
    cost: float = 1.0,
) -> ActionSpec:
    return ActionSpec(name=name, preconditions=pre or {}, effects=eff or {}, cost=cost)


def make_plan_from_start(start: PlanningState, *actions: ActionSpec) -> Plan:
    """Build a Plan by simulating forward from *start*."""
    states: list[PlanningState] = []
    sim = start
    total = 0.0
    for a in actions:
        total += a.get_cost(sim.to_dict())
        sim = sim.apply(a.effects)
        states.append(sim)
    return Plan(
        actions=actions,
        expected_states=tuple(states),
        total_cost=total,
        score=SimpleScore(scalar=total),
    )


# ---------------------------------------------------------------------------
# RepairResult dataclass
# ---------------------------------------------------------------------------


class TestRepairResultDataclass:
    def test_fields_exist(self) -> None:
        r = RepairResult(
            repaired_plan=None,
            repair_applied=False,
            valid_prefix_length=0,
            suffix_replanned=False,
            suffix_length=0,
        )
        assert r.repaired_plan is None
        assert r.repair_applied is False
        assert r.valid_prefix_length == 0
        assert r.suffix_replanned is False
        assert r.suffix_length == 0

    def test_frozen(self) -> None:
        r = RepairResult(
            repaired_plan=None,
            repair_applied=False,
            valid_prefix_length=0,
            suffix_replanned=False,
            suffix_length=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.repair_applied = True  # type: ignore[misc]

    def test_repaired_plan_field_accepts_plan(self) -> None:
        p = Plan.empty()
        r = RepairResult(
            repaired_plan=p,
            repair_applied=True,
            valid_prefix_length=2,
            suffix_replanned=True,
            suffix_length=1,
        )
        assert r.repaired_plan is p
        assert r.valid_prefix_length == 2
        assert r.suffix_length == 1


# ---------------------------------------------------------------------------
# repair_plan — goal already satisfied
# ---------------------------------------------------------------------------


class TestRepairPlanGoalAlreadySatisfied:
    def test_returns_empty_plan_when_goal_met(self) -> None:
        a = make_action("a", pre={}, eff={"x": True})
        start = PlanningState.from_dict({"done": True})
        goal = GoalSpec(conditions={"done": True})
        original = make_plan_from_start(PlanningState.from_dict({}), a)

        result = repair_plan(original, start, goal, [a])

        assert result.repaired_plan is not None
        assert len(result.repaired_plan) == 0
        assert result.repair_applied is False
        assert result.suffix_replanned is False
        assert result.valid_prefix_length == 0

    def test_goal_satisfied_suffix_length_zero(self) -> None:
        start = PlanningState.from_dict({"done": True})
        goal = GoalSpec(conditions={"done": True})
        plan = Plan.empty()
        result = repair_plan(plan, start, goal, [])
        assert result.suffix_length == 0


# ---------------------------------------------------------------------------
# repair_plan — full prefix valid (all remaining actions applicable)
# ---------------------------------------------------------------------------


class TestRepairPlanFullPrefixValid:
    def test_full_prefix_reuse_returns_repaired_plan(self) -> None:
        """All remaining actions are applicable → full prefix reused, no new suffix."""
        a = make_action("a", pre={"start": True}, eff={"a_done": True})
        b = make_action("b", pre={"a_done": True}, eff={"goal": True})
        start = PlanningState.from_dict({"start": True})
        goal = GoalSpec(conditions={"goal": True})
        original = make_plan_from_start(start, a, b)

        result = repair_plan(original, start, goal, [a, b])

        assert result.repaired_plan is not None
        assert result.repair_applied is True
        assert result.valid_prefix_length == 2
        assert result.suffix_replanned is True
        # A* from final state (goal satisfied) → empty suffix
        assert result.suffix_length == 0

    def test_repaired_plan_action_names_match_prefix(self) -> None:
        a = make_action("a", pre={}, eff={"a": True})
        b = make_action("b", pre={"a": True}, eff={"done": True})
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})
        original = make_plan_from_start(start, a, b)

        result = repair_plan(original, start, goal, [a, b])

        assert result.repaired_plan is not None
        assert result.repaired_plan.action_names == ["a", "b"]


# ---------------------------------------------------------------------------
# repair_plan — partial prefix (state drifted mid-plan)
# ---------------------------------------------------------------------------


class TestRepairPlanPartialPrefix:
    def test_partial_prefix_replans_suffix(self) -> None:
        """First action applicable, second requires drifted key → replan suffix."""
        a = make_action("a", pre={}, eff={"a": True})
        # b expects "key" which is missing in drifted state
        b = make_action("b", pre={"a": True, "key": True}, eff={"done": True})
        c = make_action("c", pre={"a": True}, eff={"done": True})  # alternate route

        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})
        original = make_plan_from_start(start, a, b)

        # current_state doesn't have "key" — b won't be applicable after a
        current = PlanningState.from_dict({})
        result = repair_plan(original, current, goal, [a, b, c])

        assert result.valid_prefix_length == 1  # only 'a' is in prefix
        assert result.repair_applied is True
        assert result.suffix_replanned is True
        assert result.repaired_plan is not None
        # Repaired plan: prefix [a] + suffix [c]
        assert result.repaired_plan.action_names == ["a", "c"]

    def test_partial_prefix_total_cost(self) -> None:
        a = make_action("a", pre={}, eff={"a": True}, cost=2.0)
        b = make_action(
            "b", pre={"a": True, "missing": True}, eff={"done": True}, cost=3.0
        )
        c = make_action("c", pre={"a": True}, eff={"done": True}, cost=1.0)

        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})
        original = make_plan_from_start(start, a, b)

        result = repair_plan(original, start, goal, [a, b, c])

        assert result.repaired_plan is not None
        # prefix cost=2.0 (action a) + suffix cost=1.0 (action c) = 3.0
        assert result.repaired_plan.total_cost == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# repair_plan — no valid prefix
# ---------------------------------------------------------------------------


class TestRepairPlanNoValidPrefix:
    def test_no_prefix_falls_back_to_full_replan(self) -> None:
        """First remaining action not applicable → full replan from current state."""
        # Plan: a → b, but current_state doesn't have precondition for a
        a = make_action("a", pre={"init": True}, eff={"a": True})
        b = make_action("b", pre={"a": True}, eff={"done": True})
        # alternative path from empty state
        c = make_action("c", pre={}, eff={"done": True})

        goal = GoalSpec(conditions={"done": True})
        current = PlanningState.from_dict({})  # missing "init"
        original = make_plan_from_start(PlanningState.from_dict({"init": True}), a, b)

        result = repair_plan(original, current, goal, [a, b, c])

        assert result.valid_prefix_length == 0
        assert result.repair_applied is False
        assert result.suffix_replanned is True
        assert result.repaired_plan is not None
        assert result.repaired_plan.action_names == ["c"]

    def test_no_prefix_and_replan_fails_returns_none(self) -> None:
        a = make_action("a", pre={"init": True}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        current = PlanningState.from_dict({})  # "init" missing, can't plan
        original = make_plan_from_start(PlanningState.from_dict({"init": True}), a)

        result = repair_plan(original, current, goal, [a])

        assert result.repaired_plan is None
        assert result.repair_applied is False
        assert result.suffix_replanned is True
        assert result.suffix_length == 0


# ---------------------------------------------------------------------------
# repair_plan — empty plan input
# ---------------------------------------------------------------------------


class TestRepairPlanEmptyInput:
    def test_empty_plan_no_goal_replans_from_scratch(self) -> None:
        c = make_action("c", pre={}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        current = PlanningState.from_dict({})
        original = Plan.empty()

        result = repair_plan(original, current, goal, [c])

        assert result.repaired_plan is not None
        assert result.repaired_plan.action_names == ["c"]
        assert result.valid_prefix_length == 0
        assert result.suffix_replanned is True


# ---------------------------------------------------------------------------
# repair_plan — current_step > 0
# ---------------------------------------------------------------------------


class TestRepairPlanCurrentStep:
    def test_current_step_skips_already_executed_actions(self) -> None:
        a = make_action("a", pre={}, eff={"a": True})
        b = make_action("b", pre={"a": True}, eff={"b": True})
        c = make_action("c", pre={"b": True}, eff={"done": True})
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})
        original = make_plan_from_start(start, a, b, c)

        # "a" was already executed; current state has "a" effect
        current = PlanningState.from_dict({"a": True})
        result = repair_plan(original, current, goal, [a, b, c], current_step=1)

        assert result.repaired_plan is not None
        # Only b and c remain; both applicable → they form the prefix
        assert "a" not in result.repaired_plan.action_names
        assert result.valid_prefix_length == 2  # b and c

    def test_current_step_beyond_plan_length_replans(self) -> None:
        a = make_action("a", pre={}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        original = make_plan_from_start(PlanningState.from_dict({}), a)
        current = PlanningState.from_dict({})  # goal not met

        result = repair_plan(original, current, goal, [a], current_step=5)

        # remaining is empty → A* replans from scratch
        assert result.valid_prefix_length == 0
        assert result.repaired_plan is not None
        assert result.repaired_plan.action_names == ["a"]


# ---------------------------------------------------------------------------
# repair_plan — blacklisted_actions
# ---------------------------------------------------------------------------


class TestRepairPlanBlacklisted:
    def test_blacklisted_excluded_from_suffix(self) -> None:
        a = make_action("a", pre={}, eff={"a": True})
        # b and c are alternate suffix routes
        b = make_action("b", pre={"a": True, "need_b": True}, eff={"done": True})
        c = make_action("c", pre={"a": True}, eff={"done": True})

        goal = GoalSpec(conditions={"done": True})
        # drift: b's extra precondition "need_b" is missing
        current = PlanningState.from_dict({})
        original = make_plan_from_start(current, a, b)

        # blacklist b → suffix must use c
        result = repair_plan(
            original, current, goal, [a, b, c], blacklisted_actions=["b"]
        )

        assert result.repaired_plan is not None
        assert "b" not in result.repaired_plan.action_names
        assert "c" in result.repaired_plan.action_names


# ---------------------------------------------------------------------------
# RepairStrategy
# ---------------------------------------------------------------------------


class TestRepairStrategy:
    def test_delegates_on_first_call_no_prior_plan(self) -> None:
        a = make_action("a", pre={}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({})

        strategy = RepairStrategy()
        result = strategy.plan(start, goal, [a])

        assert result is not None
        assert result.action_names == ["a"]

    def test_default_inner_is_astar_strategy(self) -> None:
        from langgoap.planner.strategy import AStarStrategy

        strategy = RepairStrategy()
        assert isinstance(strategy._inner, AStarStrategy)

    def test_attempts_repair_when_prior_plan_given(self) -> None:
        a = make_action("a", pre={}, eff={"a": True})
        b = make_action("b", pre={"a": True}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({})
        prior = make_plan_from_start(start, a, b)

        strategy = RepairStrategy()
        # current state matches expected (no drift) → full reuse
        result = strategy.plan(start, goal, [a, b], prior_plan=prior)

        assert result is not None
        assert result.action_names == ["a", "b"]

    def test_falls_back_to_inner_when_repair_fails(self) -> None:
        """Repair suffix fails → inner strategy rescues with a full replan."""
        a = make_action("a", pre={"init": True}, eff={"done": True})
        c = make_action("c", pre={}, eff={"done": True})  # only available rescue

        goal = GoalSpec(conditions={"done": True})
        prior = make_plan_from_start(PlanningState.from_dict({"init": True}), a)

        strategy = RepairStrategy()
        # current state: "init" missing, so neither the prefix nor A* with [a]
        # can help; but we add [c] so inner strategy can succeed.
        result = strategy.plan(
            PlanningState.from_dict({}), goal, [a, c], prior_plan=prior
        )

        assert result is not None
        # c is the only reachable action; inner strategy returns it
        assert "c" in result.action_names

    def test_repair_applied_returns_repaired_plan_not_inner(self) -> None:
        """When repair succeeds, inner strategy must NOT be called."""
        call_log: list[str] = []

        class TrackingStrategy:
            def plan(
                self,
                start: PlanningState,
                goal: GoalSpec,
                actions: list[ActionSpec],
                *,
                blacklisted_actions: list[str] | None = None,
            ) -> Plan | None:
                call_log.append("inner_called")
                from langgoap.planner.astar import plan as ap

                return ap(start, goal, actions, blacklisted_actions=blacklisted_actions)

        a = make_action("a", pre={}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({})
        prior = make_plan_from_start(start, a)

        strategy = RepairStrategy(inner=TrackingStrategy())
        result = strategy.plan(start, goal, [a], prior_plan=prior)

        assert result is not None
        assert (
            "inner_called" not in call_log
        ), "inner should not be called on repair success"

    def test_custom_inner_strategy_used_on_first_call(self) -> None:
        call_log: list[str] = []

        class SentinelStrategy:
            def plan(
                self,
                start: PlanningState,
                goal: GoalSpec,
                actions: list[ActionSpec],
                *,
                blacklisted_actions: list[str] | None = None,
            ) -> Plan | None:
                call_log.append("sentinel")
                from langgoap.planner.astar import plan as ap

                return ap(start, goal, actions, blacklisted_actions=blacklisted_actions)

        a = make_action("a", pre={}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({})

        strategy = RepairStrategy(inner=SentinelStrategy())
        result = strategy.plan(start, goal, [a])  # no prior_plan

        assert result is not None
        assert "sentinel" in call_log

    def test_current_step_forwarded_to_repair(self) -> None:
        a = make_action("a", pre={}, eff={"a": True})
        b = make_action("b", pre={"a": True}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        prior = make_plan_from_start(PlanningState.from_dict({}), a, b)

        strategy = RepairStrategy()
        # "a" already executed; current state has effect of a
        current = PlanningState.from_dict({"a": True})
        result = strategy.plan(current, goal, [a, b], prior_plan=prior, current_step=1)

        assert result is not None
        assert "a" not in result.action_names

    def test_blacklisted_forwarded_to_repair(self) -> None:
        a = make_action("a", pre={}, eff={"a": True})
        b = make_action("b", pre={"a": True, "missing": True}, eff={"done": True})
        c = make_action("c", pre={"a": True}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({})
        prior = make_plan_from_start(start, a, b)

        strategy = RepairStrategy()
        result = strategy.plan(
            start, goal, [a, b, c], prior_plan=prior, blacklisted_actions=["b"]
        )

        assert result is not None
        assert "b" not in result.action_names

    def test_repair_strategy_plan_returns_none_when_no_plan_exists(self) -> None:
        """Both repair and inner strategy fail → None."""
        a = make_action("a", pre={"impossible": True}, eff={"done": True})
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({})

        strategy = RepairStrategy()
        result = strategy.plan(start, goal, [a])

        assert result is None
