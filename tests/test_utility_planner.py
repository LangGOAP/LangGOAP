"""Unit tests for ``langgoap.planner.utility``.

Reference: research/repos/embabel-agent/embabel-agent-api/src/main/
  kotlin/com/embabel/plan/utility/UtilityPlanner.kt
"""

from __future__ import annotations

from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.utility import (
    NIRVANA_NAME,
    NirvanaGoal,
    UtilityStrategy,
    is_nirvana,
)
from langgoap.state import PlanningState

# ---------------------------------------------------------------------------
# NirvanaGoal
# ---------------------------------------------------------------------------


class TestNirvanaGoal:
    def test_subclasses_goalspec(self) -> None:
        assert isinstance(NirvanaGoal(), GoalSpec)

    def test_is_nirvana_helper_recognises_subclass(self) -> None:
        assert is_nirvana(NirvanaGoal()) is True
        assert is_nirvana(GoalSpec()) is False
        assert is_nirvana(None) is False
        assert is_nirvana("not a goal") is False

    def test_constant_matches_embabel_name(self) -> None:
        # Embabel: const val NIRVANA = "Nirvana"
        assert NIRVANA_NAME == "Nirvana"

    def test_user_kwargs_pass_through(self) -> None:
        from langgoap.goals import GoalPolicy
        from langgoap.types import ReplanStrategy

        # GoalSpec kwargs (``policy=GoalPolicy(...)``) forwarded
        # verbatim to the parent ``GoalSpec`` constructor.
        g = NirvanaGoal(
            policy=GoalPolicy(
                replan_strategy=ReplanStrategy.EVERY_ACTION,
                max_replans=999,
            ),
        )
        assert g.policy.replan_strategy is ReplanStrategy.EVERY_ACTION
        assert g.policy.max_replans == 999

    def test_conditions_default_to_empty(self) -> None:
        assert dict(NirvanaGoal().conditions) == {}


# ---------------------------------------------------------------------------
# UtilityStrategy
# ---------------------------------------------------------------------------


class TestUtilityStrategyAlgorithm:
    def _strat(self) -> UtilityStrategy:
        return UtilityStrategy()

    def test_returns_one_step_plan(self) -> None:
        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, utility=1.0)
        plan = self._strat().plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"x": True}),
            [a],
        )
        assert plan is not None
        assert len(plan) == 1

    def test_total_cost_matches_chosen_action(self) -> None:
        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, cost=3.5)
        plan = self._strat().plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"x": True}),
            [a],
        )
        assert plan is not None
        assert plan.total_cost == 3.5
        assert plan.score.scalar == 3.5

    def test_expected_state_reflects_action_effects(self) -> None:
        a = ActionSpec(
            name="a",
            preconditions={},
            effects={"x": True, "y": 7},
        )
        plan = self._strat().plan(
            PlanningState.from_dict({"prior": True}),
            GoalSpec(conditions={"x": True}),
            [a],
        )
        assert plan is not None
        end = plan.expected_states[-1].to_dict()
        assert end["x"] is True
        assert end["y"] == 7
        assert end["prior"] is True  # carried over

    def test_metadata_records_applicable_count(self) -> None:
        actions = [
            ActionSpec(name=f"act{i}", preconditions={}, effects={"x": True})
            for i in range(5)
        ]
        plan = self._strat().plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"x": True}),
            actions,
        )
        assert plan is not None
        assert plan.metadata.applicable_count == 5
        # Search-strategy field stays at its default — utility doesn't
        # do tree expansion.
        assert plan.metadata.nodes_explored == 0

    def test_strategy_name(self) -> None:
        assert self._strat().name == "UtilityStrategy"


class TestNetValueResolution:
    def test_static_utility(self) -> None:
        a = ActionSpec(
            name="a", preconditions={}, effects={"x": True}, utility=10.0, cost=2.0
        )
        b = ActionSpec(
            name="b", preconditions={}, effects={"x": True}, utility=5.0, cost=1.0
        )
        plan = UtilityStrategy().plan(
            PlanningState.from_dict({}), GoalSpec(conditions={"x": True}), [a, b]
        )
        assert plan is not None
        # net values: a=8, b=4 → a wins.
        assert plan.actions[0].name == "a"

    def test_callable_utility_receives_state_dict(self) -> None:
        captured: list[Any] = []

        def util(ws: dict[str, Any]) -> float:
            captured.append(dict(ws))
            return 5.0

        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, utility=util)
        UtilityStrategy().plan(
            PlanningState.from_dict({"k": "v"}),
            GoalSpec(conditions={"x": True}),
            [a],
        )
        assert captured == [{"k": "v"}]

    def test_callable_cost_and_utility_compose(self) -> None:
        a = ActionSpec(
            name="a",
            preconditions={},
            effects={"x": True},
            utility=lambda ws: 10.0 + ws.get("bonus", 0),
            cost=lambda ws: 2.0 - ws.get("discount", 0),
        )
        plan = UtilityStrategy().plan(
            PlanningState.from_dict({"bonus": 5, "discount": 1}),
            GoalSpec(conditions={"x": True}),
            [a],
        )
        # net value: (10+5) - (2-1) = 14
        assert plan is not None
        # Cost stored is the resolved cost.
        assert plan.total_cost == 1.0

    def test_none_utility_treated_as_zero(self) -> None:
        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, cost=2.0)
        b = ActionSpec(name="b", preconditions={}, effects={"x": True}, cost=5.0)
        plan = UtilityStrategy().plan(
            PlanningState.from_dict({}), GoalSpec(conditions={"x": True}), [a, b]
        )
        assert plan is not None
        # Net values: a=-2, b=-5 → cheapest a wins.
        assert plan.actions[0].name == "a"


class TestApplicabilityFiltering:
    def test_unsatisfied_precondition_excludes_action(self) -> None:
        a = ActionSpec(
            name="gated",
            preconditions={"k": True},
            effects={"x": True},
        )
        b = ActionSpec(name="open", preconditions={}, effects={"x": True})
        plan = UtilityStrategy().plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"x": True}),
            [a, b],
        )
        assert plan is not None
        assert plan.actions[0].name == "open"

    def test_blacklist_excludes_action(self) -> None:
        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, utility=10.0)
        b = ActionSpec(name="b", preconditions={}, effects={"x": True}, utility=1.0)
        plan = UtilityStrategy().plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"x": True}),
            [a, b],
            blacklisted_actions=["a"],
        )
        assert plan is not None
        assert plan.actions[0].name == "b"

    def test_no_applicable_returns_none(self) -> None:
        a = ActionSpec(name="a", preconditions={"k": True}, effects={"x": True})
        plan = UtilityStrategy().plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"x": True}),
            [a],
        )
        assert plan is None
