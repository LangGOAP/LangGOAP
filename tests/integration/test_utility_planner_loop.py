"""Integration tests for the utility planner.

Mirrors Embabel's contract on
``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
com/embabel/plan/utility/UtilityPlanner.kt`` and the test scenarios in
``UtilityActionTest.kt`` / ``UtilityPlannerWithStateTest.kt``:

* The utility planner is **greedy and one-step**.  At each tick it
  filters to applicable actions, scores each by
  ``net_value(state) = action.utility(state) - action.cost(state)``,
  and returns the highest-scoring action as a one-action plan.
* A **Nirvana goal** (the absent / never-satisfied goal) lets the
  planner keep moving forward indefinitely, useful for always-on
  agents that should opportunistically react to state changes.
* For a **concrete goal** the planner returns the next best action
  regardless of whether it directly satisfies the goal — the GOAP
  loop's ``ReplanStrategy.EVERY_ACTION`` keeps the planner driving
  the next decision until the observer notices satisfaction or the
  planner runs out of applicable actions.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalPolicy, GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.planner.utility import NirvanaGoal, UtilityStrategy, is_nirvana
from langgoap.state import PlanningState
from langgoap.types import ReplanStrategy

# ---------------------------------------------------------------------------
# Strategy in isolation
# ---------------------------------------------------------------------------


class TestUtilityStrategySelection:
    def test_picks_highest_net_value_action(self) -> None:
        """Among applicable actions, the one with the highest
        ``utility - cost`` is returned as a 1-step plan."""
        cheap = ActionSpec(
            name="cheap",
            preconditions={},
            effects={"a": True},
            cost=1.0,
            utility=2.0,  # net value: 1.0
        )
        valuable = ActionSpec(
            name="valuable",
            preconditions={},
            effects={"b": True},
            cost=5.0,
            utility=20.0,  # net value: 15.0
        )

        strat = UtilityStrategy()
        plan = strat.plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"any": True}),
            [cheap, valuable],
        )

        assert plan is not None
        assert len(plan) == 1
        assert plan.actions[0].name == "valuable"

    def test_default_utility_zero_means_lowest_cost_wins(self) -> None:
        """Without ``utility`` set the strategy falls back to
        ``net_value = -cost`` — cheapest action wins."""
        cheap = ActionSpec(
            name="cheap",
            preconditions={},
            effects={"a": True},
            cost=1.0,
        )
        expensive = ActionSpec(
            name="expensive",
            preconditions={},
            effects={"b": True},
            cost=5.0,
        )

        strat = UtilityStrategy()
        plan = strat.plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"any": True}),
            [cheap, expensive],
        )

        assert plan is not None
        assert plan.actions[0].name == "cheap"

    def test_callable_utility_resolved_against_state(self) -> None:
        """``utility`` may be a callable that depends on world state."""
        peak_action = ActionSpec(
            name="peak_only",
            preconditions={},
            effects={"a": True},
            cost=10.0,
            utility=lambda state: 100.0 if state.get("is_peak") else 0.0,
        )
        baseline = ActionSpec(
            name="baseline",
            preconditions={},
            effects={"b": True},
            cost=2.0,
            utility=5.0,  # net value: 3.0
        )

        strat = UtilityStrategy()

        # Off-peak: peak_only net=−10, baseline net=3 → baseline wins.
        plan = strat.plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"any": True}),
            [peak_action, baseline],
        )
        assert plan is not None and plan.actions[0].name == "baseline"

        # Peak: peak_only net=90, baseline net=3 → peak_only wins.
        plan = strat.plan(
            PlanningState.from_dict({"is_peak": True}),
            GoalSpec(conditions={"any": True}),
            [peak_action, baseline],
        )
        assert plan is not None and plan.actions[0].name == "peak_only"

    def test_no_applicable_action_returns_none(self) -> None:
        """When every action's preconditions fail, the planner returns
        ``None`` — the loop will surface ``no_plan``."""
        gated = ActionSpec(
            name="gated",
            preconditions={"unicorn": True},
            effects={"a": True},
        )
        strat = UtilityStrategy()
        plan = strat.plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"a": True}),
            [gated],
        )
        assert plan is None

    def test_blacklisted_actions_are_excluded(self) -> None:
        """Per the ``PlanningStrategy`` Protocol, blacklisted action
        names must be excluded from selection."""
        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, utility=10.0)
        b = ActionSpec(name="b", preconditions={}, effects={"x": True}, utility=5.0)
        strat = UtilityStrategy()
        plan = strat.plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"x": True}),
            [a, b],
            blacklisted_actions=["a"],
        )
        assert plan is not None
        assert plan.actions[0].name == "b"


# ---------------------------------------------------------------------------
# NirvanaGoal helper
# ---------------------------------------------------------------------------


class TestNirvanaGoal:
    def test_is_nirvana_recognises_helper(self) -> None:
        assert is_nirvana(NirvanaGoal()) is True
        assert is_nirvana(GoalSpec(conditions={"x": True})) is False

    def test_nirvana_goal_is_never_satisfied(self) -> None:
        """A Nirvana goal must not vacuously satisfy on any world state
        (the GOAP observer's ``_is_goal_satisfied`` check would
        otherwise return True for empty conditions and end the run
        before the utility planner gets a turn)."""
        from langgoap.graph.nodes import _is_goal_satisfied

        assert _is_goal_satisfied(NirvanaGoal(), {}) is False
        assert _is_goal_satisfied(NirvanaGoal(), {"anything": True}) is False


# ---------------------------------------------------------------------------
# End-to-end loop
# ---------------------------------------------------------------------------


def _two_action_no_goal_actions() -> list[ActionSpec]:
    """Mirrors Embabel's ``Utility2ActionsNoGoal``: two actions, no
    explicit goal — the agent picks the next best action each tick.

    ``can_rerun=False`` is the natural way to mark "produce X once":
    Embabel handles this implicitly through its blackboard's
    "we already have a Frog" check; LangGOAP's planner auto-blacklists
    completed single-use actions so the utility loop terminates rather
    than spinning on the still-applicable action.
    """
    return [
        ActionSpec(
            name="produce_frog",
            preconditions={},
            effects={"frog_made": True},
            execute=lambda ws: {"frog_made": True, "frog_name": "Kermit"},
            utility=10.0,
            can_rerun=False,
        ),
        ActionSpec(
            name="produce_person",
            preconditions={"frog_made": True},
            effects={"person_made": True},
            execute=lambda ws: {"person_made": True, "person_name": "Kermit"},
            utility=8.0,
            can_rerun=False,
        ),
    ]


class TestUtilityLoop:
    def test_nirvana_goal_runs_until_no_applicable_action(self) -> None:
        """Two actions, no concrete goal: both fire, then no applicable
        action remains → ``no_plan`` (the loop's natural way of saying
        STUCK without error)."""
        graph = GoapGraph(
            _two_action_no_goal_actions(),
            strategy=UtilityStrategy(),
        )

        result = graph.invoke(
            goal=NirvanaGoal(
                policy=GoalPolicy(replan_strategy=ReplanStrategy.EVERY_ACTION)
            ),
            world_state={},
        )

        assert result["status"] == "no_plan"
        executed = [h.action_name for h in result["execution_history"] if h.success]
        assert "produce_frog" in executed
        assert "produce_person" in executed
        assert result["world_state"]["frog_name"] == "Kermit"
        assert result["world_state"]["person_name"] == "Kermit"

    def test_completes_with_satisfiable_goal(self) -> None:
        """Mirrors Embabel ``completes with single explicit
        satisfiable goal``: the utility planner picks the
        right action and the observer detects goal satisfaction."""
        graph = GoapGraph(
            _two_action_no_goal_actions(),
            strategy=UtilityStrategy(),
        )

        result = graph.invoke(
            goal=GoalSpec(
                conditions={"person_made": True},
                policy=GoalPolicy(replan_strategy=ReplanStrategy.EVERY_ACTION),
            ),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        executed = [h.action_name for h in result["execution_history"] if h.success]
        assert executed == ["produce_frog", "produce_person"]


# ---------------------------------------------------------------------------
# Story test: opportunistic personal-shopper agent
# ---------------------------------------------------------------------------


class TestPersonalShopperStory:
    """Scenario for the utility planner's signature use case.

    A grocery-shopper agent has a $50 budget and a pantry to fill.
    Each candidate purchase has a base utility (how much the household
    needs it).  When a flash sale is on, the *sale_item* purchase's
    callable utility spikes — the agent should pick it over the
    baseline non-sale items.

    The agent runs in EVERY_ACTION replan mode under
    :class:`UtilityStrategy`, picking one purchase per tick until the
    pantry is full or no purchase is affordable.
    """

    def _shopper_actions(self) -> list[ActionSpec]:
        def buy(item: str, eff_key: str) -> Any:
            def fn(ws: dict[str, Any]) -> dict[str, Any]:
                spent = float(ws.get("spent", 0.0))
                price = float(ws.get(f"price_{item}", 0.0))
                return {eff_key: True, "spent": spent + price}

            return fn

        return [
            ActionSpec(
                name="buy_milk",
                preconditions={"need_milk": True},
                effects={"milk": True, "spent": 0.0},
                execute=buy("milk", "milk"),
                cost=lambda ws: float(ws.get("price_milk", 4.0)),
                utility=10.0,
                can_rerun=False,
            ),
            ActionSpec(
                name="buy_bread",
                preconditions={"need_bread": True},
                effects={"bread": True, "spent": 0.0},
                execute=buy("bread", "bread"),
                cost=lambda ws: float(ws.get("price_bread", 3.0)),
                utility=8.0,
                can_rerun=False,
            ),
            ActionSpec(
                name="buy_sale_item",
                preconditions={"sale_active": True},
                effects={"sale_item": True, "spent": 0.0},
                execute=buy("sale", "sale_item"),
                cost=lambda ws: float(ws.get("price_sale", 7.0)),
                # Sale item is 50 utility during a sale, near zero off-sale.
                utility=lambda ws: 50.0 if ws.get("sale_active") else 0.5,
                can_rerun=False,
            ),
        ]

    def test_baseline_picks_high_utility_per_dollar(self) -> None:
        """Without a sale, milk (10/4=2.5 net=6) wins over bread
        (8/3=2.67 net=5).  Net values: milk=10−4=6, bread=8−3=5,
        sale=0.5 (no precondition met → excluded)."""
        graph = GoapGraph(
            self._shopper_actions(),
            strategy=UtilityStrategy(),
        )
        result = graph.invoke(
            goal=GoalSpec(
                conditions={"milk": True, "bread": True},
                policy=GoalPolicy(replan_strategy=ReplanStrategy.EVERY_ACTION),
            ),
            world_state={"need_milk": True, "need_bread": True},
        )
        assert result["status"] == "goal_achieved"
        executed = [h.action_name for h in result["execution_history"] if h.success]
        # Higher net value first.
        assert executed == ["buy_milk", "buy_bread"]

    def test_sale_disrupts_priority(self) -> None:
        """When a sale is on, the agent reaches for the sale item first
        even though it's pricier — net value 50−7=43 dwarfs milk (6)
        and bread (5)."""
        graph = GoapGraph(
            self._shopper_actions(),
            strategy=UtilityStrategy(),
        )
        result = graph.invoke(
            goal=NirvanaGoal(
                policy=GoalPolicy(replan_strategy=ReplanStrategy.EVERY_ACTION)
            ),
            world_state={
                "need_milk": True,
                "need_bread": True,
                "sale_active": True,
            },
        )
        assert result["status"] == "no_plan"  # ran out of applicable actions
        executed = [h.action_name for h in result["execution_history"] if h.success]
        assert (
            executed[0] == "buy_sale_item"
        ), f"sale must be picked first, got {executed!r}"
        assert set(executed) == {"buy_sale_item", "buy_milk", "buy_bread"}
