"""Failing tests for MCTS tree reuse with decay gamma (Phase A).

Reference: Soemers et al., "Enhancements for Real-Time Monte-Carlo
Tree Search in General Video Game Playing" (CIG 2016), section IV-B
(``research/papers/tree-reuse-soemers-cig2016.txt:200-224``).
Algorithm: between consecutive planning ticks, retain the subtree
rooted in the child corresponding to the action just played, and
multiply every stored visit count and score by gamma in [0, 1].
See ``research/plans/tree-reuse-and-junction-graph.md`` for the
full design.

Backward-compatibility contract: ``reuse_tree=False`` (the default)
MUST leave existing ``tests/test_mcts_*.py`` bit-identical.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.state import PlanningState


# --- Fixtures --------------------------------------------------------

@dataclass
class _SlipCounterModel:
    slip_p: float = 0.3
    divergence_policy: Any = None

    def expected(self, state: Mapping[str, Any], action: ActionSpec) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(self, state: Mapping[str, Any], action: ActionSpec, rng: random.Random) -> Mapping[str, Any]:
        if rng.random() < self.slip_p:
            rem = int(state.get("remaining", 0))
            return {"remaining": rem, "done": False}
        return action.get_effects(dict(state))


def _counter_action() -> ActionSpec:
    def _eff(state: Mapping[str, Any]) -> Mapping[str, Any]:
        rem = max(0, int(state.get("remaining", 0)) - 1)
        return {"remaining": rem, "done": rem <= 0}

    return ActionSpec(
        name="decrement",
        preconditions={},
        effects=_eff,
        effect_keys=frozenset({"remaining", "done"}),
        cost=1.0,
    )


# --- API surface -----------------------------------------------------

class TestTreeReuseAPI:
    def test_strategy_exposes_reuse_tree_flag_default_false(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy
        assert MCTSStrategy().reuse_tree is False

    def test_strategy_exposes_tree_reuse_decay_default_zero_point_six(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy
        assert MCTSStrategy().tree_reuse_decay == pytest.approx(0.6)

    def test_strategy_exposes_advance_method(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy
        assert callable(getattr(MCTSStrategy(), "advance", None))

    def test_strategy_exposes_carryover_root_handle(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy
        strategy = MCTSStrategy()
        assert strategy._carryover_root is None


# --- Backward compatibility -----------------------------------------

class TestReuseOffIsBitIdentical:
    def test_reuse_off_leaves_carryover_unset(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(iterations=32, rollout_depth=4, seed=7)
        strategy.plan(start, goal, [_counter_action()])
        assert strategy.reuse_tree is False
        assert strategy._carryover_root is None


# --- advance() under a stochastic model -----------------------------

class TestAdvancePromotesObservedChildStochastic:
    def test_advance_promotes_matching_chance_successor(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        model = _SlipCounterModel(slip_p=0.3)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(
            iterations=64,
            rollout_depth=4,
            seed=7,
            transition_model=model,
            reuse_tree=True,
            tree_reuse_decay=0.6,
        )
        strategy.plan(start, goal, [_counter_action()])
        root = strategy._last_root
        assert root is not None and root.chance_children, (
            "precondition: stochastic model populates chance_children"
        )
        chance = root.chance_children[0]
        action = chance.action
        observed_decision = next(iter(chance.children_by_key.values()))
        observed_state = observed_decision.state

        strategy.advance(action, observed_state)

        new_root = strategy._carryover_root
        assert new_root is observed_decision
        assert new_root.parent is None

    def test_advance_state_mismatch_falls_back_to_cold_start(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        model = _SlipCounterModel(slip_p=0.3)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(
            iterations=32, rollout_depth=4, seed=7,
            transition_model=model, reuse_tree=True,
        )
        strategy.plan(start, goal, [_counter_action()])
        action = strategy._last_root.chance_children[0].action
        unseen = PlanningState.from_dict({"remaining": 999, "done": False})

        strategy.advance(action, unseen)

        assert strategy._carryover_root is None


# --- advance() under the deterministic fast path --------------------

class TestAdvancePromotesObservedChildDeterministic:
    def test_advance_promotes_matching_decision_child(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(
            iterations=32, rollout_depth=4, seed=7,
            reuse_tree=True, tree_reuse_decay=0.6,
        )
        strategy.plan(start, goal, [_counter_action()])
        root = strategy._last_root
        assert root is not None and root.children, (
            "precondition: deterministic model populates children"
        )
        child = root.children[0]
        action = child.action
        assert action is not None

        strategy.advance(action, child.state)

        assert strategy._carryover_root is child
        assert strategy._carryover_root.parent is None


# --- Decay semantics ------------------------------------------------

class TestDecaySemantics:
    def _build_reused_tree(self, decay: float):
        from langgoap.planner.mcts import MCTSStrategy

        model = _SlipCounterModel(slip_p=0.3)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(
            iterations=64, rollout_depth=4, seed=7,
            transition_model=model, reuse_tree=True,
            tree_reuse_decay=decay,
        )
        strategy.plan(start, goal, [_counter_action()])
        root = strategy._last_root
        chance = root.chance_children[0]
        child = next(iter(chance.children_by_key.values()))
        pre_visits = child.visits
        pre_value = child.value
        strategy.advance(chance.action, child.state)
        return strategy._carryover_root, pre_visits, pre_value

    def test_decay_one_preserves_stats_exactly(self) -> None:
        new_root, pre_visits, pre_value = self._build_reused_tree(1.0)
        assert new_root.visits == pre_visits
        assert new_root.value == pytest.approx(pre_value)

    def test_decay_zero_keeps_structure_resets_visits(self) -> None:
        new_root, _, _ = self._build_reused_tree(0.0)
        assert new_root.visits == 0

    def test_intermediate_decay_shrinks_visits_preserves_mean(self) -> None:
        new_root, pre_visits, pre_value = self._build_reused_tree(0.5)
        assert new_root.visits <= pre_visits
        if pre_visits >= 2:
            assert new_root.visits < pre_visits
        assert new_root.value == pytest.approx(pre_value)


# --- End-to-end reuse across ticks ----------------------------------

class TestReuseAccumulatesStatsAcrossTicks:
    def test_second_plan_reuses_carryover_as_new_root(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        model = _SlipCounterModel(slip_p=0.3)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(
            iterations=64, rollout_depth=4, seed=7,
            transition_model=model, reuse_tree=True,
            tree_reuse_decay=1.0,
        )
        strategy.plan(start, goal, [_counter_action()])
        root1 = strategy._last_root
        chance = root1.chance_children[0]
        observed = next(iter(chance.children_by_key.values()))
        pre_visits = observed.visits
        strategy.advance(chance.action, observed.state)

        strategy.plan(observed.state, goal, [_counter_action()])
        root2 = strategy._last_root
        assert root2 is observed
        assert root2.visits >= pre_visits + 1


# --- advance() before any plan() is a no-op -------------------------

class TestAdvanceBeforePlanIsNoOp:
    def test_advance_with_no_prior_root_leaves_carryover_none(self) -> None:
        from langgoap.planner.mcts import MCTSStrategy

        strategy = MCTSStrategy(reuse_tree=True)
        some_state = PlanningState.from_dict({"x": 1})
        strategy.advance(_counter_action(), some_state)
        assert strategy._carryover_root is None
