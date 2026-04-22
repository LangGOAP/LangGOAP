"""Failing tests for the chance-node stochastic-MCTS extension.

See ``research/plans/stochastic-mcts-expansion.md``.  The chance-node
design splits expansion into ``Decision \u2192 Chance \u2192 Decision`` layers
so UCB1 statistics at decision nodes reflect the true outcome
distribution sampled via ``TransitionModel.sample`` instead of the
deterministic ``expected`` view.  Backward-compatibility contract:
with ``DeterministicTransitionModel`` the tree stays two-layered
(decision-only) and every existing ``tests/test_mcts_*.py`` test
passes bit-identically.
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
    """``remaining`` decrements by 1 on success, stays put on slip."""

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

class TestChanceNodeAPI:
    def test_chance_node_is_importable(self) -> None:
        from langgoap.planner.mcts import ChanceNode  # noqa: F401

    def test_chance_node_has_visits_value_parent_action(self) -> None:
        from langgoap.planner.mcts import ChanceNode, MCTSNode

        parent = MCTSNode(state=PlanningState.from_dict({}))
        action = _counter_action()
        node = ChanceNode(action=action, parent=parent)
        assert node.visits == 0
        assert node.value == 0.0
        assert node.parent is parent
        assert node.action is action
        assert node.children_by_key == {}


# --- Structural tests ------------------------------------------------

class TestStochasticExpansionInsertsChanceLayer:
    def test_stochastic_model_inserts_chance_node_between_decisions(self) -> None:
        """After one strategy run with a stochastic model the root's
        children must be ChanceNodes, not direct decision grandchildren."""
        from langgoap.planner.mcts import ChanceNode, MCTSStrategy

        model = _SlipCounterModel(slip_p=0.3)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(
            iterations=16,
            rollout_depth=4,
            seed=7,
            transition_model=model,
        )
        plan = strategy.plan(start, goal, [_counter_action()])
        # The plan may be None if 16 iterations didn't hit the goal
        # under high slip, but the tree shape is what we're verifying.
        root = strategy._last_root  # type: ignore[attr-defined]
        assert root is not None
        assert root.chance_children, "stochastic model must populate chance_children"
        assert root.children == [], (
            "stochastic model must not populate the legacy children list"
        )
        # Every chance child must have at least one decision grandchild.
        for ch in root.chance_children:
            assert isinstance(ch, ChanceNode)
            assert ch.children_by_key, "chance node must have at least one sampled successor"
            assert ch.visits > 0, "chance node must receive backup visits"


class TestDeterministicFastPath:
    def test_deterministic_model_skips_chance_layer(self) -> None:
        """With ``DeterministicTransitionModel`` the tree must remain
        two-layered \u2014 ``root.chance_children`` stays empty and
        ``root.children`` is populated as today."""
        from langgoap.planner.mcts import MCTSStrategy

        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"remaining": 3, "done": False})
        strategy = MCTSStrategy(iterations=16, rollout_depth=4, seed=7)
        strategy.plan(start, goal, [_counter_action()])
        root = strategy._last_root  # type: ignore[attr-defined]
        assert root is not None
        assert root.chance_children == []
        assert len(root.children) >= 1


# --- Backup correctness ---------------------------------------------

class TestChanceNodeBackup:
    def test_backup_updates_chance_and_decision_ancestors(self) -> None:
        from langgoap.planner.mcts import ChanceNode, MCTSNode, backpropagate

        start = PlanningState.from_dict({"remaining": 2, "done": False})
        root = MCTSNode(state=start)
        chance = ChanceNode(action=_counter_action(), parent=root)
        root.chance_children.append(chance)
        leaf = MCTSNode(state=start, parent=chance, action=_counter_action())
        chance.children_by_key[hash(("a",))] = leaf

        backpropagate(leaf, reward=1.0)
        backpropagate(leaf, reward=0.0)

        for node in (root, chance, leaf):
            assert node.visits == 2
            assert node.value == pytest.approx(0.5)


# --- Convergence: risky-vs-safe MDP ---------------------------------

@dataclass
class _RiskyMDPModel:
    """``risky`` advances +2 on success, regresses to 0 on slip."""

    slip_p: float
    divergence_policy: Any = None

    def expected(self, state: Mapping[str, Any], action: ActionSpec) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(self, state: Mapping[str, Any], action: ActionSpec, rng: random.Random) -> Mapping[str, Any]:
        if action.name == "risky" and rng.random() < self.slip_p:
            return {"pos": 0, "done": False}
        return action.get_effects(dict(state))


def _risky_vs_safe_actions() -> list[ActionSpec]:
    def _safe(state: Mapping[str, Any]) -> Mapping[str, Any]:
        p = int(state.get("pos", 0)) + 1
        return {"pos": p, "done": p >= 4}

    def _risky(state: Mapping[str, Any]) -> Mapping[str, Any]:
        p = int(state.get("pos", 0)) + 2
        return {"pos": p, "done": p >= 4}

    return [
        ActionSpec(
            name="safe",
            preconditions={},
            effects=_safe,
            effect_keys=frozenset({"pos", "done"}),
            cost=1.0,
        ),
        ActionSpec(
            name="risky",
            preconditions={},
            effects=_risky,
            effect_keys=frozenset({"pos", "done"}),
            cost=1.0,
        ),
    ]


def _pos_progress_heuristic(state: PlanningState, goal: GoalSpec) -> float:
    """Continuous ``pos``-progress head so rollouts distinguish
    intermediate states along the corridor."""
    return max(-1.0, min(0.999, int(state.to_dict().get("pos", 0)) / 4.0 - 0.5))


class TestChanceNodeConvergence:
    def test_chance_node_collects_multiple_sampled_successors(self) -> None:
        """Under ``slip_p = 0.7`` the risky chance-child must collect
        both possible successor states (``pos=2`` from success,
        ``pos=0`` from slip) across repeated visits."""
        from langgoap.planner.mcts import MCTSStrategy

        model = _RiskyMDPModel(slip_p=0.7)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"pos": 0, "done": False})
        strategy = MCTSStrategy(
            iterations=64,
            rollout_depth=4,
            seed=7,
            transition_model=model,
            scalar_heuristic=_pos_progress_heuristic,
        )
        strategy.plan(start, goal, _risky_vs_safe_actions())
        root = strategy._last_root  # type: ignore[attr-defined]
        by_name = {c.action.name: c for c in root.chance_children}
        risky_successors = {
            tuple(sorted(child.state.to_dict().items()))
            for child in by_name["risky"].children_by_key.values()
        }
        assert len(risky_successors) >= 2, (
            f"risky chance-child should aggregate \u22652 sampled successors "
            f"under slip_p=0.7; got {risky_successors}"
        )
        safe_successors = {
            tuple(sorted(child.state.to_dict().items()))
            for child in by_name["safe"].children_by_key.values()
        }
        assert len(safe_successors) == 1, (
            f"safe chance-child must collapse to one successor (no slip); "
            f"got {safe_successors}"
        )

    def test_chance_node_value_weights_toward_sample_distribution(self) -> None:
        """The chance-node's running mean converges toward the
        visit-weighted mean of its decision-children's values \u2014 this
        is the core mathematical guarantee the layer delivers."""
        from langgoap.planner.mcts import MCTSStrategy

        model = _RiskyMDPModel(slip_p=0.7)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"pos": 0, "done": False})
        strategy = MCTSStrategy(
            iterations=512,
            rollout_depth=6,
            seed=7,
            transition_model=model,
            scalar_heuristic=_pos_progress_heuristic,
        )
        strategy.plan(start, goal, _risky_vs_safe_actions())
        root = strategy._last_root  # type: ignore[attr-defined]
        risky = next(c for c in root.chance_children if c.action.name == "risky")
        total = sum(c.visits for c in risky.children_by_key.values())
        expected_mean = sum(
            c.value * c.visits for c in risky.children_by_key.values()
        ) / max(total, 1)
        assert risky.value == pytest.approx(expected_mean, abs=0.02), (
            f"chance-node value {risky.value:.3f} must track its "
            f"decision children's visit-weighted mean {expected_mean:.3f}"
        )

    def test_deterministic_regression_single_chance_child(self) -> None:
        """With ``slip_p = 0.0`` both arms are deterministic; each
        chance child must have exactly one sampled successor."""
        from langgoap.planner.mcts import MCTSStrategy

        model = _RiskyMDPModel(slip_p=0.0)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"pos": 0, "done": False})
        strategy = MCTSStrategy(
            iterations=128,
            rollout_depth=6,
            seed=7,
            transition_model=model,
        )
        strategy.plan(start, goal, _risky_vs_safe_actions())
        root = strategy._last_root  # type: ignore[attr-defined]
        for ch in root.chance_children:
            assert len(ch.children_by_key) == 1, (
                f"deterministic model must produce one sampled successor "
                f"per chance child; action={ch.action.name} got "
                f"{len(ch.children_by_key)}"
            )
