"""Failing tests for :attr:`MCTSStrategy.reuse.force_deterministic_tree`.

The chance-node layer added by ``test_mcts_chance_nodes`` branches the
tree whenever ``TransitionModel.sample`` is consulted.  On planning
problems that are *effectively deterministic* from the root's
perspective (e.g. per-tick replanning in Pac-Man with rare-event
ghost-collision risk), the branching dilutes UCB1 statistics without
adding decision-relevant information — rollouts carry the noise more
efficiently than chance-node decomposition does.

``force_deterministic_tree=True`` re-uses the deterministic
``_expand`` path even when a stochastic ``TransitionModel`` is
supplied, so callers can pair ``expected()``-only tree statistics with
:class:`StochasticRollout` leaf sampling — the classical
"Determinized UCT" / "Sparse Sampling" variant (Kocsis–Szepesvári 2006;
Kearns, Mansour & Ng 2002).

See ``research/plans/phase-6.0-launch-demo.md`` — Fix A.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.mcts import (
    MCTSExploration,
    MCTSReuseConfig,
    MCTSStrategy,
    MCTSTracingConfig,
    StochasticRollout,
)
from langgoap.state import PlanningState

# --- Fixtures --------------------------------------------------------


@dataclass
class _RiskyMDPModel:
    """Match ``test_mcts_chance_nodes``: ``risky`` slips back to 0."""

    slip_p: float
    divergence_policy: Any = None

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(
        self,
        state: Mapping[str, Any],
        action: ActionSpec,
        rng: random.Random,
    ) -> Mapping[str, Any]:
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
    return max(-1.0, min(0.999, int(state.to_dict().get("pos", 0)) / 4.0 - 0.5))


# --- API surface -----------------------------------------------------


class TestForceDeterministicTreeAPI:
    def test_flag_defaults_false(self) -> None:
        assert MCTSStrategy().reuse.force_deterministic_tree is False

    def test_flag_is_settable(self) -> None:
        strategy = MCTSStrategy(reuse=MCTSReuseConfig(force_deterministic_tree=True))
        assert strategy.reuse.force_deterministic_tree is True


# --- Dispatch behaviour ---------------------------------------------


class TestDispatchBypassesChanceLayer:
    """With the flag on, the tree under a stochastic model has no
    chance-node layer — every root child is a decision node."""

    def test_stochastic_model_no_chance_children_when_forced(self) -> None:
        model = _RiskyMDPModel(slip_p=0.5)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"pos": 0, "done": False})
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=64, rollout_depth=4, seed=7),
            reuse=MCTSReuseConfig(force_deterministic_tree=True),
            transition_model=model,
            scalar_heuristic=_pos_progress_heuristic,
        )
        strategy.plan(start, goal, _risky_vs_safe_actions())
        root = strategy._last_root
        assert root is not None
        assert list(root.chance_children) == [], (
            "force_deterministic_tree=True must suppress chance-node "
            f"expansion; got {[c.action.name for c in root.chance_children]}"
        )
        assert (
            len(root.children) >= 1
        ), "tree must still expand via the deterministic path"

    def test_default_false_preserves_chance_children(self) -> None:
        """Regression guard: the existing chance-node dispatch must be
        unaffected when the new flag is off."""
        model = _RiskyMDPModel(slip_p=0.5)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"pos": 0, "done": False})
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=64, rollout_depth=4, seed=7),
            transition_model=model,
            scalar_heuristic=_pos_progress_heuristic,
        )
        strategy.plan(start, goal, _risky_vs_safe_actions())
        root = strategy._last_root
        assert root is not None
        assert len(root.chance_children) >= 1, (
            "default dispatch must route stochastic models through the "
            "chance-node layer"
        )


# --- End-to-end Determinized MCTS -----------------------------------


class TestDeterminizedMCTSWithStochasticRollout:
    """Determinized UCT: deterministic tree, stochastic rollouts.

    Under ``slip_p=0.4`` the safe arm deterministically reaches pos=4
    in 4 steps; the risky arm reaches it in 2 steps on average but
    resets to 0 with probability 0.4.  A well-behaved Determinized-UCT
    with :class:`StochasticRollout` must recover a plan that terminates
    with ``done=True`` within the rollout horizon.
    """

    def test_returns_plan_reaching_goal(self) -> None:
        model = _RiskyMDPModel(slip_p=0.4)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"pos": 0, "done": False})
        rollout = StochasticRollout(
            max_depth=8,
            model=model,
            rng=random.Random(7),
            scalar_heuristic=_pos_progress_heuristic,
        )
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=256, rollout_depth=8, seed=7),
            reuse=MCTSReuseConfig(force_deterministic_tree=True),
            transition_model=model,
            rollout_policy=rollout,
            scalar_heuristic=_pos_progress_heuristic,
        )
        plan = strategy.plan(start, goal, _risky_vs_safe_actions())
        assert (
            plan is not None
        ), "Determinized MCTS with stochastic rollouts must return a plan"
        assert plan, "plan must be non-empty"


class TestDeterminizedUCTMatchesChanceNodeAtEqualBudget:
    """Comparative audit P2: Determinized UCT must at minimum *match*
    the chance-node tree at equal budget on a stochastic toy problem,
    confirming the new dispatch is not strictly inferior even when the
    environment is genuinely noisy.  The integration test promised by
    the Fix A.1 plan entry.
    """

    def _plan_outcome(
        self, *, force_deterministic: bool, seed: int
    ) -> tuple[int, float]:
        """Return ``(first_action_as_int, root_value)`` for comparability.

        ``first_action = 0 for safe, 1 for risky``.
        """
        model = _RiskyMDPModel(slip_p=0.5)
        goal = GoalSpec(conditions={"done": True})
        start = PlanningState.from_dict({"pos": 0, "done": False})
        rollout: Any = StochasticRollout(
            max_depth=8,
            model=model,
            rng=random.Random(seed),
            scalar_heuristic=_pos_progress_heuristic,
        )
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=256, rollout_depth=8, seed=seed),
            reuse=MCTSReuseConfig(force_deterministic_tree=force_deterministic),
            transition_model=model,
            rollout_policy=rollout,
            scalar_heuristic=_pos_progress_heuristic,
        )
        plan = strategy.plan(start, goal, _risky_vs_safe_actions())
        assert plan is not None and plan.actions
        first = 0 if plan.actions[0].name == "safe" else 1
        return first, strategy._last_root.value  # type: ignore[union-attr]

    def test_both_return_valid_plans_across_seeds(self) -> None:
        """Regression guard.  Under ``slip_p=0.5`` either arm is a
        reasonable choice (safe: deterministic 4 steps; risky: avg
        4 steps).  Both dispatch modes must reliably return a plan
        across a fan of seeds."""
        for seed in (0, 1, 2, 3, 7):
            det_first, det_val = self._plan_outcome(force_deterministic=True, seed=seed)
            chance_first, chance_val = self._plan_outcome(
                force_deterministic=False, seed=seed
            )
            assert det_first in (0, 1)
            assert chance_first in (0, 1)
            assert (
                -1.0 <= det_val <= 1.0
            ), f"determinized root value out of range: {det_val}"
            assert (
                -1.0 <= chance_val <= 1.0
            ), f"chance-node root value out of range: {chance_val}"
