"""Failing unit tests for :class:`StochasticRollout` and the
``TransitionModel`` integration into :class:`MCTSStrategy`.

Scope (from ``research/experiments/2026-04-20-mcts-on-stochastic.md``
§8.1 wiring commitments):

- ``StochasticRollout`` mandates a ``TransitionModel`` and advances
  state via ``model.sample(state, action, rng)``.
- ``MCTSStrategy`` accepts a ``transition_model`` kwarg that feeds
  the rollout and tree-expansion paths.
- ``MCTSStrategy`` defaults to the deterministic model, keeping
  existing MCTS tests bit-identical under the new wiring.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.state import PlanningState


@dataclass
class _CountingModel:
    """Test double — counts ``sample`` / ``expected`` calls."""

    sample_calls: int = 0
    expected_calls: int = 0
    divergence_policy: Any = None

    def expected(self, state, action):  # type: ignore[no-untyped-def]
        self.expected_calls += 1
        return action.get_effects(dict(state))

    def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
        self.sample_calls += 1
        return action.get_effects(dict(state))


@dataclass
class _BiasedModel:
    """Test double — ``sample`` diverges from ``expected`` by a fixed
    additive shift on the declared numeric effect."""

    shift: int = 100
    divergence_policy: Any = field(default=None)

    def __post_init__(self) -> None:
        from langgoap.planner.transitions import DivergencePolicy

        self.divergence_policy = DivergencePolicy(reason="test double", kind="other")

    def expected(self, state, action):  # type: ignore[no-untyped-def]
        return action.get_effects(dict(state))

    def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
        declared = dict(action.get_effects(dict(state)))
        return {
            k: (v + self.shift if isinstance(v, int) else v)
            for k, v in declared.items()
        }


class TestStochasticRollout:
    def test_requires_transition_model(self) -> None:
        from langgoap.planner.mcts import StochasticRollout

        with pytest_raises(TypeError):
            StochasticRollout(max_depth=4)  # type: ignore[call-arg]

    def test_advances_state_via_model_sample(self) -> None:
        from langgoap.planner.mcts import StochasticRollout

        model = _CountingModel()
        policy = StochasticRollout(max_depth=3, model=model)
        goal = GoalSpec(conditions={"done": True})
        actions = [
            ActionSpec(name="a", preconditions={}, effects={"done": True}, cost=1.0),
        ]
        start = PlanningState.from_dict({})

        reward = policy.rollout(state=start, goal=goal, actions=actions)

        assert model.sample_calls >= 1
        assert reward == 1.0  # goal reached via sampled transition

    def test_sample_divergence_observable_in_rollout_trajectory(self) -> None:
        """When ``sample`` diverges from ``expected``, rollouts see
        the sampled transition — this is the whole point of MCTS under
        stochastic dynamics."""
        from langgoap.planner.mcts import StochasticRollout

        model = _BiasedModel(shift=100)
        policy = StochasticRollout(max_depth=2, model=model)
        # Goal requires x == 101 (declared effect is x=1, shift=100).
        goal = GoalSpec(conditions={"x": 101})
        actions = [
            ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0),
        ]
        start = PlanningState.from_dict({})

        reward = policy.rollout(state=start, goal=goal, actions=actions)

        # A pure get_effects-based rollout would leave x=1 and miss the
        # goal (reward < 1.0).  The sampled path lands x=101 → +1.0.
        assert reward == 1.0

    def test_rng_is_reproducible_under_same_seed(self) -> None:
        from langgoap.planner.mcts import StochasticRollout
        from langgoap.planner.transitions import DeterministicTransitionModel

        model = DeterministicTransitionModel()
        actions = [
            ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0),
            ActionSpec(name="b", preconditions={}, effects={"y": 1}, cost=1.0),
        ]
        goal = GoalSpec(conditions={"z": 1})  # never satisfied
        start = PlanningState.from_dict({})

        r1 = StochasticRollout(max_depth=5, model=model, rng=random.Random(7)).rollout(
            state=start, goal=goal, actions=actions
        )
        r2 = StochasticRollout(max_depth=5, model=model, rng=random.Random(7)).rollout(
            state=start, goal=goal, actions=actions
        )
        assert r1 == r2


class TestMCTSStrategyTransitionModel:
    def test_defaults_to_deterministic_model(self) -> None:
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )
        from langgoap.planner.transitions import DeterministicTransitionModel

        strat = MCTSStrategy(exploration=MCTSExploration(iterations=4, wall_clock_ms=0))
        assert isinstance(strat.transition_model, DeterministicTransitionModel)

    def test_accepts_custom_model(self) -> None:
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        model = _CountingModel()
        strat = MCTSStrategy(
            exploration=MCTSExploration(iterations=4, wall_clock_ms=0),
            transition_model=model,
        )
        assert strat.transition_model is model

    def test_tree_expansion_uses_model_expected(self) -> None:
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        model = _CountingModel()
        strat = MCTSStrategy(
            exploration=MCTSExploration(iterations=6, wall_clock_ms=0, seed=1),
            transition_model=model,
        )
        goal = GoalSpec(conditions={"done": True})
        actions = [
            ActionSpec(name="a", preconditions={}, effects={"done": True}, cost=1.0),
        ]
        start = PlanningState.from_dict({})

        plan = strat.plan(start, goal, actions)

        assert plan is not None
        # Tree expansion must go through model.expected() at least once.
        assert model.expected_calls >= 1

    def test_auto_selects_stochastic_rollout_for_non_default_model(self) -> None:
        """When a non-deterministic model is passed and no rollout is
        explicitly wired, the strategy should default to
        :class:`StochasticRollout` so ``sample`` is actually exercised."""
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )

        model = _CountingModel()
        strat = MCTSStrategy(
            exploration=MCTSExploration(
                iterations=4, wall_clock_ms=0, rollout_depth=3, seed=1
            ),
            transition_model=model,
        )
        # Two-step goal so rollouts do not terminate on the first
        # expansion \u2014 otherwise ``StochasticRollout.rollout`` exits on
        # the initial satisfied-check before ever calling ``sample``.
        goal = GoalSpec(conditions={"step_a": True, "step_b": True})
        actions = [
            ActionSpec(name="a", preconditions={}, effects={"step_a": True}, cost=1.0),
            ActionSpec(name="b", preconditions={}, effects={"step_b": True}, cost=1.0),
        ]
        start = PlanningState.from_dict({})

        strat.plan(start, goal, actions)

        # A deterministic-model run would leave sample_calls at 0.
        assert model.sample_calls >= 1


def pytest_raises(exc):  # tiny shim so this file has no pytest import at top
    import pytest as _pt

    return _pt.raises(exc)
