"""MCTS plans carry *expected* states, not sampled ones (issue #27).

Under a stochastic ``TransitionModel`` the search tree's decision nodes hold
``model.sample(...)`` outcomes.  A plan's ``expected_states`` must instead be
the deterministic replay of ``model.expected`` along the chosen actions,
otherwise the observer's deviation check compares the live world against
planner noise and replans spuriously.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langgoap import ActionSpec, GoalSpec
from langgoap.planner.mcts import MCTSExploration, MCTSStrategy
from langgoap.state import PlanningState


@dataclass
class _NoisyModel:
    """Every sample carries a marker key that the expected view never has."""

    divergence_policy: Any = None

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(
        self, state: Mapping[str, Any], action: ActionSpec, rng: random.Random
    ) -> Mapping[str, Any]:
        effects = dict(action.get_effects(dict(state)))
        effects["noise"] = rng.random()  # sampled outcome, never "expected"
        return effects


def _counter_action() -> ActionSpec:
    def _eff(state: Mapping[str, Any]) -> Mapping[str, Any]:
        rem = max(0, int(state.get("remaining", 0)) - 1)
        return {"remaining": rem, "done": rem <= 0}

    return ActionSpec(
        name="decrement",
        effects=_eff,
        effect_keys=frozenset({"remaining", "done"}),
        cost=1.0,
    )


class TestExpectedStatesAreDeterministicReplay:
    def test_expected_states_never_contain_sampled_keys(self) -> None:
        model = _NoisyModel()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(
                iterations=128, rollout_depth=6, seed=3, wall_clock_ms=0
            ),
            transition_model=model,
        )
        start = PlanningState.from_dict({"remaining": 2, "done": False})
        plan_obj = strategy.plan(
            start, GoalSpec(conditions={"done": True}), [_counter_action()]
        )
        assert plan_obj is not None and len(plan_obj) > 0
        assert all("noise" not in s.to_dict() for s in plan_obj.expected_states)

    def test_expected_states_match_model_expected_replay(self) -> None:
        model = _NoisyModel()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(
                iterations=128, rollout_depth=6, seed=3, wall_clock_ms=0
            ),
            transition_model=model,
        )
        start = PlanningState.from_dict({"remaining": 2, "done": False})
        plan_obj = strategy.plan(
            start, GoalSpec(conditions={"done": True}), [_counter_action()]
        )
        assert plan_obj is not None
        sim = start
        replay = []
        for action in plan_obj.actions:
            sim = sim.apply(dict(model.expected(sim.to_dict(), action)))
            replay.append(sim)
        assert list(plan_obj.expected_states) == replay
        assert replay[-1].satisfies({"done": True})
