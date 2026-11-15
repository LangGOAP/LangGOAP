"""Callables see the live world state, not just the planning slice (issue #30).

The planner hashes and compares only *planning keys* (those that appear in
some precondition, effect or goal), but ``cost=`` callables, dynamic
``effects`` and ``TransitionModel.expected`` / ``sample`` are documented to
receive "the current world state".  ``PlanningState`` therefore carries the
non-planning entries as a *context* that does not take part in equality or
hashing, and every planner hands callables ``state.as_context()``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.planner.mcts import MCTSExploration, MCTSStrategy
from langgoap.state import PlanningState


class TestPlanningStateContext:
    def test_context_is_separate_from_planning_keys(self) -> None:
        state = PlanningState.from_dict(
            {"a": 1, "vip": True, "doc": "x" * 10}, keys={"a"}
        )
        assert state.to_dict() == {"a": 1}
        assert state.as_context() == {"a": 1, "vip": True, "doc": "x" * 10}

    def test_context_does_not_affect_equality_or_hash(self) -> None:
        with_ctx = PlanningState.from_dict({"a": 1, "vip": True}, keys={"a"})
        bare = PlanningState.from_dict({"a": 1})
        assert with_ctx == bare
        assert hash(with_ctx) == hash(bare)
        assert len({with_ctx, bare}) == 1

    def test_apply_keeps_context_and_planning_keys_win(self) -> None:
        state = PlanningState.from_dict({"a": 1, "vip": True}, keys={"a"})
        after = state.apply({"a": 2, "b": 3})
        assert after.to_dict() == {"a": 2, "b": 3}
        assert after.as_context() == {"a": 2, "b": 3, "vip": True}

    def test_unhashable_context_values_are_kept(self) -> None:
        state = PlanningState.from_dict({"a": 1, "docs": ["x", "y"]}, keys={"a"})
        assert state.as_context()["docs"] == ["x", "y"]

    def test_without_keys_everything_is_planning_state(self) -> None:
        state = PlanningState.from_dict({"a": 1, "vip": True})
        assert state.as_context() == state.to_dict() == {"a": 1, "vip": True}


def _tiered_actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="basic", effects={"done": True}, cost=5.0),
        ActionSpec(
            name="premium",
            effects={"done": True},
            cost=lambda ws: 1.0 if ws.get("vip") else 100.0,
        ),
    ]


class TestCallablesSeeTheWorldState:
    def test_cost_function_reads_a_non_planning_key(self) -> None:
        result = GoapGraph(_tiered_actions()).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={"vip": True}
        )
        assert result["status"] == "goal_achieved"
        assert result["plan"].action_names == ["premium"]

    def test_cost_function_without_the_key_picks_the_other_branch(self) -> None:
        result = GoapGraph(_tiered_actions()).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["plan"].action_names == ["basic"]

    def test_dynamic_effects_read_a_non_planning_key(self) -> None:
        actions = [
            ActionSpec(
                name="unlock",
                effects=lambda ws: {"done": ws.get("tier") == "gold"},
                effect_keys=frozenset({"done"}),
                execute=lambda ws: {"done": ws.get("tier") == "gold"},
            )
        ]
        gold = GoapGraph(actions).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={"tier": "gold"}
        )
        assert gold["status"] == "goal_achieved"
        silver = GoapGraph(actions).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={"tier": "silver"}
        )
        assert silver["status"] == "no_plan"

    def test_expected_states_stay_planning_only(self) -> None:
        result = GoapGraph(_tiered_actions()).invoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={"vip": True, "report": "x" * 50},
        )
        for state in result["plan"].expected_states:
            assert "report" not in state.to_dict()
            assert "vip" not in state.to_dict()


@dataclass
class _RecordingModel:
    seen_keys: set[str] = field(default_factory=set)
    divergence_policy: Any = None

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]:
        self.seen_keys.update(state.keys())
        return action.get_effects(dict(state))

    def sample(
        self, state: Mapping[str, Any], action: ActionSpec, rng: random.Random
    ) -> Mapping[str, Any]:
        self.seen_keys.update(state.keys())
        return action.get_effects(dict(state))


class TestTransitionModelSeesTheWorldState:
    def test_mcts_passes_context_to_the_model(self) -> None:
        model = _RecordingModel()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=32, rollout_depth=3, seed=5),
            transition_model=model,
        )
        result = GoapGraph(
            _tiered_actions(), strategy=strategy, transition_model=model
        ).invoke(goal=GoalSpec(conditions={"done": True}), world_state={"vip": True})
        assert result["status"] == "goal_achieved"
        assert "vip" in model.seen_keys
