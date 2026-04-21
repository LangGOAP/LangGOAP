"""Failing tests for MCTS rollout policies.

Rollouts estimate a node's value by simulating forward from its state
for a bounded horizon and returning a scalar reward.  Two policies
are shipped in the initial MCTS strategy:

- :class:`RandomRollout` — picks a uniformly random applicable action
  per step.
- :class:`HeuristicRollout` — greedily picks the applicable action
  whose effects minimise ``h(state, goal)``.  Reuses A*'s admissible
  heuristic (count of unsatisfied conditions) so the thesis is
  about *how* the heuristic is consumed (online rollout simulation
  vs. offline best-first search), not whether the heuristic is good.

Reward shaping (shared across both policies): terminal states where
the goal is satisfied earn ``+1.0``; every other leaf scores
``-h(leaf, goal) / max_h`` so deeper / closer leaves beat shallower /
farther ones.  This keeps rewards in the ``[-1.0, +1.0]`` range that
UCB1 is best-conditioned on.
"""

from __future__ import annotations

import random

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.state import PlanningState


@pytest.fixture
def two_step_actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0),
        ActionSpec(name="b", preconditions={"x": 1}, effects={"done": True}, cost=1.0),
        ActionSpec(name="c", preconditions={}, effects={"y": 1}, cost=1.0),
    ]


class TestRandomRollout:
    def test_returns_reward_in_unit_range(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        from langgoap.planner.mcts import RandomRollout

        goal = GoalSpec(conditions={"done": True})
        policy = RandomRollout(max_depth=5, rng=random.Random(7))
        reward = policy.rollout(
            state=PlanningState.from_dict({}),
            goal=goal,
            actions=two_step_actions,
        )
        assert -1.0 <= reward <= 1.0

    def test_terminal_goal_state_returns_plus_one(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        from langgoap.planner.mcts import RandomRollout

        goal = GoalSpec(conditions={"done": True})
        policy = RandomRollout(max_depth=5, rng=random.Random(7))
        reward = policy.rollout(
            state=PlanningState.from_dict({"done": True}),
            goal=goal,
            actions=two_step_actions,
        )
        assert reward == pytest.approx(1.0)

    def test_no_applicable_action_short_circuits(self) -> None:
        from langgoap.planner.mcts import RandomRollout

        # No action applies to the start state.
        actions = [
            ActionSpec(
                name="needs_flag",
                preconditions={"flag": True},
                effects={"done": True},
                cost=1.0,
            ),
        ]
        goal = GoalSpec(conditions={"done": True})
        policy = RandomRollout(max_depth=5, rng=random.Random(7))
        reward = policy.rollout(
            state=PlanningState.from_dict({}),
            goal=goal,
            actions=actions,
        )
        # No progress possible → reward strictly negative.
        assert reward < 0.0

    def test_determinism_under_seed(self, two_step_actions: list[ActionSpec]) -> None:
        from langgoap.planner.mcts import RandomRollout

        goal = GoalSpec(conditions={"done": True})
        p1 = RandomRollout(max_depth=5, rng=random.Random(42))
        p2 = RandomRollout(max_depth=5, rng=random.Random(42))
        r1 = p1.rollout(
            state=PlanningState.from_dict({}),
            goal=goal,
            actions=two_step_actions,
        )
        r2 = p2.rollout(
            state=PlanningState.from_dict({}),
            goal=goal,
            actions=two_step_actions,
        )
        assert r1 == pytest.approx(r2)


class TestHeuristicRollout:
    def test_reaches_two_step_goal_deterministically(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        from langgoap.planner.mcts import HeuristicRollout

        goal = GoalSpec(conditions={"done": True})
        policy = HeuristicRollout(max_depth=5)
        reward = policy.rollout(
            state=PlanningState.from_dict({}),
            goal=goal,
            actions=two_step_actions,
        )
        assert reward == pytest.approx(1.0)

    def test_avoids_irrelevant_action_ahead_of_goal_action(
        self, two_step_actions: list[ActionSpec]
    ) -> None:
        from langgoap.planner.mcts import HeuristicRollout

        # Given actions a (progress), c (distractor), the heuristic
        # rollout must pick ``a`` first because ``c`` doesn't reduce
        # the unsatisfied-condition count.
        goal = GoalSpec(conditions={"done": True})
        policy = HeuristicRollout(max_depth=3)
        reward = policy.rollout(
            state=PlanningState.from_dict({}),
            goal=goal,
            actions=two_step_actions,
        )
        assert reward == pytest.approx(1.0)
