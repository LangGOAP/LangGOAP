"""Failing unit tests for the stochastic-gridworld fixture.

Fixtures live in ``tests/fixtures/stochastic_gridworld.py`` and are
consumed by the pre-registered A/B harness under
``tests/benchmarks/``.  This file covers the fixture's own correctness
contracts before that benchmark runs, per Phase 5.3.

Reference: ``research/experiments/2026-04-20-mcts-on-stochastic.md``.
"""

from __future__ import annotations

import random

import pytest

from langgoap.actions import ActionSpec
from langgoap.planner.transitions import TransitionModel


class TestTopologies:
    def test_cliff_walking_shape_and_corners(self) -> None:
        from tests.fixtures.stochastic_gridworld import cliff_walking_4x12

        topo = cliff_walking_4x12()
        assert topo.rows == 4
        assert topo.cols == 12
        assert topo.start == (3, 0)
        assert topo.goal == (3, 11)
        # Cliff spans the bottom row between start and goal, exclusive.
        expected_cliffs = {(3, c) for c in range(1, 11)}
        assert set(topo.cliffs) == expected_cliffs
        assert topo.holes == frozenset()

    def test_frozen_lake_shape_matches_gymnasium_default(self) -> None:
        from tests.fixtures.stochastic_gridworld import frozen_lake_4x4

        topo = frozen_lake_4x4()
        assert topo.rows == 4
        assert topo.cols == 4
        assert topo.start == (0, 0)
        assert topo.goal == (3, 3)
        # Default Gymnasium ``4x4`` map: F F F F / F H F H / F F F H / H F F G.
        assert set(topo.holes) == {(1, 1), (1, 3), (2, 3), (3, 0)}
        assert topo.cliffs == frozenset()

    def test_goal_reward_matches_domain_convention(self) -> None:
        from tests.fixtures.stochastic_gridworld import (
            cliff_walking_4x12,
            frozen_lake_4x4,
        )

        assert cliff_walking_4x12().goal_reward == 10.0
        assert frozen_lake_4x4().goal_reward == 1.0


class TestGridworldActions:
    def test_four_cardinal_actions(self) -> None:
        from tests.fixtures.stochastic_gridworld import (
            cliff_walking_4x12,
            make_gridworld_actions,
        )

        actions = make_gridworld_actions(cliff_walking_4x12())
        assert {a.name for a in actions} == {"north", "south", "east", "west"}

    def test_move_respects_grid_bounds(self) -> None:
        from tests.fixtures.stochastic_gridworld import (
            cliff_walking_4x12,
            make_gridworld_actions,
        )

        by_name = {a.name: a for a in make_gridworld_actions(cliff_walking_4x12())}
        # Agent at top-left corner; pressing 'north' or 'west' is a no-op.
        top_left = {"row": 0, "col": 0}
        north_eff = dict(by_name["north"].get_effects(top_left))
        west_eff = dict(by_name["west"].get_effects(top_left))
        assert north_eff["row"] == 0
        assert west_eff["col"] == 0

    def test_cliff_teleports_to_start(self) -> None:
        from tests.fixtures.stochastic_gridworld import (
            cliff_walking_4x12,
            make_gridworld_actions,
        )

        topo = cliff_walking_4x12()
        by_name = {a.name: a for a in make_gridworld_actions(topo)}
        # From (2, 1) stepping south lands on a cliff cell \u2192 declared
        # effect teleports back to the start.
        eff = dict(by_name["south"].get_effects({"row": 2, "col": 1}))
        assert (eff["row"], eff["col"]) == topo.start


class TestSlipperyTransitionModel:
    def test_conforms_to_transition_model_protocol(self) -> None:
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            cliff_walking_4x12,
        )

        model = SlipperyTransitionModel(
            topology=cliff_walking_4x12(), slip_prob=0.2
        )
        assert isinstance(model, TransitionModel)

    def test_expected_equals_declared_effects(self) -> None:
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            cliff_walking_4x12,
            make_gridworld_actions,
        )

        topo = cliff_walking_4x12()
        model = SlipperyTransitionModel(topology=topo, slip_prob=0.2)
        action = next(
            a for a in make_gridworld_actions(topo) if a.name == "north"
        )
        state = {"row": 2, "col": 3}
        assert dict(model.expected(state, action)) == dict(
            action.get_effects(state)
        )

    def test_sample_slip_frequency_matches_p(self) -> None:
        """Over many samples of a non-border move, the slip rate should
        concentrate around the declared ``slip_prob``."""
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            cliff_walking_4x12,
            make_gridworld_actions,
        )

        topo = cliff_walking_4x12()
        model = SlipperyTransitionModel(topology=topo, slip_prob=0.2)
        north = next(a for a in make_gridworld_actions(topo) if a.name == "north")
        state = {"row": 2, "col": 5}  # interior, no cliff risk
        rng = random.Random(42)

        slips = 0
        N = 2000
        for _ in range(N):
            sampled = dict(model.sample(state, north, rng))
            if (sampled["row"], sampled["col"]) != (1, 5):  # intended
                slips += 1
        # 95% CI around 0.2 over 2000 draws is roughly \u00b10.02.
        assert 0.16 <= slips / N <= 0.24

    def test_sample_is_reproducible_under_fixed_seed(self) -> None:
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            cliff_walking_4x12,
            make_gridworld_actions,
        )

        topo = cliff_walking_4x12()
        model = SlipperyTransitionModel(topology=topo, slip_prob=0.2)
        north = next(a for a in make_gridworld_actions(topo) if a.name == "north")
        state = {"row": 2, "col": 5}

        draws_a = [
            dict(model.sample(state, north, random.Random(7))) for _ in range(5)
        ]
        draws_b = [
            dict(model.sample(state, north, random.Random(7))) for _ in range(5)
        ]
        assert draws_a == draws_b


class TestEpisodeRunner:
    def test_deterministic_cliff_walking_round_trip(self) -> None:
        """With ``slip_prob=0.0`` the top-safe path (up 1 \u2192 right 11 \u2192
        down 1) should reach the goal and accrue ``13 * step_reward +
        goal_reward`` of return."""
        from langgoap.planner.strategy import AStarStrategy
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            cliff_walking_4x12,
            run_episode,
        )

        topo = cliff_walking_4x12()
        model = SlipperyTransitionModel(topology=topo, slip_prob=0.0)
        episode = run_episode(
            strategy=AStarStrategy(),
            topology=topo,
            model=model,
            max_steps=30,
            rng=random.Random(0),
        )

        assert episode.terminated is True
        assert episode.reached_goal is True
        # A* under p=0.0 picks the shortest deterministic path (13 steps
        # \u2212 one up, eleven right, one down).  Return =
        # 12 \xb7 step_reward + goal_reward = 12 \xb7 (\u22121) + 10 = \u22122.
        assert len(episode.actions_taken) == 13
        assert episode.total_return == pytest.approx(-2.0)

    def test_max_steps_truncates_non_terminating_episode(self) -> None:
        """``max_steps`` is a hard ceiling even when the agent loops."""
        from langgoap.planner.strategy import AStarStrategy
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            cliff_walking_4x12,
            run_episode,
        )

        topo = cliff_walking_4x12()
        model = SlipperyTransitionModel(topology=topo, slip_prob=0.0)
        episode = run_episode(
            strategy=AStarStrategy(),
            topology=topo,
            model=model,
            max_steps=3,  # not enough to reach the goal
            rng=random.Random(0),
        )
        assert episode.reached_goal is False
        assert len(episode.actions_taken) == 3

    def test_cliff_fall_accrues_cliff_reward_and_teleports(self) -> None:
        """With slip forced to 100% perpendicular, a plan that walks
        along row 2 will occasionally slip south into the cliff and
        pay the ``cliff_reward``."""
        from langgoap.planner.strategy import AStarStrategy
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            cliff_walking_4x12,
            run_episode,
        )

        topo = cliff_walking_4x12()
        # ``slip_prob=1.0`` guarantees slip every step \u2014 the agent's
        # intended path never plays out.  Over a short episode the
        # cliff is hit at least once.
        model = SlipperyTransitionModel(topology=topo, slip_prob=1.0)
        episode = run_episode(
            strategy=AStarStrategy(),
            topology=topo,
            model=model,
            max_steps=20,
            rng=random.Random(1),
        )
        assert any(r == topo.cliff_reward for r in episode.rewards), (
            f"expected at least one cliff-fall under p=1.0; rewards={episode.rewards}"
        )
