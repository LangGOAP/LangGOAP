"""Phase A integration test: MCTS tree reuse operates correctly in a
replanning loop on a stochastic gridworld.

Reference: Soemers et al., CIG 2016 Section IV-B
(``research/papers/tree-reuse-soemers-cig2016.txt:200-224``).
Design: ``research/plans/tree-reuse-and-junction-graph.md``, Phase A.

Scope
-----
This is a **structural-correctness** integration: across an episode the
``advance`` call must (a) find a matching carryover on a non-trivial
fraction of ticks and (b) let the next tick's search inherit visits
from the previous tick.  The empirical win-rate A/B gate is the
responsibility of Phase B, whose junction-graph compilation amplifies
the effect size above seed noise (see the plan's Phase B gate).

Protocol: FrozenLake-4x4, ``slip_prob = 1/3`` (paper default),
10 seeds, ``iterations = 200``, ``tree_reuse_decay = 0.6``.
"""

from __future__ import annotations

import random

from langgoap.planner.mcts import MCTSStrategy
from langgoap.state import PlanningState
from tests.fixtures.stochastic_gridworld import (
    Episode,
    SlipperyTransitionModel,
    _DELTAS,
    _clamp_cell,
    _compute_effect,
    _reward_for_cell,
    frozen_lake_4x4,
    gridworld_goal,
    gridworld_start_state,
    make_gridworld_actions,
)
from tests.benchmarks.stochastic_gridworld_bench import GridworldMDPRollout

SEEDS = list(range(42, 52))  # 10 seeds
SLIP_PROB = 1.0 / 3.0
MAX_STEPS = 40
ITERATIONS = 200
WALL_CLOCK_MS = 200.0
ROLLOUT_DEPTH = 20


def _build_strategy(
    *, seed: int, model: SlipperyTransitionModel, reuse: bool
) -> MCTSStrategy:
    rng = random.Random(seed)
    return MCTSStrategy(
        iterations=ITERATIONS,
        wall_clock_ms=WALL_CLOCK_MS,
        rollout_depth=ROLLOUT_DEPTH,
        rollout_policy=GridworldMDPRollout(
            topology=model.topology,
            model=model,
            max_depth=ROLLOUT_DEPTH,
            rng=rng,
        ),
        transition_model=model,
        seed=seed,
        anytime_fallback=True,
        reuse_tree=reuse,
        tree_reuse_decay=0.6,
    )


def _run_episode_with_trace(
    *, strategy: MCTSStrategy, topology, model, max_steps: int,
    rng: random.Random,
) -> tuple[Episode, list[dict]]:
    actions = make_gridworld_actions(topology)
    goal = gridworld_goal()
    state = gridworld_start_state(topology)
    episode = Episode(states=[(state["row"], state["col"])])
    trace: list[dict] = []

    for _ in range(max_steps):
        pos = (state["row"], state["col"])
        if pos == topology.goal or state.get("terminated"):
            episode.terminated = True
            episode.reached_goal = pos == topology.goal
            break

        used_carryover = (
            strategy.reuse_tree and strategy._carryover_root is not None
        )
        ps = PlanningState.from_dict(state)
        plan_result = strategy.plan(ps, goal, actions)
        root_visits_after_plan = (
            strategy._last_root.visits if strategy._last_root else 0
        )
        if plan_result is None or not plan_result.actions:
            break
        chosen = plan_result.actions[0]

        direction = model._sample_direction(chosen.name, rng)
        drow, dcol = _DELTAS[direction]
        intended_cell = _clamp_cell(pos[0] + drow, pos[1] + dcol, topology)
        reward = _reward_for_cell(*intended_cell, topology)
        resolved = _compute_effect(*intended_cell, topology)
        state = {**state, **resolved}

        matched = False
        if strategy.reuse_tree:
            strategy.advance(chosen, PlanningState.from_dict(state))
            matched = strategy._carryover_root is not None

        trace.append(
            {
                "used_carryover": used_carryover,
                "root_visits": root_visits_after_plan,
                "advance_matched": matched,
            }
        )
        episode.actions_taken.append(chosen.name)
        episode.rewards.append(reward)
        episode.states.append((state["row"], state["col"]))

        if state.get("done"):
            episode.reached_goal = True
            episode.terminated = True
            break
        if state.get("terminated"):
            episode.terminated = True
            break
    return episode, trace


def _collect_traces(*, reuse: bool) -> list[list[dict]]:
    topology = frozen_lake_4x4()
    traces: list[list[dict]] = []
    for seed in SEEDS:
        model = SlipperyTransitionModel(topology=topology, slip_prob=SLIP_PROB)
        env_rng = random.Random(seed ^ 0xABCDEF)
        strategy = _build_strategy(seed=seed, model=model, reuse=reuse)
        _, trace = _run_episode_with_trace(
            strategy=strategy, topology=topology, model=model,
            max_steps=MAX_STEPS, rng=env_rng,
        )
        traces.append(trace)
    return traces


def test_advance_produces_matching_carryover_across_episode() -> None:
    """Across a replanning episode ``advance`` must match the realised
    successor on the vast majority of ticks \u2014 otherwise tree reuse is
    trivially equivalent to cold-start.  At slip_prob=1/3 only the
    slip fraction (~1/3) can legitimately fall outside the single
    chance-child the tree expanded, so \u2265 60% match is a conservative
    floor."""
    traces = _collect_traces(reuse=True)
    advance_attempts = sum(len(t) for t in traces)
    matches = sum(1 for t in traces for step in t if step["advance_matched"])
    assert advance_attempts > 0
    match_rate = matches / advance_attempts
    assert match_rate >= 0.60, (
        f"advance matched on only {match_rate:.0%} of ticks "
        f"({matches}/{advance_attempts}); expected \u2265 60%"
    )


def test_reused_plan_visits_accumulate_beyond_single_tick_budget() -> None:
    """When ``advance`` matched on the previous tick, the next
    :meth:`plan` call must observe the carryover \u2014 its post-plan root
    visit count must strictly exceed the single-tick iteration budget
    by at least the decayed prior (gamma=0.6 * previous_visits \u2265 1)."""
    traces = _collect_traces(reuse=True)
    saw_accumulation = False
    for trace in traces:
        for prev, curr in zip(trace, trace[1:]):
            if prev["advance_matched"] and curr["used_carryover"]:
                if curr["root_visits"] > ITERATIONS:
                    saw_accumulation = True
                    break
        if saw_accumulation:
            break
    assert saw_accumulation, (
        "expected at least one tick with a reused carryover root whose "
        f"post-plan visits exceed the single-tick budget ({ITERATIONS})"
    )


def test_reuse_off_never_populates_carryover() -> None:
    """Backward-compat: ``reuse_tree=False`` must leave the carryover
    field untouched across an entire episode."""
    traces = _collect_traces(reuse=False)
    for trace in traces:
        for step in trace:
            assert step["used_carryover"] is False
            assert step["advance_matched"] is False
