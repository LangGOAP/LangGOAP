"""Pre-registered A/B benchmark: MCTS vs A* on stochastic gridworlds.

Runs the protocol declared in
``research/experiments/2026-04-20-mcts-on-stochastic.md``:

- Two domain cells \u2014 ``cliff-walking-4x12`` and ``frozen-lake-4x4``.
- Three arms \u2014 ``astar`` (baseline), ``mcts`` (treatment), and
  ``mcts-random`` (heuristic-quality ablation, only run when stage-1
  MCTS confirms).
- 30 seeds (42\u201371) at slip ``p=0.2``, with the pre-declared Pocock
  extension to 60 seeds triggered on an inconclusive stage 1.

Can be invoked two ways:

* ``uv run python -m tests.benchmarks.stochastic_gridworld_bench`` \u2014
  runs the full pre-registered protocol and writes a structured JSON
  + Markdown summary under ``research/experiments/results/``.
* ``uv run pytest tests/benchmarks/stochastic_gridworld_bench.py`` \u2014
  runs a reduced 5-seed smoke test so the harness itself stays under
  CI budget; the real numbers are produced by the script invocation.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.mcts import (
    MCTSExploration,
    MCTSReuseConfig,
    MCTSStrategy,
    MCTSTracingConfig,
)
from langgoap.planner.strategy import AStarStrategy, PlanningStrategy
from langgoap.state import PlanningState
from tests.fixtures.stochastic_gridworld import (
    Episode,
    GridworldTopology,
    SlipperyTransitionModel,
    cliff_walking_4x12,
    frozen_lake_4x4,
    make_gridworld_actions,
    run_episode,
)

# --- Pre-registered configuration -------------------------------------------

SEEDS_STAGE_1 = list(range(42, 72))  # 30 seeds
SEEDS_STAGE_2 = list(range(72, 102))  # 30 more on Pocock extension
SLIP_PROB = 0.2
MAX_EPISODE_STEPS = 100
ASTAR_BUDGET_MS = 50.0
MCTS_ITERATIONS = 200
MCTS_WALL_CLOCK_MS = 200.0
MCTS_ROLLOUT_DEPTH = 20  # long enough to reach the goal from start cells
# Pocock constant-boundary \u03b1 split.  0.025 + 0.025 keeps family-wise
# false-positive rate \u2264 0.05 across two analyses.
POCOCK_ALPHA_PER_STAGE = 0.025
BOOTSTRAP_SAMPLES = 10000


# --- Rollout that integrates per-step MDP reward ---------------------------


@dataclass(slots=True)
class GridworldMDPRollout:
    """Rollout that returns undiscounted cumulative MDP reward.

    Distinct from the library's :class:`HeuristicRollout` (goal-distance
    shape) and :class:`StochasticRollout` (same shape, sampled transitions):
    this rollout accumulates ``step_reward`` / ``cliff_reward`` /
    ``goal_reward`` / ``hole_reward`` along the sampled trajectory so
    MCTS optimises the experiment's primary dependent variable (mean
    episode return) directly.  Domain-specific by design \u2014 not a
    library-level component yet.
    """

    topology: GridworldTopology
    model: SlipperyTransitionModel
    max_depth: int
    rng: random.Random = field(default_factory=random.Random)

    def rollout(
        self,
        *,
        state: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> float:
        from tests.fixtures.stochastic_gridworld import (  # local import
            _DELTAS,
            _clamp_cell,
            _compute_effect,
            _reward_for_cell,
        )

        cursor = state.to_dict()
        total = 0.0
        for _ in range(self.max_depth):
            if cursor.get("done") or cursor.get("terminated"):
                break
            applicable = [
                a
                for a in actions
                if PlanningState.from_dict(cursor).satisfies(a.preconditions)
            ]
            if not applicable:
                break
            # Kocsis\u2013Szepesv\xe1ri canonical MCTS rollout: uniformly random
            # action choice.  The *tree* is responsible for directed
            # search; the rollout's sole job is an unbiased Monte-Carlo
            # estimate of downstream return under the noise model.
            chosen = self.rng.choice(applicable)
            direction = self.model._sample_direction(chosen.name, self.rng)
            drow, dcol = _DELTAS[direction]
            r = int(cursor.get("row", self.topology.start[0]))
            c = int(cursor.get("col", self.topology.start[1]))
            intended = _clamp_cell(r + drow, c + dcol, self.topology)
            total += _reward_for_cell(*intended, self.topology)
            cursor = {**cursor, **_compute_effect(*intended, self.topology)}
            if cursor.get("done") or cursor.get("terminated"):
                break
        return total


def _heuristic(state: PlanningState, goal: GoalSpec) -> int:
    """Count of goal conditions not yet satisfied."""
    unsatisfied = 0
    sd = state.to_dict()
    for key, required in goal.conditions.items():
        if sd.get(key) != required:
            unsatisfied += 1
    return unsatisfied


# --- Strategy factories -----------------------------------------------------


StrategyFactory = Callable[
    [int, GridworldTopology, SlipperyTransitionModel], PlanningStrategy
]


def astar_factory(
    seed: int, topology: GridworldTopology, model: SlipperyTransitionModel
) -> PlanningStrategy:
    return AStarStrategy(time_budget_ms=ASTAR_BUDGET_MS)


def mcts_factory(
    seed: int, topology: GridworldTopology, model: SlipperyTransitionModel
) -> PlanningStrategy:
    rng = random.Random(seed)
    return MCTSStrategy(
        exploration=MCTSExploration(
            iterations=MCTS_ITERATIONS,
            wall_clock_ms=MCTS_WALL_CLOCK_MS,
            rollout_depth=MCTS_ROLLOUT_DEPTH,
            seed=seed,
        ),
        # MDP replanning loop \u2014 always need a first action.
        reuse=MCTSReuseConfig(anytime_fallback=True),
        rollout_policy=GridworldMDPRollout(
            topology=topology, model=model, max_depth=MCTS_ROLLOUT_DEPTH, rng=rng
        ),
        transition_model=model,
    )


def mcts_random_factory(
    seed: int, topology: GridworldTopology, model: SlipperyTransitionModel
) -> PlanningStrategy:
    from langgoap.planner.mcts import RandomRollout

    return MCTSStrategy(
        exploration=MCTSExploration(
            iterations=MCTS_ITERATIONS,
            wall_clock_ms=MCTS_WALL_CLOCK_MS,
            rollout_depth=MCTS_ROLLOUT_DEPTH,
            seed=seed,
        ),
        reuse=MCTSReuseConfig(anytime_fallback=True),
        rollout_policy=RandomRollout(
            max_depth=MCTS_ROLLOUT_DEPTH, rng=random.Random(seed)
        ),
        transition_model=model,
    )


STRATEGIES: dict[str, StrategyFactory] = {
    "astar": astar_factory,
    "mcts": mcts_factory,
    "mcts-random": mcts_random_factory,
}


# --- Cell runner -----------------------------------------------------------


def run_cell(
    *,
    topology: GridworldTopology,
    strategy_name: str,
    seeds: list[int],
    max_steps: int = MAX_EPISODE_STEPS,
    slip_prob: float = SLIP_PROB,
) -> list[Episode]:
    """Run ``strategy_name`` across ``seeds`` on a single domain cell."""
    episodes: list[Episode] = []
    factory = STRATEGIES[strategy_name]
    for seed in seeds:
        model = SlipperyTransitionModel(topology=topology, slip_prob=slip_prob)
        # Environment RNG is seeded separately from the strategy RNG so
        # the same noise draws are observed across strategies under the
        # same ``seed`` \u2014 paired comparison, variance reduction.
        env_rng = random.Random(seed)
        strategy = factory(seed, topology, model)
        episode = run_episode(
            strategy=strategy,
            topology=topology,
            model=model,
            max_steps=max_steps,
            rng=env_rng,
        )
        episodes.append(episode)
    return episodes


# --- Summary statistics & inference ----------------------------------------


@dataclass(frozen=True, slots=True)
class CellSummary:
    cell: str
    strategy: str
    seeds: int
    mean_return: float
    std_return: float
    goal_reach_rate: float
    cliff_fall_rate: float
    hole_rate: float
    mean_episode_length: float


def summarise(
    *, cell: str, strategy: str, topology: GridworldTopology, episodes: list[Episode]
) -> CellSummary:
    returns = [e.total_return for e in episodes]
    return CellSummary(
        cell=cell,
        strategy=strategy,
        seeds=len(episodes),
        mean_return=statistics.mean(returns),
        std_return=statistics.stdev(returns) if len(returns) > 1 else 0.0,
        goal_reach_rate=sum(1 for e in episodes if e.reached_goal) / len(episodes),
        cliff_fall_rate=sum(
            1
            for e in episodes
            for r in e.rewards
            if r == topology.cliff_reward and topology.cliffs
        )
        / len(episodes),
        hole_rate=sum(
            1
            for e in episodes
            if e.terminated and not e.reached_goal and topology.holes
        )
        / len(episodes),
        mean_episode_length=statistics.mean(len(e.actions_taken) for e in episodes),
    )


# --- Statistical inference --------------------------------------------------


def welch_t_test(treatment: list[float], baseline: list[float]) -> dict[str, float]:
    """One-sided Welch's t-test, H\u2081: mean(treatment) > mean(baseline).

    Degrees of freedom approximated by the Welch\u2013Satterthwaite formula.
    P-value computed via the Normal CDF, which is accurate to \u226410\u207b\u00b3
    for df \u2265 30 (the pre-registered stage-1 sample size).
    """
    mt, mb = statistics.mean(treatment), statistics.mean(baseline)
    vt = statistics.variance(treatment) if len(treatment) > 1 else 0.0
    vb = statistics.variance(baseline) if len(baseline) > 1 else 0.0
    nt, nb = len(treatment), len(baseline)
    se = math.sqrt(vt / nt + vb / nb)
    if se == 0.0:
        return {"t": 0.0, "df": float(nt + nb - 2), "p_one_sided": 0.5}
    t = (mt - mb) / se
    denom = (vt / nt) ** 2 / max(nt - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1)
    df = (vt / nt + vb / nb) ** 2 / denom if denom > 0 else float(nt + nb - 2)
    # Normal CDF approximation (valid for df \u2265 30).
    from statistics import NormalDist

    p = 1.0 - NormalDist().cdf(t)
    return {"t": t, "df": df, "p_one_sided": p}


def bootstrap_ci(
    treatment: list[float],
    baseline: list[float],
    *,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI on ``mean(treatment) - mean(baseline)``."""
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        rt = [rng.choice(treatment) for _ in treatment]
        rb = [rng.choice(baseline) for _ in baseline]
        diffs.append(statistics.mean(rt) - statistics.mean(rb))
    diffs.sort()
    lo = diffs[int(n_bootstrap * alpha / 2)]
    hi = diffs[int(n_bootstrap * (1 - alpha / 2))]
    return (lo, hi)


# --- Verdict logic ----------------------------------------------------------


_EFFECT_THRESHOLDS = {
    "cliff_walking_4x12": 5.0,  # +5 mean-return units \u2248 1 cliff-fall saved
    "frozen_lake_4x4": 0.10,  # +10% goal-reach probability
}


def cell_verdict(
    *,
    cell: str,
    treatment_eps: list[Episode],
    baseline_eps: list[Episode],
    alpha: float = POCOCK_ALPHA_PER_STAGE,
) -> dict[str, Any]:
    treatment_returns = [e.total_return for e in treatment_eps]
    baseline_returns = [e.total_return for e in baseline_eps]
    treatment_goal = sum(1 for e in treatment_eps if e.reached_goal) / len(
        treatment_eps
    )
    baseline_goal = sum(1 for e in baseline_eps if e.reached_goal) / len(baseline_eps)
    delta = statistics.mean(treatment_returns) - statistics.mean(baseline_returns)
    t_test = welch_t_test(treatment_returns, baseline_returns)
    ci_lo, ci_hi = bootstrap_ci(treatment_returns, baseline_returns)
    effect_threshold = _EFFECT_THRESHOLDS.get(cell, 0.0)
    # Stage decision per \u00a76 of the pre-registration.
    passes_effect = delta >= effect_threshold
    passes_sig = t_test["p_one_sided"] < alpha
    passes_goal_rate = treatment_goal >= baseline_goal
    ci_crosses_zero = ci_lo <= 0.0 <= ci_hi
    if passes_effect and passes_sig and passes_goal_rate:
        verdict = "confirmed"
    elif ci_crosses_zero:
        verdict = "inconclusive"
    else:
        verdict = "disconfirmed"
    return {
        "cell": cell,
        "delta_mean_return": delta,
        "effect_threshold": effect_threshold,
        "welch_t": t_test,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "treatment_goal_rate": treatment_goal,
        "baseline_goal_rate": baseline_goal,
        "verdict": verdict,
    }


# --- Main entrypoint --------------------------------------------------------


CELLS: dict[str, Callable[[], GridworldTopology]] = {
    "cliff_walking_4x12": cliff_walking_4x12,
    "frozen_lake_4x4": frozen_lake_4x4,
}


def run_benchmark(
    seeds: list[int] = SEEDS_STAGE_1,
    *,
    include_ablation: bool = False,
) -> dict[str, Any]:
    """Run the full two-cell A/B and return a structured result dict."""
    t0 = time.monotonic()
    results: dict[str, Any] = {
        "config": {
            "seeds": seeds,
            "slip_prob": SLIP_PROB,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "astar_budget_ms": ASTAR_BUDGET_MS,
            "mcts_iterations": MCTS_ITERATIONS,
            "mcts_wall_clock_ms": MCTS_WALL_CLOCK_MS,
            "mcts_rollout_depth": MCTS_ROLLOUT_DEPTH,
            "alpha_per_stage": POCOCK_ALPHA_PER_STAGE,
        },
        "cells": {},
    }
    for cell_name, cell_factory in CELLS.items():
        topology = cell_factory()
        cell_block: dict[str, Any] = {"summaries": {}}
        per_strategy_eps: dict[str, list[Episode]] = {}
        arms = ["astar", "mcts"] + (["mcts-random"] if include_ablation else [])
        for strategy in arms:
            eps = run_cell(topology=topology, strategy_name=strategy, seeds=seeds)
            per_strategy_eps[strategy] = eps
            summary = summarise(
                cell=cell_name, strategy=strategy, topology=topology, episodes=eps
            )
            cell_block["summaries"][strategy] = asdict(summary)
        cell_block["verdict"] = cell_verdict(
            cell=cell_name,
            treatment_eps=per_strategy_eps["mcts"],
            baseline_eps=per_strategy_eps["astar"],
        )
        results["cells"][cell_name] = cell_block
    results["wall_clock_s"] = time.monotonic() - t0
    return results


def _format_markdown(results: dict[str, Any]) -> str:
    lines = ["# Stochastic Gridworld Benchmark \u2014 Results\n"]
    lines.append(f"Seeds: {len(results['config']['seeds'])}")
    lines.append(f"Slip prob: {results['config']['slip_prob']}")
    lines.append(f"Wall clock: {results['wall_clock_s']:.1f}s\n")
    for cell_name, cell_block in results["cells"].items():
        lines.append(f"## {cell_name}\n")
        lines.append(
            "| Strategy | mean\u00b1std return | goal% | cliff% | hole% | mean len |"
        )
        lines.append(
            "|----------|-----------------|-------|--------|-------|----------|"
        )
        for name, s in cell_block["summaries"].items():
            lines.append(
                f"| {name} | {s['mean_return']:.2f}\u00b1{s['std_return']:.2f} "
                f"| {100 * s['goal_reach_rate']:.0f}% "
                f"| {100 * s['cliff_fall_rate']:.1f}% "
                f"| {100 * s['hole_rate']:.0f}% "
                f"| {s['mean_episode_length']:.1f} |"
            )
        v = cell_block["verdict"]
        lines.append(
            f"\n**Verdict:** {v['verdict']} "
            f"(\u0394={v['delta_mean_return']:.2f}, "
            f"threshold={v['effect_threshold']}, "
            f"p={v['welch_t']['p_one_sided']:.4f}, "
            f"CI\u2085={v['bootstrap_ci_95'][0]:.2f}\u2026{v['bootstrap_ci_95'][1]:.2f})\n"
        )
    return "\n".join(lines)


def main() -> None:
    results = run_benchmark(SEEDS_STAGE_1, include_ablation=False)
    # Check for inconclusive cells \u2014 trigger pre-declared extension.
    inconclusive = [
        c
        for c, block in results["cells"].items()
        if block["verdict"]["verdict"] == "inconclusive"
    ]
    if inconclusive:
        extended = run_benchmark(SEEDS_STAGE_1 + SEEDS_STAGE_2, include_ablation=False)
        results["stage_2"] = extended
    # Research artifacts live at the *top-level* ``research/`` tree
    # (outside the git repo), not under ``langgoap/``.  The harness may
    # be invoked from either the workspace root or from inside
    # ``langgoap/`` \u2014 resolve both cases here.
    cwd = Path.cwd()
    if (cwd / "research").is_dir():
        base = cwd
    elif (cwd.parent / "research").is_dir():
        base = cwd.parent
    else:
        base = cwd
    out_dir = (
        base / "research" / "experiments" / "results" / "2026-04-20-mcts-on-stochastic"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "summary.md").write_text(_format_markdown(results))
    print(_format_markdown(results))


# --- Pytest smoke test ------------------------------------------------------


def test_harness_runs_end_to_end_on_smoke_seeds() -> None:
    """5-seed smoke test verifying the harness plumbs through cleanly.

    Does NOT assert direction of the effect \u2014 that's the job of
    ``main()`` against the full pre-registered seed matrix.
    """
    results = run_benchmark(list(range(42, 47)), include_ablation=False)
    for cell, block in results["cells"].items():
        assert "verdict" in block
        assert set(block["summaries"]) == {"astar", "mcts"}
        for name, s in block["summaries"].items():
            assert s["seeds"] == 5


if __name__ == "__main__":
    main()
