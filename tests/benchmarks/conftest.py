"""Shared fixtures and action-graph factories for the LangGoap benchmark suite.

All builders produce pure synthetic ActionSpecs — no LLM calls, no I/O.
Every factory is deterministic given the same parameters so benchmark results
are reproducible across runs.

Factories
---------
linear_chain(n)
    A depth-n chain: each action enables exactly the next one.
    Models best-case A* (single path, no branching).

wide_pool(n_noise, path_depth)
    A path_depth-step solution buried in n_noise irrelevant actions.
    Stresses the reachability pre-check and sorted-action pruning.

branching(depth, fan_out)
    fan_out^depth paths all leading to the same goal.
    Exercises A* backtracking under alternative-path explosion.
"""

from __future__ import annotations

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.state import PlanningState

# ---------------------------------------------------------------------------
# Action-graph factories
# ---------------------------------------------------------------------------


def linear_chain(n: int) -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    """Return (actions, goal, start) for a linear n-step chain.

    State keys:  step_0 … step_{n-1}  (all bool)
    Goal:        {step_{n-1}: True}
    Start:       {}

    A* will find exactly one path of length n.
    """
    actions: list[ActionSpec] = []
    for i in range(n):
        pre = {f"step_{i - 1}": True} if i > 0 else {}
        actions.append(
            ActionSpec(
                name=f"action_{i}",
                preconditions=pre,
                effects={f"step_{i}": True},
                cost=1.0,
            )
        )
    goal = GoalSpec(conditions={f"step_{n - 1}": True})
    start = PlanningState.from_dict({})
    return actions, goal, start


def wide_pool(
    n_noise: int, path_depth: int = 3
) -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    """Return (actions, goal, start) with a path_depth-step solution and
    n_noise irrelevant actions that cannot contribute to the goal.

    The noise actions require preconditions that are never set in the start
    state, so A*'s reachability pre-check eliminates them cheaply.
    """
    # Build the real path
    path_actions: list[ActionSpec] = []
    for i in range(path_depth):
        pre = {f"path_{i - 1}": True} if i > 0 else {}
        path_actions.append(
            ActionSpec(
                name=f"path_{i}",
                preconditions=pre,
                effects={f"path_{i}": True},
                cost=1.0,
            )
        )

    # Build noise actions (require impossible preconditions)
    noise_actions: list[ActionSpec] = [
        ActionSpec(
            name=f"noise_{j}",
            preconditions={f"impossible_{j}": True},
            effects={f"noise_out_{j}": True},
            cost=0.1,
        )
        for j in range(n_noise)
    ]

    goal = GoalSpec(conditions={f"path_{path_depth - 1}": True})
    start = PlanningState.from_dict({})
    return path_actions + noise_actions, goal, start


def branching(
    depth: int, fan_out: int
) -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    """Return (actions, goal, start) with fan_out^depth alternative paths.

    Each level has fan_out actions advancing to a level-specific flag.
    All paths are equally cheap (cost=1.0) and all reach the same goal.
    This exercises A* breadth-first exploration of many equivalent routes.
    """
    actions: list[ActionSpec] = []
    for level in range(depth):
        for branch in range(fan_out):
            pre: dict[str, bool] = (
                {f"lvl_{level - 1}_b{branch % fan_out}": True} if level > 0 else {}
            )
            actions.append(
                ActionSpec(
                    name=f"lvl_{level}_b{branch}",
                    preconditions=pre,
                    effects={f"lvl_{level}_b{branch}": True},
                    cost=1.0,
                )
            )
    # Goal: any one of the final-level flags (pick branch 0)
    goal = GoalSpec(conditions={f"lvl_{depth - 1}_b0": True})
    start = PlanningState.from_dict({})
    return actions, goal, start


# ---------------------------------------------------------------------------
# Pytest fixtures (re-usable across benchmark modules)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chain_actions_5() -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    return linear_chain(5)


@pytest.fixture(scope="session")
def chain_actions_20() -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    return linear_chain(20)


@pytest.fixture(scope="session")
def wide_100_3() -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    return wide_pool(n_noise=100, path_depth=3)


# ---------------------------------------------------------------------------
# Temporal chain fixtures (with durations + resources)
# ---------------------------------------------------------------------------


def temporal_chain(
    n: int,
) -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    """Linear chain with durations and resource usage on each action.

    Used by temporal benchmark to exercise CP-SAT IntervalVar scheduling.
    """
    from datetime import timedelta

    from langgoap.goals import ConstraintSpec

    actions: list[ActionSpec] = []
    for i in range(n):
        pre = {f"step_{i - 1}": True} if i > 0 else {}
        actions.append(
            ActionSpec(
                name=f"action_{i}",
                preconditions=pre,
                effects={f"step_{i}": True},
                cost=1.0 + (i % 5) * 0.2,
                resources={"cpu": 1.0, "memory_gb": 0.3 + (i % 4) * 0.1},
                duration=timedelta(seconds=1 + (i % 3)),
            )
        )
    goal = GoalSpec(
        conditions={f"step_{n - 1}": True},
        constraints=(
            ConstraintSpec(key="cpu", max=float(n + 10), weight=1.0),
            ConstraintSpec(key="memory_gb", max=float(n), weight=1.0),
        ),
    )
    start = PlanningState.from_dict({})
    return actions, goal, start


@pytest.fixture(scope="session")
def temporal_10() -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    return temporal_chain(10)


@pytest.fixture(scope="session")
def temporal_50() -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    return temporal_chain(50)


@pytest.fixture(scope="session")
def temporal_100() -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    return temporal_chain(100)
