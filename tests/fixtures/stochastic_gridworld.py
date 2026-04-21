"""Stochastic gridworld fixture for the MCTS-vs-A* benchmark.

Implements two topologies (``CliffWalking-v1`` and ``FrozenLake-v1``,
both following Gymnasium conventions) and a
``SlipperyTransitionModel`` that conforms to
:class:`~langgoap.planner.transitions.TransitionModel`.

See ``research/experiments/2026-04-20-mcts-on-stochastic.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from random import Random
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.strategy import PlanningStrategy
from langgoap.state import PlanningState

# Cardinal deltas.  Names kept lowercase so they compose cleanly with
# ``ActionSpec.name`` (which is stringly-typed in the GOAP core).
_DELTAS: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}

# Perpendicular-slip pairs used by :class:`SlipperyTransitionModel`.
# Matches the Gymnasium convention of slipping left-or-right relative
# to the intended direction with equal probability.
_PERPENDICULARS: dict[str, tuple[str, str]] = {
    "north": ("west", "east"),
    "south": ("west", "east"),
    "east": ("north", "south"),
    "west": ("north", "south"),
}


@dataclass(frozen=True, slots=True)
class GridworldTopology:
    """Static terrain definition for a gridworld benchmark cell.

    ``cliffs`` and ``holes`` are mutually exclusive in practice:
    CliffWalking uses the former (non-terminal penalty + teleport back
    to ``start``), FrozenLake uses the latter (terminal, episode ends).
    Reward shape is carried on the topology so the episode runner can
    score transitions without knowing the domain.
    """

    rows: int
    cols: int
    start: tuple[int, int]
    goal: tuple[int, int]
    cliffs: frozenset[tuple[int, int]] = frozenset()
    holes: frozenset[tuple[int, int]] = frozenset()
    step_reward: float = -1.0
    goal_reward: float = 10.0
    cliff_reward: float = -100.0
    hole_reward: float = 0.0


def cliff_walking_4x12() -> GridworldTopology:
    """Gymnasium's ``CliffWalking-v1`` topology (4\u00d712 grid).

    Bottom row: start at ``(3, 0)``, cliff at ``(3, 1..10)``, goal at
    ``(3, 11)``.  Rewards match the canonical Sutton-&-Barto example
    (\u22121 per step, \u2212100 per cliff-fall, +10 on goal).
    """
    return GridworldTopology(
        rows=4,
        cols=12,
        start=(3, 0),
        goal=(3, 11),
        cliffs=frozenset((3, c) for c in range(1, 11)),
    )


def frozen_lake_4x4() -> GridworldTopology:
    """Gymnasium's ``FrozenLake-v1`` default 4\u00d74 map.

    Layout (``S`` = start, ``F`` = frozen, ``H`` = hole, ``G`` = goal)::

        S F F F
        F H F H
        F F F H
        H F F G

    Sparse reward: +1 on goal, 0 everywhere else including holes
    (episode just terminates on hole-entry).
    """
    return GridworldTopology(
        rows=4,
        cols=4,
        start=(0, 0),
        goal=(3, 3),
        holes=frozenset({(1, 1), (1, 3), (2, 3), (3, 0)}),
        step_reward=0.0,
        goal_reward=1.0,
        cliff_reward=0.0,
        hole_reward=0.0,
    )


def _clamp_cell(row: int, col: int, topo: GridworldTopology) -> tuple[int, int]:
    """Clip a proposed cell to the grid bounds."""
    return (
        max(0, min(topo.rows - 1, row)),
        max(0, min(topo.cols - 1, col)),
    )


def _compute_effect(
    row: int, col: int, topo: GridworldTopology
) -> dict[str, Any]:
    """Declared-effect resolver for a single move.

    Teleports back to ``start`` on cliff entry, flags terminal on hole
    entry, flags ``done`` on goal.  A* sees this exact mapping so its
    deterministic-model plan agrees with the environment's rules \u2014 the
    stochasticity comes from :class:`SlipperyTransitionModel.sample`,
    not from these declared effects.
    """
    row, col = _clamp_cell(row, col, topo)
    if (row, col) in topo.cliffs:
        return {
            "row": topo.start[0],
            "col": topo.start[1],
            "done": False,
            "terminated": False,
        }
    if (row, col) == topo.goal:
        return {"row": row, "col": col, "done": True, "terminated": True}
    if (row, col) in topo.holes:
        return {"row": row, "col": col, "done": False, "terminated": True}
    return {"row": row, "col": col, "done": False, "terminated": False}


def _make_move(name: str, topo: GridworldTopology) -> ActionSpec:
    """Build a single cardinal-move ``ActionSpec`` for ``topo``.

    Effects are dynamic: the callable reads the agent's current
    ``row``/``col`` from the world state and returns the post-move
    state after bounds clamping, cliff teleport, and goal/hole
    flagging.  ``effect_keys`` names every key the callable may touch,
    as required by :class:`ActionSpec` for dynamic effects.
    """
    drow, dcol = _DELTAS[name]

    def _eff(state: Mapping[str, Any]) -> Mapping[str, Any]:
        r = int(state.get("row", topo.start[0]))
        c = int(state.get("col", topo.start[1]))
        return _compute_effect(r + drow, c + dcol, topo)

    return ActionSpec(
        name=name,
        preconditions={},
        effects=_eff,
        effect_keys=frozenset({"row", "col", "done", "terminated"}),
        cost=1.0,
    )


def make_gridworld_actions(topo: GridworldTopology) -> list[ActionSpec]:
    """Return the four cardinal moves for ``topo``."""
    return [_make_move(name, topo) for name in _DELTAS]


def gridworld_goal() -> GoalSpec:
    """Canonical goal spec: reach the goal cell (``done == True``)."""
    return GoalSpec(conditions={"done": True})


def gridworld_start_state(topo: GridworldTopology) -> dict[str, Any]:
    """Initial world-state dict consumed by both planners and runner."""
    return {
        "row": topo.start[0],
        "col": topo.start[1],
        "done": False,
        "terminated": False,
    }


@dataclass(frozen=True, slots=True)
class SlipperyTransitionModel:
    """``TransitionModel`` that slips perpendicular to the intended move.

    With probability ``1 - slip_prob`` the action's declared effect is
    realised.  With probability ``slip_prob`` the agent instead moves
    in one of the two perpendicular directions, chosen uniformly.

    ``divergence_policy`` is ``None`` \u2014 this model is strict: the
    *expected* transition matches the action's declared effect exactly.
    Only the *sampled* transition is noisy.
    """

    topology: GridworldTopology
    slip_prob: float
    divergence_policy: None = None

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(
        self,
        state: Mapping[str, Any],
        action: ActionSpec,
        rng: Random,
    ) -> Mapping[str, Any]:
        direction = self._sample_direction(action.name, rng)
        drow, dcol = _DELTAS[direction]
        r = int(state.get("row", self.topology.start[0]))
        c = int(state.get("col", self.topology.start[1]))
        return _compute_effect(r + drow, c + dcol, self.topology)

    def _sample_direction(self, intended: str, rng: Random) -> str:
        """Return the realised direction given the intended one."""
        if intended not in _PERPENDICULARS:
            # Non-cardinal action (future-proofing) \u2014 no slip semantics.
            return intended
        draw = rng.random()
        if draw >= self.slip_prob:
            return intended
        left, right = _PERPENDICULARS[intended]
        # Split the remaining mass 50/50 between the two perpendiculars.
        return left if (draw / max(self.slip_prob, 1e-12)) < 0.5 else right


def _reward_for_cell(
    row: int, col: int, topo: GridworldTopology
) -> float:
    """Reward for stepping onto ``(row, col)`` (before teleport)."""
    if (row, col) in topo.cliffs:
        return topo.cliff_reward
    if (row, col) == topo.goal:
        return topo.goal_reward
    if (row, col) in topo.holes:
        return topo.hole_reward
    return topo.step_reward


@dataclass(slots=True)
class Episode:
    """Trace of one full replanning episode.

    ``states`` stores ``(row, col)`` tuples, one per step including the
    initial position (so ``len(states) == len(actions_taken) + 1``).
    ``total_return`` is the undiscounted sum of ``rewards`` \u2014 the
    primary metric reported by the benchmark harness.
    """

    states: list[tuple[int, int]] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False
    reached_goal: bool = False

    @property
    def total_return(self) -> float:
        return float(sum(self.rewards))


def run_episode(
    *,
    strategy: PlanningStrategy,
    topology: GridworldTopology,
    model: SlipperyTransitionModel,
    max_steps: int,
    rng: Random,
) -> Episode:
    """Drive ``strategy`` through one stochastic-gridworld episode.

    Replans at every step from the current (possibly slipped) state so
    both A* and MCTS get the same per-tick planning budget.  Reward is
    computed from the *slipped* cell the agent stepped onto, not the
    post-teleport cell, so cliff-falls are scored correctly even when
    the declared effect bounces the agent back to ``start``.
    """
    actions = make_gridworld_actions(topology)
    goal = gridworld_goal()
    state = gridworld_start_state(topology)

    episode = Episode(states=[(state["row"], state["col"])])
    for _ in range(max_steps):
        pos = (state["row"], state["col"])
        if pos == topology.goal:
            episode.reached_goal = True
            episode.terminated = True
            break
        if state.get("terminated"):
            episode.terminated = True
            break

        ps = PlanningState.from_dict(state)
        plan_result = strategy.plan(ps, goal, actions)
        if plan_result is None or not plan_result.actions:
            break
        chosen = plan_result.actions[0]

        # Replay the slip outcome explicitly so we can score the
        # pre-teleport cell.  The model\u2019s ``_sample_direction`` is
        # consulted directly; ``sample`` is then invoked to drive the
        # teleport/termination logic on a fresh local state.
        direction = model._sample_direction(chosen.name, rng)
        drow, dcol = _DELTAS[direction]
        intended_cell = _clamp_cell(pos[0] + drow, pos[1] + dcol, topology)
        reward = _reward_for_cell(*intended_cell, topology)
        # Apply the declared-effect resolver for the same intended cell
        # so teleports/terminations propagate into the runner state.
        resolved = _compute_effect(*intended_cell, topology)
        state = {**state, **resolved}

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
    return episode


__all__ = [
    "Episode",
    "GridworldTopology",
    "SlipperyTransitionModel",
    "cliff_walking_4x12",
    "frozen_lake_4x4",
    "gridworld_goal",
    "gridworld_start_state",
    "make_gridworld_actions",
    "run_episode",
]
