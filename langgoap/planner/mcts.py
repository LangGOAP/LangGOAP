"""Monte Carlo Tree Search planning strategy.

Adapted from LATS (``research/repos/LATS/hotpot/lats.py``) with
LangGoap-specific typing: action edges are concrete
:class:`~langgoap.actions.ActionSpec` instances and each node carries
a :class:`~langgoap.state.PlanningState` so the compiled graph can
re-use the standard satisfaction / apply machinery.

The public surface is :class:`MCTSStrategy` (satisfies the
:class:`~langgoap.planner.strategy.PlanningStrategy` Protocol); node
and UCB1 primitives are exported for plan-level integration tests.

Pre-registered experiment:
``research/experiments/2026-04-20-mcts-vs-astar.md``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
from langgoap.state import PlanningState


@dataclass(slots=True)
class MCTSNode:
    """A node in the MCTS search tree.

    Mirrors LATS's ``Node`` with two LangGoap adaptations: ``state`` is
    a :class:`PlanningState` (not a free-text dict) and ``action`` is
    the :class:`~langgoap.actions.ActionSpec` that was applied to the
    parent to reach this state (``None`` at the root).
    """

    state: PlanningState
    parent: "MCTSNode | None" = None
    action: ActionSpec | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    is_terminal: bool = False
    untried_actions: list[ActionSpec] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """Number of edges from the root to this node."""
        d = 0
        cursor: "MCTSNode | None" = self.parent
        while cursor is not None:
            d += 1
            cursor = cursor.parent
        return d


def ucb1(node: MCTSNode, *, c: float = math.sqrt(2)) -> float:
    """Upper Confidence Bound applied to Trees (UCT).

    Canonical formulation matching LATS's ``Node.uct``:
    ``value/visits + c · sqrt(ln(parent.visits) / visits)``.
    Unvisited children return ``+inf`` so selection expands them first.
    """
    if node.visits == 0:
        return math.inf
    parent = node.parent
    if parent is None or parent.visits == 0:
        # No parent statistics yet — fall back to the exploitation term
        # only, as LATS does for the root in the same edge case.
        return node.value / node.visits
    exploit = node.value / node.visits
    explore = c * math.sqrt(math.log(parent.visits) / node.visits)
    return exploit + explore


def backpropagate(node: MCTSNode, *, reward: float) -> None:
    """Walk from ``node`` to the root updating running-mean statistics.

    Matches LATS's ``backpropagate``: each node's ``value`` is the
    cumulative reward divided by ``visits`` after the update, so
    ``value`` always reads as the mean reward across all rollouts
    passing through that node.
    """
    cursor: MCTSNode | None = node
    while cursor is not None:
        cursor.visits += 1
        # Running mean: new_mean = old_mean + (x - old_mean) / n.
        cursor.value = cursor.value + (reward - cursor.value) / cursor.visits
        cursor = cursor.parent


def _heuristic(state: PlanningState, goal: GoalSpec) -> int:
    """Count of goal conditions currently unsatisfied by ``state``.

    Identical in shape to ``langgoap.planner.astar._heuristic`` — kept
    local to avoid a cross-module import cycle (A* imports from
    types / state; MCTS may be composed into pipelines that A* already
    participates in).  The two implementations must agree to keep the
    heuristic-sharing thesis of the Phase 4 experiment honest.
    """
    unsatisfied = 0
    sd = state.to_dict()
    for key, required in goal.conditions.items():
        if sd.get(key) != required:
            unsatisfied += 1
    return unsatisfied


def _applicable_actions(
    state: PlanningState, actions: list[ActionSpec]
) -> list[ActionSpec]:
    return [a for a in actions if state.satisfies(a.preconditions)]


def _shape_reward(state: PlanningState, goal: GoalSpec) -> float:
    """Reward in ``[-1.0, +1.0]`` for a rollout's terminal state.

    Goal satisfied → ``+1.0``.  Otherwise ``-h(state, goal) / max_h``
    where ``max_h`` is the number of goal conditions, so the reward
    is ``0.0`` only when every condition is met and strictly negative
    otherwise.  Keeps UCB1's exploitation term in a well-conditioned
    range and makes deeper-but-failed rollouts preferable to earlier-
    but-failed ones as the tree deepens.
    """
    total = len(goal.conditions)
    if total == 0:
        return 1.0
    unsatisfied = _heuristic(state, goal)
    if unsatisfied == 0:
        return 1.0
    return -float(unsatisfied) / float(total)


@runtime_checkable
class RolloutPolicy(Protocol):
    """Pluggable rollout policy Protocol.

    Implementations simulate forward from ``state`` for at most
    ``max_depth`` steps, returning a scalar reward in the
    ``[-1.0, +1.0]`` range defined by :func:`_shape_reward`.
    """

    def rollout(
        self,
        *,
        state: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> float: ...


@dataclass(slots=True)
class RandomRollout:
    """Baseline control condition — uniformly random applicable action.

    ``rng`` is injected for deterministic reproducibility per the
    pre-registered experiment's rollout-seed protocol.
    """

    max_depth: int = 4
    rng: random.Random = field(default_factory=random.Random)

    def rollout(
        self,
        *,
        state: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> float:
        cursor = state
        for _ in range(self.max_depth):
            if state.satisfies(goal.conditions):
                return 1.0
            applicable = _applicable_actions(cursor, actions)
            if not applicable:
                break
            chosen = self.rng.choice(applicable)
            cursor = cursor.apply(chosen.get_effects(cursor.to_dict()))
            if cursor.satisfies(goal.conditions):
                return 1.0
        return _shape_reward(cursor, goal)


@dataclass(slots=True)
class HeuristicRollout:
    """Greedy rollout: pick the applicable action minimising ``h``.

    Reuses the A* admissible heuristic (count of unsatisfied goal
    conditions).  Ties are broken by action-precondition specificity,
    mirroring A*'s expansion order.
    """

    max_depth: int = 4

    def rollout(
        self,
        *,
        state: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> float:
        cursor = state
        for _ in range(self.max_depth):
            if cursor.satisfies(goal.conditions):
                return 1.0
            applicable = _applicable_actions(cursor, actions)
            if not applicable:
                break

            def _score(a: ActionSpec) -> tuple[int, int]:
                next_state = cursor.apply(a.get_effects(cursor.to_dict()))
                return (_heuristic(next_state, goal), -len(a.preconditions))

            chosen = min(applicable, key=_score)
            cursor = cursor.apply(chosen.get_effects(cursor.to_dict()))
        return _shape_reward(cursor, goal)


@dataclass(slots=True)
class MCTSStrategy:
    """Monte Carlo Tree Search planning strategy.

    Implements the classical four-phase MCTS loop (selection,
    expansion, simulation, backpropagation) over GOAP action edges.
    Adapted from LATS's ``select_node`` / ``expand_node`` /
    ``rollout`` / ``backpropagate`` quartet.

    Budgets:
        - ``iterations``: hard cap on tree iterations per ``plan()`` call.
        - ``wall_clock_ms``: hard cap on wall-clock per call.  Zero
          disables the check.  Whichever budget trips first ends the
          search.

    Knobs:
        - ``c``: UCB1 exploration constant.  LATS default √2.
        - ``rollout_depth``: max simulation horizon per rollout.
        - ``rollout_policy``: injectable :class:`RolloutPolicy`.  If
          ``None``, :class:`HeuristicRollout` is used.
        - ``seed``: base seed for the RNG driving random tie-breaks
          and the default :class:`RandomRollout` fallback.
    """

    iterations: int = 200
    wall_clock_ms: float = 200.0
    c: float = math.sqrt(2)
    rollout_depth: int = 4
    rollout_policy: RolloutPolicy | None = None
    seed: int = 0

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        if start.satisfies(goal.conditions):
            return Plan(
                actions=(),
                expected_states=(),
                total_cost=0.0,
                metadata=PlanMetadata(),
                score=SimpleScore(scalar=0.0),
            )

        filtered_actions = [
            a
            for a in actions
            if not blacklisted_actions or a.name not in blacklisted_actions
        ]
        rng = random.Random(self.seed)
        policy: RolloutPolicy = self.rollout_policy or HeuristicRollout(
            max_depth=self.rollout_depth
        )

        root = MCTSNode(state=start)
        root.untried_actions = _applicable_actions(start, filtered_actions)

        t0 = time.monotonic()
        wall_budget_s = self.wall_clock_ms / 1000.0 if self.wall_clock_ms else 0.0
        iters_done = 0
        for i in range(self.iterations):
            if wall_budget_s and (time.monotonic() - t0) >= wall_budget_s:
                break
            leaf = _select(root, c=self.c)
            leaf = _expand(leaf, goal=goal, actions=filtered_actions, rng=rng)
            reward = policy.rollout(
                state=leaf.state, goal=goal, actions=filtered_actions
            )
            backpropagate(leaf, reward=reward)
            iters_done = i + 1

        plan = _extract_plan(root, goal=goal, start=start)
        if plan is None:
            return None
        elapsed_ms = (time.monotonic() - t0) * 1000
        return Plan(
            actions=plan.actions,
            expected_states=plan.expected_states,
            total_cost=plan.total_cost,
            metadata=PlanMetadata(
                nodes_explored=iters_done,
                planning_time_ms=elapsed_ms,
            ),
            score=SimpleScore(scalar=plan.total_cost),
        )


def _select(node: MCTSNode, *, c: float) -> MCTSNode:
    """Descend via UCB1 until a node with untried actions or no children."""
    cursor = node
    while cursor.untried_actions == [] and cursor.children:
        cursor = max(cursor.children, key=lambda n: ucb1(n, c=c))
    return cursor


def _expand(
    node: MCTSNode,
    *,
    goal: GoalSpec,
    actions: list[ActionSpec],
    rng: random.Random,
) -> MCTSNode:
    """Expand one untried action into a new child, or return ``node``.

    If ``node`` has no untried actions (fully expanded leaf or
    terminal), return it unchanged so the caller rolls out from it.
    """
    if node.state.satisfies(goal.conditions):
        node.is_terminal = True
        return node
    if not node.untried_actions:
        return node
    # Pop a random untried action for diversity under the fixed seed.
    idx = rng.randrange(len(node.untried_actions))
    action = node.untried_actions.pop(idx)
    next_state = node.state.apply(action.get_effects(node.state.to_dict()))
    child = MCTSNode(state=next_state, parent=node, action=action)
    child.untried_actions = _applicable_actions(next_state, actions)
    if next_state.satisfies(goal.conditions):
        child.is_terminal = True
    node.children.append(child)
    return child


def _extract_plan(
    root: MCTSNode, *, goal: GoalSpec, start: PlanningState
) -> Plan | None:
    """Greedy descent by child visits, preferring terminal leaves.

    Matches LATS's plan-extraction heuristic: at each depth pick the
    most-visited child and stop at a terminal (goal-satisfying) node.
    Returns ``None`` if no terminal was ever discovered.
    """
    best_leaf = _find_best_terminal(root)
    if best_leaf is None:
        return None
    # Walk up from the terminal to reconstruct action / state lists.
    rev_actions: list[ActionSpec] = []
    rev_states: list[PlanningState] = []
    cursor: MCTSNode | None = best_leaf
    while cursor is not None and cursor.parent is not None:
        assert cursor.action is not None  # non-root nodes carry an action
        rev_actions.append(cursor.action)
        rev_states.append(cursor.state)
        cursor = cursor.parent
    rev_actions.reverse()
    rev_states.reverse()
    total_cost = 0.0
    sim = start
    for a in rev_actions:
        total_cost += a.get_cost(sim.to_dict())
        sim = sim.apply(a.get_effects(sim.to_dict()))
    return Plan(
        actions=tuple(rev_actions),
        expected_states=tuple(rev_states),
        total_cost=total_cost,
        metadata=PlanMetadata(),
        score=SimpleScore(scalar=total_cost),
    )


def _find_best_terminal(root: MCTSNode) -> MCTSNode | None:
    """Breadth-first search for the shallowest terminal node with the
    highest visit count along the winning branch."""
    best: MCTSNode | None = None
    stack: list[MCTSNode] = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal and (best is None or node.visits > best.visits):
            best = node
        stack.extend(node.children)
    return best


__all__ = [
    "HeuristicRollout",
    "MCTSNode",
    "MCTSStrategy",
    "RandomRollout",
    "RolloutPolicy",
    "backpropagate",
    "ucb1",
]
