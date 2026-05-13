"""Monte Carlo Tree Search planning strategy.

Adapted from LATS (``research/repos/LATS/hotpot/lats.py``) with
LangGOAP-specific typing: action edges are concrete
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

import logging
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.transitions import (
    DeterministicTransitionModel,
    TransitionModel,
)
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
from langgoap.state import PlanningState

logger = logging.getLogger("langgoap.planner.mcts")


@dataclass(slots=True)
class MCTSNode:
    """A decision node in the MCTS search tree.

    Mirrors LATS's ``Node`` with two LangGOAP adaptations: ``state`` is
    a :class:`PlanningState` (not a free-text dict) and ``action`` is
    the :class:`~langgoap.actions.ActionSpec` that was applied to the
    parent to reach this state (``None`` at the root).

    Under :class:`DeterministicTransitionModel` the tree is two-layered
    and ``children`` holds decision grandchildren directly (legacy
    shape).  Under a non-deterministic model the tree alternates
    ``Decision \u2192 Chance \u2192 Decision`` and the chance layer lives in
    ``chance_children`` \u2014 see :class:`ChanceNode`.
    """

    state: PlanningState
    parent: "MCTSNode | ChanceNode | None" = None
    action: ActionSpec | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    is_terminal: bool = False
    untried_actions: list[ActionSpec] = field(default_factory=list)
    chance_children: list["ChanceNode"] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """Number of decision edges from the root to this node."""
        d = 0
        cursor: "MCTSNode | ChanceNode | None" = self.parent
        while cursor is not None:
            if isinstance(cursor, MCTSNode):
                d += 1
            cursor = cursor.parent
        return d


@dataclass(slots=True)
class ChanceNode:
    """Aggregation layer between a decision node and its sampled
    successor states under a non-deterministic ``TransitionModel``.

    Chance nodes do *not* use UCB1 for child selection \u2014 the "arm" is
    the RNG draw from :meth:`TransitionModel.sample`, not a planner
    choice.  They maintain a state-keyed child dict so repeated samples
    of the same successor collapse onto one decision node and its
    empirical mean converges to the true outcome-averaged value.
    """

    action: ActionSpec
    parent: "MCTSNode"
    children_by_key: dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0


def ucb1(node: "MCTSNode | ChanceNode", *, c: float = math.sqrt(2)) -> float:
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


def backpropagate(node: "MCTSNode | ChanceNode", *, reward: float) -> None:
    """Walk from ``node`` to the root updating running-mean statistics.

    Matches LATS's ``backpropagate``: each node's ``value`` is the
    cumulative reward divided by ``visits`` after the update, so
    ``value`` always reads as the mean reward across all rollouts
    passing through that node.  Chance-layer ancestors are walked
    transparently because :class:`ChanceNode` exposes the same
    ``visits`` / ``value`` / ``parent`` fields as :class:`MCTSNode`.
    """
    cursor: "MCTSNode | ChanceNode | None" = node
    while cursor is not None:
        cursor.visits += 1
        # Running mean: new_mean = old_mean + (x - old_mean) / n.
        cursor.value = cursor.value + (reward - cursor.value) / cursor.visits
        cursor = cursor.parent


def _heuristic(state: PlanningState, goal: GoalSpec) -> int:
    """Count of goal conditions currently unsatisfied by ``state``.

    Identical in shape to ``langgoap.planner.astar._heuristic`` — kept
    local to avoid a cross-module import cycle.  The two
    implementations must agree so A* and MCTS share one heuristic.
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


@runtime_checkable
class ScalarHeuristic(Protocol):
    """Continuous state-value function consumed by MCTS rollouts.

    Maps ``(state, goal)`` onto ``[-1.0, +1.0]`` where higher values
    indicate states closer to satisfaction.  ``+1.0`` is reserved for
    fully satisfied states (set automatically by :func:`_shape_reward`
    when ``state.satisfies(goal.conditions)``), so implementations
    should return strictly less than ``+1.0`` for non-terminal states.

    Used instead of the default goal-condition-count reward when
    supplied to :class:`MCTSStrategy` or a rollout policy; gives UCB1
    a continuous gradient on GOAP problems whose binary condition
    reward would otherwise collapse to a single value.  Not required
    to be admissible (that's an A* property); better-is-higher is
    the only contract.
    """

    def __call__(self, state: PlanningState, goal: GoalSpec) -> float: ...


# Strict upper bar for non-satisfied states; reserves ``+1.0`` for true
# goal satisfaction so UCB1 always prefers solved leaves.
_NON_TERMINAL_UPPER = 0.999


def _shape_reward(
    state: PlanningState,
    goal: GoalSpec,
    *,
    heuristic: ScalarHeuristic | None = None,
) -> float:
    """Reward in ``[-1.0, +1.0]`` for a rollout's terminal state.

    Goal satisfied → ``+1.0`` regardless of ``heuristic``.  Otherwise
    when ``heuristic`` is supplied, its value is clamped to
    ``[-1.0, +0.999]`` and returned; when ``None`` the default
    goal-condition-count shape is used (``-unsatisfied / total``).
    If ``heuristic`` raises, a warning is logged once and the
    default shape is returned for this call.
    """
    if state.satisfies(goal.conditions):
        return 1.0
    if heuristic is not None:
        try:
            value = float(heuristic(state, goal))
        except Exception:
            logger.warning(
                "scalar heuristic raised; falling back to default reward shape",
                exc_info=True,
            )
        else:
            if value > _NON_TERMINAL_UPPER:
                return _NON_TERMINAL_UPPER
            if value < -1.0:
                return -1.0
            return value
    total = len(goal.conditions)
    if total == 0:
        return 1.0
    unsatisfied = _heuristic(state, goal)
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
    scalar_heuristic: ScalarHeuristic | None = None

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
        return _shape_reward(cursor, goal, heuristic=self.scalar_heuristic)


@dataclass(slots=True)
class HeuristicRollout:
    """Greedy rollout: pick the applicable action minimising ``h``.

    Reuses the A* admissible heuristic (count of unsatisfied goal
    conditions).  Ties are broken by action-precondition specificity,
    mirroring A*'s expansion order.  A ``scalar_heuristic`` may be
    supplied to replace the default binary reward shape at rollout
    leaves with a continuous value; action selection during rollout
    continues to use the integer admissible heuristic so the greedy
    choice rule is decoupled from the leaf value function.
    """

    max_depth: int = 4
    scalar_heuristic: ScalarHeuristic | None = None

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
        return _shape_reward(cursor, goal, heuristic=self.scalar_heuristic)


@dataclass(slots=True)
class StochasticRollout:
    """Rollout policy that advances via a :class:`TransitionModel`.

    Distinct from :class:`RandomRollout` / :class:`HeuristicRollout` in
    one dimension: state transitions are sampled through
    ``model.sample(state, action, rng)`` rather than applied directly
    from ``action.get_effects``.  This is the hook that lets MCTS
    observe transition noise (slip, learned-model drift, risk-averse
    pessimism) during simulation.

    Action selection remains greedy on the heuristic of the *expected*
    (not sampled) successor so the policy's bias is orthogonal to
    the noise source.
    """

    max_depth: int
    model: TransitionModel
    rng: random.Random = field(default_factory=random.Random)
    scalar_heuristic: ScalarHeuristic | None = None

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
                # Score successors by *expected* outcome so the rollout's
                # choice rule is noise-free; only the realised transition
                # is sampled.
                expected_effects = self.model.expected(cursor.to_dict(), a)
                next_state = cursor.apply(dict(expected_effects))
                return (_heuristic(next_state, goal), -len(a.preconditions))

            chosen = min(applicable, key=_score)
            sampled_effects = self.model.sample(cursor.to_dict(), chosen, self.rng)
            cursor = cursor.apply(dict(sampled_effects))
        return _shape_reward(cursor, goal, heuristic=self.scalar_heuristic)


@dataclass(frozen=True, slots=True)
class MCTSExploration:
    """Search budget and exploration knobs for :class:`MCTSStrategy`.

    Budgets:
        iterations: Hard cap on tree iterations per ``plan()`` call.
        wall_clock_ms: Hard cap on wall-clock per call.  Zero
            disables the check.  Whichever budget trips first ends
            the search; at least one must be > 0.

    Knobs:
        c: UCB1 exploration constant.  LATS default \u221a2.
        rollout_depth: Max simulation horizon per rollout.
        seed: Base seed for the RNG driving random tie-breaks and
            the default :class:`RandomRollout` fallback.
    """

    iterations: int = 200
    wall_clock_ms: float = 200.0
    c: float = math.sqrt(2)
    rollout_depth: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")
        if self.wall_clock_ms < 0:
            raise ValueError("wall_clock_ms must be >= 0")
        if self.iterations == 0 and self.wall_clock_ms == 0:
            raise ValueError("at least one of iterations / wall_clock_ms must be > 0")
        if self.c <= 0:
            raise ValueError("UCB1 exploration constant c must be > 0")
        if self.rollout_depth < 0:
            raise ValueError("rollout_depth must be >= 0")


@dataclass(frozen=True, slots=True)
class MCTSReuseConfig:
    """Tree-reuse and variable-depth selection config.

    Attributes:
        reuse_tree: Soemers\u2013Winands CIG 2016 \xa7IV-B tree reuse.  When
            ``True``, the subtree rooted at the action played last
            tick is retained across consecutive :meth:`MCTSStrategy.plan`
            calls via :meth:`MCTSStrategy.advance`; stored visit
            counts are multiplied by ``tree_reuse_decay`` before the
            next search.
        tree_reuse_decay: Decay factor in ``[0, 1]`` applied to
            carry-over visits.  Ignored when ``reuse_tree`` is False.
        anytime_fallback: Kocsis\u2013Szepesv\xe1ri "robust" anytime
            fallback: when ``True`` and no goal-terminal leaf was
            discovered, return a partial plan whose first action is
            the most-visited root child.
        path_length_budget: Pepels BSc \xa74.1 variable-depth selection
            budget.  When set, selection stops once the cumulative
            action cost along the root\u2192leaf path reaches this value.
        force_deterministic_tree: Kocsis\u2013Szepesv\xe1ri 2006
            Determinized UCT.  When ``True``, the tree expands through
            the deterministic path even when a stochastic
            :class:`TransitionModel` is supplied.
    """

    reuse_tree: bool = False
    tree_reuse_decay: float = 0.6
    anytime_fallback: bool = False
    path_length_budget: int | None = None
    force_deterministic_tree: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.tree_reuse_decay <= 1.0:
            raise ValueError("tree_reuse_decay must be in [0, 1]")
        if self.path_length_budget is not None and self.path_length_budget < 0:
            raise ValueError("path_length_budget must be >= 0 or None")


@dataclass(slots=True)
class MCTSTracingConfig:
    """Observability config for :class:`MCTSStrategy`.

    ``record_expansions`` is a no-op when ``tracer`` is ``None``;
    both must be set for ``on_search_expand`` events to fire.
    """

    tracer: Any = None
    record_expansions: bool = False


@dataclass(slots=True)
class MCTSStrategy:
    """Monte Carlo Tree Search planning strategy.

    Implements the classical four-phase MCTS loop (selection,
    expansion, simulation, backpropagation) over GOAP action edges.
    Adapted from LATS's ``select_node`` / ``expand_node`` /
    ``rollout`` / ``backpropagate`` quartet.

    Configuration is grouped into three small dataclasses so callers
    only construct what they need to override:

    * :class:`MCTSExploration` \u2014 search budget and UCB1 knobs.
    * :class:`MCTSReuseConfig` \u2014 tree-reuse, variable-depth, anytime.
    * :class:`MCTSTracingConfig` \u2014 observability.

    Standalone fields are pluggable strategy components (rollout
    policy, transition model, scalar heuristic) rather than flags.
    """

    exploration: MCTSExploration = field(default_factory=MCTSExploration)
    reuse: MCTSReuseConfig = field(default_factory=MCTSReuseConfig)
    tracing: MCTSTracingConfig = field(default_factory=MCTSTracingConfig)
    rollout_policy: RolloutPolicy | None = None
    transition_model: TransitionModel = field(
        default_factory=DeterministicTransitionModel
    )
    # Continuous state-value function consumed by rollouts when they
    # terminate below the goal.  When ``None`` the default
    # goal-condition-count reward shape is used; when supplied the
    # default rollout policy threads it through so UCB1 statistics
    # gain a continuous gradient on sparse-goal problems.
    scalar_heuristic: ScalarHeuristic | None = None
    # Post-search handle on the root of the last tree expanded by
    # :meth:`plan`.  Exposed so tests and observability hooks can
    # inspect chance-layer structure; ``None`` until ``plan`` has been
    # called with a non-trivial start state.
    _last_root: "MCTSNode | None" = None
    # Carryover root promoted by :meth:`advance` after the previous
    # plan's first action was executed; consumed by the next
    # :meth:`plan` call when ``reuse_tree`` is on and its state matches
    # the supplied ``start``.
    _carryover_root: "MCTSNode | None" = None

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
        prior_plan: Plan | None = None,
        current_step: int = 0,
    ) -> Plan | None:
        del prior_plan, current_step  # MCTS rebuilds the search tree
        return self._plan_with_tracer(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            tracer=self.tracing.tracer,
        )

    def _plan_with_tracer(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None,
        tracer: Any,
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
        rng = random.Random(self.exploration.seed)
        policy: RolloutPolicy = self.rollout_policy or self._default_rollout(rng)

        if (
            self.reuse.reuse_tree
            and self._carryover_root is not None
            and self._carryover_root.state == start
        ):
            root = self._carryover_root
        else:
            root = MCTSNode(state=start)
            root.untried_actions = _applicable_actions(start, filtered_actions)
        self._carryover_root = None
        self._last_root = root
        stochastic = not self.reuse.force_deterministic_tree and not isinstance(
            self.transition_model, DeterministicTransitionModel
        )

        record = tracer is not None and self.tracing.record_expansions
        on_expand = getattr(tracer, "on_search_expand", None) if record else None
        on_complete = getattr(tracer, "on_search_complete", None) if record else None
        # Stable integer ids for tracer events; root is untracked so the
        # first expanded child reports ``parent_id=None``.  ``id(node)``
        # is safe within a single ``plan`` call because nodes are retained
        # by the tree for the full duration.
        node_ids: dict[int, int] = {}
        id_counter = [0]

        t0 = time.monotonic()
        wall_budget_s = (
            self.exploration.wall_clock_ms / 1000.0
            if self.exploration.wall_clock_ms
            else 0.0
        )
        iters_done = 0
        for i in range(self.exploration.iterations):
            if wall_budget_s and (time.monotonic() - t0) >= wall_budget_s:
                break
            if stochastic:
                leaf = _select_stochastic(
                    root,
                    c=self.exploration.c,
                    actions=filtered_actions,
                    rng=rng,
                    model=self.transition_model,
                    goal=goal,
                    path_length_budget=self.reuse.path_length_budget,
                )
                leaf = _expand_stochastic(
                    leaf,
                    goal=goal,
                    actions=filtered_actions,
                    rng=rng,
                    model=self.transition_model,
                    path_length_budget=self.reuse.path_length_budget,
                )
            else:
                leaf = _select(
                    root,
                    c=self.exploration.c,
                    path_length_budget=self.reuse.path_length_budget,
                )
                leaf = _expand(
                    leaf,
                    goal=goal,
                    actions=filtered_actions,
                    rng=rng,
                    model=self.transition_model,
                    path_length_budget=self.reuse.path_length_budget,
                )
            reward = policy.rollout(
                state=leaf.state, goal=goal, actions=filtered_actions
            )
            backpropagate(leaf, reward=reward)
            iters_done = i + 1
            if on_expand is not None and leaf is not root:
                _emit_mcts_expand(
                    on_expand, leaf, node_ids, id_counter, self.exploration.c
                )

        plan = _extract_plan(
            root,
            goal=goal,
            start=start,
            model=self.transition_model,
            anytime_fallback=self.reuse.anytime_fallback,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        if on_complete is not None:
            on_complete(iters_done, elapsed_ms, plan is not None)
        if plan is None:
            return None
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

    async def aplan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
        prior_plan: Plan | None = None,
        current_step: int = 0,
    ) -> Plan | None:
        """Async variant of :meth:`plan` with parity on tracer hooks.

        The MCTS search loop is CPU-bound and synchronous; this entry
        point exists to satisfy the dual sync/async pattern enforced
        across LangGOAP.  When a tracer is attached with async hooks
        (``aon_search_expand`` / ``aon_search_complete``), the sync
        search runs through a local capture proxy and the recorded
        events are re-fired through the real tracer's async hooks.
        ``self.tracing.tracer`` is never mutated, so concurrent callers on the
        same strategy instance do not clobber each other.
        """
        del prior_plan, current_step  # MCTS rebuilds the search tree
        real_tracer = self.tracing.tracer
        if not (real_tracer is not None and self.tracing.record_expansions):
            return self.plan(
                start, goal, actions, blacklisted_actions=blacklisted_actions
            )
        proxy = _AsyncCaptureTracer()
        plan = self._plan_with_tracer(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            tracer=proxy,
        )
        for event in proxy.events:
            kind = event[0]
            if kind == "expand":
                hook = getattr(real_tracer, "aon_search_expand", None)
                if hook is not None:
                    await hook(*event[1:])
            elif kind == "complete":
                hook = getattr(real_tracer, "aon_search_complete", None)
                if hook is not None:
                    await hook(*event[1:])
        return plan

    def advance(
        self,
        action: ActionSpec,
        observed_state: "PlanningState | Mapping[str, Any]",
    ) -> None:
        """Promote the subtree matching ``(action, observed_state)`` to
        the new carryover root with visit counts decayed by
        ``tree_reuse_decay``.

        Soemers\u2013Winands CIG 2016 \xa7IV-B: retain structure and statistics
        of the child corresponding to the action just played; multiply
        every visit count by ``gamma`` so old evidence is down-weighted
        in proportion to how out-of-date it is.  When the observed
        successor does not match any expanded child (teleport, maze
        reset, unseen sample), carryover is cleared and the next
        :meth:`plan` call cold-starts.
        """
        if self._last_root is None:
            return

        if isinstance(observed_state, PlanningState):
            target = observed_state
        else:
            target = PlanningState.from_dict(dict(observed_state))

        root = self._last_root
        promoted: MCTSNode | None = None
        for chance in root.chance_children:
            if chance.action is action or chance.action.name == action.name:
                key = _state_key(target)
                match = chance.children_by_key.get(key)
                if match is not None:
                    promoted = match
                    break
        if promoted is None:
            for child in root.children:
                if child.action is None:
                    continue
                if (
                    child.action is action or child.action.name == action.name
                ) and child.state == target:
                    promoted = child
                    break

        if promoted is None:
            self._carryover_root = None
            return

        promoted.parent = None
        _decay_tree(promoted, self.reuse.tree_reuse_decay)
        self._carryover_root = promoted

    def _default_rollout(self, rng: random.Random) -> "RolloutPolicy":
        """Pick the rollout policy appropriate for ``transition_model``.

        When the model is the default
        :class:`DeterministicTransitionModel`,
        :class:`HeuristicRollout` is used.  When a non-default model
        is wired in, :class:`StochasticRollout` is selected so the
        ``sample`` path is exercised during simulation.  Both
        receive ``self.scalar_heuristic`` unchanged so the strategy's
        leaf-value function is honoured in either dynamics regime.
        """
        if isinstance(self.transition_model, DeterministicTransitionModel):
            return HeuristicRollout(
                max_depth=self.exploration.rollout_depth,
                scalar_heuristic=self.scalar_heuristic,
            )
        return StochasticRollout(
            max_depth=self.exploration.rollout_depth,
            model=self.transition_model,
            rng=rng,
            scalar_heuristic=self.scalar_heuristic,
        )


class _AsyncCaptureTracer:
    """Synchronous tracer shim that records MCTS events for later
    re-emission through a real tracer's async hooks.  Used by
    :meth:`MCTSStrategy.aplan` to preserve the ``async`` surface
    without re-implementing the (CPU-bound, synchronous) search loop.
    """

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def on_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.events.append(("expand", node_id, state, g, h, f, parent_id, action_name))

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.events.append(("complete", nodes_explored, duration_ms, found))


def _emit_mcts_expand(
    on_expand: Any,
    leaf: MCTSNode,
    node_ids: dict[int, int],
    id_counter: list[int],
    c: float,
) -> None:
    """Fire a single ``on_search_expand`` event for the most-recently
    expanded decision leaf, applying the GOAP-wide semantic remap:
    ``g`` = visits, ``h`` = UCB1 score, ``f`` = running-mean value.

    Chance-layer ancestors are transparent: the reported parent is the
    nearest :class:`MCTSNode` above ``leaf``, so downstream panels render
    a pure decision-node tree regardless of the stochastic / deterministic
    expansion regime.
    """
    # Mint a stable id for this decision leaf (idempotent on re-visits).
    key = id(leaf)
    if key not in node_ids:
        node_ids[key] = id_counter[0]
        id_counter[0] += 1
    leaf_id = node_ids[key]

    parent_id: int | None = None
    cursor: "MCTSNode | ChanceNode | None" = leaf.parent
    while cursor is not None:
        if isinstance(cursor, MCTSNode):
            # Root has no parent and is untracked; its descendants report
            # ``parent_id=None`` so consumers anchor the tree at the root.
            if cursor.parent is None:
                parent_id = None
            else:
                parent_id = node_ids.get(id(cursor))
            break
        cursor = cursor.parent

    action_name = leaf.action.name if leaf.action is not None else None
    try:
        h_value = ucb1(leaf, c=c)
    except Exception:
        h_value = math.inf
    on_expand(
        leaf_id,
        leaf.state,
        float(leaf.visits),
        float(h_value) if math.isfinite(h_value) else math.inf,
        float(leaf.value),
        parent_id,
        action_name,
    )


def _path_cost(node: MCTSNode) -> float:
    """Sum ``action.cost`` along the root\u2192``node`` edge chain.

    Used to enforce Pepels BSc \xa74.1 variable-depth selection.  Nodes
    with no parent contribute zero; chance-node parents are skipped
    because they carry the same action as the decision parent just
    above them in the alternating tree layout.  Callable costs are
    evaluated against the action's pre-state \u2014 the closest decision
    ancestor's ``state`` \u2014 matching :meth:`ActionSpec.get_cost`
    semantics elsewhere in the planner.
    """
    total = 0.0
    cur: "MCTSNode | ChanceNode | None" = node
    while cur is not None:
        if isinstance(cur, MCTSNode) and cur.action is not None:
            pre = cur.parent
            while pre is not None and not isinstance(pre, MCTSNode):
                pre = pre.parent
            world = pre.state.to_dict() if pre is not None else {}
            total += float(cur.action.get_cost(world))
        cur = cur.parent
    return total


def _over_path_length_budget(node: MCTSNode, budget: int | None) -> bool:
    if budget is None:
        return False
    return _path_cost(node) >= float(budget)


def _select(
    node: MCTSNode,
    *,
    c: float,
    path_length_budget: int | None = None,
) -> MCTSNode:
    """Descend via UCB1 until a node with untried actions or no children."""
    cursor = node
    while (
        cursor.untried_actions == []
        and cursor.children
        and not _over_path_length_budget(cursor, path_length_budget)
    ):
        cursor = max(cursor.children, key=lambda n: ucb1(n, c=c))
    return cursor


def _expand(
    node: MCTSNode,
    *,
    goal: GoalSpec,
    actions: list[ActionSpec],
    rng: random.Random,
    model: TransitionModel,
    path_length_budget: int | None = None,
) -> MCTSNode:
    """Expand one untried action into a new child, or return ``node``.

    If ``node`` has no untried actions (fully expanded leaf or
    terminal), return it unchanged so the caller rolls out from it.
    Tree edges use ``model.expected`` so the tree statistics reflect
    the planner's deterministic world view; only rollouts sample.
    """
    if node.state.satisfies(goal.conditions):
        node.is_terminal = True
        return node
    if not node.untried_actions:
        return node
    if _over_path_length_budget(node, path_length_budget):
        return node
    # Pop a random untried action for diversity under the fixed seed.
    idx = rng.randrange(len(node.untried_actions))
    action = node.untried_actions.pop(idx)
    expected_effects = model.expected(node.state.to_dict(), action)
    next_state = node.state.apply(dict(expected_effects))
    child = MCTSNode(state=next_state, parent=node, action=action)
    child.untried_actions = _applicable_actions(next_state, actions)
    if next_state.satisfies(goal.conditions):
        child.is_terminal = True
    node.children.append(child)
    return child


def _state_key(state: PlanningState) -> int:
    """Hashable key for a :class:`PlanningState` used to collapse
    repeated sampled successors onto a single chance-node child."""
    return hash(tuple(sorted(state.to_dict().items(), key=lambda kv: kv[0])))


def _decay_tree(node: MCTSNode, gamma: float) -> None:
    """Recursively multiply every stored visit count under ``node`` by
    ``gamma`` \u2208 [0, 1] (Soemers\u2013Winands CIG 2016 \xa7IV-B).

    ``value`` is stored as the running mean and is preserved: decaying
    total score and visits by the same factor leaves the mean unchanged
    while shrinking UCB1's implicit confidence in that mean.  ``gamma``
    is clamped to ``[0, 1]``.
    """
    g = 0.0 if gamma < 0.0 else (1.0 if gamma > 1.0 else gamma)
    stack: list[MCTSNode | ChanceNode] = [node]
    while stack:
        cursor = stack.pop()
        cursor.visits = int(round(cursor.visits * g))
        if isinstance(cursor, MCTSNode):
            stack.extend(cursor.children)
            stack.extend(cursor.chance_children)
        else:
            stack.extend(cursor.children_by_key.values())


def _select_stochastic(
    node: MCTSNode,
    *,
    c: float,
    actions: list[ActionSpec],
    rng: random.Random,
    model: TransitionModel,
    goal: GoalSpec,
    path_length_budget: int | None = None,
) -> MCTSNode:
    """Alternate ``Decision \u2192 Chance \u2192 Decision`` descent.

    At each decision node with no untried actions, pick the best
    chance-child by UCB1 (aggregate visit / value stats).  At that
    chance node, sample one successor via :meth:`TransitionModel.sample`;
    descend into the matching decision child when the sampled state
    hashes to a known key, otherwise materialise a fresh decision
    node under the chance parent and return it as the leaf for
    expansion + rollout.
    """
    cursor = node
    while True:
        if cursor.untried_actions or cursor.is_terminal:
            return cursor
        if _over_path_length_budget(cursor, path_length_budget):
            return cursor
        if not cursor.chance_children:
            return cursor
        chance = max(cursor.chance_children, key=lambda cn: ucb1(cn, c=c))
        sampled_effects = model.sample(cursor.state.to_dict(), chance.action, rng)
        next_state = cursor.state.apply(dict(sampled_effects))
        key = _state_key(next_state)
        existing = chance.children_by_key.get(key)
        if existing is not None:
            cursor = existing
            continue
        child = MCTSNode(state=next_state, parent=chance, action=chance.action)
        child.untried_actions = _applicable_actions(next_state, actions)
        if next_state.satisfies(goal.conditions):
            child.is_terminal = True
        chance.children_by_key[key] = child
        return child


def _expand_stochastic(
    node: MCTSNode,
    *,
    goal: GoalSpec,
    actions: list[ActionSpec],
    rng: random.Random,
    model: TransitionModel,
    path_length_budget: int | None = None,
) -> MCTSNode:
    """Expand one untried action into a chance-child and a first
    sampled decision grandchild.

    Mirrors :func:`_expand` for the deterministic case but inserts a
    :class:`ChanceNode` between the decision node and its sampled
    successor.  The sampled effects come from :meth:`TransitionModel.sample`,
    not ``expected``, so the chance node's running mean converges to
    the true outcome-averaged value.
    """
    if node.state.satisfies(goal.conditions):
        node.is_terminal = True
        return node
    if not node.untried_actions:
        return node
    if _over_path_length_budget(node, path_length_budget):
        return node
    idx = rng.randrange(len(node.untried_actions))
    action = node.untried_actions.pop(idx)
    chance = ChanceNode(action=action, parent=node)
    node.chance_children.append(chance)
    sampled_effects = model.sample(node.state.to_dict(), action, rng)
    next_state = node.state.apply(dict(sampled_effects))
    child = MCTSNode(state=next_state, parent=chance, action=action)
    child.untried_actions = _applicable_actions(next_state, actions)
    if next_state.satisfies(goal.conditions):
        child.is_terminal = True
    chance.children_by_key[_state_key(next_state)] = child
    return child


def _extract_plan(
    root: MCTSNode,
    *,
    goal: GoalSpec,
    start: PlanningState,
    model: TransitionModel,
    anytime_fallback: bool = False,
) -> Plan | None:
    """Greedy descent by child visits, preferring terminal leaves.

    Prefers a terminal (goal-satisfying) leaf when one was discovered
    in the search tree \u2014 that matches LATS's plan-extraction heuristic
    and keeps existing unit tests bit-identical.  When no terminal was
    found (common on deep MDPs where the tree cannot reach the goal
    within the per-tick iteration budget), falls back to Kocsis\u2013
    Szepesv\xe1ri's ``robust'' rule: greedy descent by visit count from
    the root, producing an anytime partial plan whose *first* action
    is the one MCTS has the most statistical evidence for.  This is
    what lets MCTS be driven in a "plan-per-tick, execute-first-action"
    replanning loop on domains where a complete plan never fits in
    the tree.
    """
    best_leaf = _find_best_terminal(root)
    if best_leaf is None:
        if not anytime_fallback or not (root.children or root.chance_children):
            return None
        best_leaf = _robust_descent(root)
    # Walk up from the chosen leaf to reconstruct action / state lists.
    # Chance-node ancestors are skipped \u2014 they have no ``state`` of
    # their own, only an action and aggregate statistics.
    rev_actions: list[ActionSpec] = []
    rev_states: list[PlanningState] = []
    cursor: "MCTSNode | ChanceNode | None" = best_leaf
    while cursor is not None and cursor.parent is not None:
        if isinstance(cursor, MCTSNode):
            assert cursor.action is not None  # non-root decision nodes carry an action
            rev_actions.append(cursor.action)
            rev_states.append(cursor.state)
        cursor = cursor.parent
    rev_actions.reverse()
    rev_states.reverse()
    total_cost = 0.0
    sim = start
    for a in rev_actions:
        total_cost += a.get_cost(sim.to_dict())
        sim = sim.apply(dict(model.expected(sim.to_dict(), a)))
    return Plan(
        actions=tuple(rev_actions),
        expected_states=tuple(rev_states),
        total_cost=total_cost,
        metadata=PlanMetadata(),
        score=SimpleScore(scalar=total_cost),
    )


def _find_best_terminal(root: MCTSNode) -> MCTSNode | None:
    """Breadth-first search for the shallowest terminal decision node
    with the highest visit count along the winning branch.

    Traverses both the deterministic ``children`` edges and the
    chance-layer ``chance_children`` so stochastic expansion trees are
    searched end-to-end.  Chance nodes themselves are skipped \u2014 only
    :class:`MCTSNode` terminals count as plan endpoints.
    """
    best: MCTSNode | None = None
    stack: list[MCTSNode] = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal and (best is None or node.visits > best.visits):
            best = node
        stack.extend(node.children)
        for chance in node.chance_children:
            stack.extend(chance.children_by_key.values())
    return best


def _robust_descent(root: MCTSNode) -> MCTSNode:
    """Greedy descent from ``root`` by visit count (Kocsis\u2013Szepesv\xe1ri
    ``robust'' action-selection rule).

    Terminates at the first decision node with no expanded children so
    the returned leaf has at least one full rollout's worth of
    statistics along every edge above it.  Under stochastic expansion
    the descent alternates decision \u2192 chance \u2192 decision, picking the
    most-visited chance child and then its most-visited sampled
    successor.
    """
    cursor: MCTSNode = root
    while True:
        if cursor.children:
            cursor = max(cursor.children, key=lambda n: n.visits)
            continue
        if cursor.chance_children:
            chance = max(cursor.chance_children, key=lambda cn: cn.visits)
            if not chance.children_by_key:
                return cursor
            cursor = max(chance.children_by_key.values(), key=lambda n: n.visits)
            continue
        return cursor


__all__ = [
    "ChanceNode",
    "HeuristicRollout",
    "MCTSNode",
    "MCTSStrategy",
    "RandomRollout",
    "RolloutPolicy",
    "ScalarHeuristic",
    "StochasticRollout",
    "backpropagate",
    "ucb1",
]
