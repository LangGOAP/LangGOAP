"""A* GOAP planner with two-pass optimization.

Forward-chaining A* search over GOAP state space. Includes:
- Reachability pre-check (fast early exit when a goal condition has no
  producing action)
- An admissible, consistent heuristic (unsatisfied goal conditions scaled
  by the largest per-action goal coverage and the smallest static cost),
  so the first goal state popped is cost-optimal
- Action specificity tie-breaking
- Two-pass plan optimization: backward relevance + forward simulation
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
from langgoap.state import PlanningState

# Monotonic clock used for time budgets; a module attribute so tests can
# inject a deterministic clock without patching ``time`` globally.
_clock = time.monotonic

#: Weight schedule for anytime search: weighted A* (f = g + w*h) with a
#: tightening ``w`` returns a complete plan early and improves it while
#: the budget lasts.  The final ``1.0`` iteration is plain (optimal) A*.
ANYTIME_WEIGHTS: tuple[float, ...] = (3.0, 2.0, 1.5, 1.0)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(order=False)
class _SearchNode:
    """A node in the A* open list."""

    f_score: float
    g_score: float
    # Tie-breaker: lower counter = earlier insertion (FIFO among equal f).
    # Also doubles as the stable ``node_id`` reported via search tracer
    # hooks so observability consumers can reconstruct parent/child
    # edges without coupling to the heap internals.
    counter: int = field(compare=False)
    state: PlanningState = field(compare=False)
    actions: tuple[ActionSpec, ...] = field(compare=False, default=())
    trace_parent_id: int | None = field(compare=False, default=None)
    trace_action_name: str | None = field(compare=False, default=None)

    def __lt__(self, other: _SearchNode) -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.counter < other.counter


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


def _heuristic_scale(
    goal_conditions: Mapping[str, Any], actions: list[ActionSpec]
) -> tuple[int, float]:
    """Return ``(max_goal_effects_per_action, min_action_cost)`` for ``actions``.

    These two scalars turn the raw unsatisfied-condition count into an
    admissible estimate (see :func:`_heuristic`):

    * ``max_goal_effects_per_action`` is the largest number of goal
      ``(key, value)`` pairs any single action can establish.  For
      dynamic-effect actions only the declared ``effect_keys`` are known,
      so every goal key they list counts.  Never below 1.
    * ``min_action_cost`` is the smallest static cost in the action set.
      A callable cost has no static lower bound, so its presence forces
      the bound to ``0.0`` (the search then degrades gracefully to
      uniform-cost search, which is still optimal).
    """
    goal_items = set(goal_conditions.items())
    goal_keys = set(goal_conditions)
    per_action = 1
    min_cost: float | None = None
    for a in actions:
        if a.has_dynamic_effects:
            n = len((a.effect_keys or frozenset()) & goal_keys)
        else:
            static_effects = cast(Mapping[str, Any], a.effects)
            n = len([kv for kv in static_effects.items() if kv in goal_items])
        if n > per_action:
            per_action = n
        if callable(a.cost):
            min_cost = 0.0
        elif min_cost is None or a.cost < min_cost:
            min_cost = float(a.cost)
    if min_cost is None or min_cost < 0.0:
        min_cost = 0.0
    return per_action, min_cost


def _heuristic(
    state: PlanningState,
    goal_conditions: Mapping[str, Any],
    *,
    per_action: int = 1,
    min_cost: float = 1.0,
) -> float:
    """Admissible, consistent heuristic for the A* search.

    ``ceil(unsatisfied / per_action) * min_cost`` where ``unsatisfied`` is
    the number of goal conditions the state does not yet meet.  At least
    ``ceil(unsatisfied / per_action)`` more actions are needed because no
    action establishes more than ``per_action`` goal conditions, and each
    of them costs at least ``min_cost``.  With the defaults
    (``per_action=1``, ``min_cost=1.0``) this is the plain count, which is
    only admissible when every action costs at least 1 and establishes at
    most one goal condition; :func:`_search` always supplies the scalars
    from :func:`_heuristic_scale` so the estimate never overshoots.
    """
    # (key, value) membership on the frozenset: no dict allocation per
    # heuristic evaluation (#63).
    conditions = state.conditions
    unsatisfied = sum(1 for kv in goal_conditions.items() if kv not in conditions)
    if unsatisfied == 0 or min_cost <= 0.0:
        return 0.0
    return math.ceil(unsatisfied / per_action) * min_cost


# ---------------------------------------------------------------------------
# Reachability pre-check
# ---------------------------------------------------------------------------


def _is_reachable(
    start: PlanningState,
    goal_conditions: Mapping[str, Any],
    actions: list[ActionSpec],
) -> bool:
    """Fast check: every unsatisfied goal condition must be producible by some action.

    This is a necessary (not sufficient) condition for plan existence.
    Used as a cheap early-exit before entering the full A* search.

    Dynamic-effect actions cannot be introspected statically; any goal
    key listed in their ``effect_keys`` is treated as producible with
    any value, and the check short-circuits conservatively to ``True``
    for those keys.
    """
    producible: set[tuple[str, Any]] = set()
    dynamic_keys: set[str] = set()
    for action in actions:
        if action.has_dynamic_effects:
            dynamic_keys.update(action.effect_keys or ())
        else:
            for k, v in action.effects.items():  # type: ignore[union-attr]
                producible.add((k, v))

    state_dict = start.to_dict()
    for k, v in goal_conditions.items():
        if k in state_dict and state_dict[k] == v:
            continue
        if k in dynamic_keys:
            continue
        if (k, v) not in producible:
            return False
    return True


# ---------------------------------------------------------------------------
# Relevance pruning
# ---------------------------------------------------------------------------


def relevant_actions(
    goal_conditions: Mapping[str, Any], actions: list[ActionSpec]
) -> list[ActionSpec]:
    """Drop the actions that cannot contribute to ``goal_conditions``.

    A key is relevant when the goal reads it or when an action that writes a
    relevant key reads it in a precondition (a backward closure over keys).
    An action is kept when it writes at least one relevant key; dynamic
    effects count through their declared ``effect_keys``, and when a relevant
    action has dynamic effects nothing is dropped at all, because such an
    effect may read any key.

    Dropping the rest is sound: preconditions and goal conditions only ever
    read relevant keys, so an action that writes none of them cannot change
    what any later action or the goal sees. All it could do in a plan is add
    cost. Searching through such actions multiplies the state space by every
    combination of the facts they write, which is what made domains with a
    few reachable but useless actions exponentially slower than their size.
    """
    relevant: set[str] = set(goal_conditions)
    footprints = [(action, action.effect_key_set()) for action in actions]
    kept: list[ActionSpec] = []
    pending = footprints
    frontier = set(relevant)
    while frontier and pending:
        still: list[tuple[ActionSpec, frozenset[str]]] = []
        newly: set[str] = set()
        for action, written in pending:
            if written & frontier:
                kept.append(action)
                newly.update(k for k in action.preconditions if k not in relevant)
            else:
                still.append((action, written))
        relevant |= newly
        frontier = newly
        pending = still
    if any(a.has_dynamic_effects for a in kept):
        # A callable effect reads whatever it likes from the state, so the keys
        # a relevant action depends on are unknown: every action may matter.
        # Same conservatism as the backward optimisation pass.
        return list(actions)
    order = {id(a): i for i, a in enumerate(actions)}
    kept.sort(key=lambda a: order[id(a)])
    return kept


# ---------------------------------------------------------------------------
# Two-pass optimization
# ---------------------------------------------------------------------------


def _backward_optimization(
    actions: tuple[ActionSpec, ...],
    goal_conditions: Mapping[str, Any],
) -> tuple[ActionSpec, ...]:
    """Backward pass: keep only actions whose effects are needed.

    Walk the plan backward. Track which conditions are still needed
    (starting from goal conditions). An action is kept only if at least
    one of its effects is needed. Its preconditions then become needed
    for earlier actions.

    Dynamic-effect actions read arbitrary state at execution time; the
    backward pass has no static way to know which keys they depend on,
    so the optimization is skipped entirely when any action has
    dynamic effects.  A* has already returned an optimal path — the
    pass is a best-effort shortener, and bailing out is safe.
    """
    dynamic = [a.name for a in actions if a.has_dynamic_effects]
    if dynamic:
        logger.debug(
            "skipping backward optimization: plan contains dynamic-effect "
            "action(s) %s whose state reads cannot be tracked statically; "
            "returning A* result unmodified",
            dynamic,
        )
        return actions

    needed: set[tuple[str, Any]] = set(goal_conditions.items())
    kept: list[ActionSpec] = []

    for action in reversed(actions):
        effect_items = set(action.effects.items())  # type: ignore[union-attr]
        if effect_items & needed:
            kept.append(action)
            # Remove satisfied conditions, add preconditions as new needs
            needed -= effect_items
            needed |= set(action.preconditions.items())

    kept.reverse()
    return tuple(kept)


def _forward_optimization(
    actions: tuple[ActionSpec, ...],
    start: PlanningState,
    goal_conditions: Mapping[str, Any],
) -> tuple[ActionSpec, ...]:
    """Forward pass: skip actions whose effects are already true.

    Simulate the plan forward. If an action's effects are already
    satisfied in the current state, skip it — it's redundant.
    Verify the final state still satisfies the goal; if not, fall back
    to the unoptimized input.
    """
    kept: list[ActionSpec] = []
    current = start

    for action in actions:
        # Check if this action actually changes state toward the goal.
        # Dynamic-effect actions are always kept — their effects depend
        # on the simulated state and may produce new values we cannot
        # evaluate without running the callable.
        current_dict = current.as_context()
        resolved = action.get_effects(current_dict)
        produces_new = any(current_dict.get(k) != v for k, v in resolved.items())
        if action.has_dynamic_effects or produces_new:
            kept.append(action)
            current = current.apply(resolved)

    # Verify goal is still satisfied
    if current.satisfies(goal_conditions):
        return tuple(kept)
    return actions  # Fall back to unoptimized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _search(
    start: PlanningState,
    goal_conditions: Mapping[str, Any],
    actions: list[ActionSpec],
    t0: float,
    deadline: float | None = None,
    *,
    tracer: Any = None,
    record_expansions: bool = False,
    attempt: int = 1,
    weight: float = 1.0,
) -> tuple[Plan | None, int]:
    """Core A* search loop.

    Separated from :func:`plan` so the blacklist fallback can call it
    without duplicating the search implementation.

    Args:
        deadline: Absolute monotonic time (seconds) after which the search
            should stop and return the best complete plan found so far.
            ``None`` means no limit (original behaviour).
        tracer: Optional :class:`~langgoap.tracing.PlanningTracer` whose
            ``on_search_expand`` / ``on_search_dead_end`` hooks receive
            per-expansion and dead-end events.  ``on_search_complete`` is
            *not* fired from this function — :func:`plan` owns the single
            completion event so retry loops do not produce duplicates.
        record_expansions: Opt-in gate for the per-expansion firehose.
            Must be ``True`` *and* a tracer supplied for any hook to fire.
        attempt: 1-based index of this search invocation within the
            enclosing :func:`plan` call.  Included in every
            ``on_search_dead_end`` detail so consumers can reconstruct
            the retry shape (e.g. blacklist fallback) from the event
            stream even though only one completion event is emitted.

    Returns:
        ``(plan, nodes_explored)`` — ``plan`` is ``None`` when no path
        exists or the deadline expired before a complete plan was found.
        ``weight`` > 1 runs weighted A* (``f = g + weight * h``): complete
        and bounded-suboptimal (cost <= weight * optimal) but faster, which
        is what the anytime schedule in :func:`plan` relies on.
        ``nodes_explored`` is always the number of nodes popped during
        this attempt and feeds the aggregate reported by :func:`plan`.
    """
    # Resolve hooks once, up-front.  Missing attributes (legacy tracers
    # pre-dating the search hooks) resolve to ``None`` and silently
    # skip at each call site — this is the "zero overhead when silent"
    # path guaranteed by the public API.
    record = tracer is not None and record_expansions
    on_expand = getattr(tracer, "on_search_expand", None) if record else None
    on_dead_end = getattr(tracer, "on_search_dead_end", None) if record else None

    # Reachability pre-check
    if not _is_reachable(start, goal_conditions, actions):
        if on_dead_end is not None:
            on_dead_end(
                "not_reachable",
                {
                    "goal_conditions": dict(goal_conditions),
                    "attempt": attempt,
                },
            )
        return None, 0

    # Only the actions that can move the goal take part in the search.
    actions = relevant_actions(goal_conditions, actions)

    # Sort actions by descending precondition count (specificity tie-breaking)
    sorted_actions = sorted(actions, key=lambda a: len(a.preconditions), reverse=True)

    # Hoist the single-use detection out of the per-expansion loop: the
    # common case (every action is rerunnable) pays no cost.
    has_single_use_actions = any(not a.can_rerun for a in sorted_actions)

    # A* search — ``counter`` is a mutable box so ``_expand_neighbors`` can
    # increment the tie-breaker and keep the open list's ordering stable
    # across calls.
    counter = [0]

    per_action, min_cost = _heuristic_scale(goal_conditions, sorted_actions)
    h0 = _heuristic(start, goal_conditions, per_action=per_action, min_cost=min_cost)
    root = _SearchNode(
        f_score=weight * h0, g_score=0.0, counter=counter[0], state=start
    )
    open_list: list[_SearchNode] = [root]
    # Map from state → best g_score seen
    best_g: dict[PlanningState, float] = {start: 0.0}
    nodes_explored = 0

    while open_list:
        # Deadline: a single A* run has no "best so far" (the first goal it
        # pops is the only complete plan it ever sees), so give up and let
        # the anytime loop in :func:`plan` keep the plan from a previous,
        # more heavily weighted iteration.
        if deadline is not None and _clock() >= deadline:
            return None, nodes_explored
        current = heapq.heappop(open_list)
        nodes_explored += 1

        if on_expand is not None:
            on_expand(
                current.counter,
                current.state,
                current.g_score,
                current.f_score - current.g_score,
                current.f_score,
                current.trace_parent_id,
                current.trace_action_name,
            )

        if current.state.satisfies(goal_conditions):
            # With ``weight == 1`` and an admissible, consistent heuristic the
            # first goal popped is optimal; with ``weight > 1`` it is a
            # complete plan whose cost is at most ``weight`` times optimal.
            built = _build_plan_from_node(
                current, start, goal_conditions, nodes_explored, t0
            )
            return built, nodes_explored

        # Skip if we've found a better path to this state
        if current.g_score > best_g.get(current.state, float("inf")):
            continue

        _expand_neighbors(
            current,
            sorted_actions,
            goal_conditions,
            best_g,
            open_list,
            counter,
            has_single_use_actions=has_single_use_actions,
            per_action=per_action,
            min_cost=min_cost,
            weight=weight,
        )

    # Open list exhausted without reaching the goal — reachability pre-check
    # was optimistic (e.g. preconditions unreachable despite effect coverage).
    if on_dead_end is not None:
        on_dead_end(
            "exhausted",
            {"nodes_explored": nodes_explored, "attempt": attempt},
        )
    return None, nodes_explored  # No path found


def _build_plan_from_node(
    node: _SearchNode,
    start: PlanningState,
    goal_conditions: Mapping[str, Any],
    nodes_explored: int,
    t0: float,
) -> Plan:
    """Materialize a :class:`Plan` from a goal-satisfying search node."""
    raw_actions = node.actions
    original_len = len(raw_actions)
    # Two-pass optimization
    optimized = _backward_optimization(raw_actions, goal_conditions)
    optimized = _forward_optimization(optimized, start, goal_conditions)
    actions_pruned = original_len - len(optimized)

    expected: list[PlanningState] = []
    sim_state = start
    total_cost = 0.0
    for a in optimized:
        sim_dict = sim_state.as_context()
        total_cost += a.get_cost(sim_dict)
        sim_state = sim_state.apply(a.get_effects(sim_dict))
        expected.append(sim_state)

    elapsed_ms = (_clock() - t0) * 1000
    return Plan(
        actions=optimized,
        expected_states=tuple(expected),
        total_cost=total_cost,
        metadata=PlanMetadata(
            nodes_explored=nodes_explored,
            planning_time_ms=elapsed_ms,
            actions_pruned=actions_pruned,
        ),
        score=SimpleScore(scalar=total_cost),
    )


def _expand_neighbors(
    current: _SearchNode,
    sorted_actions: list[ActionSpec],
    goal_conditions: Mapping[str, Any],
    best_g: dict[PlanningState, float],
    open_list: list[_SearchNode],
    counter: list[int],
    *,
    has_single_use_actions: bool = False,
    per_action: int = 1,
    min_cost: float = 1.0,
    weight: float = 1.0,
) -> None:
    """Push every improving neighbour of ``current`` onto the open list.

    ``counter`` is a single-element mutable list used as a tie-breaker
    box so the caller's value survives across invocations.

    When ``has_single_use_actions`` is ``True`` the expander tracks the
    set of action names already on the path through ``current`` and
    skips any successor whose action has ``can_rerun=False``.  The flag
    is hoisted to the caller so the common case (every action is
    rerunnable) pays no per-expansion cost.
    """
    current_dict = current.state.as_context()
    used_names: frozenset[str] = (
        frozenset(a.name for a in current.actions)
        if has_single_use_actions
        else frozenset()
    )
    for action in sorted_actions:
        if not action.can_rerun and action.name in used_names:
            continue
        if not current.state.satisfies(action.preconditions):
            continue
        effects = action.get_effects(current_dict)
        if current.state.satisfies(effects):
            continue  # no-op transition: every effect already holds, nothing to copy
        new_state = current.state.apply(effects)
        if new_state == current.state:
            continue  # no-op transition
        cost = action.get_cost(current_dict)
        tentative_g = current.g_score + cost
        if tentative_g >= best_g.get(new_state, float("inf")):
            continue
        best_g[new_state] = tentative_g
        h = _heuristic(
            new_state, goal_conditions, per_action=per_action, min_cost=min_cost
        )
        counter[0] += 1
        heapq.heappush(
            open_list,
            _SearchNode(
                f_score=tentative_g + weight * h,
                g_score=tentative_g,
                counter=counter[0],
                state=new_state,
                actions=current.actions + (action,),
                trace_parent_id=current.counter,
                trace_action_name=action.name,
            ),
        )


def _with_nodes(plan_obj: Plan, nodes_explored: int) -> Plan:
    """Return *plan_obj* with ``metadata.nodes_explored`` set to the total."""
    if plan_obj.metadata.nodes_explored == nodes_explored:
        return plan_obj
    return replace(
        plan_obj, metadata=replace(plan_obj.metadata, nodes_explored=nodes_explored)
    )


def _run_search(
    start: PlanningState,
    goal_conditions: Mapping[str, Any],
    actions: list[ActionSpec],
    t0: float,
    deadline: float | None,
    *,
    tracer: Any,
    record_expansions: bool,
    attempt: int,
) -> tuple[Plan | None, int]:
    """One logical search: plain A* without a deadline, anytime A* with one.

    The anytime loop runs weighted A* over :data:`ANYTIME_WEIGHTS`.  Each
    iteration is complete, so the first one that finishes yields a usable
    plan; later, tighter iterations can only improve it and the ``1.0``
    iteration is optimal.  When the deadline fires the best complete plan
    found so far is returned (``None`` only if not even the loosest
    iteration finished), with ``nodes_explored`` summed over iterations.
    """
    if deadline is None:
        return _search(
            start,
            goal_conditions,
            actions,
            t0,
            None,
            tracer=tracer,
            record_expansions=record_expansions,
            attempt=attempt,
        )
    best: Plan | None = None
    total = 0
    for weight in ANYTIME_WEIGHTS:
        if _clock() >= deadline:
            break
        result, n = _search(
            start,
            goal_conditions,
            actions,
            t0,
            deadline,
            tracer=tracer,
            record_expansions=record_expansions,
            attempt=attempt,
            weight=weight,
        )
        total += n
        if result is None:
            # Either the deadline interrupted this iteration (keep the best
            # so far) or weighted A*, which is complete, proved no plan
            # exists.  Both end the loop.
            break
        if best is None or result.total_cost < best.total_cost:
            best = result
    return (_with_nodes(best, total) if best is not None else None), total


def plan(
    start: PlanningState | dict[str, Any],
    goal: GoalSpec,
    actions: list[ActionSpec],
    blacklisted_actions: list[str] | None = None,
    *,
    time_budget_ms: float | None = None,
    tracer: Any = None,
    record_expansions: bool = False,
    blacklist_fallback: bool | None = None,
) -> Plan | None:
    """Find an optimal action sequence from start to goal using A*.

    Args:
        start: Current world state.  Accepts a plain ``dict`` for
            convenience; it will be coerced to ``PlanningState`` internally.
        goal: Goal specification with target conditions.
        actions: Available actions to choose from.
        blacklisted_actions: Action names to exclude from planning.
        blacklist_fallback: When the blacklist makes the goal unreachable,
            ``True`` retries with the full action set (re-admitting the
            blacklisted actions) instead of returning ``None``.  ``None``
            (default) reads ``goal.policy.blacklist_fallback``, which is
            ``False`` unless the caller opted in: a permanently failing,
            human-denied, or guard-blocked action must not be re-run just
            because it is the only path (issue #15).
        time_budget_ms: Optional wall-clock budget in milliseconds.
            When set, the search is *anytime*: weighted A* runs with the
            tightening schedule :data:`ANYTIME_WEIGHTS`, so a complete
            (bounded-suboptimal) plan is available early and is improved
            while the budget lasts; the final iteration is optimal A*.
            On expiry the best complete plan found so far is returned, or
            ``None`` when not even the loosest iteration finished.
            ``None`` (the default) means no limit — plain optimal A*.
        tracer: Optional :class:`~langgoap.tracing.PlanningTracer` that
            receives per-expansion, dead-end, and completion events when
            ``record_expansions`` is also ``True``.
        record_expansions: Opt-in gate for the search-tree firehose.
            Off by default so production planning stays zero-overhead.

    Returns:
        A Plan if a path exists, None if the goal is unreachable.
    """
    if isinstance(start, dict):
        start = PlanningState.from_dict(start)
    t0 = _clock()
    goal_conditions = goal.conditions
    deadline = (t0 + time_budget_ms / 1000.0) if time_budget_ms is not None else None

    # Single owner of ``on_search_complete`` — see :func:`_search` for
    # the rationale.  Retry loops (blacklist fallback) still produce
    # exactly one completion event per ``plan()`` invocation.
    on_complete = (
        getattr(tracer, "on_search_complete", None)
        if tracer is not None and record_expansions
        else None
    )

    def _emit_complete(found: bool, total_nodes: int) -> None:
        if on_complete is not None:
            on_complete(total_nodes, (_clock() - t0) * 1000, found)

    # Early exit: goal already satisfied
    if start.satisfies(goal_conditions):
        elapsed_ms = (_clock() - t0) * 1000
        _emit_complete(True, 0)
        return Plan(
            actions=(),
            expected_states=(),
            total_cost=0.0,
            metadata=PlanMetadata(
                nodes_explored=0,
                planning_time_ms=elapsed_ms,
            ),
            score=SimpleScore(scalar=0.0),
        )

    if blacklist_fallback is None:
        blacklist_fallback = goal.policy.blacklist_fallback

    # Filter blacklisted actions
    blacklist_set = set(blacklisted_actions) if blacklisted_actions else set()
    if blacklist_set:
        available = [a for a in actions if a.name not in blacklist_set]
        result, n1 = _run_search(
            start,
            goal_conditions,
            available,
            t0,
            deadline,
            tracer=tracer,
            record_expansions=record_expansions,
            attempt=1,
        )
        if result is not None or not blacklist_fallback:
            _emit_complete(result is not None, n1)
            return result
        # Opt-in blacklist fallback: the filtered action set made the goal
        # unreachable, so retry with all actions and let the executor
        # decide at runtime.
        result, n2 = _run_search(
            start,
            goal_conditions,
            actions,
            t0,
            deadline,
            tracer=tracer,
            record_expansions=record_expansions,
            attempt=2,
        )
        _emit_complete(result is not None, n1 + n2)
        # Report the whole planning effort, not just the second search
        # (issue #23).
        return _with_nodes(result, n1 + n2) if result is not None else None

    result, n = _run_search(
        start,
        goal_conditions,
        actions,
        t0,
        deadline,
        tracer=tracer,
        record_expansions=record_expansions,
        attempt=1,
    )
    _emit_complete(result is not None, n)
    return result
