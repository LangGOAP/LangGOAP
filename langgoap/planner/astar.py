"""A* GOAP planner with two-pass optimization.

Forward-chaining A* search over GOAP state space. Includes:
- Reachability pre-check (fast early exit when a goal condition has no
  producing action)
- Action specificity tie-breaking
- Two-pass plan optimization: backward relevance + forward simulation
"""

from __future__ import annotations

import heapq
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import SimpleScore
from langgoap.state import PlanningState

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


def _heuristic(state: PlanningState, goal_conditions: Mapping[str, Any]) -> float:
    """Admissible heuristic: count of unsatisfied goal conditions."""
    state_dict = state.to_dict()
    return sum(
        1
        for k, v in goal_conditions.items()
        if k not in state_dict or state_dict[k] != v
    )


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
        current_dict = current.to_dict()
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

    # Sort actions by descending precondition count (specificity tie-breaking)
    sorted_actions = sorted(actions, key=lambda a: len(a.preconditions), reverse=True)

    # A* search — ``counter`` is a mutable box so ``_expand_neighbors`` can
    # increment the tie-breaker and keep the open list's ordering stable
    # across calls.
    counter = [0]

    h0 = _heuristic(start, goal_conditions)
    root = _SearchNode(f_score=h0, g_score=0.0, counter=counter[0], state=start)
    open_list: list[_SearchNode] = [root]
    # Map from state → best g_score seen
    best_g: dict[PlanningState, float] = {start: 0.0}
    nodes_explored = 0
    best_complete: Plan | None = None  # best complete plan seen (anytime support)

    while open_list:
        # Anytime: stop and return best complete plan found when deadline expires.
        if deadline is not None and time.monotonic() >= deadline:
            return best_complete, nodes_explored
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
            # In anytime mode, A* explores by f-score; the first goal found
            # is optimal (admissible heuristic), so return immediately.
            # best_complete is only used if the deadline fires mid-search.
            built = _build_plan_from_node(
                current, start, goal_conditions, nodes_explored, t0
            )
            return built, nodes_explored

        # Skip if we've found a better path to this state
        if current.g_score > best_g.get(current.state, float("inf")):
            continue

        _expand_neighbors(
            current, sorted_actions, goal_conditions, best_g, open_list, counter
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
        sim_dict = sim_state.to_dict()
        total_cost += a.get_cost(sim_dict)
        sim_state = sim_state.apply(a.get_effects(sim_dict))
        expected.append(sim_state)

    elapsed_ms = (time.monotonic() - t0) * 1000
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
) -> None:
    """Push every improving neighbour of ``current`` onto the open list.

    ``counter`` is a single-element mutable list used as a tie-breaker
    box so the caller's value survives across invocations.
    """
    current_dict = current.state.to_dict()
    for action in sorted_actions:
        if not current.state.satisfies(action.preconditions):
            continue
        new_state = current.state.apply(action.get_effects(current_dict))
        if new_state == current.state:
            continue  # no-op transition
        cost = action.get_cost(current_dict)
        tentative_g = current.g_score + cost
        if tentative_g >= best_g.get(new_state, float("inf")):
            continue
        best_g[new_state] = tentative_g
        h = _heuristic(new_state, goal_conditions)
        counter[0] += 1
        heapq.heappush(
            open_list,
            _SearchNode(
                f_score=tentative_g + h,
                g_score=tentative_g,
                counter=counter[0],
                state=new_state,
                actions=current.actions + (action,),
                trace_parent_id=current.counter,
                trace_action_name=action.name,
            ),
        )


def plan(
    start: PlanningState | dict[str, Any],
    goal: GoalSpec,
    actions: list[ActionSpec],
    blacklisted_actions: list[str] | None = None,
    *,
    time_budget_ms: float | None = None,
    tracer: Any = None,
    record_expansions: bool = False,
) -> Plan | None:
    """Find an optimal action sequence from start to goal using A*.

    Args:
        start: Current world state.  Accepts a plain ``dict`` for
            convenience; it will be coerced to ``PlanningState`` internally.
        goal: Goal specification with target conditions.
        actions: Available actions to choose from.
        blacklisted_actions: Action names to exclude from planning.
            If filtering makes the goal unreachable, the planner retries
            with all actions (graceful degradation: an unreachable goal
            from the filtered set is preferred over returning ``None``
            when a plan through the full action set still exists).
        time_budget_ms: Optional wall-clock budget in milliseconds.
            When set, the search returns the best *complete* plan found
            within the budget rather than running until exhaustion.
            ``None`` (the default) means no limit — original behaviour.
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
    t0 = time.monotonic()
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
            on_complete(total_nodes, (time.monotonic() - t0) * 1000, found)

    # Early exit: goal already satisfied
    if start.satisfies(goal_conditions):
        elapsed_ms = (time.monotonic() - t0) * 1000
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

    # Filter blacklisted actions
    blacklist_set = set(blacklisted_actions) if blacklisted_actions else set()
    if blacklist_set:
        available = [a for a in actions if a.name not in blacklist_set]
        result, n1 = _search(
            start,
            goal_conditions,
            available,
            t0,
            deadline,
            tracer=tracer,
            record_expansions=record_expansions,
            attempt=1,
        )
        if result is not None:
            _emit_complete(True, n1)
            return result
        # Blacklist fallback: filtered action set made the goal unreachable
        # — retry with all actions so the executor can decide at runtime.
        result, n2 = _search(
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
        return result

    result, n = _search(
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
