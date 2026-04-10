"""A* GOAP planner with two-pass optimization.

Forward-chaining A* search adapted from Embabel and GOApy reference
implementations. Includes:
- Reachability pre-check (Embabel fast early exit)
- Action specificity tie-breaking (Embabel)
- Two-pass plan optimization: backward relevance + forward simulation
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.state import PlanningState

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(order=False)
class _SearchNode:
    """A node in the A* open list."""

    f_score: float
    g_score: float
    # Tie-breaker: lower counter = earlier insertion (FIFO among equal f)
    counter: int = field(compare=False)
    state: PlanningState = field(compare=False)
    actions: tuple[ActionSpec, ...] = field(compare=False, default=())

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
    Adapted from Embabel's reachability check.
    """
    producible: set[tuple[str, Any]] = set()
    for action in actions:
        for k, v in action.effects.items():
            producible.add((k, v))

    state_dict = start.to_dict()
    for k, v in goal_conditions.items():
        if k in state_dict and state_dict[k] == v:
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
    """
    needed: set[tuple[str, Any]] = set(goal_conditions.items())
    kept: list[ActionSpec] = []

    for action in reversed(actions):
        effect_items = set(action.effects.items())
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
        # Check if this action actually changes state toward the goal
        current_dict = current.to_dict()
        produces_new = any(current_dict.get(k) != v for k, v in action.effects.items())
        if produces_new:
            kept.append(action)
            current = current.apply(action.effects)

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
) -> Plan | None:
    """Core A* search loop.

    Separated from :func:`plan` so the blacklist fallback can call it
    without duplicating the search implementation.
    """
    # Reachability pre-check
    if not _is_reachable(start, goal_conditions, actions):
        return None

    # Sort actions by descending precondition count (specificity tie-breaking)
    sorted_actions = sorted(actions, key=lambda a: len(a.preconditions), reverse=True)

    # A* search
    counter = 0

    h0 = _heuristic(start, goal_conditions)
    root = _SearchNode(f_score=h0, g_score=0.0, counter=counter, state=start)
    open_list: list[_SearchNode] = [root]
    # Map from state → best g_score seen
    best_g: dict[PlanningState, float] = {start: 0.0}
    nodes_explored = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_explored += 1

        # Goal check
        if current.state.satisfies(goal_conditions):
            raw_actions = current.actions
            original_len = len(raw_actions)

            # Two-pass optimization
            optimized = _backward_optimization(raw_actions, goal_conditions)
            optimized = _forward_optimization(optimized, start, goal_conditions)

            actions_pruned = original_len - len(optimized)

            # Compute expected states and total cost
            expected: list[PlanningState] = []
            sim_state = start
            total_cost = 0.0
            for a in optimized:
                total_cost += a.get_cost(sim_state.to_dict())
                sim_state = sim_state.apply(a.effects)
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
            )

        # Skip if we've found a better path to this state
        if current.g_score > best_g.get(current.state, float("inf")):
            continue

        # Expand neighbors
        current_dict = current.state.to_dict()
        for action in sorted_actions:
            # Check preconditions
            if not current.state.satisfies(action.preconditions):
                continue

            new_state = current.state.apply(action.effects)

            # Skip no-op transitions
            if new_state == current.state:
                continue

            cost = action.get_cost(current_dict)
            tentative_g = current.g_score + cost

            # Only proceed if this is a better path
            if tentative_g < best_g.get(new_state, float("inf")):
                best_g[new_state] = tentative_g
                h = _heuristic(new_state, goal_conditions)
                counter += 1
                node = _SearchNode(
                    f_score=tentative_g + h,
                    g_score=tentative_g,
                    counter=counter,
                    state=new_state,
                    actions=current.actions + (action,),
                )
                heapq.heappush(open_list, node)

    return None  # No path found


def plan(
    start: PlanningState,
    goal: GoalSpec,
    actions: list[ActionSpec],
    blacklisted_actions: list[str] | None = None,
) -> Plan | None:
    """Find an optimal action sequence from start to goal using A*.

    Args:
        start: Current world state.
        goal: Goal specification with target conditions.
        actions: Available actions to choose from.
        blacklisted_actions: Action names to exclude from planning.
            If filtering makes the goal unreachable, the planner retries
            with all actions (Embabel-style graceful degradation).

    Returns:
        A Plan if a path exists, None if the goal is unreachable.
    """
    t0 = time.monotonic()
    goal_conditions = goal.conditions

    # Early exit: goal already satisfied
    if start.satisfies(goal_conditions):
        elapsed_ms = (time.monotonic() - t0) * 1000
        return Plan(
            actions=(),
            expected_states=(),
            total_cost=0.0,
            metadata=PlanMetadata(
                nodes_explored=0,
                planning_time_ms=elapsed_ms,
            ),
        )

    # Filter blacklisted actions
    blacklist_set = set(blacklisted_actions) if blacklisted_actions else set()
    if blacklist_set:
        available = [a for a in actions if a.name not in blacklist_set]
        result = _search(start, goal_conditions, available, t0)
        if result is not None:
            return result
        # Embabel fallback: blacklist made goal unreachable — retry with all actions
        return _search(start, goal_conditions, actions, t0)

    return _search(start, goal_conditions, actions, t0)
