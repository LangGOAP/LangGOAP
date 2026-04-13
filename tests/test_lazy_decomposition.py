"""Tests for LazyDecompositionStrategy (ADaPT-style incremental planning)."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.strategy import (
    AStarStrategy,
    LazyDecompositionStrategy,
    PlanningStrategy,
    TwoPhasePipelineStrategy,
)
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


def make_action(
    name: str,
    pre: dict[str, Any] | None = None,
    eff: dict[str, Any] | None = None,
    cost: float = 1.0,
) -> ActionSpec:
    return ActionSpec(name=name, preconditions=pre or {}, effects=eff or {}, cost=cost)


def _make_chain(n: int) -> tuple[list[ActionSpec], GoalSpec, PlanningState]:
    """Build a chain of N actions: a0→a1→...→a(n-1)→goal.

    Each action sets a flag that is the precondition for the next.
    """
    actions: list[ActionSpec] = []
    for i in range(n):
        pre = {f"step_{i}": True} if i > 0 else {}
        eff = {f"step_{i + 1}": True}
        actions.append(make_action(f"a{i}", pre=pre, eff=eff))
    goal = GoalSpec(conditions={f"step_{n}": True})
    start = PlanningState.from_dict({})
    return actions, goal, start


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_lazy_default_lookahead() -> None:
    strategy = LazyDecompositionStrategy()
    assert strategy._lookahead == 1


def test_lazy_custom_lookahead() -> None:
    strategy = LazyDecompositionStrategy(lookahead=3)
    assert strategy._lookahead == 3


def test_lazy_invalid_lookahead_raises() -> None:
    with pytest.raises(ValueError, match="lookahead must be >= 1"):
        LazyDecompositionStrategy(lookahead=0)


def test_lazy_conforms_to_planning_strategy_protocol() -> None:
    strategy = LazyDecompositionStrategy()
    assert isinstance(strategy, PlanningStrategy)


# ---------------------------------------------------------------------------
# Truncation behaviour
# ---------------------------------------------------------------------------


def test_lookahead_1_truncates_to_single_action() -> None:
    actions, goal, start = _make_chain(5)
    strategy = LazyDecompositionStrategy(lookahead=1)
    result = strategy.plan(start, goal, actions)

    assert result is not None
    assert len(result) == 1
    assert result.action_names == ["a0"]


def test_lookahead_3_truncates_to_3_actions() -> None:
    actions, goal, start = _make_chain(5)
    strategy = LazyDecompositionStrategy(lookahead=3)
    result = strategy.plan(start, goal, actions)

    assert result is not None
    assert len(result) == 3
    assert result.action_names == ["a0", "a1", "a2"]


def test_lookahead_larger_than_plan_returns_full_plan() -> None:
    actions, goal, start = _make_chain(3)
    strategy = LazyDecompositionStrategy(lookahead=10)
    result = strategy.plan(start, goal, actions)

    assert result is not None
    assert len(result) == 3
    assert result.action_names == ["a0", "a1", "a2"]


def test_inner_strategy_failure_returns_none() -> None:
    # No actions → A* returns None
    strategy = LazyDecompositionStrategy(lookahead=1)
    goal = GoalSpec(conditions={"impossible": True})
    start = PlanningState.from_dict({})
    result = strategy.plan(start, goal, [])
    assert result is None


# ---------------------------------------------------------------------------
# Cost and metadata
# ---------------------------------------------------------------------------


def test_cost_reflects_truncated_portion() -> None:
    actions: list[ActionSpec] = [
        make_action("cheap", eff={"step_1": True}, cost=1.0),
        make_action("expensive", pre={"step_1": True}, eff={"step_2": True}, cost=10.0),
        make_action("final", pre={"step_2": True}, eff={"done": True}, cost=5.0),
    ]
    goal = GoalSpec(conditions={"done": True})
    start = PlanningState.from_dict({})

    full_strategy = AStarStrategy()
    full_plan = full_strategy.plan(start, goal, actions)
    assert full_plan is not None
    assert full_plan.total_cost == 16.0

    lazy = LazyDecompositionStrategy(lookahead=1)
    truncated = lazy.plan(start, goal, actions)
    assert truncated is not None
    assert truncated.total_cost == 1.0


def test_metadata_preserved_from_inner_strategy() -> None:
    actions, goal, start = _make_chain(5)
    strategy = LazyDecompositionStrategy(lookahead=2)
    result = strategy.plan(start, goal, actions)

    assert result is not None
    assert result.metadata.nodes_explored > 0
    assert result.metadata.planning_time_ms >= 0.0


# ---------------------------------------------------------------------------
# Expected states
# ---------------------------------------------------------------------------


def test_expected_states_match_truncated_actions() -> None:
    actions, goal, start = _make_chain(5)
    strategy = LazyDecompositionStrategy(lookahead=2)
    result = strategy.plan(start, goal, actions)

    assert result is not None
    assert len(result.expected_states) == 2
    # After action a0: step_1=True
    assert result.expected_states[0].to_dict() == {"step_1": True}
    # After action a1: step_1=True, step_2=True
    assert result.expected_states[1].to_dict() == {"step_1": True, "step_2": True}


# ---------------------------------------------------------------------------
# Blacklisted actions forwarded
# ---------------------------------------------------------------------------


def test_blacklisted_actions_forwarded_to_inner() -> None:
    """Blacklist is forwarded to the inner strategy; preferred action is avoided."""
    # Two actions that both achieve the goal — a_cheap (cost=1) and a_pricey (cost=5).
    # Without a blacklist, A* picks a_cheap.  Blacklisting it must route planning
    # through a_pricey instead, proving the blacklist was forwarded to the inner
    # strategy rather than silently dropped.
    a_cheap = make_action("a_cheap", eff={"done": True}, cost=1.0)
    a_pricey = make_action("a_pricey", eff={"done": True}, cost=5.0)
    actions = [a_cheap, a_pricey]
    goal = GoalSpec(conditions={"done": True})
    start = PlanningState.from_dict({})
    strategy = LazyDecompositionStrategy(lookahead=1)

    # Without blacklist: cheaper action should win.
    result_default = strategy.plan(start, goal, actions)
    assert result_default is not None
    assert list(result_default.action_names) == ["a_cheap"]

    # With blacklist: a_cheap must be absent from the returned plan.
    result_blacklisted = strategy.plan(
        start, goal, actions, blacklisted_actions=["a_cheap"]
    )
    assert result_blacklisted is not None
    assert "a_cheap" not in result_blacklisted.action_names
    assert list(result_blacklisted.action_names) == ["a_pricey"]


# ---------------------------------------------------------------------------
# Works with custom inner strategy
# ---------------------------------------------------------------------------


class CountingStrategy:
    """Strategy that records how many times it was called."""

    def __init__(self) -> None:
        self.call_count = 0

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        self.call_count += 1
        return AStarStrategy().plan(
            start, goal, actions, blacklisted_actions=blacklisted_actions
        )


def test_lazy_with_custom_inner_strategy() -> None:
    actions, goal, start = _make_chain(5)
    inner = CountingStrategy()
    strategy = LazyDecompositionStrategy(lookahead=2, inner=inner)
    result = strategy.plan(start, goal, actions)

    assert result is not None
    assert len(result) == 2
    assert inner.call_count == 1


# ---------------------------------------------------------------------------
# Goal already satisfied → empty plan passthrough
# ---------------------------------------------------------------------------


def test_goal_already_satisfied_returns_empty_plan() -> None:
    actions, goal, _ = _make_chain(3)
    start = PlanningState.from_dict({"step_3": True})
    strategy = LazyDecompositionStrategy(lookahead=1)
    result = strategy.plan(start, goal, actions)

    assert result is not None
    assert len(result) == 0
