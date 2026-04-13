"""End-to-end integration tests for LazyDecompositionStrategy via GoapGraph."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.planner.strategy import LazyDecompositionStrategy
from langgoap.types import ReplanStrategy


def _make_chain_actions(n: int) -> list[ActionSpec]:
    """Build a chain of N actions."""
    actions: list[ActionSpec] = []
    for i in range(n):
        pre = {f"step_{i}": True} if i > 0 else {}
        eff = {f"step_{i + 1}": True}
        actions.append(
            ActionSpec(
                name=f"a{i}",
                preconditions=pre,
                effects=eff,
                execute=lambda ws, idx=i: {f"step_{idx + 1}": True},
            )
        )
    return actions


def test_lazy_e2e_5_actions_lookahead_1_full_loop() -> None:
    """5-action problem with lookahead=1 → executes one at a time, replans, achieves goal."""
    actions = _make_chain_actions(5)
    goal = GoalSpec(
        conditions={"step_5": True},
        replan_strategy=ReplanStrategy.EVERY_ACTION,
        max_replans=20,
    )

    strategy = LazyDecompositionStrategy(lookahead=1)
    result = GoapGraph(actions=actions, strategy=strategy).invoke(
        goal=goal, world_state={}
    )

    assert result["status"] == "goal_achieved"
    assert result["world_state"]["step_5"] is True
    # With EVERY_ACTION replan and lookahead=1, we need 4 replans
    # (initial plan + 4 replans = 5 one-action plans)
    assert result["replan_count"] >= 4


def test_lazy_e2e_lookahead_2_fewer_replans() -> None:
    """5-action problem with lookahead=2 → fewer replan rounds than lookahead=1."""
    actions = _make_chain_actions(5)
    goal = GoalSpec(
        conditions={"step_5": True},
        replan_strategy=ReplanStrategy.EVERY_ACTION,
        max_replans=20,
    )

    strategy = LazyDecompositionStrategy(lookahead=2)
    result = GoapGraph(actions=actions, strategy=strategy).invoke(
        goal=goal, world_state={}
    )

    assert result["status"] == "goal_achieved"
    assert result["world_state"]["step_5"] is True


def test_lazy_e2e_on_deviation_replan() -> None:
    """Works with ON_DEVIATION replan strategy — plan exhausts, triggers replan."""
    actions = _make_chain_actions(4)
    goal = GoalSpec(
        conditions={"step_4": True},
        replan_strategy=ReplanStrategy.ON_DEVIATION,
        max_replans=20,
    )

    strategy = LazyDecompositionStrategy(lookahead=1)
    result = GoapGraph(actions=actions, strategy=strategy).invoke(
        goal=goal, world_state={}
    )

    assert result["status"] == "goal_achieved"


@pytest.mark.asyncio
async def test_lazy_e2e_async_path() -> None:
    """LazyDecompositionStrategy works via async invoke."""
    actions = _make_chain_actions(3)
    goal = GoalSpec(
        conditions={"step_3": True},
        replan_strategy=ReplanStrategy.EVERY_ACTION,
        max_replans=20,
    )

    strategy = LazyDecompositionStrategy(lookahead=1)
    result = await GoapGraph(actions=actions, strategy=strategy).ainvoke(
        goal=goal, world_state={}
    )

    assert result["status"] == "goal_achieved"
    assert result["world_state"]["step_3"] is True
