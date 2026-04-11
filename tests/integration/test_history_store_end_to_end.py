"""End-to-end integration tests for ``StoreExecutionHistory`` through the graph.

Exercises the full planner → executor → observer loop with a history
attached and confirms that success and failure terminal states both
produce queryable records in an ``InMemoryStore``.  The same contract
is expected to hold on production-grade stores (PostgreSQL, Redis)
because :class:`StoreExecutionHistory` only uses ``get``/``put`` and
never calls ``search`` (see AD-6 NS3).
"""

from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.history import StoreExecutionHistory, compute_goal_hash


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="gather", preconditions={}, effects={"data": True}, cost=1.0),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects={"done": True},
            cost=1.0,
        ),
    ]


class TestHistoryStoreSync:
    def test_success_is_recorded(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        graph = GoapGraph(_actions(), history=history)

        goal = GoalSpec(conditions={"done": True})
        result = graph.invoke(goal=goal, world_state={})
        assert result["status"] == "goal_achieved"

        records = history.query_by_goal(compute_goal_hash(goal), limit=10)
        assert len(records) == 1
        assert records[0].outcome == "success"
        assert records[0].plan_actions == ("gather", "process")
        assert records[0].expected_cost == pytest.approx(2.0)

    def test_failure_is_recorded(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        graph = GoapGraph(_actions(), history=history)

        goal = GoalSpec(conditions={"unreachable": True})
        result = graph.invoke(goal=goal, world_state={})
        assert result["status"] == "no_plan"

        records = history.query_by_goal(compute_goal_hash(goal), limit=10)
        assert len(records) == 1
        assert records[0].outcome == "failed"
        assert records[0].plan_actions == ()


class TestHistoryStoreAsync:
    @pytest.mark.asyncio
    async def test_success_is_recorded_async(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        graph = GoapGraph(_actions(), history=history)

        goal = GoalSpec(conditions={"done": True})
        result = await graph.ainvoke(goal=goal, world_state={})
        assert result["status"] == "goal_achieved"

        records = await history.aquery_by_goal(compute_goal_hash(goal), limit=10)
        assert len(records) == 1
        assert records[0].outcome == "success"


class TestHistoryStoreMultipleGoals:
    def test_accumulates_across_runs(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        graph = GoapGraph(_actions(), history=history)

        # Two successful runs against the same goal should produce two
        # records in the goal index.
        goal = GoalSpec(conditions={"done": True})
        graph.invoke(goal=goal, world_state={})
        graph.invoke(goal=goal, world_state={})

        records = history.query_by_goal(compute_goal_hash(goal), limit=10)
        assert len(records) == 2
        assert all(r.outcome == "success" for r in records)
