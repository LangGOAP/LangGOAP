"""Tests for ExecutionRecord and StoreExecutionHistory.

Uses :class:`InMemoryStore` — the reverse-index read/write protocol
from AD-6 must work against any ``BaseStore`` via ``get()``/``put()``
and never via ``search()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore

from langgoap.history import ExecutionRecord, StoreExecutionHistory


def _record(
    goal_hash: str = "g1",
    outcome: str = "success",
    actions: tuple[str, ...] = ("a", "b"),
) -> ExecutionRecord:
    return ExecutionRecord(
        goal_hash=goal_hash,
        goal_conditions={"done": True},
        plan_actions=actions,
        expected_cost=2.0,
        actual_cost=2.0,
        outcome=outcome,
        replan_count=0,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )


class TestExecutionRecord:
    def test_is_frozen_dataclass(self) -> None:
        rec = _record()
        with pytest.raises(AttributeError):
            rec.outcome = "failed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        rec = _record()
        data = rec.to_dict()
        restored = ExecutionRecord.from_dict(data)
        assert restored == rec


class TestStoreExecutionHistorySync:
    def test_record_and_query_by_goal(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)

        rec1 = _record(goal_hash="goal_abc", actions=("a1", "a2"))
        rec2 = _record(goal_hash="goal_abc", actions=("a1", "a3"))
        history.record(rec1)
        history.record(rec2)

        results = history.query_by_goal("goal_abc", limit=10)
        assert len(results) == 2
        # Newest-first
        assert results[0].plan_actions == ("a1", "a3")
        assert results[1].plan_actions == ("a1", "a2")

    def test_query_limit_applies(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        for i in range(5):
            history.record(_record(goal_hash="g", actions=(f"a{i}",)))
        results = history.query_by_goal("g", limit=3)
        assert len(results) == 3

    def test_query_unknown_goal_returns_empty(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        assert history.query_by_goal("nothing") == []

    def test_query_failures_by_action(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)

        history.record(_record(outcome="success", actions=("good",)))
        history.record(_record(outcome="failed", actions=("bad", "also_bad")))
        history.record(_record(outcome="failed", actions=("bad",)))

        results = history.query_failures("bad")
        assert len(results) == 2
        for r in results:
            assert r.outcome == "failed"
            assert "bad" in r.plan_actions

    def test_query_failures_unknown_action_returns_empty(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        history.record(_record(outcome="success"))
        assert history.query_failures("unknown") == []

    def test_index_bound_is_respected(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store, index_limit=3)
        for i in range(10):
            history.record(_record(goal_hash="g", actions=(f"a{i}",)))
        results = history.query_by_goal("g", limit=20)
        # Only the last 3 are reachable via the index.
        assert len(results) == 3
        assert results[0].plan_actions == ("a9",)
        assert results[2].plan_actions == ("a7",)


class TestStoreExecutionHistoryAsync:
    @pytest.mark.asyncio
    async def test_arecord_and_aquery_by_goal(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        await history.arecord(_record(goal_hash="g", actions=("x",)))
        await history.arecord(_record(goal_hash="g", actions=("y",)))
        results = await history.aquery_by_goal("g", limit=5)
        assert len(results) == 2
        assert results[0].plan_actions == ("y",)

    @pytest.mark.asyncio
    async def test_aquery_failures(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        await history.arecord(_record(outcome="failed", actions=("boom",)))
        await history.arecord(_record(outcome="success", actions=("boom",)))
        results = await history.aquery_failures("boom")
        assert len(results) == 1
        assert results[0].outcome == "failed"


class TestStoreNoSearch:
    """Guard: the implementation must not call ``store.search()``.

    ``search()`` on ``AsyncPostgresStore`` and ``RedisStore`` is a
    vector-similarity query that requires an embedder.  The reverse-
    index strategy in AD-6 is specifically designed to avoid this.
    """

    def test_implementation_never_calls_search(self) -> None:
        @dataclass
        class TrackingStore:
            _real: InMemoryStore
            search_called: bool = False

            def get(self, namespace: tuple[str, ...], key: str) -> Any:
                return self._real.get(namespace, key)

            def put(self, namespace: tuple[str, ...], key: str, value: Any) -> Any:
                return self._real.put(namespace, key, value)

            def search(self, *args: Any, **kwargs: Any) -> Any:
                self.search_called = True
                raise RuntimeError("search() must not be called")

        tracking = TrackingStore(_real=InMemoryStore())
        history = StoreExecutionHistory(tracking)  # type: ignore[arg-type]
        history.record(_record(goal_hash="g"))
        history.query_by_goal("g")
        history.query_failures("anything")
        assert tracking.search_called is False
