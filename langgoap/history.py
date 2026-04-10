"""Execution history persistence on top of LangGraph ``BaseStore``.

Ships the long-deferred ``DESIGN.md`` §14.4.  ``StoreExecutionHistory``
records one :class:`ExecutionRecord` per planner-executor loop
completion and answers two queries:

* ``query_by_goal(goal_hash, limit)`` — most recent executions for a
  given goal
* ``query_failures(action_name)`` — recent executions that ended in
  ``"failed"`` and included ``action_name`` in the plan

Both queries and their async counterparts use **only** ``store.get()``
/ ``store.aget()``.  ``BaseStore.search()`` is *deliberately not
called* because ``AsyncPostgresStore`` and ``RedisStore`` interpret
``search`` as a vector-similarity query that requires an embedder —
forcing every history user to configure one would be unacceptable.

The implementation instead maintains two reverse indexes:

* goal index at ``("langgoap", "goal_index", goal_hash)`` → bounded
  newest-first list of execution UUIDs
* failure index at ``("langgoap", "failure_index", action_name)`` →
  same structure, populated only when ``outcome == "failed"``

Both indexes are bounded (``index_limit``, default 100) so their read
cost stays constant and their write cost is a single ``get`` + single
``put``.  Older IDs drop off the index but the underlying records are
left in the primary namespace, reachable via direct UUID lookup.

Trade-offs accepted for v0.1.0:

* **Read amplification** — two ``get()`` calls per returned record.
  Execution history is diagnostic, not a hot path.
* **Write amplification** — one primary write plus one or two index
  writes per record.
* **Race conditions** — concurrent writers to the same ``goal_hash``
  can lose an ID if they read-modify-write the index simultaneously.
  ``GoapObserver`` records exactly once at the end of the loop so the
  race window is small; a transactional index is post-v0.1.0.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from langgraph.store.base import BaseStore

_EXECUTIONS_NS = ("langgoap", "executions")
_GOAL_INDEX_NS = ("langgoap", "goal_index")
_FAILURE_INDEX_NS = ("langgoap", "failure_index")


@dataclass(frozen=True)
class ExecutionRecord:
    """One planner → executor → observer loop outcome.

    Attributes:
        goal_hash: Stable identifier for the goal (user-chosen or
            derived from ``GoalSpec``).
        goal_conditions: The goal's target condition dict, stored for
            later diagnostics.
        plan_actions: Tuple of action names in execution order.
        expected_cost: The plan's ``total_cost`` at planning time.
        actual_cost: Realised cost after execution.  Equal to
            ``expected_cost`` for purely deterministic runs; may
            differ when dynamic costs change mid-execution.
        outcome: One of ``"success"``, ``"failed"``,
            ``"partial"``.  Open string so new tracer implementations
            can introduce extra states without churning the schema.
        replan_count: Number of replans that happened during
            execution.
        timestamp: UTC timestamp at record creation.
        metadata: Optional free-form dict for tracer-specific extras
            (token counts, provider, model name, etc.).  Never
            required by the indexing code.
    """

    goal_hash: str
    goal_conditions: Mapping[str, Any]
    plan_actions: tuple[str, ...]
    expected_cost: float
    actual_cost: float
    outcome: str
    replan_count: int
    timestamp: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain ``dict`` suitable for ``BaseStore.put``."""
        return {
            "goal_hash": self.goal_hash,
            "goal_conditions": dict(self.goal_conditions),
            "plan_actions": list(self.plan_actions),
            "expected_cost": self.expected_cost,
            "actual_cost": self.actual_cost,
            "outcome": self.outcome,
            "replan_count": self.replan_count,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionRecord:
        """Deserialise from the format produced by :meth:`to_dict`."""
        return cls(
            goal_hash=data["goal_hash"],
            goal_conditions=dict(data.get("goal_conditions", {})),
            plan_actions=tuple(data.get("plan_actions", ())),
            expected_cost=float(data["expected_cost"]),
            actual_cost=float(data["actual_cost"]),
            outcome=data["outcome"],
            replan_count=int(data.get("replan_count", 0)),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=dict(data.get("metadata", {})),
        )


def _item_value(item: Any) -> Any:
    """Unwrap ``BaseStore.get`` / ``aget`` return values.

    LangGraph's store protocol wraps values in a ``SearchItem`` /
    ``Item`` object with a ``.value`` attribute.  Some alternate
    implementations (and older versions) return the raw value
    directly.  Normalise both shapes.
    """
    if item is None:
        return None
    return getattr(item, "value", item)


class StoreExecutionHistory:
    """Execution history backed by a LangGraph :class:`BaseStore`.

    Args:
        store: Any ``BaseStore`` implementation.  ``get``/``put`` are
            the only methods invoked (plus ``aget``/``aput`` on the
            async path).  ``search()`` is **never** called — see
            module docstring for the rationale.
        index_limit: Maximum number of execution IDs kept in a single
            reverse-index entry.  Defaults to 100 which is enough
            for most diagnostic use cases without incurring
            large per-write overhead.

    Example::

        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        history.record(ExecutionRecord(...))
        recent = history.query_by_goal("goal_abc", limit=5)
    """

    def __init__(self, store: BaseStore, *, index_limit: int = 100) -> None:
        self._store = store
        self._index_limit = index_limit

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------
    def record(self, record: ExecutionRecord) -> None:
        """Persist ``record`` and update the reverse indexes."""
        execution_id = uuid.uuid4().hex
        self._store.put(_EXECUTIONS_NS, execution_id, record.to_dict())
        self._push_id(_GOAL_INDEX_NS, record.goal_hash, execution_id)
        if record.outcome == "failed":
            for action_name in record.plan_actions:
                self._push_id(_FAILURE_INDEX_NS, action_name, execution_id)

    def query_by_goal(self, goal_hash: str, limit: int = 10) -> list[ExecutionRecord]:
        """Return up to ``limit`` most-recent records for ``goal_hash``."""
        ids = self._read_ids(_GOAL_INDEX_NS, goal_hash)
        return self._fetch_records(ids, limit)

    def query_failures(
        self, action_name: str, limit: int = 10
    ) -> list[ExecutionRecord]:
        """Return up to ``limit`` most-recent failed runs that touched ``action_name``."""
        ids = self._read_ids(_FAILURE_INDEX_NS, action_name)
        return self._fetch_records(ids, limit)

    # ------------------------------------------------------------------
    # Async API (NL2: full parity with the sync methods)
    # ------------------------------------------------------------------
    async def arecord(self, record: ExecutionRecord) -> None:
        """Async variant of :meth:`record`."""
        execution_id = uuid.uuid4().hex
        await self._store.aput(_EXECUTIONS_NS, execution_id, record.to_dict())
        await self._apush_id(_GOAL_INDEX_NS, record.goal_hash, execution_id)
        if record.outcome == "failed":
            for action_name in record.plan_actions:
                await self._apush_id(_FAILURE_INDEX_NS, action_name, execution_id)

    async def aquery_by_goal(
        self, goal_hash: str, limit: int = 10
    ) -> list[ExecutionRecord]:
        """Async variant of :meth:`query_by_goal`."""
        ids = await self._aread_ids(_GOAL_INDEX_NS, goal_hash)
        return await self._afetch_records(ids, limit)

    async def aquery_failures(
        self, action_name: str, limit: int = 10
    ) -> list[ExecutionRecord]:
        """Async variant of :meth:`query_failures`."""
        ids = await self._aread_ids(_FAILURE_INDEX_NS, action_name)
        return await self._afetch_records(ids, limit)

    # ------------------------------------------------------------------
    # Internal index helpers
    # ------------------------------------------------------------------
    def _read_ids(self, namespace: tuple[str, ...], key: str) -> list[str]:
        item = self._store.get(namespace, key)
        value = _item_value(item)
        if not value:
            return []
        return list(value.get("execution_ids", []))

    async def _aread_ids(self, namespace: tuple[str, ...], key: str) -> list[str]:
        item = await self._store.aget(namespace, key)
        value = _item_value(item)
        if not value:
            return []
        return list(value.get("execution_ids", []))

    def _push_id(self, namespace: tuple[str, ...], key: str, execution_id: str) -> None:
        ids = self._read_ids(namespace, key)
        ids.insert(0, execution_id)
        if len(ids) > self._index_limit:
            ids = ids[: self._index_limit]
        self._store.put(namespace, key, {"execution_ids": ids})

    async def _apush_id(
        self, namespace: tuple[str, ...], key: str, execution_id: str
    ) -> None:
        ids = await self._aread_ids(namespace, key)
        ids.insert(0, execution_id)
        if len(ids) > self._index_limit:
            ids = ids[: self._index_limit]
        await self._store.aput(namespace, key, {"execution_ids": ids})

    def _fetch_records(self, ids: list[str], limit: int) -> list[ExecutionRecord]:
        results: list[ExecutionRecord] = []
        for execution_id in ids:
            if len(results) >= limit:
                break
            item = self._store.get(_EXECUTIONS_NS, execution_id)
            value = _item_value(item)
            if value is None:
                continue
            results.append(ExecutionRecord.from_dict(value))
        return results

    async def _afetch_records(
        self, ids: list[str], limit: int
    ) -> list[ExecutionRecord]:
        results: list[ExecutionRecord] = []
        for execution_id in ids:
            if len(results) >= limit:
                break
            item = await self._store.aget(_EXECUTIONS_NS, execution_id)
            value = _item_value(item)
            if value is None:
                continue
            results.append(ExecutionRecord.from_dict(value))
        return results


__all__ = ["ExecutionRecord", "StoreExecutionHistory"]
