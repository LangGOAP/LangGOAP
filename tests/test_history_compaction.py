"""Opt-in compaction of the execution-history snapshots (issue #74).

Every ``ActionResult`` used to carry complete ``state_before`` and
``state_after`` copies, so a checkpoint grew as O(steps × state size) with
rich world states.  ``history_state="delta"`` keeps only the keys an action
changed (and their prior values), ``"planning"`` keeps only planning keys,
and ``"full"`` (the default) keeps today's shape.  The realised cost no
longer depends on the stored pre-state: the executor records it on the
``ActionResult`` while it still has the full state.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.graph.nodes import GoapExecutor, ParallelGoapExecutor
from langgoap.graph.state import ActionResult, GoapState
from langgoap.history import StoreExecutionHistory
from langgoap.integrations import GoapSubgraph, add_goap_subgraph, create_goap_agent
from tests.conftest import make_action, make_plan

BLOB = "x" * 20_000
GOAL = GoalSpec(conditions={"a": True})


def _state(action: ActionSpec, world: dict[str, Any]) -> GoapState:
    return {
        "world_state": dict(world),
        "plan": make_plan(action),
        "current_step": 0,
        "goal": GOAL,
    }


def _bump(ws: dict[str, Any]) -> dict[str, Any]:
    return {"a": True, "n": ws["n"] + 1}


class TestRecordedCost:
    def test_round_trip_keeps_cost(self) -> None:
        entry = ActionResult(action_name="a", success=True, cost=2.5)
        assert ActionResult.from_dict(entry.to_dict()).cost == 2.5

    def test_legacy_dicts_without_cost_rehydrate(self) -> None:
        assert (
            ActionResult.from_dict({"action_name": "a", "success": True}).cost is None
        )

    def test_executor_records_the_cost_evaluated_on_the_full_pre_state(self) -> None:
        action = make_action(
            "buy", eff={"a": True}, cost=lambda ws: ws.get("price", 0) * 2
        )
        result = GoapExecutor()(_state(action, {"price": 3}))
        assert result["execution_history"][0]["cost"] == 6.0


class TestFullIsTheDefault:
    def test_snapshots_are_complete(self) -> None:
        action = make_action("step", eff={"a": True})
        entry = GoapExecutor()(_state(action, {"blob": BLOB}))["execution_history"][0]
        assert entry["state_before"] == {"blob": BLOB}
        assert entry["state_after"] == {"blob": BLOB, "a": True}


class TestDeltaMode:
    def test_only_changed_keys_and_their_prior_values_are_stored(self) -> None:
        action = make_action("step", eff={"a": True}, execute=_bump)
        world = {"blob": BLOB, "n": 1, "flag": False}
        result = GoapExecutor(history_state="delta")(_state(action, world))

        entry = result["execution_history"][0]
        assert entry["state_after"] == {"a": True, "n": 2}
        assert entry["state_before"] == {"n": 1}
        assert (
            result["world_state"]["blob"] == BLOB
        )  # the graph state itself is untouched

    async def test_async_path(self) -> None:
        action = make_action("step", eff={"a": True}, execute=_bump)
        result = await GoapExecutor(history_state="delta").acall(
            _state(action, {"blob": BLOB, "n": 1})
        )
        entry = result["execution_history"][0]
        assert entry["state_after"] == {"a": True, "n": 2}
        assert entry["state_before"] == {"n": 1}

    def test_parallel_executor(self) -> None:
        action = make_action("step", eff={"a": True}, execute=_bump)
        result = ParallelGoapExecutor(history_state="delta")(
            _state(action, {"blob": BLOB, "n": 1})
        )
        entry = result["execution_history"][0]
        assert entry["state_after"] == {"a": True, "n": 2}
        assert entry["state_before"] == {"n": 1}

    def test_failure_records_are_compacted_too(self) -> None:
        def boom(ws: dict[str, Any]) -> None:
            raise RuntimeError("nope")

        action = make_action("step", eff={"a": True}, execute=boom)
        result = GoapExecutor(history_state="delta")(_state(action, {"blob": BLOB}))
        entry = result["execution_history"][0]
        assert entry["success"] is False
        assert entry["state_before"] == {} and entry["state_after"] == {}


class TestPlanningMode:
    def test_only_planning_keys_are_stored(self) -> None:
        action = make_action(
            "step", pre={"ready": True}, eff={"a": True}, execute=_bump
        )
        world = {"ready": True, "blob": BLOB, "n": 1}
        executor = GoapExecutor(history_state="planning", actions=[action])
        entry = executor(_state(action, world))["execution_history"][0]
        assert entry["state_before"] == {"ready": True}
        assert entry["state_after"] == {"ready": True, "a": True}


class TestGoapGraphOption:
    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="history_state"):
            GoapGraph([make_action("s", eff={"a": True})], history_state="tiny")  # type: ignore[arg-type]

    @staticmethod
    def _chain(n: int) -> list[ActionSpec]:
        return [
            make_action(
                f"s{i}", pre={f"s{i - 1}": True} if i else {}, eff={f"s{i}": True}
            )
            for i in range(n)
        ]

    def test_delta_keeps_the_history_small(self) -> None:
        actions = self._chain(30)
        goal = GoalSpec(conditions={"s29": True})

        def history_bytes(mode: str) -> int:
            graph = GoapGraph(actions, history_state=mode)  # type: ignore[arg-type]
            out = graph.compile().invoke(graph.initial_state(goal, {"blob": BLOB}))
            assert out["status"] == "goal_achieved"
            assert len(out["execution_history"]) == 30
            return len(json.dumps(out["execution_history"]))

        full, delta = history_bytes("full"), history_bytes("delta")
        assert full > 30 * 2 * len(BLOB)  # every entry carries the blob twice
        assert delta < full / 20

    def test_realised_cost_is_unchanged_under_delta(self) -> None:
        action = make_action(
            "buy", eff={"a": True}, cost=lambda ws: ws.get("price", 0.0)
        )

        def actual_cost(mode: str) -> float:
            history = StoreExecutionHistory(InMemoryStore())
            graph = GoapGraph([action], history=history, history_state=mode)  # type: ignore[arg-type]
            graph.compile().invoke(graph.initial_state(GOAL, {"price": 7.0}))
            records = history.query_by_goal(
                StoreExecutionHistory.goal_hash_for(GOAL), limit=1
            )
            return records[0].actual_cost

        assert actual_cost("delta") == actual_cost("full") == 7.0

    def test_on_ramps_accept_the_option(self) -> None:
        for fn in (create_goap_agent, add_goap_subgraph, GoapSubgraph.__init__):
            assert "history_state" in inspect.signature(fn).parameters, fn
        sub = GoapSubgraph(
            [make_action("s", eff={"a": True})], GOAL, history_state="delta"
        )
        assert sub.graph is not None
