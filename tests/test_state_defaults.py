"""Every bookkeeping key of the state schema is present in every result.

LangGraph returns only channels that were written to.  Before this fix a
happy-path run never wrote ``blacklisted_actions`` or
``action_failure_counts``, so ``result["blacklisted_actions"]`` raised
``KeyError`` precisely when nothing had gone wrong.  The defaults now come
from one table: ``GoapGraph.initial_state`` seeds them, the planner fills
any that are missing on its first pass (for callers who build the input
dict by hand), and ``rehydrate_state`` fills them on the way out.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from langgoap import ActionSpec, GoalSpec, GoapGraph, rehydrate_state
from langgoap.graph.state import STATE_DEFAULTS, GoapStateChannels

BOOKKEEPING = {
    "current_step": 0,
    "execution_history": [],
    "replan_count": 0,
    "replan_reason": None,
    "blacklisted_actions": [],
    "action_failure_counts": {},
    "current_subgoal_index": 0,
    "skipped_subgoals": [],
    "no_plan_explanation": None,
    "reflection_context": [],
    "prepared_wait": None,
}


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="do",
            preconditions={},
            effects={"done": True},
            execute=lambda ws: {"done": True},
        )
    ]


GOAL = GoalSpec(conditions={"done": True})


class TestDefaultsTable:
    def test_table_matches_the_schema(self) -> None:
        assert STATE_DEFAULTS == BOOKKEEPING
        channel_keys = set(GoapStateChannels.__annotations__)
        assert set(STATE_DEFAULTS) <= channel_keys
        # The planner owns these; they are never defaulted.
        assert {"goal", "world_state", "plan", "status", "wall_clock_started_at"} == (
            channel_keys - set(STATE_DEFAULTS)
        )

    def test_defaults_are_fresh_copies(self) -> None:
        a = GoapGraph(_actions()).initial_state(GOAL)
        b = GoapGraph(_actions()).initial_state(GOAL)
        a["blacklisted_actions"].append("x")
        assert b["blacklisted_actions"] == []
        assert STATE_DEFAULTS["blacklisted_actions"] == []


class TestInitialState:
    def test_seeds_every_bookkeeping_key(self) -> None:
        state = GoapGraph(_actions()).initial_state(GOAL, {"k": 1})
        assert state["goal"] == GOAL.to_dict()
        assert state["world_state"] == {"k": 1}
        for key, default in BOOKKEEPING.items():
            assert state[key] == default, key


class TestResultsAreComplete:
    def test_happy_path_result_has_every_key(self) -> None:
        graph = GoapGraph(_actions())
        result = graph.compile(checkpointer=MemorySaver()).invoke(
            graph.initial_state(GOAL), config={"configurable": {"thread_id": "t"}}
        )
        assert result["status"] == "goal_achieved"
        assert result["blacklisted_actions"] == []
        assert result["action_failure_counts"] == {}
        assert result["replan_count"] == 0
        assert result["skipped_subgoals"] == []
        assert result["reflection_context"] == []
        assert result["no_plan_explanation"] is None
        assert result["replan_reason"] is None
        assert result["current_subgoal_index"] == 0

    def test_hand_built_input_gets_the_defaults_from_the_planner(self) -> None:
        graph = GoapGraph(_actions())
        result = graph.compile(checkpointer=MemorySaver()).invoke(
            {"goal": GOAL.to_dict(), "world_state": {}},
            config={"configurable": {"thread_id": "t"}},
        )
        assert result["status"] == "goal_achieved"
        for key, default in BOOKKEEPING.items():
            if key in ("current_step", "execution_history"):
                continue  # written by the run itself
            assert result[key] == default, key

    def test_convenience_invoke_result_has_every_key(self) -> None:
        result = GoapGraph(_actions()).invoke(goal=GOAL)
        for key in BOOKKEEPING:
            assert key in result, key
        assert result["blacklisted_actions"] == []

    @pytest.mark.asyncio
    async def test_async_result_has_every_key(self) -> None:
        graph = GoapGraph(_actions())
        result = await graph.compile(checkpointer=MemorySaver()).ainvoke(
            {"goal": GOAL.to_dict(), "world_state": {}},
            config={"configurable": {"thread_id": "t"}},
        )
        assert result["blacklisted_actions"] == []
        assert result["action_failure_counts"] == {}


class TestRehydrateFillsDefaults:
    def test_missing_keys_are_filled(self) -> None:
        typed = rehydrate_state({"world_state": {"a": 1}, "status": "goal_achieved"})
        assert typed["blacklisted_actions"] == []
        assert typed["execution_history"] == []
        assert typed["replan_count"] == 0

    def test_present_keys_are_kept(self) -> None:
        typed = rehydrate_state({"blacklisted_actions": ["x"], "replan_count": 3})
        assert typed["blacklisted_actions"] == ["x"]
        assert typed["replan_count"] == 3

    def test_filled_values_are_fresh_copies(self) -> None:
        a: dict[str, Any] = dict(rehydrate_state({}))
        b: dict[str, Any] = dict(rehydrate_state({}))
        a["blacklisted_actions"].append("x")
        assert b["blacklisted_actions"] == []
