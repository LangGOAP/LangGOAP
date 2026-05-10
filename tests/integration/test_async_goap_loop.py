"""Integration tests for the async GOAP planning-execution loop.

These tests verify end-to-end behavior through GoapGraph.ainvoke(),
including async action callables, mixed sync/async chains, replanning,
and rich world state.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langgoap.actions import ActionSpec, GoapAction
from langgoap.goals import GoalPolicy, GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.types import ReplanStrategy
from tests.conftest import make_action as _action


class TestAsyncGoapLoop:
    async def test_async_pipeline_achieves_goal(self) -> None:
        """Full async pipeline: gather → clean → analyze → report."""

        async def gather(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"has_raw_data": True}

        async def clean(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"has_clean_data": True}

        async def analyze(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"has_analysis": True}

        async def report(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"report_complete": True}

        actions = [
            ActionSpec(name="gather", effects={"has_raw_data": True}, aexecute=gather),
            ActionSpec(
                name="clean",
                preconditions={"has_raw_data": True},
                effects={"has_clean_data": True},
                aexecute=clean,
            ),
            ActionSpec(
                name="analyze",
                preconditions={"has_clean_data": True},
                effects={"has_analysis": True},
                aexecute=analyze,
            ),
            ActionSpec(
                name="report",
                preconditions={"has_analysis": True},
                effects={"report_complete": True},
                aexecute=report,
            ),
        ]

        result = await GoapGraph(actions=actions).ainvoke(
            goal=GoalSpec(conditions={"report_complete": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["report_complete"] is True
        history = result["execution_history"]
        names = [h.action_name for h in history if h.success]
        assert names == ["gather", "clean", "analyze", "report"]

    async def test_mixed_sync_async_pipeline(self) -> None:
        """Pipeline with alternating sync and async actions."""

        async def async_gather(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"has_data": True, "data": [1, 2, 3]}

        def sync_process(ws: dict[str, Any]) -> dict[str, Any]:
            data = ws.get("data", [])
            return {"processed": True, "total": sum(data)}

        async def async_report(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"report_ready": True}

        actions = [
            ActionSpec(
                name="gather", effects={"has_data": True}, aexecute=async_gather
            ),
            ActionSpec(
                name="process",
                preconditions={"has_data": True},
                effects={"processed": True},
                execute=sync_process,
            ),
            ActionSpec(
                name="report",
                preconditions={"processed": True},
                effects={"report_ready": True},
                aexecute=async_report,
            ),
        ]

        result = await GoapGraph(actions=actions).ainvoke(
            goal=GoalSpec(conditions={"report_ready": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["total"] == 6
        assert result["world_state"]["report_ready"] is True

    async def test_async_replanning_on_failure(self) -> None:
        """Async action failure triggers replanning and eventual success."""
        call_count = 0

        async def flaky_action(ws: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError(f"transient failure #{call_count}")
            return {"done": True}

        actions = [
            ActionSpec(name="flaky", effects={"done": True}, aexecute=flaky_action),
        ]

        result = await GoapGraph(actions=actions).ainvoke(
            goal=GoalSpec(conditions={"done": True}, policy=GoalPolicy(max_replans=5)),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) == 2

    async def test_async_with_rich_world_state(self) -> None:
        """Async actions passing rich data (lists, dicts) through world_state."""

        async def fetch_documents(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {
                "has_documents": True,
                "documents": [
                    {"title": "GOAP Planning", "content": "..."},
                    {"title": "LangGraph", "content": "..."},
                ],
            }

        async def summarize(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            docs = ws.get("documents", [])
            titles = [d["title"] for d in docs]
            return {
                "has_summary": True,
                "summary": f"Summary of {len(docs)} docs: {', '.join(titles)}",
            }

        actions = [
            ActionSpec(
                name="fetch",
                preconditions={"has_query": True},
                effects={"has_documents": True},
                aexecute=fetch_documents,
            ),
            ActionSpec(
                name="summarize",
                preconditions={"has_documents": True},
                effects={"has_summary": True},
                aexecute=summarize,
            ),
        ]

        result = await GoapGraph(actions=actions).ainvoke(
            goal=GoalSpec(conditions={"has_summary": True}),
            world_state={"has_query": True, "query": "What is GOAP?"},
        )

        assert result["status"] == "goal_achieved"
        assert "GOAP Planning" in result["world_state"]["summary"]
        assert len(result["world_state"]["documents"]) == 2

    async def test_async_never_strategy_stops_on_failure(self) -> None:
        """NEVER replan strategy with async action stops immediately on failure."""

        async def failing_action(ws: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("permanent async failure")

        actions = [
            ActionSpec(name="fail", effects={"done": True}, aexecute=failing_action),
        ]

        result = await GoapGraph(actions=actions).ainvoke(
            goal=GoalSpec(
                conditions={"done": True},
                policy=GoalPolicy(replan_strategy=ReplanStrategy.NEVER),
            ),
            world_state={},
        )

        assert result["status"] == "failed"

    async def test_async_goap_action_subclass_pipeline(self) -> None:
        """End-to-end with GoapAction subclasses using aexecute."""

        class AsyncFetch(GoapAction):
            preconditions: dict[str, Any] = {}
            effects = {"has_data": True}

            async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
                await asyncio.sleep(0)
                return {"has_data": True}

        class AsyncProcess(GoapAction):
            preconditions = {"has_data": True}
            effects = {"done": True}

            async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
                await asyncio.sleep(0)
                return {"done": True}

        actions = [AsyncFetch().to_spec(), AsyncProcess().to_spec()]

        result = await GoapGraph(actions=actions).ainvoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"

    async def test_async_goap_action_with_sync_execute_via_executor(self) -> None:
        """GoapAction subclass with only sync execute works through ainvoke.

        The default aexecute delegates to sync execute via run_in_executor.
        """

        class SyncStep(GoapAction):
            effects = {"done": True}

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"done": True, "source": "sync_execute"}

        actions = [SyncStep().to_spec()]
        result = await GoapGraph(actions=actions).ainvoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["source"] == "sync_execute"
