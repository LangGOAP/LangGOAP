"""End-to-end integration tests for ``LangSmithTracer`` through ``GoapGraph``.

A ``MagicMock`` stands in for ``langsmith.Client`` so no network traffic
is generated.  We compile a ``GoapGraph`` with the mocked tracer and
assert that a successful planner → executor → observer cycle emits the
expected run tree:

* one root run of type ``"chain"`` for the whole plan
* one child run of type ``"tool"`` per executed action, each with
  ``parent_run_id`` pointing at the root
* a final ``update_run`` on the root reporting ``goal_achieved``

Both ``invoke`` and ``ainvoke`` paths are exercised — the NS2
``RunnableLambda(func, afunc)`` wiring means the same tracer must see
both sync and async hooks without any code duplication in the tracer.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.tracing import LangSmithTracer


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


def _root_run_call(calls: list[Any]) -> Any:
    for call in calls:
        if call.kwargs.get("run_type") == "chain":
            return call
    raise AssertionError("No chain-type create_run call found")


def _tool_run_calls(calls: list[Any]) -> list[Any]:
    return [c for c in calls if c.kwargs.get("run_type") == "tool"]


class TestLangSmithTracerSyncPath:
    def test_successful_run_emits_root_and_per_action_child_runs(self) -> None:
        client = MagicMock(name="langsmith.Client")
        tracer = LangSmithTracer(client=client, project_name="test-e2e")
        graph = GoapGraph(_actions(), tracer=tracer)

        result = graph.invoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )
        assert result["status"] == "goal_achieved"

        create_calls = client.create_run.call_args_list
        # 1 root run + 2 tool runs (one per action).
        assert len(create_calls) == 3

        root = _root_run_call(create_calls)
        assert root.kwargs["name"] == "goap_plan"
        assert root.kwargs["project_name"] == "test-e2e"
        root_id = root.kwargs["id"]

        tool_calls = _tool_run_calls(create_calls)
        assert len(tool_calls) == 2
        tool_names = [c.kwargs["name"] for c in tool_calls]
        assert tool_names == ["goap_action:gather", "goap_action:process"]
        for tc in tool_calls:
            assert tc.kwargs["parent_run_id"] == root_id

        # Final root update must report goal_achieved.
        root_updates = [
            c
            for c in client.update_run.call_args_list
            if c.args and c.args[0] == root_id
        ]
        goal_updates = [
            c
            for c in root_updates
            if c.kwargs.get("outputs", {}).get("status") == "goal_achieved"
        ]
        assert len(goal_updates) == 1

        # Each tool run was closed with an outputs payload.
        for tc in tool_calls:
            tool_id = tc.kwargs["id"]
            closes = [
                c
                for c in client.update_run.call_args_list
                if c.args and c.args[0] == tool_id and "outputs" in c.kwargs
            ]
            assert len(closes) == 1

    def test_unreachable_goal_fires_plan_failed_error(self) -> None:
        client = MagicMock(name="langsmith.Client")
        tracer = LangSmithTracer(client=client)
        graph = GoapGraph(_actions(), tracer=tracer)

        result = graph.invoke(
            goal=GoalSpec(conditions={"impossible": True}),
            world_state={},
        )
        assert result["status"] == "no_plan"

        # Exactly one root run was created.
        create_calls = client.create_run.call_args_list
        root_calls = [c for c in create_calls if c.kwargs.get("run_type") == "chain"]
        assert len(root_calls) == 1
        root_id = root_calls[0].kwargs["id"]

        # The root run was finalized with an error.
        error_updates = [
            c
            for c in client.update_run.call_args_list
            if c.args
            and c.args[0] == root_id
            and "plan_failed" in (c.kwargs.get("error") or "")
        ]
        assert len(error_updates) == 1


class TestLangSmithTracerAsyncPath:
    def test_ainvoke_routes_async_hooks_to_same_tracer(self) -> None:
        client = MagicMock(name="langsmith.Client")
        tracer = LangSmithTracer(client=client, project_name="test-e2e-async")
        graph = GoapGraph(_actions(), tracer=tracer)

        async def run() -> Any:
            return await graph.ainvoke(
                goal=GoalSpec(conditions={"done": True}),
                world_state={},
            )

        result = asyncio.run(run())
        assert result["status"] == "goal_achieved"

        create_calls = client.create_run.call_args_list
        assert len(create_calls) == 3
        tool_calls = _tool_run_calls(create_calls)
        assert len(tool_calls) == 2

        # Root still finalises even though the async path is used.
        root = _root_run_call(create_calls)
        root_id = root.kwargs["id"]
        goal_updates = [
            c
            for c in client.update_run.call_args_list
            if c.args
            and c.args[0] == root_id
            and c.kwargs.get("outputs", {}).get("status") == "goal_achieved"
        ]
        assert len(goal_updates) == 1


class TestLangSmithTracerSurvivesBrokenClient:
    def test_broken_create_run_does_not_break_planner(self) -> None:
        client = MagicMock(name="langsmith.Client")
        client.create_run.side_effect = RuntimeError("langsmith down")
        tracer = LangSmithTracer(client=client)
        graph = GoapGraph(_actions(), tracer=tracer)

        # Planner must still succeed — observability must never break
        # the planning loop.
        result = graph.invoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )
        assert result["status"] == "goal_achieved"
