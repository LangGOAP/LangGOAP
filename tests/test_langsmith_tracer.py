"""Unit tests for ``LangSmithTracer``.

These tests substitute a ``MagicMock`` for ``langsmith.Client`` so no
network traffic is generated.  They verify four invariants:

1. Every hook translates to the expected ``create_run`` /
   ``update_run`` sequence with stable field semantics.
2. Async hooks produce the same call sequence as sync hooks.
3. Tracer exceptions never propagate to the caller — the planner
   must never be broken by observability.
4. A missing ``LANGCHAIN_API_KEY`` / ``LANGSMITH_API_KEY`` disables
   the tracer entirely and logs a single warning.

End-to-end wiring through ``GoapGraph`` lives in
``tests/integration/test_langsmith_end_to_end.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.types import Plan
from langgoap.tracing import LangSmithTracer, PlanningTracer


@dataclass
class _StubResult:
    action_name: str
    success: bool = True
    state_after: dict[str, Any] | None = None
    error: str | None = None


@pytest.fixture
def mock_client() -> MagicMock:
    """Fresh ``langsmith.Client`` mock per test."""
    return MagicMock(name="langsmith.Client")


@pytest.fixture
def tracer(mock_client: MagicMock) -> LangSmithTracer:
    return LangSmithTracer(client=mock_client, project_name="test-project")


@pytest.fixture
def goal() -> GoalSpec:
    return GoalSpec(conditions={"done": True})


@pytest.fixture
def actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="gather", preconditions={}, effects={"data": True}, cost=1.0),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects={"done": True},
            cost=1.0,
        ),
    ]


@pytest.fixture
def plan(actions: list[ActionSpec]) -> Plan:
    return Plan(actions=tuple(actions), total_cost=2.0)


class TestProtocolConformance:
    def test_satisfies_planning_tracer_protocol(self, tracer: LangSmithTracer) -> None:
        assert isinstance(tracer, PlanningTracer)


class TestPlanLifecycle:
    def test_plan_start_creates_root_run(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        tracer.on_plan_start(goal, {"ready": True}, "a_star")

        mock_client.create_run.assert_called_once()
        kwargs = mock_client.create_run.call_args.kwargs
        assert kwargs["name"] == "goap_plan"
        assert kwargs["run_type"] == "chain"
        assert kwargs["project_name"] == "test-project"
        assert "id" in kwargs
        assert "start_time" in kwargs
        # parent_run_id should be absent for the initial plan
        assert "parent_run_id" not in kwargs
        # Inputs include goal, state, strategy name
        inputs = kwargs["inputs"]
        assert inputs["strategy"] == "a_star"
        assert "goal" in inputs
        assert "state" in inputs

    def test_plan_complete_updates_root_with_plan_metadata(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
        plan: Plan,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        root_id = mock_client.create_run.call_args.kwargs["id"]

        tracer.on_plan_complete(plan, duration_ms=42.0)

        mock_client.update_run.assert_called_once()
        call = mock_client.update_run.call_args
        assert call.args[0] == root_id
        extra = call.kwargs["extra"]
        assert extra["metadata"]["planning_duration_ms"] == 42.0
        assert extra["metadata"]["plan"]["action_names"] == ["gather", "process"]
        assert extra["metadata"]["plan"]["total_cost"] == 2.0

    def test_plan_failed_finalizes_root_with_error(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        root_id = mock_client.create_run.call_args.kwargs["id"]

        tracer.on_plan_failed("no_plan", duration_ms=5.0)

        mock_client.update_run.assert_called_once()
        call = mock_client.update_run.call_args
        assert call.args[0] == root_id
        assert "end_time" in call.kwargs
        assert call.kwargs["error"] == "plan_failed: no_plan"
        assert call.kwargs["outputs"]["reason"] == "no_plan"

    def test_goal_achieved_finalizes_root(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        root_id = mock_client.create_run.call_args.kwargs["id"]

        tracer.on_goal_achieved({"done": True})

        call = mock_client.update_run.call_args
        assert call.args[0] == root_id
        assert call.kwargs["outputs"]["status"] == "goal_achieved"
        assert call.kwargs["outputs"]["final_state"] == {"done": True}
        assert "end_time" in call.kwargs

        # After finalization, no further update_run should target the old id.
        mock_client.update_run.reset_mock()
        tracer.on_plan_complete(None, duration_ms=1.0)
        mock_client.update_run.assert_not_called()


class TestActionChildRuns:
    def test_action_start_creates_child_run_under_root(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        root_id = mock_client.create_run.call_args.kwargs["id"]

        tracer.on_action_start(actions[0], {"ready": True})

        # Two create_run calls total: root + action
        assert mock_client.create_run.call_count == 2
        kwargs = mock_client.create_run.call_args.kwargs
        assert kwargs["name"] == "goap_action:gather"
        assert kwargs["run_type"] == "tool"
        assert kwargs["parent_run_id"] == root_id
        assert kwargs["inputs"]["action"]["name"] == "gather"

    def test_action_complete_updates_child_run(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        tracer.on_action_start(actions[0], {})
        action_id = mock_client.create_run.call_args.kwargs["id"]

        result = _StubResult(
            action_name="gather", success=True, state_after={"data": True}
        )
        tracer.on_action_complete(result)

        call = mock_client.update_run.call_args
        assert call.args[0] == action_id
        assert call.kwargs["outputs"]["result"]["action_name"] == "gather"
        assert call.kwargs["outputs"]["result"]["success"] is True
        assert call.kwargs["error"] is None

    def test_action_complete_with_failure_sets_error(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        tracer.on_action_start(actions[0], {})

        result = _StubResult(
            action_name="gather", success=False, error="network_timeout"
        )
        tracer.on_action_complete(result)

        call = mock_client.update_run.call_args
        assert call.kwargs["error"] == "network_timeout"


class TestReplanNestedRoots:
    def test_replan_opens_new_root_chained_to_previous(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
        plan: Plan,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        first_root = mock_client.create_run.call_args.kwargs["id"]
        tracer.on_plan_complete(plan, duration_ms=1.0)

        # Second on_plan_start signals a replan cycle.
        tracer.on_plan_start(goal, {"data": True}, "a_star")
        second_root = mock_client.create_run.call_args.kwargs["id"]

        # The previous root was closed before the new one opened.
        close_calls = [
            c
            for c in mock_client.update_run.call_args_list
            if c.args and c.args[0] == first_root and c.kwargs.get("outputs")
        ]
        assert any(
            c.kwargs["outputs"].get("status") == "superseded_by_replan"
            for c in close_calls
        )
        # New root's parent is the previous root — forming a chain.
        create_kwargs = mock_client.create_run.call_args.kwargs
        assert create_kwargs["parent_run_id"] == first_root
        assert second_root != first_root

    def test_on_replan_updates_current_root_metadata(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
        plan: Plan,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        tracer.on_plan_complete(plan, duration_ms=1.0)
        tracer.on_plan_start(goal, {"data": True}, "a_star")
        second_root = mock_client.create_run.call_args.kwargs["id"]
        mock_client.update_run.reset_mock()

        tracer.on_replan("action_failure", plan)

        mock_client.update_run.assert_called_once()
        call = mock_client.update_run.call_args
        assert call.args[0] == second_root
        assert call.kwargs["extra"]["metadata"]["replan_reason"] == "action_failure"


class TestAsyncParity:
    def test_async_hooks_produce_same_sequence_as_sync(
        self,
        mock_client: MagicMock,
        goal: GoalSpec,
        actions: list[ActionSpec],
        plan: Plan,
    ) -> None:
        async_tracer = LangSmithTracer(client=mock_client, project_name="test")

        async def run() -> None:
            await async_tracer.aon_plan_start(goal, {}, "a_star")
            await async_tracer.aon_plan_complete(plan, 1.0)
            await async_tracer.aon_action_start(actions[0], {})
            await async_tracer.aon_action_complete(
                _StubResult(action_name="gather", success=True)
            )
            await async_tracer.aon_goal_achieved({"done": True})

        asyncio.run(run())

        # 2 create_run calls (root + action)
        assert mock_client.create_run.call_count == 2
        # update_run calls: plan_complete, action_complete, goal_achieved
        assert mock_client.update_run.call_count == 3


class TestExceptionIsolation:
    def test_create_run_failure_does_not_propagate(
        self,
        mock_client: MagicMock,
        goal: GoalSpec,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_client.create_run.side_effect = RuntimeError("langsmith down")
        tracer = LangSmithTracer(client=mock_client)

        with caplog.at_level(logging.WARNING, logger="langgoap.tracing"):
            tracer.on_plan_start(goal, {}, "a_star")  # must not raise

        assert any(
            "LangSmithTracer create_run raised" in rec.message for rec in caplog.records
        )

    def test_update_run_failure_does_not_propagate(
        self,
        mock_client: MagicMock,
        goal: GoalSpec,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_client.update_run.side_effect = RuntimeError("langsmith flaky")
        tracer = LangSmithTracer(client=mock_client)
        tracer.on_plan_start(goal, {}, "a_star")

        with caplog.at_level(logging.WARNING, logger="langgoap.tracing"):
            tracer.on_goal_achieved({})  # must not raise

        assert any(
            "LangSmithTracer update_run raised" in rec.message for rec in caplog.records
        )


class TestDegradedMode:
    def test_missing_api_key_disables_tracer_with_single_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        goal: GoalSpec,
    ) -> None:
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING, logger="langgoap.tracing"):
            tracer = LangSmithTracer()

        warning_count = sum(
            1
            for rec in caplog.records
            if "LangSmithTracer" in rec.message and "no-op" in rec.message
        )
        assert warning_count == 1

        # All hooks are no-ops and must not raise or access a client.
        tracer.on_plan_start(goal, {}, "a_star")
        tracer.on_plan_complete(None, 1.0)
        tracer.on_plan_failed("x", 1.0)
        tracer.on_action_start(None, {})
        tracer.on_action_complete(None)
        tracer.on_replan("r", None)
        tracer.on_goal_achieved({})

    def test_api_key_present_constructs_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_pt_fake")
        # We do not actually want to construct a real Client here —
        # patch the import target at the call site.
        fake_client_instance = MagicMock()
        fake_client_cls = MagicMock(return_value=fake_client_instance)
        import langsmith

        monkeypatch.setattr(langsmith, "Client", fake_client_cls)

        tracer = LangSmithTracer()

        fake_client_cls.assert_called_once_with()
        assert tracer._client is fake_client_instance  # type: ignore[attr-defined]


class TestSearchEvents:
    """A* search hooks attach as events/metadata to the open ``goap_plan`` run.

    These follow the OpenTelemetry GenAI convention of capturing
    high-volume content as span *events* rather than nested child spans
    — LangSmith maps OTel events to LangSmith run events natively.
    """

    def test_search_expand_updates_root_with_event_metadata(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        tracer.on_plan_start(goal, {"a": True}, "a_star")
        root_id = mock_client.create_run.call_args.kwargs["id"]
        mock_client.update_run.reset_mock()

        tracer.on_search_expand(
            node_id=1,
            state={"a": True, "b": True},
            g=1.0,
            h=2.0,
            f=3.0,
            parent_id=0,
            action_name="a_to_b",
        )

        mock_client.update_run.assert_called_once()
        call = mock_client.update_run.call_args
        assert call.args[0] == root_id
        events = call.kwargs["extra"]["metadata"]["search_events"]
        assert isinstance(events, list) and len(events) == 1
        e = events[0]
        assert e["kind"] == "search_expand"
        assert e["node_id"] == 1
        assert e["parent_id"] == 0
        assert e["action_name"] == "a_to_b"
        assert e["g"] == 1.0 and e["h"] == 2.0 and e["f"] == 3.0

    def test_search_complete_emits_aggregate_metadata(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        mock_client.update_run.reset_mock()

        tracer.on_search_complete(nodes_explored=42, duration_ms=5.5, found=True)

        mock_client.update_run.assert_called_once()
        meta = mock_client.update_run.call_args.kwargs["extra"]["metadata"]
        assert meta["search_summary"] == {
            "nodes_explored": 42,
            "duration_ms": 5.5,
            "found": True,
        }

    def test_search_dead_end_records_event(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        mock_client.update_run.reset_mock()

        tracer.on_search_dead_end(reason="not_reachable", detail={"missing": ["b"]})

        mock_client.update_run.assert_called_once()
        events = mock_client.update_run.call_args.kwargs["extra"]["metadata"][
            "search_events"
        ]
        assert events[-1]["kind"] == "search_dead_end"
        assert events[-1]["reason"] == "not_reachable"
        assert events[-1]["detail"] == {"missing": ["b"]}

    def test_search_events_are_noops_without_open_root(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
    ) -> None:
        """No root run open → no update_run issued."""
        tracer.on_search_expand(1, {}, 0.0, 0.0, 0.0, None, None)
        tracer.on_search_dead_end("exhausted", {})
        tracer.on_search_complete(0, 0.0, False)
        mock_client.update_run.assert_not_called()

    def test_search_events_disabled_when_no_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        goal: GoalSpec,
    ) -> None:
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        tracer = LangSmithTracer()
        # Must not raise and must not attempt any client access.
        tracer.on_plan_start(goal, {}, "a_star")
        tracer.on_search_expand(1, {}, 0.0, 0.0, 0.0, None, None)
        tracer.on_search_dead_end("exhausted", {})
        tracer.on_search_complete(0, 0.0, False)

    @pytest.mark.asyncio
    async def test_async_search_hooks_delegate_to_sync(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        tracer.on_plan_start(goal, {}, "a_star")
        mock_client.update_run.reset_mock()

        await tracer.aon_search_expand(1, {}, 1.0, 2.0, 3.0, 0, "move")
        await tracer.aon_search_dead_end("exhausted", {"nodes_explored": 11})
        await tracer.aon_search_complete(11, 0.5, False)

        # Three updates total: expand, dead_end (event append), complete (summary).
        assert mock_client.update_run.call_count == 3
