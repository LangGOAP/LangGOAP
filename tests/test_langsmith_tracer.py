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


class TestBoundedSearchBuffer:
    """Enterprise-hardening: the search-event buffer must be bounded
    and batched to protect LangSmith quotas and avoid O(n^2) upload
    growth for long plans.

    - ``max_search_events`` caps how many ``search_expand`` events are
      retained for a single root run.  When the cap fires, exactly one
      synthetic ``search_truncated`` marker is appended.
    - ``search_dead_end`` events bypass the cap (always low-cardinality
      and diagnostically valuable).
    - ``flush_every`` batches uploads to amortise the quadratic cost of
      resending the full list on every update.  Terminal lifecycle
      events (``on_plan_complete`` / ``on_plan_failed`` /
      ``on_goal_achieved``) force a final flush so no pending events
      are ever lost.
    """

    def test_default_cap_truncates_after_max_search_events(
        self, mock_client: MagicMock, goal: GoalSpec
    ) -> None:
        tracer = LangSmithTracer(
            client=mock_client,
            project_name="test",
            max_search_events=3,
        )
        tracer.on_plan_start(goal, {}, "a_star")
        for i in range(5):
            tracer.on_search_expand(
                node_id=i,
                state={},
                g=0.0,
                h=0.0,
                f=0.0,
                parent_id=None,
                action_name=None,
            )
        tracer.on_plan_complete(
            Plan(actions=(), total_cost=0.0),
            duration_ms=1.0,
        )

        # Find the last update_run call that carries search_events.
        search_event_calls = [
            c
            for c in mock_client.update_run.call_args_list
            if "search_events" in (c.kwargs.get("extra", {}).get("metadata", {}) or {})
        ]
        assert search_event_calls, "no update_run carried search_events"
        final_events = search_event_calls[-1].kwargs["extra"]["metadata"][
            "search_events"
        ]
        expand_events = [e for e in final_events if e["kind"] == "search_expand"]
        truncated = [e for e in final_events if e["kind"] == "search_truncated"]
        assert len(expand_events) == 3
        assert len(truncated) == 1
        assert truncated[0]["dropped"] == 2

    def test_truncation_event_recorded_once(
        self, mock_client: MagicMock, goal: GoalSpec
    ) -> None:
        tracer = LangSmithTracer(
            client=mock_client,
            project_name="test",
            max_search_events=2,
        )
        tracer.on_plan_start(goal, {}, "a_star")
        for i in range(10):
            tracer.on_search_expand(i, {}, 0.0, 0.0, 0.0, None, None)
        tracer.on_plan_complete(Plan(actions=(), total_cost=0.0), 0.0)

        search_event_calls = [
            c
            for c in mock_client.update_run.call_args_list
            if "search_events" in (c.kwargs.get("extra", {}).get("metadata", {}) or {})
        ]
        final_events = search_event_calls[-1].kwargs["extra"]["metadata"][
            "search_events"
        ]
        truncated = [e for e in final_events if e["kind"] == "search_truncated"]
        assert len(truncated) == 1
        assert truncated[0]["dropped"] == 8

    def test_flush_every_batches_uploads(
        self, mock_client: MagicMock, goal: GoalSpec
    ) -> None:
        tracer = LangSmithTracer(
            client=mock_client,
            project_name="test",
            flush_every=3,
        )
        tracer.on_plan_start(goal, {}, "a_star")
        mock_client.update_run.reset_mock()

        for i in range(5):
            tracer.on_search_expand(i, {}, 0.0, 0.0, 0.0, None, None)

        # With flush_every=3, only one batched upload has fired after
        # 5 expands (at the 3rd event); the remaining 2 are pending
        # in the buffer.  Default (flush_every=1) would issue 5.
        assert mock_client.update_run.call_count == 1

    def test_plan_complete_forces_final_flush(
        self, mock_client: MagicMock, goal: GoalSpec, plan: Plan
    ) -> None:
        tracer = LangSmithTracer(
            client=mock_client,
            project_name="test",
            flush_every=100,
        )
        tracer.on_plan_start(goal, {}, "a_star")
        for i in range(3):
            tracer.on_search_expand(i, {}, 0.0, 0.0, 0.0, None, None)
        # Nothing flushed yet under flush_every=100.
        mock_client.update_run.reset_mock()
        tracer.on_plan_complete(plan, duration_ms=1.0)

        # on_plan_complete must flush the pending events.
        payloads = [
            c.kwargs.get("extra", {}).get("metadata", {})
            for c in mock_client.update_run.call_args_list
        ]
        flushed = [p for p in payloads if "search_events" in p]
        assert flushed, "pending search_events were not flushed on plan_complete"
        assert len(flushed[-1]["search_events"]) == 3

    def test_dead_end_events_bypass_truncation_cap(
        self, mock_client: MagicMock, goal: GoalSpec
    ) -> None:
        tracer = LangSmithTracer(
            client=mock_client,
            project_name="test",
            max_search_events=2,
        )
        tracer.on_plan_start(goal, {}, "a_star")
        for i in range(10):  # blow past the cap
            tracer.on_search_expand(i, {}, 0.0, 0.0, 0.0, None, None)
        tracer.on_search_dead_end("exhausted", {"nodes_explored": 10})
        tracer.on_plan_complete(Plan(actions=(), total_cost=0.0), 0.0)

        search_event_calls = [
            c
            for c in mock_client.update_run.call_args_list
            if "search_events" in (c.kwargs.get("extra", {}).get("metadata", {}) or {})
        ]
        final_events = search_event_calls[-1].kwargs["extra"]["metadata"][
            "search_events"
        ]
        dead_ends = [e for e in final_events if e["kind"] == "search_dead_end"]
        assert len(dead_ends) == 1
        assert dead_ends[0]["reason"] == "exhausted"


class TestConcurrentActionTracing:
    """``LangSmithTracer`` must correlate concurrent ``on_action_start``
    / ``on_action_complete`` pairs by ``action_name`` so a parallel wave
    of N actions emits N distinct child runs whose outputs land on the
    correct run id.
    """

    def test_parallel_action_runs_do_not_clobber_each_other(
        self,
        mock_client: MagicMock,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> None:
        import threading

        tracer = LangSmithTracer(client=mock_client, project_name="test")
        tracer.on_plan_start(goal, {}, "a_star")

        # Two actions whose start/complete pairs interleave from
        # different threads.  Without per-action correlation the second
        # ``on_action_start`` would overwrite the first run id and both
        # ``on_action_complete`` calls would target the same child run.
        a, b = actions

        start = threading.Barrier(2)
        results: dict[str, _StubResult] = {
            "gather": _StubResult(action_name="gather", success=True),
            "process": _StubResult(action_name="process", success=False, error="boom"),
        }

        def _run(action: ActionSpec) -> None:
            start.wait()
            tracer.on_action_start(action, {})
            tracer.on_action_complete(results[action.name])

        threads = [threading.Thread(target=_run, args=(act,)) for act in (a, b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Two child create_run calls (plus the root) and two action
        # update_run calls — one per action, addressed to its own id.
        action_creates = [
            c
            for c in mock_client.create_run.call_args_list
            if c.kwargs.get("run_type") == "tool"
        ]
        assert len(action_creates) == 2
        name_to_id = {
            c.kwargs["name"].removeprefix("goap_action:"): c.kwargs["id"]
            for c in action_creates
        }
        assert set(name_to_id) == {"gather", "process"}

        # Each action update_run targets exactly one of the freshly
        # created child run ids.  The error/success payloads must not
        # be cross-wired between actions.
        action_updates = {
            c.args[0]: c.kwargs
            for c in mock_client.update_run.call_args_list
            if c.args and c.args[0] in name_to_id.values()
        }
        assert len(action_updates) == 2
        assert action_updates[name_to_id["gather"]]["error"] is None
        assert action_updates[name_to_id["process"]]["error"] == "boom"

    def test_action_complete_without_matching_start_is_noop(
        self,
        tracer: LangSmithTracer,
        mock_client: MagicMock,
        goal: GoalSpec,
    ) -> None:
        """Stray ``on_action_complete`` (no matching start) must not
        update an unrelated run id; in the legacy single-slot impl this
        could update the most recent action by accident.
        """
        tracer.on_plan_start(goal, {}, "a_star")
        mock_client.update_run.reset_mock()
        tracer.on_action_complete(_StubResult(action_name="never_started"))
        # No update_run targeting an action run id should fire.
        assert mock_client.update_run.call_count == 0

    def test_action_complete_accepts_state_update_dict_payload(
        self,
        mock_client: MagicMock,
        goal: GoalSpec,
        actions: list[ActionSpec],
    ) -> None:
        """The graph executor passes its state-update dict (with
        ``execution_history: [ActionResult]``) to ``on_action_complete``
        rather than a raw :class:`ActionResult`.  The tracer must
        resolve ``action_name`` from the inner :class:`ActionResult`
        so the corresponding child run is closed.

        Pins the regression that surfaced when
        :class:`~langgoap.tracing.langsmith.LangSmithTracer` was
        refactored to look up runs by ``result.action_name`` directly:
        the dict payload silently bypassed the close because
        ``getattr(dict, 'action_name', '')`` returned ``''``.
        """
        tracer = LangSmithTracer(client=mock_client, project_name="test")
        tracer.on_plan_start(goal, {}, "a_star")
        a = actions[0]

        tracer.on_action_start(a, {})
        # Mimic what GoapExecutor.acall actually passes \u2014 a state-update
        # dict with execution_history wrapping the ActionResult.
        executor_payload = {
            "world_state": {"x": True},
            "current_step": 1,
            "execution_history": [
                _StubResult(action_name=a.name, success=True),
            ],
        }
        tracer.on_action_complete(executor_payload)

        action_creates = [
            c
            for c in mock_client.create_run.call_args_list
            if c.kwargs.get("run_type") == "tool"
        ]
        assert len(action_creates) == 1
        run_id = action_creates[0].kwargs["id"]
        closes = [
            c
            for c in mock_client.update_run.call_args_list
            if c.args and c.args[0] == run_id and "outputs" in c.kwargs
        ]
        assert len(closes) == 1, "tracer must close the action run when given a dict"
