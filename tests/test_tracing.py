"""Tests for the PlanningTracer Protocol and concrete tracers."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.tracing import (
    LoggingTracer,
    MultiTracer,
    NullTracer,
    PlanningTracer,
)


class RecordingTracer:
    """Test helper that records every hook invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.calls.append(("on_plan_start", (goal, state, strategy_name)))

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.calls.append(("on_plan_complete", (plan, duration_ms)))

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        self.calls.append(("on_plan_failed", (reason, duration_ms)))

    def on_action_start(self, action: Any, state: Any) -> None:
        self.calls.append(("on_action_start", (action, state)))

    def on_action_complete(self, result: Any) -> None:
        self.calls.append(("on_action_complete", (result,)))

    def on_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        self.calls.append(
            ("on_action_retry", (action, attempt, exception, backoff_ms))
        )

    def on_strategy_chosen(self, strategy_name: str) -> None:
        self.calls.append(("on_strategy_chosen", (strategy_name,)))

    def on_replan(self, reason: str, new_plan: Any) -> None:
        self.calls.append(("on_replan", (reason, new_plan)))

    def on_goal_achieved(self, final_state: Any) -> None:
        self.calls.append(("on_goal_achieved", (final_state,)))

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        self.calls.append(("on_sensor_complete", (sensor_name, updates)))

    async def aon_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.calls.append(("aon_plan_start", (goal, state, strategy_name)))

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.calls.append(("aon_plan_complete", (plan, duration_ms)))

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
        self.calls.append(("aon_plan_failed", (reason, duration_ms)))

    async def aon_action_start(self, action: Any, state: Any) -> None:
        self.calls.append(("aon_action_start", (action, state)))

    async def aon_action_complete(self, result: Any) -> None:
        self.calls.append(("aon_action_complete", (result,)))

    async def aon_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        self.calls.append(
            ("aon_action_retry", (action, attempt, exception, backoff_ms))
        )

    async def aon_strategy_chosen(self, strategy_name: str) -> None:
        self.calls.append(("aon_strategy_chosen", (strategy_name,)))

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        self.calls.append(("aon_replan", (reason, new_plan)))

    async def aon_goal_achieved(self, final_state: Any) -> None:
        self.calls.append(("aon_goal_achieved", (final_state,)))

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        self.calls.append(("aon_sensor_complete", (sensor_name, updates)))

    # A* search-tree hooks (sync + async parity).
    def on_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.calls.append(
            ("on_search_expand", (node_id, state, g, h, f, parent_id, action_name))
        )

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        self.calls.append(("on_search_dead_end", (reason, detail)))

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.calls.append(("on_search_complete", (nodes_explored, duration_ms, found)))

    async def aon_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.calls.append(
            ("aon_search_expand", (node_id, state, g, h, f, parent_id, action_name))
        )

    async def aon_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        self.calls.append(("aon_search_dead_end", (reason, detail)))

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.calls.append(("aon_search_complete", (nodes_explored, duration_ms, found)))


class TestProtocol:
    def test_null_tracer_is_a_tracer(self) -> None:
        assert isinstance(NullTracer(), PlanningTracer)

    def test_logging_tracer_is_a_tracer(self) -> None:
        assert isinstance(LoggingTracer(), PlanningTracer)

    def test_recording_tracer_is_a_tracer(self) -> None:
        assert isinstance(RecordingTracer(), PlanningTracer)


class TestNullTracer:
    def test_all_hooks_are_noops(self) -> None:
        tracer = NullTracer()
        # Sync hooks
        tracer.on_plan_start(None, None, "A*")
        tracer.on_plan_complete(None, 1.0)
        tracer.on_plan_failed("no_plan", 1.0)
        tracer.on_action_start(None, None)
        tracer.on_action_complete(None)
        tracer.on_replan("failed", None)
        tracer.on_goal_achieved({})
        tracer.on_search_expand(0, None, 0.0, 0.0, 0.0, None, None)
        tracer.on_search_dead_end("exhausted", {})
        tracer.on_search_complete(0, 0.0, False)

    @pytest.mark.asyncio
    async def test_async_hooks_are_noops(self) -> None:
        tracer = NullTracer()
        await tracer.aon_plan_start(None, None, "A*")
        await tracer.aon_plan_complete(None, 1.0)
        await tracer.aon_plan_failed("no_plan", 1.0)
        await tracer.aon_action_start(None, None)
        await tracer.aon_action_complete(None)
        await tracer.aon_replan("failed", None)
        await tracer.aon_goal_achieved({})
        await tracer.aon_search_expand(0, None, 0.0, 0.0, 0.0, None, None)
        await tracer.aon_search_dead_end("exhausted", {})
        await tracer.aon_search_complete(0, 0.0, False)


class TestLoggingTracer:
    def test_logs_plan_events(self, caplog: pytest.LogCaptureFixture) -> None:
        tracer = LoggingTracer()
        with caplog.at_level("INFO", logger="langgoap.tracing"):
            tracer.on_plan_start(goal=None, state={"a": True}, strategy_name="A*")
            tracer.on_plan_complete(plan=None, duration_ms=12.3)
        assert any("plan_start" in r.message for r in caplog.records)
        assert any("plan_complete" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_async_delegates_to_sync(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracer = LoggingTracer()
        with caplog.at_level("INFO", logger="langgoap.tracing"):
            await tracer.aon_plan_start(goal=None, state={}, strategy_name="A*")
        assert any("plan_start" in r.message for r in caplog.records)

    def test_logs_search_events(self, caplog: pytest.LogCaptureFixture) -> None:
        tracer = LoggingTracer()
        with caplog.at_level("INFO", logger="langgoap.tracing"):
            tracer.on_search_expand(
                node_id=3,
                state={"a": True},
                g=1.0,
                h=2.0,
                f=3.0,
                parent_id=1,
                action_name="step",
            )
            tracer.on_search_dead_end(reason="exhausted", detail={"nodes": 42})
            tracer.on_search_complete(nodes_explored=42, duration_ms=5.5, found=False)
        messages = [r.message for r in caplog.records]
        assert any("search_expand" in m for m in messages)
        assert any("search_dead_end" in m for m in messages)
        assert any("search_complete" in m for m in messages)


class TestMultiTracer:
    def test_fans_out_sync_calls(self) -> None:
        r1 = RecordingTracer()
        r2 = RecordingTracer()
        multi = MultiTracer([r1, r2])
        multi.on_plan_start(None, None, "A*")
        assert len(r1.calls) == 1
        assert len(r2.calls) == 1

    @pytest.mark.asyncio
    async def test_fans_out_async_calls(self) -> None:
        r1 = RecordingTracer()
        r2 = RecordingTracer()
        multi = MultiTracer([r1, r2])
        await multi.aon_action_complete({"ok": True})
        assert r1.calls == [("aon_action_complete", ({"ok": True},))]
        assert r2.calls == [("aon_action_complete", ({"ok": True},))]

    def test_one_failing_tracer_does_not_break_others(self) -> None:
        class BrokenTracer:
            def on_plan_start(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("boom")

            # stub the rest
            def on_plan_complete(self, *a: Any, **k: Any) -> None: ...
            def on_plan_failed(self, *a: Any, **k: Any) -> None: ...
            def on_action_start(self, *a: Any, **k: Any) -> None: ...
            def on_action_complete(self, *a: Any, **k: Any) -> None: ...
            def on_replan(self, *a: Any, **k: Any) -> None: ...
            def on_goal_achieved(self, *a: Any, **k: Any) -> None: ...
            def on_sensor_complete(self, *a: Any, **k: Any) -> None: ...
            def on_search_expand(self, *a: Any, **k: Any) -> None: ...
            def on_search_dead_end(self, *a: Any, **k: Any) -> None: ...
            def on_search_complete(self, *a: Any, **k: Any) -> None: ...
            async def aon_plan_start(self, *a: Any, **k: Any) -> None: ...
            async def aon_plan_complete(self, *a: Any, **k: Any) -> None: ...
            async def aon_plan_failed(self, *a: Any, **k: Any) -> None: ...
            async def aon_action_start(self, *a: Any, **k: Any) -> None: ...
            async def aon_action_complete(self, *a: Any, **k: Any) -> None: ...
            async def aon_replan(self, *a: Any, **k: Any) -> None: ...
            async def aon_goal_achieved(self, *a: Any, **k: Any) -> None: ...
            async def aon_sensor_complete(self, *a: Any, **k: Any) -> None: ...
            async def aon_search_expand(self, *a: Any, **k: Any) -> None: ...
            async def aon_search_dead_end(self, *a: Any, **k: Any) -> None: ...
            async def aon_search_complete(self, *a: Any, **k: Any) -> None: ...

        good = RecordingTracer()
        multi = MultiTracer([BrokenTracer(), good])
        # Must not raise — tracer exceptions are swallowed.
        multi.on_plan_start(None, None, "A*")
        assert len(good.calls) == 1

    def test_fans_out_search_hooks(self) -> None:
        r1 = RecordingTracer()
        r2 = RecordingTracer()
        multi = MultiTracer([r1, r2])
        multi.on_search_expand(7, {"k": 1}, 1.0, 2.0, 3.0, 4, "move")
        multi.on_search_dead_end("exhausted", {"nodes_explored": 11})
        multi.on_search_complete(11, 4.2, False)
        assert [c[0] for c in r1.calls] == [
            "on_search_expand",
            "on_search_dead_end",
            "on_search_complete",
        ]
        assert [c[0] for c in r2.calls] == [
            "on_search_expand",
            "on_search_dead_end",
            "on_search_complete",
        ]

    @pytest.mark.asyncio
    async def test_fans_out_async_search_hooks(self) -> None:
        r1 = RecordingTracer()
        r2 = RecordingTracer()
        multi = MultiTracer([r1, r2])
        await multi.aon_search_expand(1, {"k": 0}, 0.0, 1.0, 1.0, None, None)
        await multi.aon_search_complete(1, 0.1, True)
        assert [c[0] for c in r1.calls] == ["aon_search_expand", "aon_search_complete"]
        assert [c[0] for c in r2.calls] == ["aon_search_expand", "aon_search_complete"]
