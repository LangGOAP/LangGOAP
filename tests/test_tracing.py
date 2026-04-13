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

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        self.calls.append(("aon_replan", (reason, new_plan)))

    async def aon_goal_achieved(self, final_state: Any) -> None:
        self.calls.append(("aon_goal_achieved", (final_state,)))

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        self.calls.append(("aon_sensor_complete", (sensor_name, updates)))


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
            async def aon_plan_start(self, *a: Any, **k: Any) -> None: ...
            async def aon_plan_complete(self, *a: Any, **k: Any) -> None: ...
            async def aon_plan_failed(self, *a: Any, **k: Any) -> None: ...
            async def aon_action_start(self, *a: Any, **k: Any) -> None: ...
            async def aon_action_complete(self, *a: Any, **k: Any) -> None: ...
            async def aon_replan(self, *a: Any, **k: Any) -> None: ...
            async def aon_goal_achieved(self, *a: Any, **k: Any) -> None: ...
            async def aon_sensor_complete(self, *a: Any, **k: Any) -> None: ...

        good = RecordingTracer()
        multi = MultiTracer([BrokenTracer(), good])
        # Must not raise — tracer exceptions are swallowed.
        multi.on_plan_start(None, None, "A*")
        assert len(good.calls) == 1
