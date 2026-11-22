"""Routine planning and execution log lines are DEBUG, so a host's INFO handler never
runs inside the planner's timed span (issue #109)."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.tracing import NullTracer

GOAL = GoalSpec(conditions={"done": True})


def _actions() -> list[ActionSpec]:
    """Two steps, and a re-plan: the first run of ``half`` reports the opposite of what
    it declared, so the observer sends the graph back to the planner once."""
    runs: list[int] = []

    def half(ws: dict[str, Any]) -> dict[str, Any]:
        runs.append(1)
        return {"half": len(runs) > 1}

    def finish(ws: dict[str, Any]) -> dict[str, Any]:
        return {"done": True}

    return [
        ActionSpec(name="half", preconditions={}, effects={"half": True}, execute=half),
        ActionSpec(
            name="finish",
            preconditions={"half": True},
            effects={"done": True},
            execute=finish,
        ),
    ]


def _run(executor: str, tracer: Any = None) -> Any:
    graph = GoapGraph(_actions(), goal=GOAL, executor=executor, tracer=tracer)  # type: ignore[arg-type]
    return graph.invoke(world_state={"half": False, "done": False})


class DurationTracer(NullTracer):
    def __init__(self) -> None:
        self.durations: list[float] = []

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.durations.append(duration_ms)

    def on_replan(self, reason: str, new_plan: Any, duration_ms: float = 0.0) -> None:
        self.durations.append(duration_ms)


class SleepingHandler(logging.Handler):
    """What an expensive host handler does to a log record: takes its time."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        time.sleep(self.seconds)


@pytest.mark.parametrize("executor", ["sequential", "parallel"])
class TestRoutineLinesAreDebug:
    def test_a_routine_run_says_nothing_at_info(
        self, executor: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracer = DurationTracer()
        with caplog.at_level(logging.INFO, logger="langgoap"):
            result = _run(executor, tracer)
        assert result.succeeded
        assert len(tracer.durations) == 2  # the plan and the re-plan both happened
        assert [(r.name, r.getMessage()) for r in caplog.records] == []

    def test_the_lines_are_still_there_at_debug(
        self, executor: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="langgoap"):
            _run(executor)
        messages = [r.getMessage() for r in caplog.records]
        for start in ("Planning for goal", "Plan found", "Replanning (replan #1)"):
            assert any(m.startswith(start) for m in messages), (start, messages)
        step = (
            "Executing action"
            if executor == "sequential"
            else "ParallelExecutor: running"
        )
        assert any(m.startswith(step) for m in messages), (step, messages)
        assert all(r.levelno == logging.DEBUG for r in caplog.records), [
            (r.levelname, r.getMessage())
            for r in caplog.records
            if r.levelno != logging.DEBUG
        ]


class TestTheTimedSpan:
    """A host that puts a slow handler on the root logger at INFO (mcp's ``MCPServer`` does,
    with rich) must not see it counted as planning time."""

    SLEEP = 0.2

    def _with_root_handler(
        self, level: int, executor: str
    ) -> tuple[DurationTracer, SleepingHandler]:
        root = logging.getLogger()
        handler = SleepingHandler(self.SLEEP)
        handler.setLevel(level)
        old_level = root.level
        root.setLevel(level)
        root.addHandler(handler)
        try:
            tracer = DurationTracer()
            assert _run(executor, tracer).succeeded
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
        return tracer, handler

    @pytest.mark.parametrize("executor", ["sequential", "parallel"])
    def test_an_info_handler_stays_out_of_the_planners_time(
        self, executor: str
    ) -> None:
        tracer, handler = self._with_root_handler(logging.INFO, executor)
        assert handler.records == []
        assert max(tracer.durations) < self.SLEEP * 1000

    def test_the_span_does_contain_the_log_calls(self) -> None:
        """The control: at DEBUG the same handler fires inside the span, which is why the
        routine lines must not be INFO."""
        tracer, handler = self._with_root_handler(logging.DEBUG, "sequential")
        assert handler.records
        assert max(tracer.durations) >= self.SLEEP * 1000
