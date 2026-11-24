"""Event-driven path benchmarks: GoapGraph.start / notify on waits (the thread API).

A thread paused in a wait is notified on every platform event; most events change none of
the wait's facts, so the common case is a notify that looks and keeps waiting.  Measured
with the in-memory saver and with SQLite, one thread and fifty.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph

GOAL = GoalSpec(conditions={"x": True})


def _waiting_graph(checkpointer: Any = None) -> tuple[GoapGraph, dict[str, bool]]:
    """A graph whose only action is a wait on ``x``; the sensor reads ``cell['x']``."""
    cell = {"x": False}

    def sense(ws: dict[str, Any]) -> dict[str, Any]:
        return {"x": cell["x"]}

    graph = GoapGraph(
        [ActionSpec.wait("hold", {"x": False}, {"x": True})],
        goal=GOAL,
        sensors=[sense],
        effects="observed",
        checkpointer=checkpointer,
    )
    return graph, cell


def _sqlite() -> Any:
    saver = pytest.importorskip(
        "langgraph.checkpoint.sqlite",
        reason="langgraph-checkpoint-sqlite not installed",
    )
    return saver.SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))


@pytest.mark.parametrize("saver", ["memory", "sqlite"])
def test_bench_notify_unchanged(benchmark: object, saver: str) -> None:
    """One thread in a wait; a notify that changes nothing."""
    graph, cell = _waiting_graph(_sqlite() if saver == "sqlite" else None)
    graph.start(world_state={"x": False}, thread_id="t")
    assert graph.waiting_action("t") == "hold"

    def one_notify() -> None:
        graph.notify("t", {"kind": "tick"})

    benchmark(one_notify)  # type: ignore[operator]
    assert graph.waiting_action("t") == "hold"
    cell["x"] = True
    assert graph.notify("t", {"kind": "tick"}).succeeded


@pytest.mark.parametrize("saver", ["memory", "sqlite"])
def test_bench_notify_fifty_threads(benchmark: object, saver: str) -> None:
    """Fifty threads in a wait, all notified once (a world change wakes every thread)."""
    graph, cell = _waiting_graph(_sqlite() if saver == "sqlite" else None)
    ids = [f"t{i}" for i in range(50)]
    for tid in ids:
        graph.start(world_state={"x": False}, thread_id=tid)

    def notify_all() -> None:
        for tid in ids:
            graph.notify(tid, {"kind": "world.changed"})

    benchmark(notify_all)  # type: ignore[operator]
    cell["x"] = True
    assert all(graph.notify(tid, {"kind": "tick"}).succeeded for tid in ids)


def test_bench_waiting_lookup(benchmark: object) -> None:
    """``waiting_action`` alone: the checkpoint read behind every notify."""
    graph, _ = _waiting_graph()
    graph.start(world_state={"x": False}, thread_id="t")
    assert benchmark(graph.waiting_action, "t") == "hold"  # type: ignore[operator]


def test_bench_start_to_goal(benchmark: object) -> None:
    """A fresh thread each round: start, one notify that changes nothing, one that ends it."""
    graph, cell = _waiting_graph()
    counter = {"n": 0}

    def run() -> None:
        cell["x"] = False
        counter["n"] += 1
        tid = f"t{counter['n']}"
        graph.start(world_state={"x": False}, thread_id=tid)
        graph.notify(tid, None)
        cell["x"] = True
        assert graph.notify(tid, None).succeeded

    benchmark(run)  # type: ignore[operator]
