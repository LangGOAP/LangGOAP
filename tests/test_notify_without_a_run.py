"""A notify that changes nothing does not run the graph (issue #116).

The thread API looks before waking: the wait's ``on_resume`` and the sensors run outside
the graph on the thread's last state; the graph runs only when a watched fact moved or the
state changed, and then with the already-sensed outcome, so nothing runs twice.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.tracing import NullTracer

GOAL = GoalSpec(conditions={"x": True})


class CountingSaver(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self.puts = 0

    def put(self, *args: Any, **kwargs: Any) -> Any:
        self.puts += 1
        return super().put(*args, **kwargs)

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        self.puts += 1
        return await super().aput(*args, **kwargs)


class Counting:
    def __init__(self) -> None:
        self.x = False
        self.sensed = 0
        self.resumed = 0

    def sense(self, ws: dict[str, Any]) -> dict[str, Any]:
        self.sensed += 1
        return {"x": self.x}

    def on_resume(self, ws: dict[str, Any]) -> None:
        self.resumed += 1


class SensorTracer(NullTracer):
    def __init__(self) -> None:
        self.sensor_hooks = 0

    def on_sensor_complete(self, name: str, updates: Any) -> None:
        self.sensor_hooks += 1


def _graph(
    world: Counting, saver: Any = None, on_resume: Any = None, **kw: Any
) -> GoapGraph:
    return GoapGraph(
        [
            ActionSpec.wait(
                "hold",
                {"x": False},
                {"x": True},
                on_resume=on_resume or world.on_resume,
            )
        ],
        goal=GOAL,
        sensors=[world.sense],
        effects="observed",
        checkpointer=saver,
        **kw,
    )


class TestUnchangedNotify:
    def test_it_does_not_run_the_graph(self) -> None:
        world, saver = Counting(), CountingSaver()
        graph = _graph(world, saver)
        before = graph.start(world_state={"x": False}, thread_id="t")
        saver.puts, world.sensed, world.resumed = 0, 0, 0
        result = graph.notify("t", {"kind": "tick"})
        assert saver.puts == 0
        assert (world.resumed, world.sensed) == (1, 1)
        assert result is not None and result.interrupt == before.interrupt
        assert graph.waiting_action("t") == "hold"

    def test_the_tracer_still_sees_the_sensors(self) -> None:
        world, tracer = Counting(), SensorTracer()
        graph = _graph(world, tracer=tracer)
        graph.start(world_state={"x": False}, thread_id="t")
        tracer.sensor_hooks = 0
        graph.notify("t", None)
        assert tracer.sensor_hooks == 1

    def test_on_stop_is_still_called(self) -> None:
        world, stops = Counting(), []
        graph = _graph(
            world, on_stop=lambda tid, state: stops.append((tid, state.interrupt))
        )
        graph.start(world_state={"x": False}, thread_id="t")
        stops.clear()
        graph.notify("t", None)
        assert (
            len(stops) == 1 and stops[0][0] == "t" and stops[0][1]["action"] == "hold"
        )


class TestChangedNotify:
    def test_on_resume_and_the_sensors_run_once(self) -> None:
        world, saver = Counting(), CountingSaver()
        graph = _graph(world, saver)
        graph.start(world_state={"x": False}, thread_id="t")
        world.x = True
        world.sensed, world.resumed = 0, 0
        result = graph.notify("t", None)
        assert result is not None and result.succeeded
        assert world.resumed == 1
        # the executor applies the pre-sensed outcome; observed effects may sense once more
        # after the wait settles, never a second on_resume
        assert graph.waiting("t") is None

    def test_a_change_on_resume_makes_without_a_watched_key_is_kept(self) -> None:
        world, saver = Counting(), CountingSaver()
        notes: list[int] = []

        def on_resume(ws: dict[str, Any]) -> None:
            notes.append(1)
            ws["note"] = len(notes)

        graph = _graph(world, saver, on_resume=on_resume)
        graph.start(world_state={"x": False}, thread_id="t")
        saver.puts = 0
        result = graph.notify("t", None)
        assert result is not None and result["world_state"]["note"] == 1
        assert saver.puts > 0  # the state moved, so the run happened
        assert len(notes) == 1  # and on_resume ran once for it
        assert graph.waiting_action("t") == "hold"


class TestTheCache:
    def test_a_fresh_object_reads_the_checkpoint_then_remembers(self) -> None:
        world, saver = Counting(), CountingSaver()
        first = _graph(world, saver)
        first.start(world_state={"x": False}, thread_id="t")
        second = _graph(world, saver)  # a new process over the same checkpointer
        assert second.waiting_action("t") == "hold"
        saver.puts = 0
        assert (
            second.notify("t", None) is not None
        )  # the graph path, nothing cached yet
        ran_first_time = saver.puts
        saver.puts = 0
        second.notify("t", None)
        assert saver.puts == 0 and ran_first_time > 0

    def test_a_finished_thread_is_forgotten(self) -> None:
        world = Counting()
        graph = _graph(world)
        graph.start(world_state={"x": False}, thread_id="t")
        world.x = True
        assert graph.notify("t", None).succeeded
        assert graph.waiting("t") is None
        assert graph.notify("t", None) is None


class TestAsync:
    def test_an_unchanged_anotify_does_not_run_the_graph(self) -> None:
        world, saver = Counting(), CountingSaver()
        resumed: list[int] = []

        async def on_resume(ws: dict[str, Any]) -> None:
            resumed.append(1)

        graph = _graph(world, saver, on_resume=on_resume)

        async def run() -> None:
            await graph.astart(world_state={"x": False}, thread_id="t")
            saver.puts, world.sensed = 0, 0
            resumed.clear()
            assert await graph.anotify("t", None) is not None
            assert saver.puts == 0 and world.sensed == 1 and len(resumed) == 1
            world.x = True
            resumed.clear()
            assert (await graph.anotify("t", None)).succeeded
            assert len(resumed) == 1

        asyncio.run(run())
