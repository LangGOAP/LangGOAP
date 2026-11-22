"""Tracer hooks know their run, and every planning round reports its duration (issue #96)."""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoapGraph, LoggingTracer, MultiTracer, NullTracer
from langgoap.tracing import current_thread_id


class Shop:
    def __init__(self) -> None:
        self.kitchen = "cooking"

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        return {"kitchen": self.kitchen}


def _actions(shop: Shop) -> list[ActionSpec]:
    def cook(world_state: dict[str, Any]) -> dict[str, Any]:
        shop.kitchen = "rejected"
        return {"kitchen": "rejected"}

    return [
        ActionSpec("cook", {"kitchen": "cooking"}, {"kitchen": "ready"}, execute=cook),
        ActionSpec("dispatch", {"kitchen": "ready"}, {"delivered": True}),
        ActionSpec("refund", {"kitchen": "rejected"}, {"delivered": True}, cost=9.0),
    ]


class TestReplanDuration:
    def test_on_replan_receives_duration_ms(self) -> None:
        seen: list[Any] = []

        class Timed(NullTracer):
            def on_replan(
                self, reason: str, new_plan: Any, duration_ms: float | None = None
            ) -> None:
                seen.append(duration_ms)

        shop = Shop()
        graph = GoapGraph(
            _actions(shop),
            goal={"delivered": True},
            sensors=[shop.sense],
            tracer=Timed(),
        )
        assert graph.start(thread_id="t")["status"] == "goal_achieved"  # type: ignore[index]
        assert len(seen) == 1
        assert isinstance(seen[0], float) and seen[0] >= 0

    def test_a_tracer_with_the_old_signature_still_works(self) -> None:
        seen: list[str] = []

        class Old(NullTracer):
            def on_replan(self, reason: str, new_plan: Any) -> None:  # type: ignore[override]
                seen.append(reason)

        shop = Shop()
        graph = GoapGraph(
            _actions(shop), goal={"delivered": True}, sensors=[shop.sense], tracer=Old()
        )
        assert graph.start(thread_id="t")["status"] == "goal_achieved"  # type: ignore[index]
        assert len(seen) == 1

    def test_multi_tracer_passes_it_to_those_that_take_it(self) -> None:
        new: list[Any] = []
        old: list[str] = []

        class New(NullTracer):
            def on_replan(
                self, reason: str, new_plan: Any, duration_ms: float | None = None
            ) -> None:
                new.append(duration_ms)

        class Old(NullTracer):
            def on_replan(self, reason: str, new_plan: Any) -> None:  # type: ignore[override]
                old.append(reason)

        shop = Shop()
        graph = GoapGraph(
            _actions(shop),
            goal={"delivered": True},
            sensors=[shop.sense],
            tracer=MultiTracer([New(), Old(), LoggingTracer()]),
        )
        graph.start(thread_id="t")
        assert len(new) == 1 and isinstance(new[0], float)
        assert len(old) == 1

    async def test_async_path_reports_it_too(self) -> None:
        seen: list[Any] = []

        class Timed(NullTracer):
            async def aon_replan(
                self, reason: str, new_plan: Any, duration_ms: float | None = None
            ) -> None:
                seen.append(duration_ms)

        shop = Shop()
        graph = GoapGraph(
            _actions(shop),
            goal={"delivered": True},
            sensors=[shop.sense],
            tracer=Timed(),
        )
        await graph.astart(thread_id="t")
        assert len(seen) == 1 and isinstance(seen[0], float)


class TestCurrentThreadId:
    def test_hooks_can_ask_which_thread_they_run_for(self) -> None:
        seen: set[str | None] = set()

        class Who(NullTracer):
            def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
                seen.add(current_thread_id())

            def on_action_complete(self, result: Any) -> None:
                seen.add(current_thread_id())

        graph = GoapGraph(
            [ActionSpec("finish", {}, {"done": True})],
            goal={"done": True},
            tracer=Who(),
        )
        graph.start(thread_id="order-7")
        assert seen == {"order-7"}

    def test_outside_a_run_there_is_no_thread(self) -> None:
        assert current_thread_id() is None

    def test_actions_and_sensors_can_ask_too(self) -> None:
        seen: list[str | None] = []

        def sense(world_state: dict[str, Any]) -> dict[str, Any]:
            seen.append(current_thread_id())
            return {}

        graph = GoapGraph(
            [
                ActionSpec(
                    "finish",
                    {},
                    {"done": True},
                    execute=lambda ws: seen.append(current_thread_id()),
                )
            ],
            goal={"done": True},
            sensors=[sense],
        )
        graph.start(thread_id="order-9")
        assert seen and set(seen) == {"order-9"}


class TestThreadIdReachesSyncCodeOnTheAsyncPath:
    async def test_sync_actions_and_sensors_run_in_a_worker_still_know_the_thread(
        self,
    ) -> None:
        """The async path runs sync callables in a worker thread; the run's
        context must travel with them."""
        seen: dict[str, str | None] = {}

        def sense(world_state: dict[str, Any]) -> dict[str, Any]:
            seen["sensor"] = current_thread_id()
            return {}

        def finish(world_state: dict[str, Any]) -> None:
            seen["action"] = current_thread_id()

        graph = GoapGraph(
            [ActionSpec("finish", {}, {"done": True}, execute=finish)],
            goal={"done": True},
            sensors=[sense],
        )
        await graph.astart(thread_id="order-3")
        assert seen == {"sensor": "order-3", "action": "order-3"}
