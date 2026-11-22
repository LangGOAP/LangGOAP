"""An event-driven order, end to end, on the API of issue #96.

A small delivery platform: the payment and the kitchen are waited for, the
kitchen may reject, a courier is dispatched, support approves a refund.  The
platform publishes events while actions run; the graph object routes them.
Nothing in the actions re-reads the platform: the sensor does.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoapGraph, NullTracer
from langgoap.tracing import current_thread_id


class Platform:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.listeners: list[Any] = []
        self.log: list[str] = []

    def place(self, order_id: str) -> None:
        self.orders[order_id] = {"paid": False, "kitchen": "idle", "courier": None}

    def publish(self, order_id: str, event: str) -> None:
        self.log.append(f"{order_id}:{event}")
        for listener in self.listeners:
            listener(order_id, event)

    def pay(self, order_id: str) -> None:
        self.orders[order_id]["paid"] = True
        self.publish(order_id, "payment.completed")

    def kitchen(self, order_id: str, status: str) -> None:
        self.orders[order_id]["kitchen"] = status
        self.publish(order_id, f"kitchen.{status}")

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        order = self.orders[world_state["order_id"]]
        return {
            "paid": order["paid"],
            "kitchen": order["kitchen"],
            "courier": order["courier"] is not None,
            "refunded": order.get("refunded", False),
        }


class Decisions(NullTracer):
    def __init__(self) -> None:
        self.by_thread: dict[str, list[str]] = {}

    def _note(self, text: str) -> None:
        self.by_thread.setdefault(str(current_thread_id()), []).append(text)

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self._note("plan: " + " > ".join(a.name for a in plan.actions))

    def on_replan(
        self, reason: str, new_plan: Any, duration_ms: float | None = None
    ) -> None:
        assert duration_ms is not None
        self._note("replan: " + " > ".join(a.name for a in new_plan.actions))


def build(platform: Platform, tracer: Decisions) -> GoapGraph:
    def order_of(world_state: dict[str, Any]) -> dict[str, Any]:
        return platform.orders[world_state["order_id"]]

    def start_cooking(world_state: dict[str, Any]) -> None:
        # the kitchen starts as soon as it hears of the payment: an event inside an action
        platform.kitchen(world_state["order_id"], "cooking")

    def dispatch(world_state: dict[str, Any]) -> None:
        order_of(world_state)["courier"] = "c-1"

    def refund(world_state: dict[str, Any]) -> None:
        order_of(world_state)["refunded"] = True

    actions = [
        ActionSpec.wait("await_payment", {"paid": False}, {"paid": True}),
        ActionSpec(
            "send_to_kitchen",
            {"paid": True, "kitchen": "idle"},
            {"kitchen": "cooking"},
            execute=start_cooking,
        ),
        ActionSpec.wait("await_kitchen", {"kitchen": "cooking"}, {"kitchen": "ready"}),
        ActionSpec(
            "dispatch",
            {"kitchen": "ready"},
            {"courier": True, "settled": True},
            execute=dispatch,
        ),
        ActionSpec(
            "refund",
            {"kitchen": "rejected"},
            {"refunded": True, "settled": True},
            cost=10.0,
            execute=refund,
            require_human_approval=True,
        ),
    ]
    graph = GoapGraph(
        actions,
        goal={"settled": True},
        sensors=[
            platform.sense,
            lambda ws: {"settled": ws["courier"] or ws["refunded"]},
        ],
        effects="observed",
        tracer=tracer,
        max_actions=20,
    )
    platform.listeners.append(lambda order_id, event: graph.notify(order_id, event))
    return graph


def test_two_orders_one_delivered_one_refunded() -> None:
    platform, tracer = Platform(), Decisions()
    graph = build(platform, tracer)
    for order_id in ("o-1", "o-2"):
        platform.place(order_id)
        graph.start({"order_id": order_id}, thread_id=order_id)
        assert graph.waiting(order_id)["action"] == "await_payment"  # type: ignore[index]

    platform.pay("o-1")  # wakes o-1; send_to_kitchen publishes kitchen.cooking mid-run
    assert graph.waiting("o-1")["action"] == "await_kitchen"  # type: ignore[index]
    assert graph.waiting("o-2")["action"] == "await_payment"  # type: ignore[index]

    platform.kitchen("o-1", "ready")
    assert graph.waiting("o-1") is None
    assert platform.orders["o-1"]["courier"] == "c-1"

    platform.pay("o-2")
    platform.kitchen("o-2", "rejected")
    waiting = graph.waiting("o-2")
    assert waiting is not None and waiting["type"] == "goap_action_approval"
    assert waiting["action"] == "refund"
    platform.publish("o-2", "noise")  # an event is not an answer
    assert graph.waiting("o-2")["type"] == "goap_action_approval"  # type: ignore[index]
    done = graph.resume(True, thread_id="o-2")
    assert done is not None and done["status"] == "goal_achieved"
    assert platform.orders["o-2"]["refunded"] is True

    assert tracer.by_thread["o-1"][0].startswith("plan: await_payment")
    assert any(
        line.startswith("replan:") and "refund" in line
        for line in tracer.by_thread["o-2"]
    )
    assert not any("refund" in line for line in tracer.by_thread["o-1"])


async def test_the_same_order_on_the_async_path() -> None:
    platform, tracer = Platform(), Decisions()
    graph = build(platform, tracer)
    platform.listeners.clear()  # events are delivered by hand on the async path
    platform.place("o-1")
    await graph.astart({"order_id": "o-1"}, thread_id="o-1")
    platform.pay("o-1")
    await graph.anotify("o-1")
    assert graph.waiting("o-1")["action"] == "await_kitchen"  # type: ignore[index]
    platform.kitchen("o-1", "ready")
    done = await graph.anotify("o-1")
    assert done is not None and done["status"] == "goal_achieved"
