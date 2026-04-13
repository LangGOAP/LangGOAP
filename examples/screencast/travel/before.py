"""Travel disruption: Hardcoded LangGraph workflow with manual fallback edges.

This version handles disruptions — but look at how much routing code is needed.
Every failure mode requires a dedicated conditional edge. Adding a new transport
option means touching the graph structure, not just the business logic.

Run: uv run python examples/screencast/travel/before.py
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TravelState(TypedDict, total=False):
    destination: str
    ceremony_time: str
    status: str
    transport_mode: str | None
    accommodation: str | None
    disruptions: dict[str, bool]
    log: list[str]


# ---------------------------------------------------------------------------
# Node functions — the actual business logic (same in both versions)
# ---------------------------------------------------------------------------


def search_flights(state: TravelState) -> dict[str, Any]:
    print("  [search_flights] Searching for available flights...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("flights_cancelled"):
        return {"status": "no_flights", "log": ["Searched flights: all cancelled (weather)"]}
    return {
        "status": "flights_found",
        "log": ["Searched flights: direct option available"],
    }


def book_direct_flight(state: TravelState) -> dict[str, Any]:
    print("  [book_direct_flight] Booking direct flight...")
    return {
        "status": "travel_booked",
        "transport_mode": "direct_flight",
        "log": ["Booked direct flight — arrives 11am"],
    }


def book_connecting_flight(state: TravelState) -> dict[str, Any]:
    print("  [book_connecting_flight] Booking connecting flight...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("flights_cancelled"):
        return {"status": "no_flights", "log": ["Connecting flights also cancelled"]}
    return {
        "status": "travel_booked",
        "transport_mode": "connecting_flight",
        "log": ["Booked connecting flight — arrives 1pm"],
    }


def search_ground_transport(state: TravelState) -> dict[str, Any]:
    print("  [search_ground_transport] Searching trains, rental cars...")
    return {
        "status": "ground_options_found",
        "log": ["Found ground options: train (4h), rental car (5h)"],
    }


def book_train(state: TravelState) -> dict[str, Any]:
    print("  [book_train] Booking train tickets...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("train_sold_out"):
        return {"status": "train_unavailable", "log": ["Train sold out"]}
    return {
        "status": "travel_booked",
        "transport_mode": "train",
        "log": ["Booked train — arrives 12:30pm"],
    }


def book_rental_car(state: TravelState) -> dict[str, Any]:
    print("  [book_rental_car] Booking rental car...")
    return {
        "status": "travel_booked",
        "transport_mode": "rental_car",
        "log": ["Booked rental car — drive time 5h, arrive ~1pm"],
    }


def book_hotel(state: TravelState) -> dict[str, Any]:
    print("  [book_hotel] Booking hotel near venue...")
    return {
        "status": "ready",
        "accommodation": "Hotel near venue",
        "log": ["Booked hotel — 0.5 miles from venue"],
    }


def confirm_arrival(state: TravelState) -> dict[str, Any]:
    mode = state.get("transport_mode", "unknown")
    print(f"  [confirm_arrival] Confirmed: arriving via {mode}")
    return {
        "status": "confirmed",
        "log": [f"Arrival confirmed via {mode} — will make the 3pm ceremony"],
    }


# ---------------------------------------------------------------------------
# Routing functions — THIS is the pain point. Every branch is hand-wired.
# ---------------------------------------------------------------------------


def route_after_flight_search(
    state: TravelState,
) -> Literal["book_direct_flight", "book_connecting_flight", "search_ground_transport"]:
    """Three-way branch: direct → connecting → ground. Each path is explicit."""
    if state.get("status") == "flights_found":
        return "book_direct_flight"
    # No direct? Try connecting (maybe partial cancellation)
    disruptions = state.get("disruptions", {})
    if not disruptions.get("flights_cancelled"):
        return "book_connecting_flight"
    # All flights down → ground transport
    return "search_ground_transport"


def route_after_connecting(
    state: TravelState,
) -> Literal["book_hotel", "search_ground_transport"]:
    """If connecting also failed, fall through to ground."""
    if state.get("status") == "travel_booked":
        return "book_hotel"
    return "search_ground_transport"


def route_after_ground_search(
    state: TravelState,
) -> Literal["book_train", "book_rental_car"]:
    """Prefer train over car."""
    disruptions = state.get("disruptions", {})
    if disruptions.get("train_sold_out"):
        return "book_rental_car"
    return "book_train"


def route_after_train(
    state: TravelState,
) -> Literal["book_hotel", "book_rental_car"]:
    """Train sold out at booking time → fall back to car."""
    if state.get("status") == "travel_booked":
        return "book_hotel"
    return "book_rental_car"


# ---------------------------------------------------------------------------
# Graph assembly — count the routing edges
# ---------------------------------------------------------------------------


def build_travel_graph() -> StateGraph:
    builder = StateGraph(TravelState)

    # Nodes (the business logic)
    builder.add_node("search_flights", search_flights)
    builder.add_node("book_direct_flight", book_direct_flight)
    builder.add_node("book_connecting_flight", book_connecting_flight)
    builder.add_node("search_ground_transport", search_ground_transport)
    builder.add_node("book_train", book_train)
    builder.add_node("book_rental_car", book_rental_car)
    builder.add_node("book_hotel", book_hotel)
    builder.add_node("confirm_arrival", confirm_arrival)

    # Edges — every fallback path is a hand-wired conditional edge
    builder.add_edge(START, "search_flights")
    builder.add_conditional_edges("search_flights", route_after_flight_search)
    builder.add_edge("book_direct_flight", "book_hotel")
    builder.add_conditional_edges("book_connecting_flight", route_after_connecting)
    builder.add_conditional_edges("search_ground_transport", route_after_ground_search)
    builder.add_conditional_edges("book_train", route_after_train)
    builder.add_edge("book_rental_car", "book_hotel")
    builder.add_edge("book_hotel", "confirm_arrival")
    builder.add_edge("confirm_arrival", END)

    return builder


# ---------------------------------------------------------------------------
# Demo runs
# ---------------------------------------------------------------------------


def run(disruptions: dict[str, bool] | None = None) -> None:
    label = "HAPPY PATH" if not disruptions else f"DISRUPTIONS: {disruptions}"
    print(f"\n{'='*60}")
    print(f"  BEFORE (Hardcoded LangGraph) — {label}")
    print(f"{'='*60}")

    graph = build_travel_graph().compile()
    result = graph.invoke(
        {
            "destination": "Chicago",
            "ceremony_time": "3pm Saturday",
            "disruptions": disruptions or {},
            "log": [],
        }
    )

    print(f"\n  Result: {result['status']}")
    print(f"  Transport: {result.get('transport_mode', 'none')}")
    for entry in result.get("log", []):
        print(f"    • {entry}")


if __name__ == "__main__":
    # Happy path — direct flight
    run()

    # Disruption 1: all flights cancelled → ground transport → train
    run({"flights_cancelled": True})

    # Disruption 2: flights cancelled + train sold out → rental car
    run({"flights_cancelled": True, "train_sold_out": True})

    # Count the routing code
    print(f"\n{'='*60}")
    print("  ROUTING FUNCTIONS: 4 (route_after_flight_search,")
    print("    route_after_connecting, route_after_ground_search, route_after_train)")
    print("  CONDITIONAL EDGES: 4")
    print("  To add a new transport option (bus, rideshare)?")
    print("  → Edit routing functions, add conditional edges, update type hints")
    print(f"{'='*60}")
