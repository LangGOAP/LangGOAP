"""Travel disruption: GOAP version — declare actions, let the planner route.

Same business logic functions. Zero routing code. The planner discovers the
cheapest path to the goal. When disruptions occur (after_disrupted.py), it
replans automatically through alternative actions.

Run: uv run python examples/screencast/travel/after.py
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph

# ---------------------------------------------------------------------------
# Execute functions — the business logic (identical purpose to before.py)
# ---------------------------------------------------------------------------


def search_flights(ws: dict[str, Any]) -> dict[str, Any]:
    """Search for available flights to the destination."""
    print("  [search_flights] Searching for available flights...")
    return {
        "flight_options_found": True,
        "flight_info": "Direct flight available — arrives 11am",
    }


def book_direct_flight(ws: dict[str, Any]) -> dict[str, Any]:
    """Book the cheapest direct flight."""
    print("  [book_direct_flight] Booking direct flight...")
    return {
        "travel_booked": True,
        "transport_mode": "direct_flight",
        "arrival_time": "11:00am",
    }


def book_connecting_flight(ws: dict[str, Any]) -> dict[str, Any]:
    """Book a connecting flight when direct isn't available."""
    print("  [book_connecting_flight] Booking connecting flight...")
    return {
        "travel_booked": True,
        "transport_mode": "connecting_flight",
        "arrival_time": "1:00pm",
    }


def search_ground_transport(ws: dict[str, Any]) -> dict[str, Any]:
    """Search for trains, rental cars, and rideshares."""
    print("  [search_ground_transport] Searching ground transport options...")
    return {
        "ground_options_found": True,
        "ground_info": "Train (4h), rental car (5h) available",
    }


def book_train(ws: dict[str, Any]) -> dict[str, Any]:
    """Book train tickets."""
    print("  [book_train] Booking train tickets...")
    return {
        "travel_booked": True,
        "transport_mode": "train",
        "arrival_time": "12:30pm",
    }


def book_rental_car(ws: dict[str, Any]) -> dict[str, Any]:
    """Book a rental car as last resort."""
    print("  [book_rental_car] Booking rental car...")
    return {
        "travel_booked": True,
        "transport_mode": "rental_car",
        "arrival_time": "1:00pm",
    }


def book_hotel(ws: dict[str, Any]) -> dict[str, Any]:
    """Book a hotel near the wedding venue."""
    print("  [book_hotel] Booking hotel near venue...")
    return {
        "accommodation_ready": True,
        "hotel": "Hilton — 0.5 miles from venue",
    }


def confirm_arrival(ws: dict[str, Any]) -> dict[str, Any]:
    """Validate the end-to-end plan gets us there on time."""
    mode = ws.get("transport_mode", "unknown")
    arrival = ws.get("arrival_time", "unknown")
    print(f"  [confirm_arrival] Confirmed: {mode}, arriving {arrival}")
    return {
        "at_venue": True,
        "on_time": True,
    }


# ---------------------------------------------------------------------------
# Action declarations — preconditions, effects, cost. No routing.
# ---------------------------------------------------------------------------

travel_actions = [
    ActionSpec(
        name="search_flights",
        preconditions={"has_destination": True},
        effects={"flight_options_found": True},
        cost=1.0,
        execute=search_flights,
    ),
    ActionSpec(
        name="book_direct_flight",
        preconditions={"flight_options_found": True},
        effects={"travel_booked": True},
        cost=2.0,
        execute=book_direct_flight,
    ),
    ActionSpec(
        name="book_connecting_flight",
        preconditions={"flight_options_found": True},
        effects={"travel_booked": True},
        cost=3.0,
        execute=book_connecting_flight,
    ),
    ActionSpec(
        name="search_ground_transport",
        preconditions={"has_destination": True},
        effects={"ground_options_found": True},
        cost=1.0,
        execute=search_ground_transport,
    ),
    ActionSpec(
        name="book_train",
        preconditions={"ground_options_found": True},
        effects={"travel_booked": True},
        cost=2.0,
        execute=book_train,
    ),
    ActionSpec(
        name="book_rental_car",
        preconditions={"ground_options_found": True},
        effects={"travel_booked": True},
        cost=3.0,
        execute=book_rental_car,
    ),
    ActionSpec(
        name="book_hotel",
        preconditions={"travel_booked": True},
        effects={"accommodation_ready": True},
        cost=1.0,
        execute=book_hotel,
    ),
    ActionSpec(
        name="confirm_arrival",
        preconditions={"travel_booked": True, "accommodation_ready": True},
        effects={"at_venue": True, "on_time": True},
        cost=1.0,
        execute=confirm_arrival,
    ),
]


# ---------------------------------------------------------------------------
# Demo run
# ---------------------------------------------------------------------------


def run() -> None:
    print(f"\n{'='*60}")
    print("  AFTER (GOAP) — Happy Path")
    print(f"{'='*60}")

    result = GoapGraph(actions=travel_actions).invoke(
        goal=GoalSpec(conditions={"at_venue": True, "on_time": True}),
        world_state={"has_destination": True, "destination": "Chicago"},
    )

    print(f"\n  Status: {result['status']}")
    print(f"  Transport: {result['world_state'].get('transport_mode')}")
    print(f"  Hotel: {result['world_state'].get('hotel')}")

    plan_path = [h.action_name for h in result["execution_history"] if h.success]
    print(f"  Plan: {' → '.join(plan_path)}")
    print(f"\n  ROUTING FUNCTIONS: 0")
    print(f"  CONDITIONAL EDGES: 0")
    print(f"  The planner found the cheapest path automatically.")


if __name__ == "__main__":
    run()
