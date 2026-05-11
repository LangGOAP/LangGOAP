"""Travel disruption: GOAP handles cascading failures — zero routing code.

Same actions as after.py. Disruptions are injected via execute functions that
raise exceptions. The observer detects the failure, blacklists the action, and
the planner finds an alternative path automatically.

Run: uv run python examples/screencast/travel/after_disrupted.py
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalPolicy, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Reusable execute functions (from after.py)
# ---------------------------------------------------------------------------


def search_flights(ws: dict[str, Any]) -> dict[str, Any]:
    """Search flights — fails when weather disruption is active."""
    print("  [search_flights] Searching for available flights...")
    if ws.get("weather_grounded"):
        raise RuntimeError("All flights grounded due to regional storm")
    return {
        "flight_options_found": True,
        "flight_info": "Direct flight available — arrives 11am",
    }


def book_direct_flight(ws: dict[str, Any]) -> dict[str, Any]:
    """Book direct flight — fails when direct is cancelled."""
    print("  [book_direct_flight] Booking direct flight...")
    if ws.get("direct_cancelled"):
        raise RuntimeError("Direct flight cancelled — weather delay at hub")
    return {
        "travel_booked": True,
        "transport_mode": "direct_flight",
        "arrival_time": "11:00am",
    }


def book_connecting_flight(ws: dict[str, Any]) -> dict[str, Any]:
    """Book connecting flight."""
    print("  [book_connecting_flight] Booking connecting flight...")
    return {
        "travel_booked": True,
        "transport_mode": "connecting_flight",
        "arrival_time": "1:00pm",
    }


def search_ground_transport(ws: dict[str, Any]) -> dict[str, Any]:
    """Search ground transport options."""
    print("  [search_ground_transport] Searching ground transport options...")
    return {
        "ground_options_found": True,
        "ground_info": "Train (4h), rental car (5h) available",
    }


def book_train(ws: dict[str, Any]) -> dict[str, Any]:
    """Book train — fails when sold out."""
    print("  [book_train] Booking train tickets...")
    if ws.get("train_sold_out"):
        raise RuntimeError("Train sold out — weekend demand")
    return {
        "travel_booked": True,
        "transport_mode": "train",
        "arrival_time": "12:30pm",
    }


def book_rental_car(ws: dict[str, Any]) -> dict[str, Any]:
    """Book rental car — the last-resort option."""
    print("  [book_rental_car] Booking rental car...")
    return {
        "travel_booked": True,
        "transport_mode": "rental_car",
        "arrival_time": "1:00pm",
    }


def book_hotel(ws: dict[str, Any]) -> dict[str, Any]:
    """Book hotel near venue."""
    print("  [book_hotel] Booking hotel near venue...")
    return {
        "accommodation_ready": True,
        "hotel": "Hilton — 0.5 miles from venue",
    }


def confirm_arrival(ws: dict[str, Any]) -> dict[str, Any]:
    """Validate the plan gets us there on time."""
    mode = ws.get("transport_mode", "unknown")
    arrival = ws.get("arrival_time", "unknown")
    print(f"  [confirm_arrival] Confirmed: {mode}, arriving {arrival}")
    return {"at_venue": True, "on_time": True}


# ---------------------------------------------------------------------------
# Action declarations — same structure, disruptions come from world_state
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
# Demo runs — same goal, different disruptions
# ---------------------------------------------------------------------------


def run(label: str, world_state: dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  AFTER (GOAP) — {label}")
    print(f"{'='*60}")

    result = GoapGraph(actions=travel_actions).invoke(
        goal=GoalSpec(
            conditions={"at_venue": True, "on_time": True},
            policy=GoalPolicy(replan_strategy=ReplanStrategy.ON_DEVIATION),
        ),
        world_state=world_state,
    )

    print(f"\n  Status: {result['status']}")
    print(f"  Transport: {result['world_state'].get('transport_mode')}")
    print(f"  Replans: {result.get('replan_count', 0)}")

    plan_path = [h.action_name for h in result["execution_history"] if h.success]
    print(f"  Path taken: {' → '.join(plan_path)}")

    failures = [h for h in result["execution_history"] if not h.success]
    if failures:
        print(f"  Failures recovered from:")
        for f in failures:
            print(f"    ✗ {f.action_name}: {f.error}")

    print(f"\n  Zero routing code. The planner adapted automatically.")


if __name__ == "__main__":
    base_ws = {"has_destination": True, "destination": "Chicago"}

    # Disruption 1: Direct flight cancelled → replans through connecting
    run(
        "Direct flight cancelled",
        {**base_ws, "direct_cancelled": True},
    )

    # Disruption 2: All flights grounded → replans through ground transport → train
    run(
        "All flights grounded (storm)",
        {**base_ws, "weather_grounded": True},
    )

    # Disruption 3: Flights grounded + train sold out → rental car
    run(
        "Flights grounded + train sold out",
        {**base_ws, "weather_grounded": True, "train_sold_out": True},
    )
