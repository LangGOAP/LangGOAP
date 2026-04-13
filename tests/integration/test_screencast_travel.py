"""Integration tests for the "Get to the Wedding" screencast scenario.

Verifies that the GOAP planner finds optimal travel paths and recovers
from cascading disruptions (flight cancellations, train sellouts) by
replanning through alternative transport modes.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Shared execute functions
# ---------------------------------------------------------------------------


def search_flights(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("weather_grounded"):
        raise RuntimeError("All flights grounded due to regional storm")
    return {"flight_options_found": True}


def book_direct_flight(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("direct_cancelled"):
        raise RuntimeError("Direct flight cancelled")
    return {"travel_booked": True, "transport_mode": "direct_flight"}


def book_connecting_flight(ws: dict[str, Any]) -> dict[str, Any]:
    return {"travel_booked": True, "transport_mode": "connecting_flight"}


def search_ground_transport(ws: dict[str, Any]) -> dict[str, Any]:
    return {"ground_options_found": True}


def book_train(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("train_sold_out"):
        raise RuntimeError("Train sold out")
    return {"travel_booked": True, "transport_mode": "train"}


def book_rental_car(ws: dict[str, Any]) -> dict[str, Any]:
    return {"travel_booked": True, "transport_mode": "rental_car"}


def book_hotel(ws: dict[str, Any]) -> dict[str, Any]:
    return {"accommodation_ready": True}


def confirm_arrival(ws: dict[str, Any]) -> dict[str, Any]:
    return {"at_venue": True, "on_time": True}


def _travel_actions() -> list[ActionSpec]:
    return [
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


_GOAL = GoalSpec(
    conditions={"at_venue": True, "on_time": True},
    replan_strategy=ReplanStrategy.ON_DEVIATION,
)

_BASE_WS: dict[str, Any] = {"has_destination": True, "destination": "Chicago"}


class TestTravelHappyPath:
    """Planner discovers optimal path without disruptions."""

    def test_prefers_direct_flight(self) -> None:
        """Direct flight is cheapest air option (cost=2 vs connecting=3)."""
        result = GoapGraph(actions=_travel_actions()).invoke(
            goal=_GOAL, world_state=_BASE_WS
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["at_venue"] is True
        assert result["world_state"]["on_time"] is True

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "search_flights" in successful
        assert "book_direct_flight" in successful
        assert "book_hotel" in successful
        assert "confirm_arrival" in successful

    def test_flight_path_cheaper_than_ground(self) -> None:
        """Flight path (1+2+1+1=5) is cheaper than ground (1+2+1+1=5) but
        flight is tried first due to lower subtotal at the booking step."""
        result = GoapGraph(actions=_travel_actions()).invoke(
            goal=_GOAL, world_state=_BASE_WS
        )

        successful = [h.action_name for h in result["execution_history"] if h.success]
        # Should pick flight path, not ground
        assert (
            "book_direct_flight" in successful or "book_connecting_flight" in successful
        )
        assert "book_train" not in successful
        assert "book_rental_car" not in successful


class TestTravelDisruptions:
    """Planner recovers from cascading failures via replanning."""

    def test_direct_cancelled_replans_to_connecting(self) -> None:
        """Direct flight cancelled → planner replans through connecting."""
        result = GoapGraph(actions=_travel_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "direct_cancelled": True},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "book_connecting_flight" in successful

        failures = [h for h in result["execution_history"] if not h.success]
        assert any("Direct flight cancelled" in (f.error or "") for f in failures)

    def test_all_flights_grounded_replans_to_train(self) -> None:
        """All flights grounded → planner replans through ground transport."""
        result = GoapGraph(actions=_travel_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "weather_grounded": True},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "search_ground_transport" in successful
        assert "book_train" in successful

    def test_flights_grounded_and_train_sold_out_uses_rental_car(self) -> None:
        """All flights + train fail → planner falls through to rental car."""
        result = GoapGraph(actions=_travel_actions()).invoke(
            goal=_GOAL,
            world_state={
                **_BASE_WS,
                "weather_grounded": True,
                "train_sold_out": True,
            },
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert result["world_state"]["transport_mode"] == "rental_car"

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "book_rental_car" in successful

        # Both flights and train should have failed
        failures = [h for h in result["execution_history"] if not h.success]
        failed_names = [f.action_name for f in failures]
        assert "search_flights" in failed_names or "book_train" in failed_names

    def test_blacklisted_actions_tracked(self) -> None:
        """Failed actions appear in blacklisted_actions."""
        result = GoapGraph(actions=_travel_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "weather_grounded": True},
        )

        assert "search_flights" in result.get("blacklisted_actions", [])


class TestTravelNoPath:
    """When no action chain can reach the goal, planner reports no_plan."""

    def test_no_plan_with_insufficient_actions(self) -> None:
        """Only search actions, no booking → goal unreachable."""
        actions = [
            ActionSpec(
                name="search_flights",
                preconditions={"has_destination": True},
                effects={"flight_options_found": True},
                cost=1.0,
                execute=search_flights,
            ),
        ]
        result = GoapGraph(actions=actions).invoke(goal=_GOAL, world_state=_BASE_WS)
        assert result["status"] == "no_plan"
