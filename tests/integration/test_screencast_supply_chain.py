"""Integration tests for the "The Order Must Ship" screencast scenario.

Verifies that the GOAP planner finds optimal vendor + shipping combinations
and recovers from vendor failures and shipping delays by replanning through
alternative paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Shared execute functions
# ---------------------------------------------------------------------------


def check_preferred_vendor(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("preferred_out_of_stock"):
        raise RuntimeError("Preferred vendor out of stock")
    return {"vendor_quoted": True, "vendor_name": "Acme Supplies"}


def search_alternate_vendors(ws: dict[str, Any]) -> dict[str, Any]:
    return {"vendor_quoted": True, "vendor_name": "Global Parts Co"}


def negotiate_terms(ws: dict[str, Any]) -> dict[str, Any]:
    return {"terms_agreed": True}


def place_purchase_order(ws: dict[str, Any]) -> dict[str, Any]:
    return {"order_placed": True}


def arrange_standard_shipping(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("standard_shipping_late"):
        raise RuntimeError("Standard shipping misses Friday deadline")
    return {"shipment_scheduled": True, "shipping_method": "standard"}


def arrange_express_shipping(ws: dict[str, Any]) -> dict[str, Any]:
    return {"shipment_scheduled": True, "shipping_method": "express"}


def confirm_fulfillment(ws: dict[str, Any]) -> dict[str, Any]:
    return {"order_fulfilled": True}


def _supply_chain_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="check_preferred_vendor",
            preconditions={"order_received": True},
            effects={"vendor_quoted": True},
            cost=1.0,
            execute=check_preferred_vendor,
        ),
        ActionSpec(
            name="search_alternate_vendors",
            preconditions={"order_received": True},
            effects={"vendor_quoted": True},
            cost=3.0,
            execute=search_alternate_vendors,
        ),
        ActionSpec(
            name="negotiate_terms",
            preconditions={"vendor_quoted": True},
            effects={"terms_agreed": True},
            cost=1.0,
            execute=negotiate_terms,
        ),
        ActionSpec(
            name="place_purchase_order",
            preconditions={"terms_agreed": True},
            effects={"order_placed": True},
            cost=1.0,
            execute=place_purchase_order,
        ),
        ActionSpec(
            name="arrange_standard_shipping",
            preconditions={"order_placed": True},
            effects={"shipment_scheduled": True},
            cost=1.0,
            execute=arrange_standard_shipping,
        ),
        ActionSpec(
            name="arrange_express_shipping",
            preconditions={"order_placed": True},
            effects={"shipment_scheduled": True},
            cost=4.0,
            execute=arrange_express_shipping,
        ),
        ActionSpec(
            name="confirm_fulfillment",
            preconditions={"shipment_scheduled": True},
            effects={"order_fulfilled": True},
            cost=1.0,
            execute=confirm_fulfillment,
        ),
    ]


_GOAL = GoalSpec(
    conditions={"order_fulfilled": True},
    replan_strategy=ReplanStrategy.ON_DEVIATION,
)

_BASE_WS: dict[str, Any] = {
    "order_received": True,
    "product": "Widget-X",
    "quantity": 500,
    "deadline": "Friday",
}


class TestSupplyChainHappyPath:
    """Planner discovers the cheapest vendor + shipping combination."""

    def test_prefers_preferred_vendor_and_standard_shipping(self) -> None:
        """Cheapest path: preferred vendor(1) + negotiate(1) + PO(1) +
        standard(1) + confirm(1) = total cost 5."""
        result = GoapGraph(actions=_supply_chain_actions()).invoke(
            goal=_GOAL, world_state=_BASE_WS
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["order_fulfilled"] is True

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "check_preferred_vendor" in successful
        assert "negotiate_terms" in successful
        assert "place_purchase_order" in successful
        assert "arrange_standard_shipping" in successful
        assert "confirm_fulfillment" in successful
        # Should NOT use expensive alternatives
        assert "search_alternate_vendors" not in successful
        assert "arrange_express_shipping" not in successful

    def test_vendor_name_flows_through_world_state(self) -> None:
        """Rich data (vendor name) propagates through the execution chain."""
        result = GoapGraph(actions=_supply_chain_actions()).invoke(
            goal=_GOAL, world_state=_BASE_WS
        )
        assert result["world_state"]["vendor_name"] == "Acme Supplies"


class TestSupplyChainDisruptions:
    """Planner recovers from vendor and shipping failures."""

    def test_preferred_vendor_out_of_stock_uses_alternate(self) -> None:
        """Preferred vendor fails → planner replans through alternate."""
        result = GoapGraph(actions=_supply_chain_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "preferred_out_of_stock": True},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert result["world_state"]["vendor_name"] == "Global Parts Co"

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "search_alternate_vendors" in successful

    def test_standard_shipping_late_uses_express(self) -> None:
        """Standard shipping fails → planner replans through express."""
        result = GoapGraph(actions=_supply_chain_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "standard_shipping_late": True},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert result["world_state"]["shipping_method"] == "express"

    def test_both_vendor_and_shipping_disrupted(self) -> None:
        """Vendor out of stock + standard shipping late → alternate + express."""
        result = GoapGraph(actions=_supply_chain_actions()).invoke(
            goal=_GOAL,
            world_state={
                **_BASE_WS,
                "preferred_out_of_stock": True,
                "standard_shipping_late": True,
            },
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert result["world_state"]["vendor_name"] == "Global Parts Co"
        assert result["world_state"]["shipping_method"] == "express"

    def test_failure_details_in_execution_history(self) -> None:
        """Failed actions appear in execution history with error messages."""
        result = GoapGraph(actions=_supply_chain_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "preferred_out_of_stock": True},
        )

        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) >= 1
        assert any("out of stock" in (f.error or "") for f in failures)
