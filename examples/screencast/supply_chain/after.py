"""Supply chain: GOAP version — declare actions, planner finds the path.

Same business logic. Zero routing code. The planner picks the cheapest vendor
and shipping combination automatically.

Run: uv run python examples/screencast/supply_chain/after.py
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph

# ---------------------------------------------------------------------------
# Execute functions
# ---------------------------------------------------------------------------


def check_preferred_vendor(ws: dict[str, Any]) -> dict[str, Any]:
    """Check preferred vendor for availability and pricing."""
    print("  [check_preferred_vendor] Checking preferred vendor...")
    return {
        "vendor_quoted": True,
        "vendor_name": "Acme Supplies",
        "quote_amount": 6000.0,
        "lead_time_days": 2,
    }


def search_alternate_vendors(ws: dict[str, Any]) -> dict[str, Any]:
    """Search and evaluate alternate vendors."""
    print("  [search_alternate_vendors] Searching alternate vendors...")
    return {
        "vendor_quoted": True,
        "vendor_name": "Global Parts Co",
        "quote_amount": 7500.0,
        "lead_time_days": 3,
    }


def negotiate_terms(ws: dict[str, Any]) -> dict[str, Any]:
    """Negotiate payment and delivery terms with the selected vendor."""
    vendor = ws.get("vendor_name", "Unknown")
    print(f"  [negotiate_terms] Negotiating with {vendor}...")
    return {
        "terms_agreed": True,
        "payment_terms": "net-30",
    }


def place_purchase_order(ws: dict[str, Any]) -> dict[str, Any]:
    """Submit the purchase order."""
    vendor = ws.get("vendor_name", "Unknown")
    qty = ws.get("quantity", 0)
    print(f"  [place_purchase_order] PO to {vendor} for {qty} units...")
    return {
        "order_placed": True,
        "po_number": "PO-2024-001",
    }


def arrange_standard_shipping(ws: dict[str, Any]) -> dict[str, Any]:
    """Book standard freight — cheapest option."""
    print("  [arrange_standard_shipping] Booking standard freight...")
    return {
        "shipment_scheduled": True,
        "shipping_method": "standard",
        "delivery_day": "Thursday",
    }


def arrange_express_shipping(ws: dict[str, Any]) -> dict[str, Any]:
    """Book express shipping — more expensive but faster."""
    print("  [arrange_express_shipping] Booking express shipping...")
    return {
        "shipment_scheduled": True,
        "shipping_method": "express",
        "delivery_day": "Wednesday",
    }


def confirm_fulfillment(ws: dict[str, Any]) -> dict[str, Any]:
    """Send confirmation to the customer."""
    vendor = ws.get("vendor_name", "Unknown")
    method = ws.get("shipping_method", "unknown")
    print(f"  [confirm_fulfillment] Confirmed: {vendor} via {method}")
    return {"order_fulfilled": True}


# ---------------------------------------------------------------------------
# Action declarations
# ---------------------------------------------------------------------------

supply_chain_actions = [
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run() -> None:
    print(f"\n{'='*60}")
    print("  AFTER (GOAP) — Happy Path")
    print(f"{'='*60}")

    result = GoapGraph(actions=supply_chain_actions).invoke(
        goal=GoalSpec(conditions={"order_fulfilled": True}),
        world_state={
            "order_received": True,
            "product": "Widget-X",
            "quantity": 500,
            "deadline": "Friday",
        },
    )

    print(f"\n  Status: {result['status']}")
    print(f"  Vendor: {result['world_state'].get('vendor_name')}")
    print(f"  Shipping: {result['world_state'].get('shipping_method')}")

    plan_path = [h.action_name for h in result["execution_history"] if h.success]
    print(f"  Plan: {' → '.join(plan_path)}")
    print(f"\n  ROUTING FUNCTIONS: 0")
    print(f"  CONDITIONAL EDGES: 0")


if __name__ == "__main__":
    run()
