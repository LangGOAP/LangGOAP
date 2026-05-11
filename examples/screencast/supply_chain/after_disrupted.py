"""Supply chain: GOAP handles vendor failures + shipping delays automatically.

Same actions as after.py. Disruptions injected via world_state flags that cause
execute functions to raise exceptions. The planner replans through alternatives.

Run: uv run python examples/screencast/supply_chain/after_disrupted.py
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalPolicy, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Execute functions — disruption-aware
# ---------------------------------------------------------------------------


def check_preferred_vendor(ws: dict[str, Any]) -> dict[str, Any]:
    """Check preferred vendor — fails when out of stock."""
    print("  [check_preferred_vendor] Checking preferred vendor...")
    if ws.get("preferred_out_of_stock"):
        raise RuntimeError("Preferred vendor out of stock — no ETA on restock")
    return {
        "vendor_quoted": True,
        "vendor_name": "Acme Supplies",
        "quote_amount": 6000.0,
        "lead_time_days": 2,
    }


def search_alternate_vendors(ws: dict[str, Any]) -> dict[str, Any]:
    """Search alternate vendors."""
    print("  [search_alternate_vendors] Searching alternate vendors...")
    return {
        "vendor_quoted": True,
        "vendor_name": "Global Parts Co",
        "quote_amount": 7500.0,
        "lead_time_days": 3,
    }


def negotiate_terms(ws: dict[str, Any]) -> dict[str, Any]:
    """Negotiate terms with selected vendor."""
    vendor = ws.get("vendor_name", "Unknown")
    print(f"  [negotiate_terms] Negotiating with {vendor}...")
    return {"terms_agreed": True, "payment_terms": "net-30"}


def place_purchase_order(ws: dict[str, Any]) -> dict[str, Any]:
    """Submit purchase order."""
    vendor = ws.get("vendor_name", "Unknown")
    print(f"  [place_purchase_order] PO to {vendor}...")
    return {"order_placed": True, "po_number": "PO-2024-001"}


def arrange_standard_shipping(ws: dict[str, Any]) -> dict[str, Any]:
    """Standard shipping — fails when delivery would miss deadline."""
    print("  [arrange_standard_shipping] Booking standard freight...")
    if ws.get("standard_shipping_late"):
        raise RuntimeError("Standard shipping: earliest delivery Monday — misses Friday deadline")
    return {
        "shipment_scheduled": True,
        "shipping_method": "standard",
        "delivery_day": "Thursday",
    }


def arrange_express_shipping(ws: dict[str, Any]) -> dict[str, Any]:
    """Express shipping — more expensive but guaranteed to meet deadline."""
    print("  [arrange_express_shipping] Booking express shipping...")
    return {
        "shipment_scheduled": True,
        "shipping_method": "express",
        "delivery_day": "Wednesday",
    }


def confirm_fulfillment(ws: dict[str, Any]) -> dict[str, Any]:
    """Send fulfillment confirmation."""
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
# Demo runs
# ---------------------------------------------------------------------------


def run(label: str, world_state: dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  AFTER (GOAP) — {label}")
    print(f"{'='*60}")

    result = GoapGraph(actions=supply_chain_actions).invoke(
        goal=GoalSpec(
            conditions={"order_fulfilled": True},
            policy=GoalPolicy(replan_strategy=ReplanStrategy.ON_DEVIATION),
        ),
        world_state=world_state,
    )

    print(f"\n  Status: {result['status']}")
    print(f"  Vendor: {result['world_state'].get('vendor_name')}")
    print(f"  Shipping: {result['world_state'].get('shipping_method')}")
    print(f"  Replans: {result.get('replan_count', 0)}")

    plan_path = [h.action_name for h in result["execution_history"] if h.success]
    print(f"  Path taken: {' → '.join(plan_path)}")

    failures = [h for h in result["execution_history"] if not h.success]
    if failures:
        print(f"  Failures recovered from:")
        for f in failures:
            print(f"    ✗ {f.action_name}: {f.error}")


if __name__ == "__main__":
    base_ws = {
        "order_received": True,
        "product": "Widget-X",
        "quantity": 500,
        "deadline": "Friday",
    }

    # Disruption 1: Preferred vendor out of stock → alternate vendor
    run(
        "Preferred vendor out of stock",
        {**base_ws, "preferred_out_of_stock": True},
    )

    # Disruption 2: Standard shipping can't meet deadline → express
    run(
        "Standard shipping delayed",
        {**base_ws, "standard_shipping_late": True},
    )

    # Disruption 3: Both — vendor switch + express shipping
    run(
        "Vendor out of stock + shipping delayed",
        {**base_ws, "preferred_out_of_stock": True, "standard_shipping_late": True},
    )
