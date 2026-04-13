"""Supply chain: Hardcoded LangGraph workflow with manual fallback edges.

This version handles vendor failures and shipping disruptions — but every
fallback is a hand-wired conditional edge. Adding a new vendor or shipping
option means editing routing functions and graph structure.

Run: uv run python examples/screencast/supply_chain/before.py
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class OrderState(TypedDict, total=False):
    product: str
    quantity: int
    deadline: str
    status: str
    vendor: str | None
    shipping_method: str | None
    disruptions: dict[str, bool]
    log: list[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def check_preferred_vendor(state: OrderState) -> dict[str, Any]:
    print("  [check_preferred_vendor] Checking preferred vendor availability...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("preferred_out_of_stock"):
        return {"status": "vendor_unavailable", "log": ["Preferred vendor: OUT OF STOCK"]}
    return {
        "status": "vendor_quoted",
        "vendor": "Acme Supplies",
        "log": ["Preferred vendor: Acme Supplies — 500 units available, $12/unit"],
    }


def search_alternate_vendors(state: OrderState) -> dict[str, Any]:
    print("  [search_alternate_vendors] Searching alternate vendors...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("all_vendors_fail"):
        return {"status": "no_vendors", "log": ["No alternate vendors can deliver by Friday"]}
    return {
        "status": "vendor_quoted",
        "vendor": "Global Parts Co",
        "log": ["Alternate vendor: Global Parts Co — 500 units, $15/unit, 2-day lead"],
    }


def negotiate_terms(state: OrderState) -> dict[str, Any]:
    vendor = state.get("vendor", "Unknown")
    print(f"  [negotiate_terms] Negotiating with {vendor}...")
    return {
        "status": "terms_agreed",
        "log": [f"Terms agreed with {vendor}: net-30, FOB origin"],
    }


def place_purchase_order(state: OrderState) -> dict[str, Any]:
    vendor = state.get("vendor", "Unknown")
    print(f"  [place_purchase_order] Placing PO with {vendor}...")
    return {
        "status": "order_placed",
        "log": [f"PO submitted to {vendor}: 500 units"],
    }


def arrange_standard_shipping(state: OrderState) -> dict[str, Any]:
    print("  [arrange_standard_shipping] Arranging standard freight...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("standard_shipping_late"):
        return {
            "status": "shipping_delayed",
            "log": ["Standard shipping: delivery Monday — MISSES Friday deadline"],
        }
    return {
        "status": "shipment_scheduled",
        "shipping_method": "standard",
        "log": ["Standard shipping: delivery Thursday"],
    }


def arrange_express_shipping(state: OrderState) -> dict[str, Any]:
    print("  [arrange_express_shipping] Arranging express shipping...")
    return {
        "status": "shipment_scheduled",
        "shipping_method": "express",
        "log": ["Express shipping: delivery Wednesday — meets Friday deadline"],
    }


def confirm_fulfillment(state: OrderState) -> dict[str, Any]:
    vendor = state.get("vendor", "Unknown")
    method = state.get("shipping_method", "unknown")
    print(f"  [confirm_fulfillment] Order fulfilled: {vendor} via {method}")
    return {
        "status": "fulfilled",
        "log": [f"Order fulfilled: {vendor}, {method} shipping"],
    }


# ---------------------------------------------------------------------------
# Routing functions — every fallback is explicit
# ---------------------------------------------------------------------------


def route_after_preferred_vendor(
    state: OrderState,
) -> Literal["negotiate_terms", "search_alternate_vendors"]:
    if state.get("status") == "vendor_quoted":
        return "negotiate_terms"
    return "search_alternate_vendors"


def route_after_alternate_vendor(
    state: OrderState,
) -> Literal["negotiate_terms"]:
    # In a real system you'd need another fallback here if no vendors found.
    # That's yet another conditional edge to maintain.
    return "negotiate_terms"


def route_after_standard_shipping(
    state: OrderState,
) -> Literal["confirm_fulfillment", "arrange_express_shipping"]:
    if state.get("status") == "shipment_scheduled":
        return "confirm_fulfillment"
    return "arrange_express_shipping"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_order_graph() -> StateGraph:
    builder = StateGraph(OrderState)

    builder.add_node("check_preferred_vendor", check_preferred_vendor)
    builder.add_node("search_alternate_vendors", search_alternate_vendors)
    builder.add_node("negotiate_terms", negotiate_terms)
    builder.add_node("place_purchase_order", place_purchase_order)
    builder.add_node("arrange_standard_shipping", arrange_standard_shipping)
    builder.add_node("arrange_express_shipping", arrange_express_shipping)
    builder.add_node("confirm_fulfillment", confirm_fulfillment)

    builder.add_edge(START, "check_preferred_vendor")
    builder.add_conditional_edges("check_preferred_vendor", route_after_preferred_vendor)
    builder.add_conditional_edges("search_alternate_vendors", route_after_alternate_vendor)
    builder.add_edge("negotiate_terms", "place_purchase_order")
    builder.add_edge("place_purchase_order", "arrange_standard_shipping")
    builder.add_conditional_edges("arrange_standard_shipping", route_after_standard_shipping)
    builder.add_edge("arrange_express_shipping", "confirm_fulfillment")
    builder.add_edge("confirm_fulfillment", END)

    return builder


# ---------------------------------------------------------------------------
# Demo runs
# ---------------------------------------------------------------------------


def run(disruptions: dict[str, bool] | None = None) -> None:
    label = "HAPPY PATH" if not disruptions else f"DISRUPTIONS: {disruptions}"
    print(f"\n{'='*60}")
    print(f"  BEFORE (Hardcoded LangGraph) — {label}")
    print(f"{'='*60}")

    graph = build_order_graph().compile()
    result = graph.invoke(
        {
            "product": "Widget-X",
            "quantity": 500,
            "deadline": "Friday",
            "disruptions": disruptions or {},
            "log": [],
        }
    )

    print(f"\n  Result: {result['status']}")
    print(f"  Vendor: {result.get('vendor', 'none')}")
    print(f"  Shipping: {result.get('shipping_method', 'none')}")
    for entry in result.get("log", []):
        print(f"    • {entry}")


if __name__ == "__main__":
    # Happy path
    run()

    # Disruption 1: Preferred vendor out of stock
    run({"preferred_out_of_stock": True})

    # Disruption 2: Preferred out of stock + standard shipping late
    run({"preferred_out_of_stock": True, "standard_shipping_late": True})

    print(f"\n{'='*60}")
    print("  ROUTING FUNCTIONS: 3")
    print("  CONDITIONAL EDGES: 3")
    print("  To add a new vendor tier or shipping option?")
    print("  → Edit routing functions, add edges, update type hints")
    print(f"{'='*60}")
