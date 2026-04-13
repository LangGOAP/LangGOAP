# The Order Must Ship

A customer ordered 500 units. Delivery is promised by Friday. Your job is to
source the product, negotiate terms, place the order, ship it, and confirm
fulfillment. Straightforward — until a vendor goes out of stock or a shipping
lane misses the deadline.

## The traditional approach

In a standard LangGraph workflow, you check the preferred vendor first. If
they're out of stock, a conditional edge routes to an alternate vendor search.
If standard shipping can't meet the deadline, another conditional edge routes
to express. Each fallback is an explicit branch you anticipated and wired at
design time.

This works fine for the failure modes you predicted. The problem is
combinatorial: every new vendor tier or shipping partner means another routing
function, another set of conditional edges, another set of type annotations
to maintain. The graph structure encodes your recovery strategy, not just
your business logic.

```
before.py — 3 routing functions, 3 conditional edges
```

## The GOAP approach

With GOAP, you declare two ways to get a vendor quote: `check_preferred_vendor`
at cost 1, `search_alternate_vendors` at cost 3. Both produce
`vendor_quoted: True`. You declare two shipping options:
`arrange_standard_shipping` at cost 1, `arrange_express_shipping` at cost 4.
Both produce `shipment_scheduled: True`.

The planner sees multiple paths to each intermediate state and picks the
cheapest combination end-to-end. On a clean run: preferred vendor, negotiate,
place PO, standard shipping, confirm. Total cost 5.

```
after.py — 0 routing functions, 0 conditional edges
```

## When things go wrong

The preferred vendor is out of stock. The execute function raises an
exception. The observer blacklists that action and sends the state back to
the planner. The planner finds the alternate path: `search_alternate_vendors`
(cost 3) feeds into the same downstream chain — negotiate, PO, ship, confirm.
The vendor changed; the fulfillment pipeline didn't.

Then standard shipping can't meet the Friday deadline. Same pattern: the
planner blacklists standard shipping and replans through express. Higher cost,
but the order ships on time.

Both disruptions in the same run? The planner adapts to each independently.
Alternate vendor plus express shipping. Two replans, two failures recovered,
order fulfilled.

```
after_disrupted.py — same actions, vendor + shipping failures, all recovered
```

## What this demonstrates

In a supply chain, disruptions are the norm. Vendors go out of stock.
Shipping lanes get congested. Lead times shift. The traditional approach
handles each of these — if you predicted them. GOAP handles them generically:
any action that fails gets blacklisted, and the planner re-searches for a
path using whatever actions remain.

The cost structure does the prioritization. Preferred vendor is cheap,
alternate is expensive. Standard shipping is cheap, express is expensive.
You don't encode "try preferred first, then alternate" as a routing rule —
the planner derives that ordering from the costs. Adding a third vendor tier
or a new shipping partner is a single `ActionSpec`, not a graph restructuring.

The business logic stays in the execute functions. The recovery strategy
lives in the cost assignments. The routing code doesn't exist.
