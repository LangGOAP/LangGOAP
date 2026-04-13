# Script: "The Order Must Ship"

**Duration:** ~7 minutes

---

## 0:00–0:30 — Hook

"500 units. Delivery promised by Friday. Your preferred vendor just went out
of stock. Standard shipping can't make the deadline. Your customer is waiting.

This is a $1.2 trillion industry. Supply chain disruptions are the norm, not
the exception. Let me show you how GOAP handles it."

---

## 0:30–1:30 — The "Before" (Hardcoded LangGraph)

Open `before.py`. Walk through:

- **7 node functions** — check vendor, negotiate, place PO, ship, confirm
- **3 routing functions** — route after preferred vendor, after alternate, after shipping
- **3 conditional edges**

"Every vendor fallback and shipping escalation is a hand-wired decision.
'If preferred vendor fails, search alternates.' 'If standard shipping is late,
use express.' You hardcode the recovery strategy."

Run happy path.

---

## 1:30–2:30 — Show the Fallback Complexity

Run with disruptions: preferred vendor out, then both vendor + shipping.

"It works — because we anticipated these exact failures. But what about a new
vendor tier? A new shipping partner? Each one means editing routing functions."

---

## 2:30–3:00 — The Transition

"What if you just declared what's possible — vendors with costs, shipping
options with costs — and the system picked the best combination? And when one
option fails, it picks the next best. Automatically."

---

## 3:00–5:30 — Build the "After"

Open `after.py`. Highlight:

1. Actions as declarations: `check_preferred_vendor` (cost=1) vs
   `search_alternate_vendors` (cost=3). Both produce `vendor_quoted=True`.
2. `arrange_standard_shipping` (cost=1) vs `arrange_express_shipping` (cost=4).
   Both produce `shipment_scheduled=True`.
3. "The planner sees two ways to get `vendor_quoted` and two ways to get
   `shipment_scheduled`. It picks the cheapest combination."
4. One-line invocation with `GoalSpec(conditions={"order_fulfilled": True})`.

Run: preferred vendor + standard shipping selected (cheapest path).

---

## 5:30–6:30 — Break It

Open `after_disrupted.py`. Run preferred vendor out of stock.

"Preferred vendor raises an exception → planner blacklists it → replans through
alternate vendor. Rest of the chain continues: negotiate, PO, ship, confirm."

---

## 6:30–7:30 — Escalate

Run both disruptions: vendor out + shipping late.

"Two failures in different parts of the chain. Planner adapted to both: alternate
vendor AND express shipping. Total cost is higher — but the order ships on time."

---

## 7:30–8:00 — Close

"The business logic is the same. The routing code is gone. Adding a new vendor
or shipping option? Add an ActionSpec. That's it. The planner handles the rest.

`pip install langgoap`."
