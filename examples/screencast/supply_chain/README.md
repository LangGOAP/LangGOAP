# Screencast: "The Order Must Ship" (Supply Chain)

Customer ordered 500 units. Delivery promised by Friday. Vendor fails.
Shipping delays. How does the system adapt?

## The Contrast

**before.py** — Hardcoded LangGraph with 3 routing functions and 3 conditional
edges. Every vendor fallback and shipping escalation is a hand-wired branch.

**after.py** — GOAP version. Same business logic. Zero routing code. The
planner picks preferred vendor + standard shipping (cheapest path).

**after_disrupted.py** — Preferred vendor out of stock? Planner replans through
alternate vendor. Standard shipping late? Planner switches to express. Both?
Handles it. No routing code changed.

## Run

```bash
cd langgoap
uv run python examples/screencast/supply_chain/before.py
uv run python examples/screencast/supply_chain/after.py
uv run python examples/screencast/supply_chain/after_disrupted.py
```

## Test

```bash
uv run pytest tests/integration/test_screencast_supply_chain.py -vv
```
