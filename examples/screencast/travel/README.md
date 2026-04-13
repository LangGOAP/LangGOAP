# Screencast: "Get to the Wedding" (Travel Disruption)

A wedding ceremony starts at 3pm Saturday in another city. Your travel plans
keep falling apart. How does the system adapt?

## The Contrast

**before.py** — Hardcoded LangGraph with 4 routing functions and 4 conditional
edges. Every fallback (connecting flight, train, rental car) is a hand-wired
branch. Works, but adding a new transport option means editing graph structure.

**after.py** — GOAP version. Same business logic. Zero routing functions, zero
conditional edges. The planner picks the cheapest path automatically.

**after_disrupted.py** — Same GOAP actions with disruptions injected. Direct
flight cancelled? Planner replans through connecting. All flights grounded?
Planner switches to ground transport. Train sold out? Rental car. No routing
code changed.

## Run

```bash
cd langgoap
uv run python examples/screencast/travel/before.py
uv run python examples/screencast/travel/after.py
uv run python examples/screencast/travel/after_disrupted.py
```

## Test

```bash
uv run pytest tests/integration/test_screencast_travel.py -vv
```
