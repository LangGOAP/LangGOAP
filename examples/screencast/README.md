# Screencasts

Three paired demos that contrast a hand-written LangGraph routing graph
with the same business logic expressed as a LangGOAP plan, then run the
GOAP version through realistic disruptions to show automatic replanning.

Each scenario ships four pieces:

| File | Purpose |
| --- | --- |
| `before.py` | The baseline. A hand-wired `StateGraph` with conditional edges encoding every fallback branch. Runs without LangGOAP installed. |
| `after.py` | The same scenario as a LangGOAP graph. Identical observable behaviour on the happy path; the routing code is gone. |
| `after_disrupted.py` | The GOAP version run under perturbation (vendor outage, partial outage, weather, …). No routing code changed; the planner replans. |
| `script.md` / `narrative.md` | Production artifacts used to record the screencast. Not required for running the demo. |

Every scenario has a corresponding integration test under
[`tests/integration/`](../../tests/integration/) so the scripts stay
runnable as the codebase evolves.

## Scenarios

### [`incident/`](incident/) — Incident response

A pager fires. The agent triages, runs diagnostics, applies a remediation,
and notifies stakeholders. In `after_disrupted.py`, the primary
remediation fails its post-condition and the planner re-routes to a
fallback path without touching the graph definition.

- Integration test: [`tests/integration/test_screencast_incident.py`](../../tests/integration/test_screencast_incident.py)

### [`supply_chain/`](supply_chain/) — Supply-chain disruption

A 500-unit order must ship by Friday. The happy path picks the preferred
vendor and standard shipping (cheapest plan). In `after_disrupted.py`,
the preferred vendor is out of stock and standard shipping is late; the
planner switches to an alternate vendor and express shipping
automatically.

- Integration test: [`tests/integration/test_screencast_supply_chain.py`](../../tests/integration/test_screencast_supply_chain.py)

### [`travel/`](travel/) — Travel rebooking

A multi-leg itinerary needs to land on time. Weather grounds the
preferred flight in `after_disrupted.py`; the planner reroutes through
an alternate carrier and rebooks the downstream hotel without any
hand-written fallback logic.

- Integration test: [`tests/integration/test_screencast_travel.py`](../../tests/integration/test_screencast_travel.py)

## Running a screencast locally

```bash
uv run python examples/screencast/incident/before.py
uv run python examples/screencast/incident/after.py
uv run python examples/screencast/incident/after_disrupted.py
```

Or run the integration tests to verify all three variants in one shot:

```bash
uv run pytest tests/integration/test_screencast_incident.py -vv
uv run pytest tests/integration/test_screencast_supply_chain.py -vv
uv run pytest tests/integration/test_screencast_travel.py -vv
```
