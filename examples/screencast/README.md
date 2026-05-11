# Screencasts

Live-API and hermetic demos that contrast hand-written LangGraph agents with the same business logic expressed as LangGOAP plans, then run the GOAP versions through realistic disruptions to show automatic replanning.

## Flagship \u2014 the YouTube intro

### [`research_agent/`](research_agent/) \u2014 cost-bounded research agent

The featured demo. One brief, four agent shapes, real OpenAI + Tavily:

| Version | Shape | What it shows |
| --- | --- | --- |
| React baseline | `create_react_agent(llm, tools)` | Canonical LangGraph idiom \u2014 LLM decides everything |
| Hand-wired | Explicit `StateGraph` with dispatcher node | Senior-dev pattern after react proves unpredictable |
| LangGOAP planned | `GoapGraph` + `ConstraintSpec(cost_usd <= 2.00)` | Cost-aware, self-healing |
| Disrupted | All three, run under a revoked `TAVILY_API_KEY` | The climax: which one survives? |

Real OpenAI + Tavily calls, real USD costs on screen, real disruption (Tavily key revocation) handled without retry logic. Backed by [`tests/integration/test_screencast_research_agent.py`](../../tests/integration/test_screencast_research_agent.py) with both a fast mocked path (CI) and a gated live-API path.

```bash
# Mocked (fast, no API keys, runs in CI)
uv run pytest tests/integration/test_screencast_research_agent.py

# Live four-way head-to-head
export OPENAI_API_KEY=sk-... TAVILY_API_KEY=tvly-...
uv run python -m examples.screencast.research_agent.disrupted
```

Full breakdown: [`research_agent/README.md`](research_agent/README.md). Notebook companion: [`research_agent/screencast.ipynb`](research_agent/screencast.ipynb). Video script with timecodes: [`research_agent/script.md`](research_agent/script.md).

---

## Further case studies

Three paired demos focused on a single domain each. They use hermetic fixtures (no API keys required) and pair a hand-wired LangGraph baseline with the same logic as a LangGOAP plan, plus a disruption variant.

Each ships four files:

| File | Purpose |
| --- | --- |
| `before.py` | Hand-wired `StateGraph` with conditional edges encoding every fallback branch. Runs without LangGOAP. |
| `after.py` | The same scenario as a LangGOAP graph. Identical observable behaviour on the happy path; the routing code is gone. |
| `after_disrupted.py` | The GOAP version run under perturbation. No routing code changed; the planner replans. |
| `script.md` / `narrative.md` | Production artifacts. Not required for running the demo. |

Every case study has an integration test under [`tests/integration/`](../../tests/integration/) so the scripts stay runnable as the codebase evolves.

### [`incident/`](incident/) \u2014 emergent cost-based escalation

A pager fires. The agent triages, runs diagnostics, applies a remediation, and notifies stakeholders. In `after_disrupted.py`, the primary remediation fails its post-condition and the planner re-routes to a fallback path. The most explicit demonstration of GOAP discovering a multi-step recovery chain (`analyze_logs \u2192 apply_hotfix`) the dev did not script.

Integration test: [`tests/integration/test_screencast_incident.py`](../../tests/integration/test_screencast_incident.py).

### [`supply_chain/`](supply_chain/) \u2014 constraint optimisation under disruption

A 500-unit order must ship by Friday. The happy path picks the preferred vendor and standard shipping. Under disruption, the preferred vendor is out of stock and standard shipping is late; the planner switches to an alternate vendor and express shipping. Best illustration of a `ConstraintSpec`-style deadline driving the replan.

Integration test: [`tests/integration/test_screencast_supply_chain.py`](../../tests/integration/test_screencast_supply_chain.py).

### [`travel/`](travel/) \u2014 multi-modal rebooking

A multi-leg itinerary needs to land on time. Weather grounds the preferred flight; the planner reroutes through an alternate carrier and rebooks the downstream hotel. The most relatable narrative of the three.

Integration test: [`tests/integration/test_screencast_travel.py`](../../tests/integration/test_screencast_travel.py).

---

## Running the case studies locally

```bash
uv run python examples/screencast/incident/before.py
uv run python examples/screencast/incident/after.py
uv run python examples/screencast/incident/after_disrupted.py
```

Or verify all three case-study variants in one shot:

```bash
uv run pytest tests/integration/test_screencast_incident.py -vv
uv run pytest tests/integration/test_screencast_supply_chain.py -vv
uv run pytest tests/integration/test_screencast_travel.py -vv
```
