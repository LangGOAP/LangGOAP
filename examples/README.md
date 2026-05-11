# Examples

Runnable documentation for LangGOAP, organized in three layers:

- [`basics/`](basics/) — Short primers that each exercise a single mechanic.
- [`tutorials/`](tutorials/) — End-to-end walkthroughs on real problems, tiered by complexity.
- [`screencast/`](screencast/) — Before / after / after-disrupted demos contrasting a hand-wired LangGraph workflow with the same logic in GOAP.

Every featured notebook has a corresponding integration test under
[`tests/integration/`](../tests/integration/) that runs on every
`make check`. The notebook is the explanation; the integration test is
the source of truth.

## Start here

- **New to GOAP?** Open [`tutorials/directory_handler.ipynb`](tutorials/directory_handler.ipynb) — a minimal GOAP loop in fewer than 30 cells.
- **Coming from a routing graph?** Open [`screencast/incident/`](screencast/incident/) — a hand-wired LangGraph incident-response workflow rewritten as a four-line GOAP graph, then perturbed at runtime.
- **Building a LangChain agent today?** Open [`tutorials/deep_research_agent.ipynb`](tutorials/deep_research_agent.ipynb) — drop-in `create_goap_agent` over LangChain tools with `LangSmithTracer` tracing.

## Basics

Single-mechanic primers, each runnable end-to-end without an LLM unless explicitly noted.

- [`basics/plan_visualization.ipynb`](basics/plan_visualization.ipynb) — Mermaid / DOT / ASCII rendering of plans and CSP schedules.
- [`basics/nl_goal_interpreter.ipynb`](basics/nl_goal_interpreter.ipynb) — Plain-English request → `GoalSpec` via `GoalInterpreter`.
- [`basics/tracing.ipynb`](basics/tracing.ipynb) — Observability hooks with `LoggingTracer`, `LangSmithTracer`, and `MultiTracer`.
- [`basics/cli.ipynb`](basics/cli.ipynb) — The `langgoap` CLI: `plan`, `actions`, `explain`, `visualize`, `deploy-init`.
- [`basics/stochastic_gridworld.ipynb`](basics/stochastic_gridworld.ipynb) — `TransitionModel` separating declared effects from sampled dynamics on FrozenLake-4x4.
- [`basics/mcts_vs_astar_stochastic.ipynb`](basics/mcts_vs_astar_stochastic.ipynb) — When to use `MCTSStrategy` instead of A* on stochastic domains.

## Tutorials

See [`tutorials/README.md`](tutorials/README.md) for the full 22-tutorial catalog across three tiers.

## Screencasts

Five-minute paired demos. Each scenario ships three runnable scripts:
a hand-written LangGraph baseline (`before.py`), the same logic as a
GOAP graph (`after.py`), and the GOAP version run under disruption
(`after_disrupted.py`). The contrast is the point: identical behaviour
when nothing goes wrong, divergent behaviour when reality drifts.

- [`screencast/incident/`](screencast/incident/) — Incident response. Backed by [`test_screencast_incident.py`](../tests/integration/test_screencast_incident.py).
- [`screencast/supply_chain/`](screencast/supply_chain/) — Supply-chain disruption. Backed by [`test_screencast_supply_chain.py`](../tests/integration/test_screencast_supply_chain.py).
- [`screencast/travel/`](screencast/travel/) — Travel rebooking. Backed by [`test_screencast_travel.py`](../tests/integration/test_screencast_travel.py).

## Running notebooks

```bash
pip install langgoap jupyter
jupyter lab
```

A Docker Compose setup is also provided in this directory:

```bash
cd examples && docker compose up
```

It mounts `../` and installs the local library automatically; open
<http://127.0.0.1:8888/tree> when the container is ready.
