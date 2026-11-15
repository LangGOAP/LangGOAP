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

- **First time here?** Open [`screencast/research_agent/`](screencast/research_agent/) — the flagship four-way head-to-head (react baseline vs. hand-wired LangGraph vs. LangGOAP vs. LangGOAP-under-disruption) with real OpenAI + Tavily costs.
- **New to GOAP?** Open [`tutorials/directory_handler.ipynb`](tutorials/directory_handler.ipynb) — a minimal GOAP loop in fewer than 30 cells.
- **Coming from a routing graph?** Open [`screencast/incident/`](screencast/incident/) — a hand-wired LangGraph incident-response workflow rewritten as a four-line GOAP graph, then perturbed at runtime.
- **Building a LangChain agent today?** Open [`tutorials/deep_research_agent.ipynb`](tutorials/deep_research_agent.ipynb) — drop-in `create_goap_agent` over LangChain tools with `LangSmithTracer` tracing.

## Basics

Single-mechanic primers, each runnable end-to-end without an LLM unless explicitly noted.

- [`basics/plan_visualization.ipynb`](basics/plan_visualization.ipynb) — Mermaid / DOT / ASCII rendering of plans and CSP schedules.
- [`basics/nl_goal_interpreter.ipynb`](basics/nl_goal_interpreter.ipynb) — Plain-English request → `GoalSpec` via `GoalInterpreter` (needs an LLM).
- [`basics/create_goap_agent_quickstart.ipynb`](basics/create_goap_agent_quickstart.ipynb) — The one-liner: LangChain tools + goal → compiled `StateGraph`.
- [`basics/termination_policies.ipynb`](basics/termination_policies.ipynb) — `MaxCostPolicy`, `MaxActionsPolicy`, and `FirstOfPolicy` composition.
- [`basics/stuck_handlers.ipynb`](basics/stuck_handlers.ipynb) — `FunctionalStuckHandler` and `MulticastStuckHandler` recovering planning failures.
- [`basics/typed_form_hitl.ipynb`](basics/typed_form_hitl.ipynb) — Typed-form `interrupt()` with Pydantic validation on resume.
- [`basics/action_qos.ipynb`](basics/action_qos.ipynb) — In-executor retry with `ActionQos` and exponential backoff.
- [`basics/goap_subgraph.ipynb`](basics/goap_subgraph.ipynb) — Embedding a `GoapGraph` as a node of a larger `StateGraph`.
- [`basics/tracing.ipynb`](basics/tracing.ipynb) — Observability hooks with `LoggingTracer`, `LangSmithTracer`, and `MultiTracer`.
- [`basics/cli.ipynb`](basics/cli.ipynb) — The `langgoap` CLI: `plan`, `actions`, `explain`, `visualize`, `deploy-init`.
- [`basics/stochastic_gridworld.ipynb`](basics/stochastic_gridworld.ipynb) — `TransitionModel` separating declared effects from sampled dynamics on FrozenLake-4x4.
- [`basics/mcts_vs_astar_stochastic.ipynb`](basics/mcts_vs_astar_stochastic.ipynb) — When to use `MCTSStrategy` instead of A* on stochastic domains.

## Tutorials

See [`tutorials/README.md`](tutorials/README.md) for the 22 featured tutorials across three tiers plus the five LangGraph-comparison notebooks.

## Screencasts

**Flagship**: [`screencast/research_agent/`](screencast/research_agent/) — four-way comparison (react baseline / hand-wired LangGraph / LangGOAP / LangGOAP-disrupted) with real OpenAI + Tavily costs and a Tavily-key-revocation climax. Backed by [`test_screencast_research_agent.py`](../tests/integration/test_screencast_research_agent.py).

**Further case studies** — single-domain paired demos with hermetic fixtures (no API keys). Each ships `before.py` (hand-wired LangGraph), `after.py` (same logic as GOAP), and `after_disrupted.py` (GOAP under perturbation):

- [`screencast/incident/`](screencast/incident/) — Incident response. Backed by [`test_screencast_incident.py`](../tests/integration/test_screencast_incident.py).
- [`screencast/supply_chain/`](screencast/supply_chain/) — Supply-chain disruption. Backed by [`test_screencast_supply_chain.py`](../tests/integration/test_screencast_supply_chain.py).
- [`screencast/travel/`](screencast/travel/) — Travel rebooking. Backed by [`test_screencast_travel.py`](../tests/integration/test_screencast_travel.py).

**LangGraph tutorial refactors** — official LangGraph tutorials rewritten as GOAP, each with a narrated notebook that runs the condensed hand-wired original, the GOAP version and the GOAP version under disruption (no API keys):

- [`screencast/llm_compiler/`](screencast/llm_compiler/) — LLMCompiler: `$N` dependency references, a hand-written scheduler and a `Replan`-emitting joiner become preconditions, `executor="parallel"` and blacklist-and-replan. Notebook: [`llm_compiler_screencast.ipynb`](screencast/llm_compiler/llm_compiler_screencast.ipynb). Backed by [`test_screencast_llm_compiler.py`](../tests/integration/test_screencast_llm_compiler.py).
- [`screencast/email_assistant/`](screencast/email_assistant/) — executive-ai-assistant: three message-shape routers with `ValueError` fall-throughs and four Agent Inbox handlers become triage facts as preconditions, `require_human_approval=True` and a Pydantic form; a denied send is re-planned around; a new process resumes a paused thread. Notebook: [`email_assistant_screencast.ipynb`](screencast/email_assistant/email_assistant_screencast.ipynb). Backed by [`test_screencast_email_assistant.py`](../tests/integration/test_screencast_email_assistant.py).

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
