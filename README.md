<div align="center">
  <a href="https://github.com/LangGOAP/LangGOAP">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="LangGOAP Logo" src=".github/images/logo-light.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>Goal-oriented planning for LangGraph agents.</h3>
</div>

<div align="center">
  <a href="LICENSE" target="_blank"><img src="https://img.shields.io/pypi/l/langgoap" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/langgoap" target="_blank"><img src="https://img.shields.io/pepy/dt/langgoap" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/langgoap/" target="_blank"><img src="https://img.shields.io/pypi/v/langgoap.svg?label=%20" alt="Version"></a>
  <a href="pyproject.toml" target="_blank"><img src="https://img.shields.io/pypi/pyversions/langgoap" alt="Python versions"></a>
  <a href="https://github.com/bsbodden/mfcqi" target="_blank"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/LangGOAP/LangGOAP/main/.github/badges/mfcqi.json" alt="MFCQI Score"></a>
</div>

<br>

LangGOAP turns a goal and a set of LangChain tools into a compiled `StateGraph` that plans before it acts, replans on failure, and stays deterministic by default. The planner is classical A* with optional OR-Tools CP-SAT refinement; the runtime is plain LangGraph, so checkpointing, streaming, `interrupt()`, and LangSmith all just work.

```bash
pip install -U langgoap
```

> [!TIP]
> For developing, debugging, and deploying agents, see [LangSmith](https://docs.langchain.com/langsmith/home). LangGOAP ships a `LangSmithTracer` that maps plan / replan / goal-achieved events to LangSmith runs alongside LangGraph's automatic node-level tracing.

## Why LangGOAP?

LangGOAP provides a planning layer for *any* agent that has to choose tools in a particular order under hard constraints:

- **Deterministic planning** — A classical A* search over your action set produces a checked plan before any tool runs; the same inputs always yield the same plan.
- **Constraint optimization built in** — Hard resource caps, soft objectives, temporal `IntervalVar` scheduling, and multi-plan Pareto selection via OR-Tools CP-SAT, with no extra configuration.
- **The plan _is_ a `StateGraph`** — Every plan compiles to a real LangGraph graph, so checkpointers, stores, streaming, `interrupt()`, and LangSmith all just work.
- **Replans automatically** — When an action fails, the world drifts, or a sensor invalidates a precondition, the executor blacklists the offender and the planner picks a new path without any routing code.
- **LLM where it earns its keep** — Natural-language goals are parsed once by `GoalInterpreter`; the loop itself stays symbolic. No ReAct, no agentic reasoning between tool calls.

### In Plain English...

You hand LangGOAP a goal in plain English and a bag of tools, and an LLM reads the goal exactly once to turn it into a symbolic target — that's the only place a model gets to make decisions. From there, a classical A* search picks the shortest sequence of tool calls that reaches the goal, and because the search is deterministic, the same goal and the same tools always produce the same plan. If you've declared resource caps, scheduling windows, or objectives to optimize, those are handed to OR-Tools CP-SAT to refine the plan against real constraints, all without extra wiring. The plan then compiles down to an ordinary LangGraph `StateGraph`, so checkpointing, streaming, `interrupt()`, LangSmith tracing, and everything else in the LangGraph ecosystem work the way you'd expect. When execution hits the real world and something breaks — a tool errors out, a precondition no longer holds, an external sensor disagrees — the executor blacklists the offending action and asks the planner for a new path, so recovery happens automatically rather than through hand-written routing code.

> [!TIP]
> See [`examples/screencast/research_agent/`](examples/screencast/research_agent/) for a head-to-head comparison of `create_react_agent`, a hand-wired `StateGraph`, and LangGOAP — same brief, same tools, real OpenAI + Tavily costs, a revoked API key as the climax.

## Quickstart

The snippet below wraps three LangChain tools and asks LangGOAP to publish an article. The LLM parses the natural-language goal exactly once into a symbolic target like `{"published": True}`, and from there A* takes over. The `preconditions` and `effects` dictionaries describe how each tool changes the world: `write_article` can only run once `have_brief` is true, and `research_topic` is what makes `have_brief` true in the first place. From this static graph the planner derives the chain `research_topic → write_article → publish_article` without any LLM reasoning between tool calls. Every action also has a `cost` — A* minimizes the total cost of the chosen path. We pass `costs={...}` explicitly here so you can see the shape; values that omitted default to `1.0`. Costs only change the plan when multiple chains can reach the goal (e.g., a cached lookup at `1.0` vs. a paid API at `100.0`), and they feed directly into the CSP layer when you want hard resource budgets like `cost_usd` or `tokens` — see [`examples/screencast/research_agent/`](examples/screencast/research_agent/) for that.

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgoap import create_goap_agent

@tool
def research_topic(topic: str) -> str:
    """Produce a short research brief for a topic."""
    return f"Brief on {topic}"

@tool
def write_article(brief: str) -> str:
    """Turn a research brief into an article draft."""
    return f"Article from: {brief}"

@tool
def publish_article(draft: str) -> str:
    """Publish an article draft."""
    return f"Published: {draft}"

agent = create_goap_agent(
    tools=[research_topic, write_article, publish_article],
    goal="Publish an article about GOAP for LangGraph",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    # Preconditions/effects keep the planner honest — never LLM-inferred.
    preconditions={
        "write_article":   {"have_brief": True},
        "publish_article": {"have_draft": True},
    },
    effects={
        "research_topic":  {"have_brief": True},
        "write_article":   {"have_draft": True},
        "publish_article": {"published":  True},
    },
    # A* minimizes total cost. Omitted tools default to cost=1.0.
    costs={
        "research_topic":  1.0,
        "write_article":   3.0,
        "publish_article": 1.0,
    },
)

result = agent.invoke({"world_state": {}, "goal": agent.goap_goal})
```

`agent` is a compiled LangGraph graph. Use it with streaming,
checkpointers, `interrupt()`, or any LangGraph feature.

## Three ways to use it

LangGOAP ships three on-ramps so you can adopt as much or as little as
you need without rewriting your action definitions.

- **`create_goap_agent`** — Natural-language goal plus LangChain tools.
  The shortest path. Used in
  [`examples/tutorials/deep_research_agent.ipynb`](examples/tutorials/deep_research_agent.ipynb).
- **`goapify_tool` / `GoapGraph`** — Hand-authored `ActionSpec` objects
  with explicit preconditions, effects, costs, and validators. The
  workhorse API used by every tutorial.
- **`GoapSubgraph` / `add_goap_subgraph`** — Drop a GOAP loop into an
  existing `StateGraph` as a sealed node. Useful when GOAP is one
  reasoning mode among many.

Each layer is covered by an integration test:
[`test_prebuilt.py`](tests/integration/test_prebuilt.py),
[`test_goapify_tool.py`](tests/integration/test_goapify_tool.py),
[`test_subgraph.py`](tests/integration/test_subgraph.py).

## Features

### Planning

- **A* planner** with custom cost functions, effect validators, and per-action retry budgets.
- **Two-phase pipeline** — A* produces a candidate plan, then OR-Tools CP-SAT refines or replaces it when the goal carries constraints or objectives.
- **Pluggable strategies** via the `PlanningStrategy` Protocol — built-in `AStarStrategy`, `MCTSStrategy`, `CSPRefinementStrategy`, `TwoPhasePipelineStrategy`, `UtilityStrategy`, `AnytimePlanningStrategy`, and `RepairStrategy`.
- **`StrategyRouter`** that dispatches to the right strategy based on cheap-to-compute `ProblemFeatures` (hard constraints, stochasticity, branching factor). Reproducible — routing is a pure function of the problem.
- **`TransitionModel`** to separate declared effects from sampled dynamics, so A* plans under expected values while MCTS rollouts and the graph runtime see the actual sampled world.

### Constraints and scheduling

- **`Score` hierarchy** — `SimpleScore`, `HardSoftScore`, `BendableScore`, with lexicographic comparison and a consistent penalize/reward sign convention.
- **Fluent `ConstraintBuilder`** for resource caps, soft objectives, and weighted penalties: `for_each_action().where(...).sum_resource("gpu_hours").bounded(max=budget)...`
- **Temporal scheduling** — CP-SAT `IntervalVar` per action, precedence from the dependency graph, makespan minimization. Render Gantt charts from `CSPMetadata.schedule`.

### LangGraph-native execution

- **`GoapGraph`** compiles a real `StateGraph` with planner / executor / observer nodes. Sync and async variants of each node so tracer hooks fire from both `.invoke()` and `.ainvoke()`.
- **`StoreExecutionHistory`** persists execution traces in any LangGraph `BaseStore` (`InMemoryStore`, `AsyncPostgresStore`, custom) without requiring an embedder.
- **`MultiGoal`** for sequential and `any`-mode goal decomposition, dispatched at the observer/planner level.
- **Plan visualization** via `render_mermaid`, `render_mermaid_gantt`, `render_dot`, `render_ascii`, `render_ascii_gantt`, and the top-level `visualize` dispatcher — pure Python, no binary dependencies for Mermaid / ASCII.

### Reliability and recovery

- **`ActionQos`** — per-action retry policy, idempotency markers (`FIRE_ONCE`, `can_rerun`, `read_only`).
- **`require_human_approval`** — pass a Pydantic `BaseModel` to collect a typed form from a human via LangGraph's `interrupt()`; pass `True` for a plain approve/deny gate. Validated on resume.
- **Stuck handlers** — when planning fails, `MulticastStuckHandler` runs ordered recovery handlers that can mutate world state, swap in a relaxed goal, or escalate to a human. The first handler that returns `REPLAN` wins.
- **Early-termination policies** — `MaxCostPolicy`, `MaxWallClockPolicy`, `MaxLLMCallsPolicy`, `MaxTokensPolicy`, `MaxActionsPolicy`, `OnStuckPolicy`, composed with `FirstOfPolicy` / `AllOfPolicy`.

### Observability

- **`PlanningTracer` Protocol** with sync + async hooks (`on_plan_complete`, `on_action_retry`, `on_strategy_chosen`, …). `NullTracer`, `LoggingTracer`, `MultiTracer`, and `LangSmithTracer` ship in-tree. Custom tracers (OpenTelemetry, Prometheus) are ordinary classes that implement the protocol; tracer exceptions never propagate into the planner.
- **`CostAccumulator`** + `DEFAULT_COST_PER_1K_TOKENS` for live LLM token / USD accounting, plumbed into world state for `MaxCostPolicy` and the `cost_bounded_research_agent` tutorial.

### Checkpointing

- **All three LangGraph backends supported and tested**: `MemorySaver`, `AsyncPostgresSaver`, `RedisSaver` / `AsyncRedisSaver`. Custom `ormsgpack` serde round-trips frozen dataclasses, `MappingProxyType`, `frozenset`, `timedelta`, and `tuple` correctly. Install with `pip install langgoap[checkpoint-postgres]` or `pip install langgoap[checkpoint-redis]`.

### Natural-language goals

- **`GoalInterpreter`** parses plain-English requests into a `GoalSpec` via any LangChain chat model that supports structured output. `GoapGraph.invoke_nl(request, llm=...)` / `ainvoke_nl()` accept a string directly.

### CLI and deployment

- **`langgoap` CLI** — `plan`, `actions`, `explain`, `visualize`, `deploy-init`. Loads actions, goals, and world state from `module:variable` references.
- **LangGraph deployment scaffold** — `scaffold_deployment` (and `langgoap deploy-init`) generate a complete `langgraph dev`-ready directory in one command. Every `langgraph` deployment serves an `/mcp` endpoint, so a LangGOAP graph becomes an MCP tool with no extra wiring.

### DeepAgents and tool interop

- **`create_goap_tool`** wraps a `GoapGraph` as a LangChain `StructuredTool`.
- **`create_goap_subagent`** returns a `CompiledSubAgent`-compatible dict for [Deep Agents](https://github.com/langchain-ai/deepagents).
- Both call `GoalInterpreter` under the hood so the calling agent sends a natural-language request.

## Examples

The [`examples/`](examples/) directory holds three flavours of runnable
documentation. Every featured tutorial has a corresponding integration
test under [`tests/integration/`](tests/integration/) — the notebook is
the explanation; the test is the source of truth.

- **[`examples/basics/`](examples/basics/)** — short primers that each
  exercise a single mechanic: quickstart, CLI, plan visualization,
  natural-language goals, tracing, termination policies, stuck
  handlers, typed-form HITL, action QoS, MCTS vs A* on stochastic
  domains.
- **[`examples/tutorials/`](examples/tutorials/)** — end-to-end
  walkthroughs in three tiers. Tier 1 introduces GOAP on toy domains;
  Tier 2 covers constraint-optimization workflows; Tier 3 showcases
  the full stack on substantial problems
  (`deep_research_agent`, `flexible_job_shop`, `supply_chain_disruption_mediator`,
  `code_review_agent_mcp_deployment`).
- **[`examples/screencast/`](examples/screencast/)** — the flagship
  YouTube companion lives at
  [`examples/screencast/research_agent/`](examples/screencast/research_agent/):
  four agents (`create_react_agent` baseline, hand-wired `StateGraph`,
  LangGOAP, LangGOAP-under-disruption) on the same brief with real
  OpenAI + Tavily costs and a Tavily-key revocation as the climax.
  Three hermetic case studies (incident, supply chain, travel) sit
  alongside it for vertical-specific walk-throughs.

Start with
[`examples/screencast/research_agent/`](examples/screencast/research_agent/)
to see GOAP replace a routing graph and survive a runtime disruption
in one sitting, or
[`examples/tutorials/directory_handler.ipynb`](examples/tutorials/directory_handler.ipynb)
if you are new to GOAP and want a smaller starting point.

## LangGraph ecosystem

While LangGOAP can be used wherever LangChain tools are available, it integrates seamlessly with the rest of the LangChain stack:

- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)** — the runtime substrate. Every LangGOAP plan compiles to a real `StateGraph` and works with streaming, checkpointing, and `interrupt()`.
- **[LangSmith](https://docs.langchain.com/langsmith/home)** — `LangSmithTracer` emits GOAP plan / replan / goal-achieved events to LangSmith alongside LangGraph's automatic node-level traces.
- **[LangGraph deployment](https://docs.langchain.com/langsmith/deployments)** — `langgoap deploy-init` scaffolds a `langgraph dev`-ready directory; the generated deployment serves an `/mcp` endpoint so a LangGOAP graph is callable from any MCP client (Claude Desktop, Cursor, …).
- **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)** — `create_goap_tool` and `create_goap_subagent` embed a LangGOAP graph as a tool or subagent inside a Deep Agents harness.

---

## Documentation

- [`examples/screencast/research_agent/`](examples/screencast/research_agent/) – Flagship walkthrough: ReAct vs. hand-wired `StateGraph` vs. LangGOAP on the same brief, with measured OpenAI + Tavily costs.
- [`examples/tutorials/`](examples/tutorials/) – End-to-end notebooks across three tiers (toy domains, constraint-optimization, full-stack).
- [`examples/basics/`](examples/basics/) – Short primers, one mechanic per notebook (CLI, visualization, NL goals, tracing, termination policies, stuck handlers, typed-form HITL, MCTS vs A*).
- [`langgoap/__init__.py`](langgoap/__init__.py) – Authoritative list of public symbols. Anything not re-exported from the top-level package is internal and subject to change.
- [OptaPlanner concept mapping](docs/optaplanner_mapping.md) – How LangGOAP's `Score` hierarchy and `ConstraintBuilder` map onto OptaPlanner.
- [Changelog](CHANGELOG.md) – Release notes for every public version.

## Contributing

LangGOAP is developed test-first: every feature starts with a failing integration test that uses real infrastructure via TestContainers, never mocks. Notebooks are runnable documentation of what the tests already verify — never the other way around.

```bash
uv sync
make check                                              # full lint + test suite
uv run pytest tests/integration/test_flexible_job_shop.py -vv
uv run pytest -m api                                    # requires OPENAI_API_KEY / ANTHROPIC_API_KEY
```

---

## Acknowledgements

LangGOAP is inspired by [GOAP](https://alumni.media.mit.edu/~jorkin/gdc2006_orkin_jeff_fear.pdf) (Jeff Orkin / F.E.A.R.) and [Embabel](https://github.com/embabel/embabel-agent), which first applied GOAP planning to agentic LLM workflows. The `Score` hierarchy and fluent `ConstraintBuilder` are adapted from [OptaPlanner](https://www.optaplanner.org/); the CSP pipeline is built on [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver); A* correctness was checked against [GOApy](https://github.com/leopoldmaillard/GOApy); and the action/effect model draws on [unified-planning](https://github.com/aiplan4eu/unified-planning). LangGOAP is built on [LangGraph](https://langchain-ai.github.io/langgraph/) by [Integrallis Software](https://integrallis.com), but can be used wherever LangChain tools are available.

## License

MIT. See [LICENSE](LICENSE).
