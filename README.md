<div align="center">
  <h1>LangGOAP</h1>
  <h3>GOAP planning for LangGraph agents.</h3>
</div>

<div align="center">

[![License](https://img.shields.io/pypi/l/langgoap)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/langgoap)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/langgoap?label=%20)](https://pypi.org/project/langgoap/)
[![Downloads](https://img.shields.io/pepy/dt/langgoap)](https://pypistats.org/packages/langgoap)
[![MFCQI Score](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/langgoap/main/.github/badges/mfcqi.json)](https://github.com/bsbodden/mfcqi)

</div>

<br>

LangGOAP plans before it acts. Give it a goal and a set of LangChain
tools; it returns a [compiled `StateGraph`](https://langchain-ai.github.io/langgraph/concepts/low_level/)
that picks the cheapest valid action sequence, executes it, and replans
on failure — no hand-written routing, no free-form ReAct loop,
deterministic by default.

```bash
pip install -U langgoap
```

> [!TIP]
> For developing, debugging, and deploying agents, see
> [LangSmith](https://docs.langchain.com/langsmith/home). LangGOAP ships
> a `LangSmithTracer` that maps plan / replan / goal-achieved events to
> LangSmith runs alongside LangGraph's automatic node-level tracing.

## Why LangGOAP?

- **[Deterministic planning](https://docs.langchain.com/oss/python/langgraph/overview)** — A classical A* search over your action set produces a checked plan before any tool runs; the same inputs always yield the same plan.
- **Constraint optimization built in** — Hard resource caps, soft objectives, temporal `IntervalVar` scheduling, and multi-plan Pareto selection via OR-Tools CP-SAT, with no extra configuration.
- **The plan _is_ a `StateGraph`** — Every plan compiles to a real LangGraph graph, so checkpointers, stores, streaming, `interrupt()`, and LangSmith all just work.
- **Replans automatically** — Action fails, world drifts, or a sensor invalidates a precondition; the executor blacklists the offender and the planner picks a new path without any routing code.
- **LLM where it earns its keep** — Natural-language goals are parsed once by `GoalInterpreter`; the loop itself stays symbolic. No ReAct, no agentic reasoning between tool calls.

## Quickstart

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

LangGOAP is built on LangGraph and integrates with the rest of the
LangChain stack:

- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)** — the runtime substrate. Every LangGOAP plan compiles to a real `StateGraph` and works with streaming, checkpointing, and `interrupt()`.
- **[LangSmith](https://docs.langchain.com/langsmith/home)** — `LangSmithTracer` emits GOAP plan / replan / goal-achieved events to LangSmith alongside LangGraph's automatic node-level traces.
- **[LangGraph deployment](https://docs.langchain.com/langsmith/deployments)** — `langgoap deploy-init` scaffolds a `langgraph dev`-ready directory; the generated deployment serves an `/mcp` endpoint so a LangGOAP graph is callable from any MCP client (Claude Desktop, Cursor, …).
- **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)** — `create_goap_tool` and `create_goap_subagent` embed a LangGOAP graph as a tool or subagent inside a Deep Agents harness.

## Documentation

- [Changelog](CHANGELOG.md) – Release notes for every public version.
- [`langgoap/__init__.py`](langgoap/__init__.py) – Authoritative list of public symbols. Anything not re-exported from the top-level package is internal and subject to change.
- [OptaPlanner concept mapping](docs/optaplanner_mapping.md) – How LangGOAP's `Score` hierarchy and `ConstraintBuilder` map onto OptaPlanner.

## Contributing

LangGOAP is developed test-first: every feature starts with a failing
integration test that uses real infrastructure via TestContainers, never
mocks. Notebooks are runnable documentation of what the tests already
verify — never the other way around.

```bash
uv sync
make check                                              # full lint + test suite
uv run pytest tests/integration/test_flexible_job_shop.py -vv
uv run pytest -m api                                    # requires OPENAI_API_KEY / ANTHROPIC_API_KEY
```

## Acknowledgements

LangGOAP builds on ideas and implementations from several projects:

- **[GOAP](https://alumni.media.mit.edu/~jorkin/gdc2006_orkin_jeff_fear.pdf)** (Jeff Orkin / F.E.A.R.) — the classical game-AI technique that drives LangGOAP's A* planner.
- **[Embabel](https://github.com/embabel/embabel-agent)** — first to apply GOAP planning to agentic LLM workflows.
- **[OptaPlanner](https://www.optaplanner.org/)** — the `Score` hierarchy and fluent `ConstraintBuilder` are adapted from OptaPlanner's constraint-solving API.
- **[OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)** — the constraint solver behind LangGOAP's CSP pipeline.
- **[GOApy](https://github.com/leopoldmaillard/GOApy)** — pure-Python GOAP implementation used as a reference for A* correctness.
- **[unified-planning](https://github.com/aiplan4eu/unified-planning)** — formal AI planning concepts (temporal, numeric, PDDL interop) that informed LangGOAP's action/effect model.
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — the runtime substrate.

Built by [Integrallis Software](https://integrallis.com).

## License

MIT. See [LICENSE](LICENSE).
