# LangGoap

**Goal-Oriented Action Planning for [LangGraph](https://langchain-ai.github.io/langgraph/), with constraint optimization.**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

LangGoap turns a declarative goal and a set of actions into a **compiled
LangGraph `StateGraph`** that plans, executes, and replans. It combines
classical GOAP A\* search with OR-Tools CP-SAT constraint optimization
and LLM-driven natural-language goal interpretation, and ships as a
first-class citizen of the LangChain ecosystem.

Built by [Integrallis Software](https://integrallis.com).

---

## Why LangGoap

| Capability                              | LangGoap | `create_react_agent` |
|-----------------------------------------|:--------:|:--------------------:|
| A\* planning over explicit actions      |    Yes   |          No          |
| Resource constraints (hard + soft)      |    Yes   |          No          |
| Multi-objective optimization (CP-SAT)   |    Yes   |          No          |
| Temporal scheduling (`IntervalVar`)     |    Yes   |          No          |
| Natural-language goal interpretation    |    Yes   |          —           |
| Human-in-the-loop (`interrupt()`)       |    Yes   |         Yes          |
| Compiled `StateGraph` as the plan       |    Yes   |         Yes          |
| Sync + async parity                     |    Yes   |         Yes          |
| Multi-goal sequential decomposition     |    Yes   |          No          |
| Execution history in `BaseStore`        |    Yes   |          —           |
| Checkpointing (Memory / Postgres / Redis) |  Yes   |         Yes          |
| Plan visualization (Mermaid / DOT)      |    Yes   |          No          |

---

## Installation

```bash
pip install langgoap

# Add CP-SAT constraint optimization (recommended for Tier 2+ tutorials):
pip install "langgoap[optimization]"
```

Requires Python 3.10+. OR-Tools is an **optional** dependency; core
A\* planning works without it and CSP features degrade gracefully with
a clear `ImportError` when invoked.

---

## Quickstart — the one-liner

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
    # Pre/post conditions keep the planner honest — never LLM-inferred.
    effects={
        "research_topic":   {"have_brief": True},
        "write_article":    {"have_draft": True},
        "publish_article":  {"published":  True},
    },
    preconditions={
        "write_article":    {"have_brief": True},
        "publish_article":  {"have_draft": True},
    },
)

result = agent.invoke({"world_state": {}, "goal": agent.goap_goal})
```

No free-form ReAct loop — the planner produces a deterministic action
sequence before a single tool executes, and re-plans on failure.

### With checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

graph = GoapGraph(actions).compile(checkpointer=MemorySaver())
result = graph.invoke(
    {"world_state": {}, "goal": goal},
    config={"configurable": {"thread_id": "run-1"}},
)
```

<details>
<summary>With Redis or Postgres</summary>

```python
# Redis
from langgraph.checkpoint.redis import RedisSaver

with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
    graph = GoapGraph(actions).compile(checkpointer=saver)

# Postgres
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
    await saver.setup()
    graph = GoapGraph(actions).compile(checkpointer=saver)
```

</details>

---

## Three-layer low-code on-ramp

LangGoap ships three integration layers so you can start simple and
graduate without rewriting action definitions.

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Layer A — create_goap_agent(tools, goal, llm=, …)                   │
│   The one-liner. NL goal + tools → compiled StateGraph.             │
├─────────────────────────────────────────────────────────────────────┤
│ Layer B — goapify_tool(tool, preconditions=, effects=, …)           │
│   Explicit BaseTool → ActionSpec adapter. Fully deterministic.      │
├─────────────────────────────────────────────────────────────────────┤
│ Layer C — GoapSubgraph / add_goap_subgraph(parent, …)               │
│   Drop a GOAP loop into an existing StateGraph as a sealed node.    │
└─────────────────────────────────────────────────────────────────────┘
```

Internally, Layer A delegates to Layer B, and Layer C uses the same
`GoapGraph` builder — there is no duplicated wiring code. End-to-end
use of each layer is covered by
[`tests/integration/test_prebuilt.py`](tests/integration/test_prebuilt.py),
[`tests/integration/test_goapify_tool.py`](tests/integration/test_goapify_tool.py),
and [`tests/integration/test_subgraph.py`](tests/integration/test_subgraph.py);
the Tier 2 tutorial notebooks below exercise Layer A (`create_goap_agent`)
and Layer B (`goapify_tool`) on real problems.

---

## What's in the box

### Planning

- **A\* planner** (`langgoap.plan`) — classical GOAP search with
  customizable cost functions, effect validators, and per-action retry
  budgets.
- **Two-phase pipeline** (`langgoap.pipeline_plan`) — A\* produces a
  candidate plan, then CSP refines or replaces it with a better
  alternative when the goal has constraints or objectives.
- **Planning strategy hierarchy** — `PlanningStrategy` Protocol with
  built-in `AStarStrategy`, `CSPRefinementStrategy`, and
  `TwoPhasePipelineStrategy`. Pass your own strategy to
  `GoapPlanner(strategy=...)`.

### Constraints and scoring

- **`Score` hierarchy** — `SimpleScore`, `HardSoftScore`,
  `BendableScore`. Lexicographic comparison, sign convention matches
  OptaPlanner (`hard ≤ 0`).
- **Fluent `ConstraintBuilder`** — `for_each_action().where(...)
  .sum_resource("gpu_hours").bounded(max=budget).penalize(level="hard",
  weight=1.0).as_constraint("gpu_budget")`.
- **Hard vs soft constraints** — hard violations mark plans
  `INFEASIBLE`; soft bounds contribute weighted penalties to the
  CP-SAT objective via non-negative violation variables.
- **Temporal scheduling** — CP-SAT `IntervalVar` per action, precedence
  from the dependency graph, makespan minimization. Render Gantt charts
  from `CSPMetadata.schedule`.

### LangGraph-native execution

- **`GoapGraph`** — builds and compiles a real `StateGraph` with
  planner / executor / observer nodes. Every node has sync and async
  variants so tracer hooks fire from both `.invoke()` and `.ainvoke()`.
- **Execution history in `BaseStore`** — `StoreExecutionHistory` with
  reverse-index storage that works on `InMemoryStore`,
  `AsyncPostgresStore`, and any custom `BaseStore` without requiring an
  embedder.
- **`MultiGoal`** — sequential and `any`-mode multi-goal decomposition,
  dispatched at the observer/planner level.

### Observability

- **`PlanningTracer` Protocol** with sync + async hooks (`on_*` /
  `aon_*`). `NullTracer`, `LoggingTracer`, `MultiTracer`, and
  `LangSmithTracer` ship in-tree. `LangSmithTracer` maps GOAP domain
  events (plan/replan/goal-achieved) to LangSmith runs alongside
  LangGraph's automatic node-level tracing. Custom tracers
  (OpenTelemetry, Prometheus) are ordinary Python classes that implement
  the protocol. Tracer exceptions never propagate into the planner.
- **Plan visualization** — `render_mermaid`, `render_mermaid_gantt`,
  `render_dot`, `render_ascii`, `render_ascii_gantt`, `visualize`.
  Pure Python; no binary dependencies for Mermaid / ASCII output.

### Human-in-the-loop

- **`ActionSpec.require_human_approval`** — when `True`, the executor
  calls `interrupt()` before running the action and waits for a resume.
  Denied actions are immediately blacklisted and the planner replans.
  Requires a checkpointer (`MemorySaver`, `AsyncPostgresSaver`,
  `RedisSaver`, etc.).

### Checkpointing

- **Tested with all three LangGraph backends**: `MemorySaver`,
  `AsyncPostgresSaver`, and `RedisSaver`/`AsyncRedisSaver`. Custom
  ormsgpack serializers round-trip frozen dataclasses,
  `MappingProxyType`, `frozenset`, `timedelta`, and `tuple` correctly.
  Install optional extras: `pip install langgoap[checkpoint-postgres]`
  or `pip install langgoap[checkpoint-redis]`.

### Natural-language goals

- **`GoalInterpreter`** — provider-agnostic via `BaseChatModel`
  structured output. `GoapGraph.invoke_nl(request, llm=...)` /
  `ainvoke_nl()` accept a plain string and produce a `GoalSpec` on
  the fly.

---

## Tutorial catalog

All 15 tutorials in `examples/tutorials/` have a corresponding passing
integration test under `tests/integration/`. Tutorial helpers and
domain data live in the shared
`examples/tutorials/tutorial_examples/` package.

### Tier 1 — Primers

- (1) `directory_handler.ipynb` — GOAP basics from `GOApy`.
- (2) `robot_navigation.ipynb` — A\* primer from `unified-planning`.
- (3) `hungry_agent.ipynb` — NL goal interpreter walk-through.

### Tier 2 — OptaPlanner GOAPifications + workflow agents

- (4) `cloud_balancing.ipynb` — VM bin-packing with the one-liner.
- (5) `vehicle_routing.ipynb` — capacity-constrained routing with Gantt.
- (6) `nurse_rostering.ipynb` — skill matching with `HardSoftScore`.
- (7) `project_job_scheduling.ipynb` — RCPSP and critical path.
- (8) `task_assigning.ipynb` — fluent `ConstraintBuilder`.
- (9) `sql_query_agent.ipynb` — schema → generate → test → refine.
- (10) `vulnerability_scanner.ipynb` — phased discovery and
  blacklisting.

### Tier 3 — Full-feature showcases

- (11) `deep_research_agent.ipynb` — `StoreExecutionHistory` + tracing.
- (12) `hierarchical_product_launch.ipynb` — `MultiGoal` decomposition.
- (13) `content_builder_agent.ipynb` — multi-objective CSP.
- (14) `temporal_match_cellar.ipynb` — durative overlapping actions.
- (15) `flexible_job_shop.ipynb` — every v0.1.0 feature in one notebook.

Basics notebooks (`examples/basics/`) cover plan visualization
(`plan_visualization.ipynb`) and the natural-language goal interpreter
(`nl_goal_interpreter.ipynb`). The three integration layers, tracing,
and execution history are covered by the tutorial notebooks above and
by their corresponding integration tests under `tests/integration/`.

---

## Relationship to OptaPlanner

LangGoap does **not** re-implement OptaPlanner's `Move` / `Phase` /
`ScoreDirector` / `Tabu` / `SimulatedAnnealing` machinery — OR-Tools
CP-SAT already implements the equivalent search at a lower, more
efficient level via branch-and-bound with no-good learning,
propagation, and restart. Instead, LangGoap exposes a coherent Python
surface that maps OptaPlanner concepts onto CP-SAT and the LangChain
ecosystem.

See [`docs/optaplanner_mapping.md`](docs/optaplanner_mapping.md) for
the full class-to-concept mapping and a worked custom-strategy example.

---

## Development

```bash
# Set up the environment
uv sync

# Run the full lint + test suite
make check

# Run a single integration test file
uv run pytest tests/integration/test_flexible_job_shop.py -vv

# Run live API tests (requires OPENAI_API_KEY / ANTHROPIC_API_KEY)
uv run pytest -m api
```

LangGoap is developed test-first: every feature starts with a failing
integration test that uses real infrastructure via TestContainers,
never mocks. Notebooks are runnable documentation of what the tests
already verify — never the other way around.

---

## Public API

All symbols exported from `langgoap` are part of the stable v0.1.0
public surface. The full list is in
[`langgoap/__init__.py`](langgoap/__init__.py) and mirrored in the
[changelog](CHANGELOG.md). Anything not re-exported from the top-level
package is internal and subject to change.

---

## Reference

- **Changelog**: [`CHANGELOG.md`](CHANGELOG.md)
- **OptaPlanner mapping**: [`docs/optaplanner_mapping.md`](docs/optaplanner_mapping.md)
- **LangGraph**: <https://langchain-ai.github.io/langgraph/>
- **OR-Tools CP-SAT**: <https://developers.google.com/optimization/cp/cp_solver>

---

## License

MIT. See [LICENSE](LICENSE).
