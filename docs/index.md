# LangGOAP

**Goal-Oriented Action Planning for [LangGraph](https://langchain-ai.github.io/langgraph/), with constraint optimization.**

LangGOAP turns a goal and a set of LangChain tools into a compiled
`StateGraph` that plans before it acts, replans on failure, and stays
deterministic by default. The planner is classical [A\*](https://en.wikipedia.org/wiki/A*_search_algorithm) with optional
OR-Tools CP-SAT refinement; the runtime is plain LangGraph, so
checkpointing, streaming, `interrupt()`, and LangSmith all just work.

## Quickstart

The snippet below wraps four LangChain tools and asks LangGOAP to
publish an article. The LLM parses the natural-language goal exactly
once into a symbolic target like `{"published": True}`, and from there
A\* takes over. Two writers compete to satisfy the `have_draft`
precondition — `write_article_fast` at `cost=1.0` and
`write_article_premium` at `cost=5.0` — and A\* picks the cheaper one.
If the cheap writer fails at execution time, the executor blacklists it
and the planner re-derives a fresh plan through the premium writer
without any routing code. `result_keys` plumbs each tool's return value
into `world_state` under a chosen key, so `research_topic`'s output
lands at `world_state["brief"]` where the next tool's `brief` argument
can pick it up.

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgoap import create_goap_agent

# A small counter so the demo can simulate the cheap writer flaking out
# on its first N invocations.  Flip `fail_fast_n_times` to 1 to see the
# planner replan through the premium writer.
def make_writers(fail_fast_n_times: int = 0):
    state = {"fast_calls": 0}

    @tool
    def write_article_fast(brief: str) -> str:
        """Quickly draft an article from a brief. Cheaper, occasionally flaky."""
        state["fast_calls"] += 1
        if state["fast_calls"] <= fail_fast_n_times:
            raise RuntimeError(f"upstream rate limit (attempt {state['fast_calls']})")
        return f"Fast draft: {brief}"

    @tool
    def write_article_premium(brief: str) -> str:
        """Premium-quality article. Higher cost, always reliable."""
        return f"Premium draft: {brief}"

    return write_article_fast, write_article_premium

@tool
def research_topic(topic: str) -> str:
    """Produce a short research brief for a topic."""
    return f"Brief on {topic}"

@tool
def publish_article(draft: str) -> str:
    """Publish an article draft."""
    return f"Published: {draft}"

write_article_fast, write_article_premium = make_writers(fail_fast_n_times=0)

agent = create_goap_agent(
    tools=[research_topic, write_article_fast, write_article_premium, publish_article],
    goal="Publish an article about GOAP for LangGraph",
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    # Preconditions/effects keep the planner honest — never LLM-inferred.
    preconditions={
        "write_article_fast":    {"have_brief": True},
        "write_article_premium": {"have_brief": True},
        "publish_article":       {"have_draft": True},
    },
    effects={
        "research_topic":        {"have_brief": True},
        "write_article_fast":    {"have_draft": True},
        "write_article_premium": {"have_draft": True},
        "publish_article":       {"published":  True},
    },
    # A* minimizes total cost. Omitted tools default to cost=1.0.
    costs={
        "research_topic":        1.0,
        "write_article_fast":    1.0,
        "write_article_premium": 5.0,
        "publish_article":       1.0,
    },
    # Wire each tool's return value into the next tool's input.
    result_keys={
        "research_topic":        "brief",
        "write_article_fast":    "draft",
        "write_article_premium": "draft",
    },
)

result = agent.invoke(
    {
        "world_state": {"topic": "GOAP for LangGraph"},
        "goal": agent.goap_goal,
    }
)
```

`agent` is a compiled LangGraph graph. Use it with streaming,
checkpointers, `interrupt()`, or any LangGraph feature. The LangGraph
cycle that `create_goap_agent` compiles to is small and fixed — every
plan flows through the same `planner → executor → observer` loop:

```{image} _static/images/quickstart-stategraph.png
:alt: Compiled StateGraph: planner → executor → observer
:width: 120px
:align: center
```

### Scenario 1 — Happy path (cheap writer wins on cost)

A\* sees two paths to `have_draft: True` and picks the cheaper one:

```{image} _static/images/quickstart-plan-happy.png
:alt: "Plan: research_topic → write_article_fast → publish_article"
:width: 220px
:align: center
```

```text
status:              'goal_achieved'
replan_count:        0
blacklisted_actions: []
plan.action_names:   ['research_topic', 'write_article_fast', 'publish_article']
plan.total_cost:     3.0
execution_history:
   1. [ok ] research_topic
   2. [ok ] write_article_fast
   3. [ok ] publish_article
world_state (relevant keys): {'topic': 'GOAP for LangGraph', 'brief': 'Brief on GOAP for LangGraph', 'draft': 'Fast draft: Brief on GOAP for LangGraph'}
```

### Scenario 2 — Cheap writer flakes, planner replans through premium

Set `fail_fast_n_times=1` and run again. `write_article_fast` raises on
its first call, the executor blacklists it, and the observer hands
control back to the planner. A\* re-derives a new plan from the current
world state (`have_brief` is already `True` because `research_topic`
succeeded), so the remaining work is just the premium writer plus
publish:

```{image} _static/images/quickstart-plan-replan.png
:alt: "Replan: write_article_premium → publish_article"
:width: 320px
:align: center
```

```text
status:              'goal_achieved'
replan_count:        1
blacklisted_actions: ['write_article_fast']
plan.action_names:   ['write_article_premium', 'publish_article']
plan.total_cost:     6.0
execution_history:
   1. [ok ] research_topic
   2. [FAIL] write_article_fast  (upstream rate limit (attempt 1))
   3. [ok ] write_article_premium
   4. [ok ] publish_article
world_state (relevant keys): {'topic': 'GOAP for LangGraph', 'brief': 'Brief on GOAP for LangGraph', 'draft': 'Premium draft: Brief on GOAP for LangGraph'}
```

## What's in the box

- **Cost-aware planner** — given your tools and a goal, finds the
  cheapest sequence of tool calls that reaches it. You supply
  per-action costs, optional effect validators that double-check
  what each tool actually changed, and per-action retry budgets.
- **Constraint-aware refinement** — when your goal carries hard
  resource caps (budgets, time windows) or objectives to minimize or
  maximize, the candidate plan is handed to a constraint solver
  ([OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver))
  that refines or replaces it to satisfy them.
- **Plan scoring** — rank candidate plans by a single number
  (`SimpleScore`), by feasibility-first hard/soft tradeoffs
  (`HardSoftScore` — hard violations make a plan infeasible, soft
  scores rank the survivors), or by weighted priority levels
  (`BendableScore`).
- **Declarative constraints** — a fluent `ConstraintBuilder` API for
  declaring resource limits and optimization objectives without
  writing solver code.
- **Temporal scheduling** — when actions have durations and deadlines,
  LangGOAP solves a scheduling problem alongside the plan and can
  render the result as a Gantt chart.
- **Natural-language goals** — describe the goal in plain English and
  any `BaseChatModel` (OpenAI, Anthropic, local, …) translates it
  into a symbolic goal exactly once, before any tool runs.
- **Replanning memory** — each step's outcome is recorded in a
  LangGraph `BaseStore` keyed by tool name, so on replan the planner
  can avoid actions that already failed in this run. No vector
  embeddings or extra infrastructure required.
- **Pluggable tracing** — a `PlanningTracer` Protocol with sync and
  async hooks streams planning events to LangSmith, OpenTelemetry,
  or any logger you wire up.
- **Plan visualization** — render any plan as a Mermaid diagram,
  GraphViz DOT graph, ASCII tree, or Gantt chart.
- **Multi-goal decomposition** — a `MultiGoal` wrapper sequences a
  list of subgoals into a single executable plan that satisfies
  them in order.
- **Progressive API** — three entry points for different needs:
  `create_goap_agent` (one-liner from a list of LangChain tools),
  `goapify_tool` (decorate an existing tool with planning metadata),
  and `GoapSubgraph` (embed planning inside a larger LangGraph).

## Install

```bash
pip install langgoap
```

Requires Python 3.10+.

## Documentation layout

- **[Concepts](concepts/index.md)** — GOAP planning, constraint
  optimization, and the LangGraph-native execution model.
- **[Examples](examples/index.md)** — 27 tutorial notebooks and
  12 basics notebooks.
- **[API reference](api/index.md)** — every symbol in the public
  `langgoap.__all__`.

```{toctree}
:maxdepth: 2
:hidden:

concepts/index
examples/index
api/index
```

```{toctree}
:caption: Links
:maxdepth: 1
:hidden:

Changelog <https://github.com/LangGOAP/LangGOAP/releases>
PyPI <https://pypi.org/project/langgoap/>
GitHub <https://github.com/LangGOAP/LangGOAP>
```
