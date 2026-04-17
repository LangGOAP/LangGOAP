# langgoap

**Goal-Oriented Action Planning for [LangGraph](https://langchain-ai.github.io/langgraph/), with constraint optimization.**

`langgoap` turns a declarative goal and a set of actions into a compiled
LangGraph `StateGraph` that plans, executes, and replans. It combines
classical GOAP A\* search with OR-Tools CP-SAT constraint optimization
and LLM-driven natural-language goal interpretation, and ships as a
first-class citizen of the LangChain ecosystem.

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
    effects={
        "research_topic":  {"have_brief": True},
        "write_article":   {"have_draft": True},
        "publish_article": {"published":  True},
    },
    preconditions={
        "write_article":   {"have_brief": True},
        "publish_article": {"have_draft": True},
    },
)

result = agent.invoke({"world_state": {}, "goal": agent.goap_goal})
```

The planner produces a deterministic action sequence before a single
tool executes and re-plans on failure — no free-form ReAct loop.

## What's in the box

- **A\* planner** with customizable cost functions, effect validators,
  and per-action retry budgets.
- **Two-phase A\* → CSP pipeline**: A\* produces a candidate plan, then
  CP-SAT refines or replaces it when the goal has constraints or
  objectives.
- **Score hierarchy**: `SimpleScore`, `HardSoftScore`, `BendableScore`
  with hard/soft sign convention (`hard <= 0` for feasibility).
- **Fluent `ConstraintBuilder`** for hard/soft resource constraints and
  weighted objectives.
- **Temporal scheduling** with CP-SAT `IntervalVar` and Gantt rendering.
- **Natural-language goal interpreter** backed by any `BaseChatModel`.
- **Execution history** in `BaseStore` via reverse indexes — no
  embedder required.
- **`PlanningTracer` Protocol** with sync + async hooks.
- **Plan visualization**: Mermaid, DOT, ASCII, Gantt.
- **`MultiGoal`** for sequential multi-goal decomposition.
- **Three-layer low-code on-ramp**: `create_goap_agent`,
  `goapify_tool`, `GoapSubgraph`.

## Install

```bash
pip install langgoap
```

Requires Python 3.10+. OR-Tools CP-SAT is included as a core dependency.

## Documentation layout

- **[Concepts](concepts/index.md)** — GOAP planning, constraint
  optimization, and the LangGraph-native execution model.
- **[Examples](examples/index.md)** — 15 tutorial notebooks and
  two basics notebooks.
- **[API reference](api/index.md)** — every symbol in the public
  `langgoap.__all__`.

```{toctree}
:maxdepth: 2
:hidden:

concepts/index
examples/index
api/index
optaplanner_mapping
```

```{toctree}
:caption: Links
:maxdepth: 1
:hidden:

Changelog <https://github.com/integrallis/langgoap/releases>
PyPI <https://pypi.org/project/langgoap/>
GitHub <https://github.com/integrallis/langgoap>
```
