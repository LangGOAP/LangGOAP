# Concepts

`langgoap` combines four building blocks: **Goal-Oriented Action
Planning** (GOAP) search, **constraint optimization** with OR-Tools
CP-SAT, **natural-language goal interpretation**, and **LangGraph-native
execution**. This page is a top-down tour of how those pieces fit
together.

## Goal-Oriented Action Planning

A GOAP problem is defined by three things:

1. A **world state** — a dictionary of boolean/string/numeric facts.
2. A set of **actions**, each with *preconditions* (what must be true
   for the action to run), *effects* (how the action changes the
   world state), and a *cost*.
3. A **goal** — a set of conditions the world must satisfy.

The planner searches the state space with A\* from the initial state,
expanding actions whose preconditions are satisfied, applying effects,
and stopping when the goal conditions hold. Cost functions drive the
heuristic; `CostFunction` is a `Protocol`, so users can supply dynamic,
state-dependent costs.

Actions are declared as `ActionSpec` frozen dataclasses. Every
`ActionSpec` field is immutable (`MappingProxyType` for preconditions
and effects) so the same action can be reused across concurrent
planning runs without mutation hazards.

## The plan *is* a compiled `StateGraph`

`langgoap` does not treat planning and execution as separate systems.
`GoapGraph.compile()` returns a real
`langgraph.graph.state.CompiledStateGraph` whose nodes are the
planner, executor, and observer. The plan lives inside the graph
state (`GoapState.plan`), and execution, replanning, and goal-check
all happen as node transitions inside the same graph.

Every node has both a sync (`__call__`) and async (`acall`) variant,
wrapped by `RunnableLambda(func=..., afunc=...)`, so
`compiled.invoke()` and `compiled.ainvoke()` both fire the tracer
hooks exactly once per transition.

## Constraint optimization

When a goal carries `constraints` or `objectives`, `GoapPlanner`
routes through a two-phase pipeline (`langgoap.pipeline_plan`):

1. A\* produces a candidate plan ignoring constraints — the
   *construction heuristic*.
2. CP-SAT validates or replaces that plan against resource budgets,
   temporal precedence, and weighted objectives — the *refinement
   phase*.

The CP-SAT layer supports:

- **Hard resource constraints** — `model.add(sum(...) <= budget)`.
  Violations mark the plan `INFEASIBLE`.
- **Soft resource constraints** — non-negative violation variables
  whose weighted sum feeds the objective.
- **Temporal scheduling** — one `IntervalVar` per action, precedence
  from the dependency graph, makespan minimization.
- **Multi-plan selection** — `BoolVar` per candidate, lexicographic
  objective with user-supplied weights.

OR-Tools CP-SAT is a core dependency, installed automatically with
`pip install langgoap`.

## Score hierarchy

Every finished plan carries a `Score`:

- `SimpleScore(value)` — scalar cost, used by A\*-only plans.
- `HardSoftScore(hard, soft)` — hard/soft sign convention.
  `hard <= 0` (feasible plan → `hard == 0`); `soft` has no sign
  restriction, so both penalties and rewards are expressible.
- `BendableScore(hard_levels, soft_levels)` — layered scores for
  lexicographic multi-criteria decisions.

Scores compare lexicographically (hard first, then soft) so that
`min(plans, key=lambda p: p.score)` returns the best feasible plan.

## Natural-language goals

`GoalInterpreter(llm, actions)` converts a plain-English request into
a `GoalSpec` by asking a `BaseChatModel` for structured output. The
interpreter is provider-agnostic: any `langchain_core.language_models
.BaseChatModel` with structured-output support works (`ChatOpenAI`,
`ChatAnthropic`, `ChatVertexAI`, etc.).

`GoapGraph.invoke_nl(request, llm=llm)` is the one-liner convenience
for single-shot NL execution.

## Execution history and tracing

`StoreExecutionHistory(store)` persists `ExecutionRecord`s to any
LangGraph `BaseStore` — `InMemoryStore`, `AsyncPostgresStore`, Redis,
or a custom implementation — using `get()`/`put()` reverse indexes.
No embedder is required.

`PlanningTracer` is a `runtime_checkable` Protocol with matching
`on_*` / `aon_*` hooks for every planner event. `NullTracer`,
`LoggingTracer`, and `MultiTracer` ship in-tree; custom tracers
(OpenTelemetry, LangSmith, Prometheus) are ordinary Python classes
that implement the protocol. Tracer exceptions never propagate into
the planner — observability is a hard invariant.

## Related reading

- {doc}`../api/index` — public API reference.
- {doc}`../examples/index` — runnable tutorial notebooks.
- The `optaplanner_mapping.md` document in the repository root
  explains how constraint-solver concepts map onto CP-SAT primitives.

```{toctree}
:maxdepth: 2
:hidden:
```
