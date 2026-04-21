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

## Transition models

Classical GOAP treats an action's declared `effects` as both the
planning view *and* the runtime view — "what I expect" equals "what
happens." That works for deterministic domains but breaks as soon as
the world pushes back: API calls jitter, grid tiles are slippery,
tool outputs are non-deterministic, learned models return
distributions.

`TransitionModel` decouples the two views:

- `expected(state, action)` — the planner's deterministic view. A\*,
  CSP, and MCTS tree expansion all consume this. It must equal the
  action's declared effects unless a `DivergencePolicy` opts out.
- `sample(state, action, rng)` — one draw from the effect
  distribution. MCTS rollouts and the graph action-executor consume
  this. Free to diverge from `expected`; that divergence is the
  whole point of a non-deterministic model.

`DeterministicTransitionModel` is the zero-configuration default:
`expected` and `sample` both return the action's declared effects,
regardless of RNG state. Every call site that does not wire a
transition model sees this instance and behaves bit-identically to
the pre-`TransitionModel` code.

`DivergencePolicy` is the structured opt-out when a planner
legitimately wants `expected() != action.get_effects(state)` — for
example, a CVaR or robust-control planner whose point estimate is
pessimistically shaded relative to the nominal effect. The policy
carries a non-empty `reason`, a `kind` taxonomy slot
(`"risk-averse"`, `"learned"`, `"hierarchical"`, `"other"`), an
optional `max_relative_deviation` bound, and free-form `extra`
configuration. `assert_expected_matches_declared` enforces the bound
when present and the non-empty-reason invariant when not.

## Strategy routing

`StrategyRouter` dispatches to the right `PlanningStrategy` based on
the problem. It reads cheap `ProblemFeatures` at `plan()` time —
action count, goal-condition count, hard-constraint / soft-objective
presence, trajectory-metric presence, `is_stochastic`, and
`risk_profile` — and asks its `classifier` which registered
strategy to invoke. Because `StrategyRouter` itself satisfies
`PlanningStrategy`, callers that already accept a strategy (e.g.
`GoapPlanner(strategy=...)`) can opt in by passing a router with no
other changes.

The default `RuleBasedClassifier` is lexicographic and conservative:

1. Hard constraints, soft objectives, or trajectory metrics →
   `"csp-pipeline"` (CP-SAT refinement).
2. `risk_profile == "risk-averse"` → `"mcts"` (explicit user opt-in
   via `DivergencePolicy(kind="risk-averse")`).
3. `is_stochastic` **and** `prefer_mcts_for_stochastic=True` →
   `"mcts"` (flag-gated opt-in; see the outcome-3 gate in
   `research/experiments/2026-04-20-mcts-on-stochastic.md`).
4. High branching × deep horizon → `"mcts"`.
5. Otherwise → `"astar"`.

Routing is a pure function of the problem, so decisions are
reproducible and easy to test. Custom classifiers are ordinary
callables: any `Callable[[ProblemFeatures], str]` satisfies the
`StrategyClassifier` Protocol.

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
