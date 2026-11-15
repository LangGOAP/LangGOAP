# Concepts

LangGOAP combines four building blocks: **Goal-Oriented Action
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
and stopping when the goal conditions hold. The heuristic is admissible
and consistent — `ceil(unsatisfied goal conditions / most goal
conditions any single action establishes) × cheapest static action
cost` — so the first plan A\* returns is a cheapest plan. `CostFunction`
is a `Protocol`, so users can supply dynamic, state-dependent costs;
when any cost is dynamic the heuristic falls back to zero (plain
Dijkstra) rather than risk overestimating.

With a `time_budget_ms`, planning is *anytime*: weighted A\* runs with a
tightening weight schedule, so a complete (bounded-suboptimal) plan is
available early and is improved while the budget lasts; on expiry the
best complete plan found so far is returned. `AnytimePlanningStrategy`
wraps this for strategies that do not take a budget themselves.

Actions are declared as `ActionSpec` frozen dataclasses. Every
`ActionSpec` field is immutable (`MappingProxyType` for preconditions
and effects) so the same action can be reused across concurrent
planning runs without mutation hazards. Precondition, effect and goal
values must be hashable (scalars, tuples, frozensets); a list or dict
is rejected at construction with the offending key named.

The planner only hashes and compares *planning keys* (those that appear
in some precondition, effect or goal condition), but the rest of the
world state is not lost: `PlanningState` carries it as a `context`, and
cost functions, dynamic effects and `TransitionModel.expected` /
`sample` receive `state.as_context()`, the full world state with the
planning keys layered on top. Rich data (documents, tool output) can
therefore drive a dynamic cost or effect without becoming part of the
search space.

## The plan *is* a compiled `StateGraph`

LangGOAP does not treat planning and execution as separate systems.
`GoapGraph.compile()` returns a real
`langgraph.graph.state.CompiledStateGraph` whose nodes are the
planner, executor, and observer. The plan lives inside the graph
state (`GoapState.plan`), and execution, replanning, and goal-check
all happen as node transitions inside the same graph.

Every node has both a sync (`__call__`) and async (`acall`) variant,
wrapped by `RunnableLambda(func=..., afunc=...)`, so
`compiled.invoke()` and `compiled.ainvoke()` both fire the tracer
hooks exactly once per transition.

### The state is plain JSON

LangGraph hands every channel value and every node write to the
checkpointer's serializer, so the graph is built from
`GoapStateChannels`, a schema whose values are dicts, lists and scalars
only. That is what lets *any* `BaseCheckpointSaver` — in-memory,
Postgres, Redis, SQLite or your own — persist a GOAP run with its stock
serializer; LangGOAP ships no serializer and imports no backend. Typed
objects live at the edges: `GoapGraph.invoke()` converts the goal on
the way in and returns a `GoapState` whose `goal`, `plan` and
`execution_history` are `GoalSpec` / `Plan` / `ActionResult` objects;
the nodes rehydrate on entry and serialise on exit, so strategies,
tracers, guards and stuck handlers never see the JSON form. Code that
drives the compiled graph directly sees the JSON form and can call
`GoapGraph.rehydrate(result)` (or `langgoap.graph.state.rehydrate_state`)
to get objects back. Two consequences follow:

- With a checkpointer attached, build the input with
  `GoapGraph.initial_state(goal, world_state)` or pass `goal.to_dict()`.
  LangGraph persists the raw input before the state reducers can convert
  a `GoalSpec` object, and the stock serializer refuses it.
- Callables are never read from a checkpoint. Plan actions are stored by
  their declarative fields and rebound by name to the live action list;
  a goal's callable `value` and its plan-quality `metrics` are kept
  reachable for the lifetime of the process, and a checkpoint restored
  in another process falls back to `value=1.0` and drops the metrics
  with a warning.

### Replanning, retries and blacklisting

The observer compares the world state after each action with the
planner's expected state and routes back to the planner on a failure or
a deviation. A failed action is retried up to `ActionSpec.max_retries`
times (or per its `ActionQos` policy) and then blacklisted for the rest
of the run; the planner excludes blacklisted actions and finds another
path if one exists. When every remaining path needs a blacklisted
action the run ends with `status="no_plan"` and a `no_plan_explanation`
that names the blacklisted actions and the knobs that would unblock the
goal. `GoalPolicy(blacklist_fallback=True)` opts into the alternative:
re-admit the failed actions and plan through them again, bounded by
`max_replans`.

`RepairStrategy` reuses the already-executed prefix of the previous plan
and re-plans the suffix; blacklisted actions and spent `can_rerun=False`
actions never survive in the prefix, and when the repair cannot reach the
goal the strategy delegates to its inner strategy instead of returning an
unrepaired plan.

`ParallelGoapExecutor` runs independent plan actions as a wave. Each
action's world-state delta is merged independently, a failure aborts the
wave after the successful siblings' deltas are kept, and an action that
requires human approval always runs in a wave of its own so the
`interrupt()` is never raised mid-wave.

## Constraint optimization

When a goal carries `constraints`, `objectives`, `soft_goals` or
`metrics` (`pipeline.needs_csp`), `GoapPlanner` routes through a
two-phase pipeline (`langgoap.pipeline_plan`):

1. A\* produces a candidate plan ignoring constraints — the
   *construction heuristic*.
2. CP-SAT validates or replaces that plan against resource budgets,
   temporal precedence, and weighted objectives — the *refinement
   phase*.

A goal that states only hard limits is satisfied by the first feasible
A\* plan. A goal that states *preferences* — objectives, soft
constraints or soft goals (`pipeline.needs_selection`) — compares that
plan against alternative plans and lets CP-SAT pick the best one, with
total cost as a strict tie-breaker. Plan-quality `metrics` are arbitrary
Python callables that CP-SAT cannot express; they re-score the chosen
plan but do not drive selection. When the only plan is `INFEASIBLE` the
run ends with `no_plan` and an explanation listing the plan and its
per-resource shortfalls; `GoalPolicy(on_infeasible="execute")` runs it
anyway and reports the status in `plan.metadata.csp`.

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

### Runtime consumption

`GoapGraph` accepts `transition_model=` and `rng=` and threads them
through to both `GoapExecutor` and `ParallelGoapExecutor`. When an
action's `execute` callable returns a dict, that dict is
authoritative — real runtime data (LLM outputs, tool responses,
sensor readings) always wins. Only when `execute` is absent or
returns `None` does the executor fall through to
`transition_model.sample(state, action, rng)`; without a model it
falls through to `action.get_effects(state)` as before. This keeps
the declared-effects path bit-identical to the pre-`TransitionModel`
runtime while letting stochastic domains (slip, jitter, ghost noise,
retry churn) drive the executor from the same distribution the MCTS
rollouts sampled from.

## Strategy routing

`StrategyRouter` dispatches to the right `PlanningStrategy` based on
the problem. It reads cheap `ProblemFeatures` at `plan()` time —
action count, goal-condition count, relaxed-plan horizon,
hard-constraint / soft-objective presence, trajectory-metric presence,
`is_stochastic`, and `risk_profile` — and asks its `classifier` which
registered strategy to invoke. Because `StrategyRouter` itself satisfies
`PlanningStrategy`, callers that already accept a strategy (e.g.
`GoapPlanner(strategy=...)`) can opt in by passing a router with no
other changes.

The default `RuleBasedClassifier` is lexicographic and conservative:

1. Hard constraints, soft objectives, or trajectory metrics →
   `"csp-pipeline"` (CP-SAT refinement).
2. `risk_profile == "risk-averse"` → `"mcts"` (explicit user opt-in
   via `DivergencePolicy(kind="risk-averse")`).
3. `is_stochastic` **and** `prefer_mcts_for_stochastic=True` →
   `"mcts"` (flag-gated opt-in).
4. High branching (30+ actions) **and** a deep horizon → `"mcts"`.
   The horizon is the depth of the relaxed planning graph from the
   start state to the goal (delete effects ignored), so catalog size
   alone never promotes a short chain to MCTS.
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

`LangSmithTracer` ships the same hooks to LangSmith. A tracer that
implements only some hooks is fine: the nodes wrap every tracer in
`SafeTracerProxy`, which turns missing hooks into no-ops. Pass
`record_expansions=True` to receive `on_search_expand` for every A*
node; explicit `AStarStrategy()` / `TwoPhasePipelineStrategy()`
instances inherit the planner's tracer, so tracing does not disappear
when you name the default strategy. `CostAccumulator` is a LangChain
callback that totals tokens and USD per run from a bundled price
snapshot, warns once per unknown model, and raises instead when
`require_known_pricing=True`. `ReflexionTracer` turns action failures
into `Reflection`s and persists them in the history's store so later
runs can read them back.

## Pluggable strategies

`GoapGraph(strategy=...)` accepts any object implementing the
`PlanningStrategy` protocol: `plan(start, goal, actions, *,
blacklisted_actions, prior_plan, current_step)`. The built-in strategies
cover the common shapes:

- `AStarStrategy` is plain A*; `TwoPhasePipelineStrategy` runs A* and then
  CSP validation and selection, which is what the planner does by default
  when the goal carries constraints or objectives.
- `MCTSStrategy` searches stochastic domains with Monte-Carlo tree search
  over a `TransitionModel`. `MCTSExploration`, `MCTSReuseConfig` and
  `MCTSTracingConfig` tune exploration, tree reuse between episodes and
  per-node tracing; the rollout policies (`RandomRollout`,
  `HeuristicRollout`, `StochasticRollout`) and `ScalarHeuristic` shape the
  value estimate.
- `AnytimePlanningStrategy(time_budget_ms)` runs weighted A* with a
  tightening weight schedule and returns the best complete plan found
  within the budget.
- `LazyDecompositionStrategy(lookahead=...)` plans only the next few steps
  and re-plans as it goes; the truncated plan is re-validated against the
  goal's constraints.
- `RepairStrategy` patches the in-flight plan after a failure (see
  *Replanning, retries and blacklisting*).
- `UtilityStrategy` ranks applicable actions by expected value instead of
  planning toward a fixed goal. Pair it with `NirvanaGoal`, a goal that is
  never satisfied, for always-on agents that stop only when no action
  applies or a termination policy fires.
- `refine_with_csp(plan, goal)` validates an existing plan against a goal's
  constraints and attaches CSP scoring without searching.

`StrategyRouter` picks among these automatically (see *Strategy routing*).

## Multi-goal decomposition

`MultiGoal` composes several `GoalSpec`s. In `sequential` mode the planner
works through them in order and tracks `current_subgoal_index`; a sub-goal
the world already satisfies is skipped, and one that cannot be planned
ends the run with `status="no_plan"` unless `on_infeasible="skip"`, which
records it in `skipped_subgoals` and finishes with `status="partial"`. In
`any` mode every sub-goal is planned and the cheapest feasible plan runs;
`best_value` picks the sub-goal whose `GoalSpec.value` net of plan cost is
highest. When no sub-goal can be planned, `no_plan_explanation` carries
each sub-goal's own explanation.

## Reliability and recovery

### Retries and `ActionQos`

`ActionSpec.max_retries` is the observer's budget: a failed action is
re-planned up to that many times and then blacklisted. `ActionQos` adds
in-executor retries without a replan: `max_attempts`, exponential back-off
(`backoff_initial_ms`, `backoff_multiplier`, `backoff_max_ms`, `jitter`)
and the exceptions that count as transient (`retryable_exceptions`, or
every exception when `idempotent=True`). `DEFAULT_RETRY_POLICY` and
`FIRE_ONCE` are ready-made policies. Back-off jitter never draws from the
`rng` you pass for stochastic sampling, so retries keep seeded runs
reproducible.

### Human-in-the-loop and typed forms

`ActionSpec(require_human_approval=True)` makes the executor call
LangGraph's `interrupt()` before the action runs. Resume with
`Command(resume=True)` (or `{"approved": True}`) to approve and `False` to
deny; a denial is a failure that counts toward the action's retry budget.
Passing a Pydantic model class instead of `True` turns the pause into a
typed form: the resume payload is validated against the model and stored
under `human_input_key` in the world state; a payload that fails validation
is asked again, with `validation_errors` and the rejected payload in the
new interrupt, and `False` (or `{"approved": False}`) declines the form
like a denied boolean gate. The answer survives a failed
attempt: a retry reuses an answer that still validates instead of asking
again, and a valid answer already present in the world state skips the
prompt. Approval actions always run alone in a parallel wave.

### Stuck handlers

When planning fails, the planner node consults the `stuck_handlers` given
to `GoapGraph`, wrapped in a `MulticastStuckHandler`. A handler receives
the state and the `no_plan_explanation` and returns a
`StuckHandlerResult`: `REPLAN` with `state_updates` and/or a `new_goal`
makes the planner try again, up to `max_stuck_iterations` times;
`NO_RESOLUTION` lets the run end with `status="no_plan"`.
`FunctionalStuckHandler` wraps a plain function, and `AsyncStuckHandler`
implementations are awaited on the async path.

### Early termination

`termination_policies` stop a run before the goal is reached:
`MaxActionsPolicy`, `MaxCostPolicy`, `MaxWallClockPolicy`,
`MaxTokensPolicy`, `MaxLLMCallsPolicy` and `OnStuckPolicy`, combined with
`FirstOfPolicy` or `AllOfPolicy`. The observer checks goal satisfaction
first, so a run that reaches the goal on the step that exhausts a cap
reports `goal_achieved`; otherwise it ends with `status="terminated"` and
an `EarlyTermination` record naming the policy that fired. The wall-clock
anchor is an epoch timestamp stored in the state, so budgets survive a
resume in another process.

## Sensors, guards and condition resolvers

Three hooks let a graph check the world instead of trusting its state.
`sensors` (`Sensor`, `AsyncSensor`, `FunctionalSensor`) run in the planner
node before each planning pass and merge fresh observations into
`world_state`. `guards` (`ActionGuard`, `FunctionalGuard`) run before an
action executes and return `GuardResult`s: a `GuardSeverity.BLOCK` result
fails the action and counts toward its retry budget, a `WARN` is logged
and the action proceeds. `resolvers` (`ConditionResolver`,
`FunctionalConditionResolver`, `PromptConditionResolver`) answer the
preconditions no action can establish with a three-valued
`ConditionStatus`; resolved values are written back to the state so a
resumed run does not ask again.

## Checkpointing

Any LangGraph checkpointer works with its stock serializer because the
state is JSON (see *The state is plain JSON*). Compile with
`checkpointer=MemorySaver()`, or the Postgres and Redis savers from the
`checkpoint-postgres` and `checkpoint-redis` extras, build the input with
`GoapGraph.initial_state(goal, world_state)`, and invoke with a
`thread_id`. A run interrupted mid-plan, by a crash or by a
human-in-the-loop pause, resumes from the last checkpoint with its plan,
history and failure counts intact; `rehydrate_state` turns a checkpointed
state back into typed objects.

Every `ActionResult` in `execution_history` stores the world state before
and after its action, so with large non-planning values a checkpoint grows
with every step. `GoapGraph(history_state="delta")` keeps only the keys an
action changed (with their prior values) and `"planning"` keeps only the
keys preconditions, effects and the goal read; `"full"` is the default. The
realised cost of a run does not depend on these snapshots, because each
result records its cost at execution time.

## Plan explanations

A failed planning pass explains itself. `explain_no_plan` performs a
static reachability analysis and reports the goal values no action
produces and the precondition values nothing establishes;
`explain_infeasibility` reports every violated constraint of a
CSP-infeasible plan with its `ResourceShortfall`s. The planner stores the
result under `no_plan_explanation`, stuck handlers receive it as their
`reason`, and `langgoap explain` prints it from the command line.

## Visualization

`render_mermaid`, `render_dot` and `render_ascii` draw a plan as a graph;
`render_mermaid_gantt` and `render_ascii_gantt` draw the CSP schedule of
durative actions; `visualize(plan, format="auto")` picks a renderer for
the current front end. Step costs are evaluated against each step's
pre-state, and a cost function that cannot be evaluated is shown without
a number rather than failing the render. In Jupyter a `Plan` displays as
Markdown offline; set `LANGGOAP_DISPLAY_PNG=1` to fetch a PNG from
mermaid.ink instead.

## Command line and deployment

The `langgoap` CLI loads actions from a Python module. `langgoap plan`
plans only (add `--run` to execute), `langgoap actions` lists the catalog,
`langgoap explain` diagnoses an unreachable goal, `langgoap visualize`
renders a plan, and `langgoap deploy-init` writes a LangGraph deployment.
`scaffold_deployment` is the function behind the last one: it emits
`langgraph.json`, a `graph.py` entrypoint, `.env.example` and a README so
`langgraph dev` serves the graph with an `/mcp` endpoint and `langgraph
deploy` ships it to LangGraph Platform.

## Deep Agents and tool interop

LangGOAP graphs compose with the LangChain ecosystem in both directions.
`goapify_tool` turns a LangChain tool into an `ActionSpec`, with an
optional `EffectValidator` to check the tool's effects, and
`create_goap_agent` builds a graph from a list of tools and a goal.
`create_goap_tool` wraps a compiled graph as a `StructuredTool`,
`create_goap_subagent` exposes it as a Deep Agents sub-agent, and
`format_goap_result` turns a run's result into a compact report for the
calling agent. `GoapSubgraph` and `add_goap_subgraph` embed a graph as a
node of a larger `StateGraph`.

## Related reading

- {doc}`../api/index` — public API reference.
- {doc}`../examples/index` — runnable tutorial notebooks.

```{toctree}
:maxdepth: 2
:hidden:
```
