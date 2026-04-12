# Changelog

All notable changes to LangGoap are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-11

Initial public release. LangGoap is a Goal-Oriented Action Planning
framework for LangGraph with constraint optimization, natural-language
goal interpretation, and a LangChain-first execution model in which the
plan **is** a compiled `StateGraph`.

### Planning, optimization, and interpretation core

- **Constraint satisfaction and optimization via OR-Tools CP-SAT**
  (`langgoap.planner.csp`). Hard/soft resource constraints, weighted
  multi-objective optimization, temporal scheduling with `IntervalVar`,
  multi-plan selection with `BoolVar`, and a pure-Python fast path when
  ortools is not installed. Install the optional extra with
  `pip install langgoap[optimization]`.
- **Natural-language goal interpretation** (`langgoap.interpreter`).
  Provider-agnostic `GoalInterpreter` backed by `BaseChatModel`
  structured output. `GoapGraph.invoke_nl()` / `ainvoke_nl()` accept a
  plain string request and produce a `GoalSpec` on the fly.
- **LangGraph-native execution.** `GoapGraph.compile()` returns a real
  `CompiledStateGraph`; planning and execution are one system. The
  planner, executor, and observer are `RunnableLambda`-wrapped so sync
  and async call paths both fire tracer hooks without duplication.
- **Immutable `PlanningState`** with frozenset-backed conditions,
  `MappingProxyType`-backed `ActionSpec` fields, and `dataclasses
  .replace` for scoring updates — safe to share across concurrent
  planner invocations.
- **Dynamic `CostFunction` protocol**, `ReplanStrategy` enum, per-action
  `effect_validator`, `max_retries`, `duration`, and `resources` fields.
- **Error recovery**: the observer tracks per-action failures,
  blacklists actions that exceed a configurable threshold, and the
  planner filters out blacklisted actions with a fallback re-plan.
- **Graceful ortools fallback** via `_require_ortools()` with a clear
  `ImportError` message. CSP features degrade cleanly when the extra is
  missing.

### Low-code path for existing LangGraph applications

Three layered on-ramps from "figure it out for me" to "full control",
implemented in `langgoap.integrations`:

- **Layer A — `create_goap_agent(tools, goal, ...)`** — the
  `create_react_agent` equivalent. Accepts a list of LangChain
  `BaseTool`s, a natural-language string goal (interpreted via
  `GoalInterpreter`) or a `GoalSpec`, optional per-tool preconditions /
  effects / resources, and returns a compiled `StateGraph`.
  Preconditions and effects are **never** LLM-inferred — the user
  either declares them explicitly or accepts empty-pre/empty-effect
  actions (with a `WARNING` log) for trivial prototyping.
- **Layer B — `goapify_tool(tool, preconditions=, effects=, ...)`** —
  explicit `BaseTool → ActionSpec` adapter, fully deterministic. Also
  usable as a `@goapify_tool(...)` decorator over `@tool`-decorated
  functions.
- **Layer C — `GoapSubgraph` and `add_goap_subgraph(parent, ...)`** —
  drop a GOAP planning loop into an existing `StateGraph` as a sealed
  subgraph. Internal GOAP keys (`plan`, `current_step`,
  `blacklisted_actions`, `execution_history`) stay private; the parent
  graph sees only the declared `input_key` and `output_key`.

### OptaPlanner-style scoring unified with GOAP and LangGraph

- **Score hierarchy** (`langgoap.score`): `Score` ← `SimpleScore` /
  `HardSoftScore` / `BendableScore`. Every subclass exposes a
  `.value` property for scalar logging and an `is_feasible()` method
  for hard-constraint checks. Lexicographic comparison (hard first,
  then soft); cross-subclass comparisons raise `TypeError`.
  - **Sign convention**: `hard <= 0` — a feasible plan has
    `hard == 0`; violations subtract from `hard`. `soft` has no sign
    restriction so that `MAXIMIZE` objectives can push it positive.
- **Hard/soft constraint bifurcation**. `ConstraintSpec.level:
  Literal["hard", "soft"]` (default `"hard"`, backward compatible).
  `validate_plan()` marks only hard-constraint violations `INFEASIBLE`;
  `optimize_plans()` adds hard constraints to the CP-SAT model directly
  and models soft bounds via non-negative violation variables whose
  contribution to the objective is `violation_amount * weight`.
- **Fluent `ConstraintBuilder`** (`langgoap.constraints`) with
  `for_each_action` / `for_plan`, `.where(...)`, `.sum_resource(...)`,
  `.bounded(min=..., max=...)`, `.penalize(level=..., weight=...)`,
  `.minimize()` / `.maximize()`, `.as_constraint(name)` /
  `.as_objective(name)`, and a `BuilderOutput` feed into
  `GoalSpec.from_builder(...)`.
- **Planning strategy hierarchy** (`langgoap.planner.strategy`):
  `PlanningStrategy` Protocol with concrete `AStarStrategy`,
  `CSPRefinementStrategy`, and `TwoPhasePipelineStrategy`. User-defined
  strategies conform to the Protocol and are accepted by
  `GoapPlanner(strategy=...)`. OptaPlanner's `Move`/`Phase`/
  `ScoreDirector` concepts map to CP-SAT internals — see
  `docs/optaplanner_mapping.md` for the full table.
- **Two-phase pipeline** (`langgoap.planner.pipeline`): A* produces a
  candidate plan, then CSP (if the goal has constraints or
  objectives) refines or replaces it with a better alternative.
  `pipeline_plan` is exported as a first-class public function.

### Tutorial catalog (15 notebooks)

Every notebook has a corresponding passing integration test under
`tests/integration/`. Tutorial helpers and domain data live in the
shared `examples/tutorials/tutorial_examples/` package.

**Tier 1 — Primers**

1. `examples/tutorials/directory_handler.ipynb` — GOAP primer from
   `GOApy`: file-system predicates, `CreateDir` and `CreateToken`
   actions.
2. `examples/tutorials/robot_navigation.ipynb` — A* primer from
   `unified-planning/01`: linear graph, heuristic, state-space search.
3. `examples/tutorials/hungry_agent.ipynb` — `GoalInterpreter` +
   cost-driven action selection on the classic "I'm tired and hungry"
   example.

**Tier 2 — OptaPlanner GOAPifications + workflow agents**

4. `cloud_balancing.ipynb` — VM bin-packing with resource constraints,
   `create_goap_agent` one-liner.
5. `vehicle_routing.ipynb` — capacity-constrained routing with CSP
   temporal scheduling and Gantt visualization.
6. `nurse_rostering.ipynb` — shift assignments, skill matching,
   hard/soft decomposition via `HardSoftScore`.
7. `project_job_scheduling.ipynb` — RCPSP with precedence constraints,
   makespan minimization, critical-path visualization.
8. `task_assigning.ipynb` — ticket routing with the fluent
   `ConstraintBuilder`, weighted delay minimization.
9. `sql_query_agent.ipynb` — schema-explore → generate → test → refine
   loop with `effect_validator` and replanning on test failure.
10. `vulnerability_scanner.ipynb` — phased discovery with sensor
    integration and action blacklisting.

**Tier 3 — Full-feature showcases**

11. `deep_research_agent.ipynb` — LLM-heavy research loop with
    `create_goap_agent`, `StoreExecutionHistory`, and tracing.
12. `hierarchical_product_launch.ipynb` — `MultiGoal` sequential
    decomposition end-to-end.
13. `content_builder_agent.ipynb` — multi-objective CSP with
    conditional format generation and `ConstraintBuilder`.
14. `temporal_match_cellar.ipynb` — durative actions with overlap
    constraints and parallel scheduling.
15. `flexible_job_shop.ipynb` — every v0.1.0 feature in one notebook:
    NL intake, A* → CSP, hard/soft scoring, temporal scheduling,
    visualization, and tracing.

### OptaPlanner example GOAPifications

Notebooks 4–8 in the Tier 2 list are direct GOAPifications of
OptaPlanner example problems (cloud balancing, vehicle routing, nurse
rostering, project job scheduling, task assigning). Every instance is
derived from a small fixture committed under
`examples/tutorials/tutorial_examples/data/` with provenance headers.
`docs/optaplanner_mapping.md` explains how each OptaPlanner concept
maps to a LangGoap class.

### Observability, history, and visualization

- **Execution history in LangGraph `BaseStore`** (`langgoap.history`):
  `StoreExecutionHistory` with sync and async `record` / `query_by_goal`
  / `query_failures` methods. Reverse-index design using `get()` / `put
  ()` only — works with every `BaseStore` implementation
  (`InMemoryStore`, `AsyncPostgresStore`, custom stores) without
  requiring an embedder. `compute_goal_hash` is exported as a
  first-class public helper.
- **Tracing protocol** (`langgoap.tracing`): `PlanningTracer` Protocol
  with full sync + async parity (`on_*` and `aon_*` hooks for plan
  start/complete/failed, action start/complete, replan, and
  goal-achieved). `NullTracer`, `LoggingTracer`, `MultiTracer`, and
  `LangSmithTracer` ship in-tree. `LangSmithTracer` maps domain events
  (plan/replan/goal-achieved) to LangSmith runs alongside LangGraph's
  automatic node-level tracing — `langsmith` is already a transitive
  dependency via `langchain-core`, so no extra install is needed.
  OpenTelemetry adapters are documented but not bundled; custom tracers
  are ordinary Python classes that implement the protocol. Tracer
  exceptions never propagate into the planner.
- **Plan visualization** (`langgoap.viz`): `render_mermaid`,
  `render_mermaid_gantt`, `render_dot`, `render_ascii`,
  `render_ascii_gantt`, and `visualize` with auto-format detection.
  Pure-Python — Mermaid and DOT require no extra dependencies; DOT
  rendering falls back to returning the raw DOT string when the
  `graphviz` binary is missing.

### Human-in-the-loop

- **`ActionSpec.require_human_approval`** (`bool`, default `False`). When
  `True`, `GoapExecutor` calls LangGraph's `interrupt()` before executing
  the action, surfacing the pending action name, preconditions, effects,
  and current world state to the caller. Resume with
  `Command(resume={"approved": True})` to continue or
  `Command(resume={"approved": False, "reason": "..."})` to deny — a
  denied action is immediately blacklisted (bypasses `max_retries`) and
  the planner replans around it. Requires a checkpointer
  (`MemorySaver`, `AsyncPostgresSaver`, `RedisSaver`, etc.) to persist
  the interrupt across the pause/resume boundary.

### Checkpointer support

- **Custom ormsgpack serializers** for LangGoap's frozen dataclasses.
  `MappingProxyType`, `frozenset`, `timedelta`, and `tuple` are all
  round-tripped correctly through LangGraph's checkpoint wire format.
  Callable fields (`execute`, `aexecute`, `effect_validator`) are
  serialized as `None` — they are re-bound from the compile-time action
  specs, not from checkpoints.
- **Tested with all three LangGraph checkpoint backends**: `MemorySaver`
  (in-memory), `AsyncPostgresSaver` (via TestContainers), and
  `RedisSaver`/`AsyncRedisSaver` (via TestContainers with `redis:8`).
  Install optional extras with `pip install langgoap[checkpoint-postgres]`
  or `pip install langgoap[checkpoint-redis]`.

### Performance

- **`slots=True`** on all 18 frozen dataclasses across the core package.
  Removes the per-instance `__dict__` overhead (typically 48–112 bytes on
  CPython) for every hot-path object: `PlanningState`, `ActionSpec`,
  `Plan`, `PlanMetadata`, `GoalSpec`, `ConstraintSpec`, `Score` hierarchy,
  `CSPMetadata`, `ResourceUsage`, `ScheduleEntry`, `ExecutionRecord`, and
  constraint-builder internals.
- **Benchmark suite** (`tests/benchmarks/`): memory footprint profiling,
  A* and A*→CSP temporal scheduling at 10/50/100-action scale, and a
  head-to-head latency comparison against
  `langgraph.prebuilt.create_react_agent`. GOAP baselines are
  regression-gated via `make benchmark-compare`.

### Hierarchical / multi-goal planning

- **`MultiGoal`** (`langgoap.goals`) with `mode="sequential"` and
  `mode="any"`. Sequential mode plans and executes each sub-goal in
  order, threading the resulting world state forward. `any` mode plans
  all goals and picks the lowest-score feasible plan. `GoapObserver`
  and `GoapPlanner` dispatch on `MultiGoal` at the top of their
  routing; the A* planner and CSP pipeline never see it directly.
  `GoapState.current_subgoal_index` tracks sub-goal progress.
  Recursive HTN-style decomposition is out of scope for v0.1.0.

### Public API

Every symbol in `langgoap.__all__` is part of the stable v0.1.0 public
surface:

- **Core**: `GoapGraph`, `ActionSpec`, `GoapAction`, `goap_action`,
  `GoalSpec`, `Goal`, `MultiGoal`, `ConstraintSpec`, `PlanningState`,
  `ActionResult`, `GoapState`.
- **Planner**: `plan` (A* only), `pipeline_plan` (A* → CSP), `Plan`,
  `PlanMetadata`, `PlanningStrategy`, `AStarStrategy`,
  `CSPRefinementStrategy`, `TwoPhasePipelineStrategy`.
- **Scoring**: `Score`, `SimpleScore`, `HardSoftScore`, `BendableScore`.
- **Constraints**: `ConstraintBuilder`, `ConstraintChain`,
  `ChainOutput`, `BuilderOutput`.
- **CSP**: `CSPMetadata`, `CSPStatus`, `ResourceUsage`, `ScheduleEntry`.
- **Integrations**: `create_goap_agent`, `goapify_tool`, `GoapSubgraph`,
  `add_goap_subgraph`.
- **Interpreter**: `GoalInterpreter`, `InterpretedGoal`,
  `InterpretedConstraint`, `InterpretedObjective`.
- **Graph nodes**: `GoapPlanner`, `GoapExecutor`, `GoapObserver`.
- **Tracing**: `PlanningTracer`, `NullTracer`, `LoggingTracer`,
  `MultiTracer`, `LangSmithTracer`.
- **History**: `ExecutionRecord`, `StoreExecutionHistory`,
  `compute_goal_hash`.
- **Types**: `CostFunction`, `Maximize`, `Minimize`,
  `ObjectiveDirection`, `ReplanStrategy`.
- **Visualization**: `render_mermaid`, `render_mermaid_gantt`,
  `render_dot`, `render_ascii`, `render_ascii_gantt`, `visualize`.

### Explicitly out of scope for v0.1.0

Documented in the release plan and `docs/optaplanner_mapping.md`:
utility AI planner, annotation-based `@Agent`/`@Action` reflection API,
recursive HTN-style decomposition, learned cost functions from
execution history, bundled OpenTelemetry adapter (LangSmith adapter
ships in-tree as `LangSmithTracer`),
multi-agent peer coordination beyond sequential `MultiGoal`, interactive
HTML/D3 visualization, custom `Move`/`Phase`/`Tabu`/`SimulatedAnnealing`
classes (CP-SAT implements these at a lower level), `CostNormalizer`
(`DESIGN.md` §14.1 — superseded by the Score hierarchy),
strict `WorldState` `TypedDict` schema enforcement (`DESIGN.md` §14.2
— `ActionSpec.effect_validator` already covers the common need), and a
Sphinx / ReadTheDocs site (the README, docstrings, and notebooks are
the v0.1.0 documentation surface).

[0.1.0]: https://github.com/integrallis/langgoap/releases/tag/v0.1.0
