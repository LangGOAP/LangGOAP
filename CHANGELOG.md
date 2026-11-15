# Changelog

All notable changes to LangGOAP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — Unreleased

The first batch of fixes from the v0.2.0 audit (issues #10–#18). Several
default behaviours change; each has an explicit opt-in that restores the
0.1.x behaviour where that behaviour was ever useful.

### Added

- **Low-code on-ramps reach the full API (#19).** `create_goap_agent`,
  `GoapSubgraph` and `add_goap_subgraph` accept every `GoapGraph` option
  (`strategy`, `tracer`, `history`, `sensors`, `guards`, `resolvers`,
  `record_expansions`, `transition_model`, `rng`, `stuck_handlers`,
  `max_stuck_iterations`, `termination_policies`) as explicit keyword
  arguments; `**kwargs` on `create_goap_agent` and
  `GoapSubgraph.compile` are compile options only (`checkpointer`,
  `store`, `interrupt_before`, `interrupt_after`) and any other name
  raises a `TypeError` that lists both sets. `goapify_tool` exposes the
  remaining `ActionSpec` fields (`qos`, `require_human_approval`,
  `human_input_key`, `can_rerun`, `read_only`, `utility`, `metadata`) and
  `create_goap_agent` takes per-tool maps for each. The compiled agent
  carries `goap_actions` next to `goap_goal` for `rehydrate_state`.
- `GoapGraph.initial_state`, `GoapGraph.rehydrate`,
  `langgoap.rehydrate_state`, `GoapStateChannels`, and `to_dict` /
  `from_dict` on every state type (see #17 below).

### Changed

- **`GoapState` is JSON-native (#17).** LangGraph channels now hold plain
  dicts, lists and scalars, so every `BaseCheckpointSaver` (in-memory,
  Postgres, Redis, SQLite, your own) works with its **stock** serializer
  and `langgoap/` contains no backend-specific code (a test enforces this;
  the `checkpoint-postgres` / `checkpoint-redis` extras are plain
  pass-throughs). `GoapGraph.invoke` / `ainvoke` / `invoke_nl` still
  return typed objects (`GoalSpec`, `Plan`, `ActionResult`), so most
  callers see no difference. Callers that drive a **compiled** graph
  directly now receive the JSON form: `result["plan"]` is a dict and
  `execution_history` entries are dicts. Use `GoapGraph.rehydrate(result)`
  (or `langgoap.graph.state.rehydrate_state` / `plan_from_state`) to get
  objects back; `successful_action_names` accepts both forms. When a
  checkpointer is attached, build the input with
  `GoapGraph.initial_state(goal, world_state)` or pass `goal.to_dict()`:
  LangGraph persists the raw input before the state reducers run, so a
  `GoalSpec` object in the input now fails with LangGraph's
  `Type is not msgpack serializable` error. Every state type gained
  `to_dict` / `from_dict` (`GoalSpec`, `GoalPolicy`, `ConstraintSpec`,
  `SoftGoal`, `MultiGoal`, `NirvanaGoal`, `ActionSpec`, `ActionQos`,
  `Plan`, `PlanMetadata`, `CSPMetadata`, `ResourceUsage`, `ScheduleEntry`,
  `InfeasibilityExplanation`, `ResourceShortfall`, `ActionResult`, the
  `Score` classes via `score_from_dict`). Plan actions are stored by their
  declarative fields and rebound by name to the live action list, so
  callables are never read from a checkpoint. A goal's callable `value`
  and its `metrics` expressions are kept reachable for the lifetime of the
  process; a checkpoint restored in a different process falls back to
  `value=1.0` and drops the metrics with a warning. The LangGraph schema
  class is `GoapStateChannels`; `GoapState` is the typed view.
- **Declared effect keys are verified when `execute` returns a dict (#31).**
  A returned dict is authoritative for the world state; if it never sets
  one of the action's declared effect keys the step is now
  `action_failed` with an error naming the missing keys, so the observer
  blacklists and replans instead of recording a success and looping the
  planner into the same action until `max_replans`. Values are not
  compared (a declared `{"computed": 0}` stays a placeholder for a runtime
  number; drift from the plan is the observer's deviation check). Effects
  the runtime applies itself (declared effects when `execute` returns
  `None`, `TransitionModel.sample` outcomes) are not checked. An explicit
  `effect_validator` keeps its full verdict; opt out per action with
  `effect_validator=lambda pre, post: True`.
- **Sequential `MultiGoal` is strict (#35).** A sub-goal that cannot be
  planned ends the run with `no_plan` at that stage, as the docstring
  always promised, instead of being skipped and the run reported as
  `goal_achieved`. The former best-effort behaviour is the explicit
  `MultiGoal(on_infeasible="skip")`: skipped indices are recorded in
  `GoapState.skipped_subgoals` and the run finishes with
  `status="partial"`.
- **Blacklist fallback is opt-in (#15).** When every remaining path to the
  goal needs a blacklisted action, the planner used to clear the blacklist
  and re-plan through the failed action, bounded only by `max_replans`.
  It now ends the run with `status="no_plan"` and an explanation that
  names the blacklisted actions and suggests `ActionSpec.max_retries`, an
  `ActionQos` retry policy, an alternative action, or
  `GoalPolicy(blacklist_fallback=True)`. The opt-in fallback clears only
  the blacklist and keeps `action_failure_counts`. Transient failures of a
  sole-path action should declare `max_retries`.
- **Infeasible plans are not executed by default (#13).** When CP-SAT
  marks the only available plan `INFEASIBLE`, the run ends with
  `status="no_plan"` and a `no_plan_explanation` carrying
  `infeasible_plan` and per-resource `resource_shortfalls`.
  `GoalPolicy(on_infeasible="execute")` restores the previous advisory
  behaviour (the plan runs and `plan.metadata.csp.status` reports
  `INFEASIBLE`).
- **Preferences now influence plan selection (#12).** Objectives, soft
  constraints and soft goals trigger multi-plan selection even when the
  first A* plan is feasible, and total cost is a strict tie-breaker
  rather than being added to the preference objective. Previously a
  feasible first plan was returned unchanged and `Minimize` / `Maximize`
  objectives never changed which plan ran. Plan-quality `metrics` still
  only re-score the chosen plan (CP-SAT cannot express them). The
  job-shop, project-scheduling and content-builder tutorial tests were
  rewritten to the corrected semantics; their notebooks need
  re-execution.
- `pipeline.needs_csp` is the public name of the former `_needs_csp`
  (kept as an alias) and includes soft goals; `pipeline.needs_selection`
  is new.

### Fixed

- **Anytime planning works (#23).** `plan(..., time_budget_ms=...)` runs
  weighted A* with a tightening weight schedule (`ANYTIME_WEIGHTS`, ending
  in plain optimal A*), so a complete, bounded-suboptimal plan exists early
  and is improved while the budget lasts; on expiry the best complete plan
  found so far is returned. Previously the deadline could only ever yield
  `None`. `AnytimePlanningStrategy` uses that native path for A* and, for a
  black-box inner strategy, returns `None` with a `RuntimeWarning` as soon
  as the budget expires instead of blocking until the inner strategy
  finishes. After a blacklist fallback `PlanMetadata.nodes_explored` now
  counts both searches.
- **Strategy routing no longer sends every large catalog to MCTS (#24).**
  `ProblemFeatures.horizon_estimate` is the depth of the relaxed planning
  graph (`relaxed_plan_depth`) instead of `unsatisfied × action_count`, so
  a five-step chain over a thirty-five-action catalog routes to A*; deep
  chains over large catalogs still route to MCTS.
- **Infeasibility explanations list every violated constraint (#25).**
  `conflicting_constraints`, `resource_shortfalls`, the suggestion text and
  the Mermaid note cover all violated hard bounds; the greedy
  minimal-unsatisfiable-subset pass that kept only one is gone (fixed
  resource totals make each violated bound independently unsatisfiable).
- **CP-SAT plan selection is numerically exact (#26).** Hard feasibility of
  each candidate is decided with the same float check as `validate_plan`
  before the integer model is built, so a plan over a bound by less than
  one scaling tick can no longer be selected as `OPTIMAL`; resource
  variables have signed domains so negative totals (net balances) are
  representable; integer bounds are loose so a feasible plan at a
  fractional bound is never rejected; `plans_evaluated` counts every
  candidate.
- **MCTS plans carry expected states, not sampled ones (#27).** Under a
  stochastic `TransitionModel` the plan's `expected_states` are now the
  deterministic `model.expected` replay along the chosen actions, so the
  observer no longer replans on the planner's own sampling noise.
- **MCTS tree reuse respects the blacklist and is per-thread (#28).** A
  carried-over tree is pruned of every branch through a newly
  blacklisted action and its frontier is recomputed against the current
  action set, and the carryover / last-root handles live in thread-local
  storage so one `MCTSStrategy` shared by concurrent graphs keeps one tree
  per thread.
- **Callables see the live world state (#30).** `PlanningState` keeps the
  non-planning entries of the world state as a `context` that takes no
  part in equality or hashing, and every planner (A*, pipeline, repair,
  utility, MCTS, truncation) hands `state.as_context()` to cost
  functions, dynamic effects and `TransitionModel.expected` / `sample`.
  A `cost=lambda ws: ... ws.get("vip")` now sees `vip` even though it is
  not a precondition, effect or goal key. `expected_states` stay
  planning-only.
- **Condition resolvers persist and stay in their lane (#32).** Resolved
  values are written into the executed `world_state` (previously only
  sensor results were), so the deviation check no longer fires after
  every step, and resolvers are consulted only for precondition keys no
  action can produce, never for effect keys or the goal itself.
- **Policy failures merge into `action_failure_counts` (#33).** Guard
  blocks, human denials and invalid typed forms add the failing action at
  `max_retries + 1` to the existing counts instead of replacing the dict,
  so other actions keep their spent retry budgets.
- **Unhashable planning values fail at construction (#34).** `GoalSpec`,
  `SoftGoal` and `ActionSpec` (preconditions and static effects) raise a
  `ValueError` naming the offending key when a value is a list, dict or
  other unhashable, instead of a `TypeError` deep inside A*.
- **Partial tracers never break execution (#36).** `SafeTracerProxy`
  resolves a hook the wrapped tracer lacks to a no-op (async by prefix),
  so a tracer written against an older protocol no longer raises
  `AttributeError` mid-run and turns an `ActionQos` retry into a failure.
  `ReflexionTracer` implements the whole protocol (`on_action_retry`,
  `on_strategy_chosen`, the `on_search_*` family and their async twins).
- **Tool failures are action failures (#37).** `goapify_tool` invokes
  tools created with `handle_tool_error` / `handle_validation_error`
  through a copy with those options off, so a `ToolException` fails the
  action and the observer blacklists and replans instead of applying the
  declared effects and storing the error text as a result.
- **`goapify_tool` input filtering (#38)** handles a JSON-schema
  (`dict`) `args_schema` via `tool.args` and forwards Pydantic aliased
  fields under their alias, accepting either the field name or the alias
  from `world_state`.
- **`add_goap_subgraph` runs async and inherits the parent's config (#20).**
  The node is a `RunnableLambda` with sync and async variants, so
  async-only actions work under `parent.ainvoke`, and it passes the
  parent's `RunnableConfig` to the child, so a parent checkpointer,
  callbacks and a human-approval `interrupt()` inside the child resume
  through the parent. The child input is built with
  `GoapGraph.initial_state`, so a parent checkpointer can persist it.
- **Soft goals are planned on and scored (#22).** `needs_csp` includes
  `soft_goals`, achieved soft goals add their weight to the soft score,
  and a soft goal triggers plan selection, so a plan that achieves one
  is preferred at equal cost.
- **`NirvanaGoal` survives a checkpoint (#29).** Its JSON form carries
  `kind="nirvana"`, `is_nirvana` is true after rehydration, and `repr`
  names the type.
- **Restored plans are usable (#39).** Rehydration rebinds plan actions
  to the live action list, so a restored plan renders, repairs and
  executes with its callable costs intact; a plan rebuilt without live
  actions falls back to `cost=1.0` instead of `None`.
- **A* heuristic is admissible and consistent (#11).** The heuristic was
  the number of unsatisfied goal conditions, which overestimates when one
  action satisfies several conditions or when the cheapest action costs
  less than 1, so A* could return a non-optimal plan. It is now
  `ceil(unsatisfied / max goal conditions any action establishes) * min
  static action cost` (zero when any action cost is dynamic).
- **`StrategyRouter.plan` accepts `prior_plan` and `current_step` (#10)**
  and forwards them to the selected strategy, so a `RepairStrategy` behind
  a router receives the plan to repair instead of planning from scratch.
- **`RepairStrategy` (#14)** no longer keeps blacklisted actions or spent
  `can_rerun=False` actions in the reused prefix, and when the repair
  cannot reach the goal it delegates to the inner strategy instead of
  returning the unrepaired plan.
- **Parallel executor (#16).** A failure inside a wave now aborts the
  wave after merging the successful siblings' deltas (previously the
  failure record could be lost or later siblings kept running), the
  world-state merge applies per-action deltas instead of
  last-writer-wins, and an action with `require_human_approval` runs in a
  wave of its own so the interrupt cannot be raised mid-wave.
- **No class references in checkpoints (#18).** The checkpoint format
  used to store a typed-form `require_human_approval` class by module and
  qualified name and import it on load. With the JSON-native state the
  class is recorded as `True` and rebound from the live action list, so
  nothing is imported by name when a checkpoint is restored.

### CLI

- **`langgoap plan` only plans (#40).** It no longer runs the actions'
  `execute` callables; pass `--run` to execute through the GOAP graph as
  before. `-s auto` (the new default) routes to the CSP pipeline when the
  goal carries constraints, objectives or soft goals and to A* otherwise;
  `-s astar` now really runs A*. The JSON output names the strategy that
  ran. `plan -f gantt` on a plan without a schedule reports a clean error,
  and `langgoap actions` renders a callable cost as `dyn`.

### Packaging and CI

- Direct imports are declared: `langchain-core`, `pydantic` and
  `typing-extensions` join `dependencies` (#44). The wheel ships
  `py.typed`; the sdist contains the package, README, LICENSE and
  CHANGELOG only; the version is read from `langgoap/_version.py`
  (`dynamic = ["version"]`); CONTRIBUTING no longer lists a `cli` extra.
- Release and TestPyPI workflows run the suite against the published
  wheel in a throwaway virtualenv with the checkout's package moved aside,
  the minimum-versions job uses `uv sync --resolution lowest-direct`, and
  the `deepagents` extra is installed on the main test matrix (#42).
- `@pytest.mark.api` tests skip when `OPENAI_API_KEY` is unset instead of
  erroring, the `MemorySaver` checkpoint tests live in their own module
  so a missing Docker daemon no longer hides them, the Docker probe times
  out after two seconds, and the public-API test compares the version to
  the installed distribution metadata (#43).

### Dependencies

- `uv.lock` refreshed to the current releases (#21): langgraph 1.2.22,
  langgraph-checkpoint 4.3.2, langgraph-checkpoint-postgres 3.2.1,
  langgraph-checkpoint-redis 0.6.0, langchain-core 1.8.0, langsmith
  0.14.17. Declared ranges are unchanged. A nightly CI job
  (`test-latest-dependencies`) now runs the suite against the newest
  resolvable versions.

### Deprecated

- `langgoap.install_langgoap_serde`, `langgoap.LangGOAPSerializer` and
  `langgoap.LANGGOAP_ALLOWED_MSGPACK_TYPES` are no-op shims that emit
  `DeprecationWarning`; they will be removed in 0.3.0.
  `LangGOAPRedisSerializer` is gone (it imported the Redis backend).

## [0.1.1] — Unreleased

Initial release. LangGOAP is a Goal-Oriented Action Planning framework
for LangGraph with constraint optimization, natural-language goal
interpretation, and a LangChain-first execution model in which the plan
*is* a compiled `StateGraph`.

`0.1.0` was uploaded to TestPyPI with a broken console script (`click`
sat under the `[cli]` extra while the `langgoap` entry point was
always registered) and has been yanked.  `0.1.1` is the first
installable release; no public API or behavior differs from the
`0.1.0` source tree.
