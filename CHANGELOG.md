# Changelog

All notable changes to LangGOAP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3] — 2026-11-22

### Changed

- Release workflow (#111): every job that installs the fresh upload from
  Test PyPI first waits for the index to list the version's files, up to
  five minutes, and the validation install retries five times at growing
  intervals. The 0.3.2 run had asked within a second of the upload and
  failed on timing alone. No change to the package.
- Release drafter configuration in the v7 shape: list fields written as lists and
  the autolabeler as its own workflow step, ahead of the release-drafter 7 bump.

## [0.3.2] — 2026-11-22

### Changed

- Routine log lines are DEBUG (#109): "Planning for goal", "Plan found",
  "Replanning (replan #n)", the MultiGoal sub-goal rounds, the CSP pipeline's
  validation and selection lines, and the executors' "Executing action" /
  "running action" lines fire on every round or step and are now emitted at
  DEBUG. Lines that report an event (a stuck handler resolving a failure, the
  blacklist fallback, an infeasible primary plan, an early termination, a
  rejected approval or form) stay at INFO. The routine lines are emitted
  inside the planner's timed span, so a host that configures an expensive
  handler on the root logger at INFO (mcp's `MCPServer` installs a rich one)
  saw its rendering counted in `duration_ms`; now nothing routine reaches such
  a handler. Set the `langgoap` logger to DEBUG to see the lines again.

## [0.3.1] — 2026-11-22

### Added

- Tracer hooks receive `thread_id` (#104): every hook whose signature accepts
  a `thread_id` keyword gets the run's thread id (`None` outside a run), also
  through `MultiTracer`. Hooks that do not declare it are called as before, so
  a tracer shared by many threads needs no `current_thread_id()` calls. The
  signature lookup is remembered per hook function, so hooks that fire per
  search expansion stay cheap.
- `GoapGraph.waiting_action(thread_id)` (#105): the name of the wait action a
  thread is paused on, or `None` when it is not in a wait (finished, unknown,
  or paused on an approval or form).
- Observed effects (`GoapGraph(effects="observed")`, #96). After an action
  runs, its declared effects and any dict it returned are assumed, the sensors
  are asked, and what they report for the action's keys wins: the effect keys,
  the new `ActionSpec.observe` and the graph's `always_observe`. Actions no
  longer re-read and return their own facts, and a decision-only action needs
  no `execute`. Works with both executors on the sync and async paths. The
  default stays `effects="declared"`.
- `ActionSpec.wait(...)`: an action that pauses the graph (interrupt payload
  type `goap_action_wait`) until a sensed value of one of its keys changes,
  with `on_resume` and `payload` options. Each resume runs `on_resume` and the
  sensors once; a resume that changed nothing pauses the wait again in a fresh
  task (the state key `prepared_wait` remembers that its `execute` has run),
  so a notification costs the same however many came before it. The parallel
  executor runs a wait alone, like an approval gate. New fields
  `ActionSpec.is_wait`, `on_resume` and `wait_payload`;
  `@goap_action(is_wait=True, ...)` and a `GoapAction` subclass with
  `is_wait = True` declare a wait too.
- Thread API on `GoapGraph`: `goal=`, `checkpointer=`, `on_stop=` and
  `max_actions=` on the constructor, and `start`, `notify`, `resume`, `waiting`
  (plus `astart`, `anotify`, `aresume`, `awaiting`) per thread id. Runs are
  serialised through a mailbox, so a call made while a run is in progress is
  delivered after it, including calls from other OS threads; `notify` never
  answers a question and `resume` never ends a wait.
- `sensors=` accepts plain functions `(world_state) -> updates`
  (`langgoap.sensors.as_sensor`).
- `current_thread_id()` tells a tracer hook, action or sensor which thread it
  is running for.
- `GoapGraph(effects="merged")` (#97): the declared effects are applied and a
  returned dict is merged on top, so an action returns only what it adds or
  what differs. The default stays `"declared"`, where a returned dict must set
  every declared effect key.
- `ActionSpec.name` may be omitted when `execute` or `aexecute` is a named
  function or method; the action is named after it (#98). A lambda or a
  `functools.partial` still needs an explicit name.
- `GoapGraph.invoke` and `ainvoke` accept a plain mapping of conditions as the
  goal and fall back to the graph's `goal=` (#100).
- `GoapResult` (#101): `invoke`, `ainvoke`, `rehydrate` and the thread API
  return the state dict as a `dict` subclass with `status`, `succeeded`,
  `actions`, `world_state`, `interrupt` and `summary()`. Every existing lookup
  on the result keeps working.

### Fixed

- Synchronous actions, sensors, guards and condition resolvers that the async
  path runs in a worker thread now run in the caller's context, so LangGraph's
  run config (and `current_thread_id()`) reaches them.
- `on_replan` and `aon_replan` receive `duration_ms`, like `on_plan_complete`.
  It is passed by keyword and only to hooks that declare it, so tracers written
  against the earlier signature keep working.

### Documentation

- `GoapGraph.initial_state` and the concepts page say that the initial world
  state may be partial, because sensors run before the first plan too.
- Examples: the human-in-the-loop scripts for the email assistant and the
  weather alert use the thread API (#102), with unchanged output; `cost=1.0`
  is dropped from the five example files where every action has the default
  cost (#99).


## [0.3.0] — 2026-11-17

### Performance

- A* drops the actions that cannot contribute to the goal before searching
  (`relevant_actions`: a backward closure over the keys the goal reads and the
  precondition keys of actions that write relevant keys). Reachable but useless
  actions used to multiply the state space by every combination of the facts
  they set; a chain of 6 layers with 2 such actions per layer explored 377
  nodes and took 93 ms, now 7 nodes and 0.4 ms, and shapes that did not finish
  within 10 minutes (20 layers, 5 alternatives and 6 distractors each) plan in
  6 ms. Pruning is skipped whenever a relevant action has callable effects,
  which may read any key. Plans are unchanged where the search finished before.
- A* no longer copies the state for an action whose effects already hold, and
  `PlanningState.apply` builds the new state without re-probing every value for
  hashability (values are validated when actions and states are constructed).
- A plan rebuilt from the state channel materialises its expected states, and
  the history its `ActionResult` entries, only when they are read
  (`LazyStates`, `LazyHistory`); both compare, index, slice and concatenate like
  the tuple and list they replace. Every graph node rehydrates the state, so
  rebuilding the whole plan and history per node made a run quadratic in its
  length: in the benchmark suite a 60-step run went from 306 ms to 99 ms and a
  120-step run from 1121 ms to 198 ms on the same machine (the full before and
  after table is in the pull request).
- The planning-key set is remembered per action list instead of being recomputed
  by every node on every step.

### Changed

- `Plan.expected_states` is typed `Sequence[PlanningState]` and
  `GoapState["execution_history"]` `Sequence[ActionResult]`; both are tuples
  and lists when built in-process and lazy sequences when rehydrated from a
  checkpoint or the state channel.

### Fixed

- `interrupt()` inside an action's `execute` or `aexecute` now pauses the graph
  instead of being recorded as a failed action. Every executor handler
  (sequential and parallel, sync and async, with and without `ActionQos`
  retries) lets LangGraph's `GraphBubbleUp` exceptions propagate, so an action
  can wait for an external event and `Command(resume=...)` resumes it with the
  value it returns applied as usual; a wait that returns a different outcome
  than its declared effect makes the planner replan, as drift always does (#89).
- `AnytimePlanningStrategy` forwards the planner's tracer. Its fast path
  called A* directly and dropped the tracer that `rebind_inner_tracer` had
  bound into the inner strategy, so a tracer given to `GoapGraph` saw no
  search expansions from an anytime run while the lazy, repair and
  two-phase wrappers reported theirs (#63).

### Tests

- The gaps the v0.2.0 audit listed are closed: a run resumed by a **new**
  process on the same checkpointer and thread keeps its plan, history,
  replan count, failure counts and blacklist, sub-goal index and skipped
  sub-goals, wall-clock anchor, no-plan explanation and reflections, and a
  finished thread stays finished (`tests/integration/test_crash_resume.py`);
  `ainvoke` matches `invoke` for termination policies, the three
  `MultiGoal` modes and the strict sequential default, stuck recovery,
  failure counts, handled tool errors, the parallel wave merge and the
  blacklist fallback, plus the goal-object-with-checkpointer error, the
  compile-option `TypeError` wording and the report key cap
  (`tests/test_async_parity.py`); every codec round-trips every field,
  including `ActionResult.cost`, `MultiGoal.on_infeasible`, `PlanMetadata`
  and `CSPMetadata`, `ActionQos`, typed-form `ActionSpec`s and
  `NirvanaGoal` (`tests/test_codec_roundtrips.py`); the wrapper strategies
  forward the tracer and the pipeline, repair and utility planners see
  non-planning context keys (`tests/test_tracer_forwarding_and_context.py`).

### CI

- `make lint` checks and never rewrites: it runs `black --check`,
  `isort --check-only`, mypy and `ruff check --select F821,F401,F811`
  (undefined names, unused imports, redefinitions). It used to run the
  formatters in write mode, so an unsorted import block passed CI and
  surfaced only in the release job's format step; and no check caught an
  undefined name, which is how the `NameError` in #71's first CI run got
  through. `make format` still rewrites. The 150 findings ruff reported
  on the existing tree are fixed (unused imports, duplicate `Mapping`
  imports, test annotations naming types that were only imported inside
  the function).
- The release notes drop the Contributors block: release-drafter unions
  pull-request authors with commit authors, which listed the maintainer
  twice and dependabot twice.

## [0.2.0] — 2026-11-15

Resolves the 58 findings of the 2026-11-14 pre-release audit (#10–#67):
planner correctness, a JSON-native graph state that works with every
LangGraph checkpointer, stricter defaults, async parity, and a documentation
and catalog sweep. Several default behaviours change; each has an explicit
opt-in that restores the 0.1.x behaviour where that behaviour was ever
useful.

### Breaking changes

- **The graph state is JSON.** A compiled graph returns and checkpoints
  plain dicts: `result["plan"]` and `execution_history` entries are dicts
  (`GoapGraph.rehydrate` / `rehydrate_state` convert them back), and with a
  checkpointer the input is `GoapGraph.initial_state(goal, world_state)` or
  `goal.to_dict()`, never a `GoalSpec` object (#17, #18).
- **Defaults are strict.** A CSP-infeasible plan is not executed
  (`GoalPolicy(on_infeasible="execute")` restores it, #13); the blacklist
  fallback is off (`GoalPolicy(blacklist_fallback=True)`, #15); an
  `execute()` dict must set every declared effect key (#31); a sequential
  `MultiGoal` fails on an unreachable sub-goal
  (`MultiGoal(on_infeasible="skip")`, #35); objectives, soft constraints
  and soft goals change which plan runs (#12); `langgoap plan` only plans
  (`--run` executes, #40).
- **Termination and reporting.** A run that reaches the goal on the step
  that exhausts a cap reports `goal_achieved` (#47); a final `no_plan` is
  reported as `no_plan` (#64); `ExecutionRecord.plan_actions` and
  `actual_cost` describe executed steps (#51); `Plan` display is offline
  Markdown unless `LANGGOAP_DISPLAY_PNG=1` (#58).
- **Removed and deprecated API:** see the sections at the end of this entry.

### Added

- `GoapGraph(history_state="delta" | "planning")` (also on `GoapExecutor`,
  `ParallelGoapExecutor`, `create_goap_agent`, `GoapSubgraph` and
  `add_goap_subgraph`) compacts the world-state snapshots stored in every
  `ActionResult`: `"delta"` keeps only the keys an action changed, with
  their prior values, and `"planning"` keeps only planning-relevant keys,
  so checkpoints stop growing with every step when the world state carries
  large non-planning data. `"full"` stays the default. Each `ActionResult`
  now records `cost`, the action's cost on the full pre-state, and
  `ExecutionRecord.actual_cost` uses it, so cost accounting no longer
  depends on the snapshots (#74).
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

- **The boolean approval gate asks again instead of guessing.** Any
  non-empty string, any non-zero number, a dict without `approved` or
  with a non-boolean `approved` all used to count as approval, so
  `Command(resume="no")` approved the action. Only `True` / `False` and
  `{"approved": <bool>, "reason": ...}` are decisions now; any other
  resume value re-raises the interrupt with `attempt`, `validation_errors`
  and `rejected_payload`, exactly like the typed-form gate. Both gate
  payloads carry `attempt` and `effects`.
- Planner hygiene (#63): `assert_expected_matches_declared` raises
  `ValueError` instead of relying on `assert`; `explain_no_plan` compares
  `(key, value)` pairs, so a goal wanting `x=False` is unreachable when
  actions only set `x=True` and start values such as `0` count as
  present; `pareto_plans` annotates `plans_evaluated` and solver time
  without objectives too; `PlanningState.get` / `in` use a cached dict
  view and the A* heuristic no longer allocates a dict per evaluation;
  `AStarStrategy` and `TwoPhasePipelineStrategy` accept `tracer` /
  `record_expansions` and inherit the planner's tracer when passed
  explicitly (the lazy, anytime and repair wrappers forward it);
  `refine_with_csp(plan, goal)` replaces `CSPRefinementStrategy`.
- Graph-runtime hygiene (#64): an `execute()` return that is neither a
  dict nor `None` is logged as a warning before the declared effects are
  applied; QoS back-off jitter uses its own RNG so a retry never perturbs
  seeded `TransitionModel.sample` sequences; `GoapObserver` built without
  action specs reads `max_retries` from the plan's specs; an any/best_value
  `MultiGoal` that cannot be planned emits a `no_plan_explanation` with one
  entry per sub-goal; a final `no_plan` is reported as `no_plan` rather
  than `max_replans_exceeded`; dead helpers (`_safe_tracer_call`,
  `_safe_tracer_acall`, `GoapPlanner._resolve_sequential_subgoal`) are
  removed.
- Public exports (#65): `PlanQualityMetric` and the `Minimize*` metrics,
  `RepairResult` / `repair_plan`, `AgentModel` / `UniformRandomAgentModel`,
  `ScalarHeuristic` and the rollout policies,
  `assert_expected_matches_declared`, `SafeTracerProxy`,
  `async_execute_action` and `EffectValidator` are importable from
  `langgoap`; `langgoap.planner` re-exports the planner-level ones.
  `StoreExecutionHistory.store` exposes the backing store, and
  `ReflexionTracer` uses it instead of a private attribute.
- The deployment scaffold (#65) lists `"."` first in `langgraph.json`
  dependencies so the factory module ships, installs
  `langgraph-cli[inmem]`, uses the `LANGSMITH_*` variables, names the
  generated agent in its README and no longer tells users to pass a
  checkpointer that LangGraph Platform ignores.
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
  rewritten to the corrected semantics and their notebooks re-executed.
  Alternatives are enumerated breadth-first over excluded actions (one
  swap away from the A* plan, then two, ...) up to `max_alternatives`,
  whose default rises from 5 to 8. Excluding one action at a time could
  never reach a plan two swaps away: with uniform action costs the
  objective settled on the $1,506 cloud-balancing placement while a
  feasible $1,092 one existed, and the project-scheduling soft budget of
  $20 looked unattainable although a two-swap plan meets it exactly.
- `pipeline.needs_csp` is the public name of the former `_needs_csp`
  (kept as an alias) and includes soft goals; `pipeline.needs_selection`
  is new.

### Fixed

- **A typed form that fails validation is asked again, not blacklisted.**
  A resume payload rejected by the Pydantic model (or one of the wrong
  shape, such as `True`) was recorded as a `human_form_invalid` failure
  with the failure count pre-set past `max_retries`, so a typo in a form
  blacklisted the action and, with no alternative, ended the run in
  `no_plan`. It is an input error, not a decision: the executor now raises
  the interrupt again with `attempt`, `validation_errors` (`loc` / `msg` /
  `type` per error) and `rejected_payload` in the payload, and the
  corrected submission goes through. A form can now be declined
  explicitly: `False`, or `{"approved": False, "reason": ...}` when the
  model has no `approved` field, is recorded and blacklisted exactly like
  a denied boolean gate (`human_approval_denied`), which the form gate
  previously had no way to express.
- `GoalSpec(conditions={...})`, `SoftGoal(conditions={...})` and
  `objectives={...}` type-check: the fields are annotated as `Mapping`
  (what a caller writes) and still converted to an immutable
  `MappingProxyType` at construction. They were annotated as
  `MappingProxyType`, so every dict literal in user code was a mypy error.
- **Every bookkeeping key is present in every result.** LangGraph returns
  only the channels a run wrote to, so `result["blacklisted_actions"]`
  and `result["action_failure_counts"]` raised `KeyError` precisely when
  nothing had gone wrong (and `replan_reason`, `skipped_subgoals`,
  `reflection_context`, `current_subgoal_index`, `no_plan_explanation`
  likewise). The defaults now live in one table,
  `langgoap.graph.state.STATE_DEFAULTS`: `GoapGraph.initial_state` seeds
  them, the planner fills any that an input built by hand
  (`{"goal": ..., "world_state": ...}`) left out, and `rehydrate_state`
  fills them on the way out, so typed access is total in sync and async
  runs and after a resume in a new process.
- **An operator's denial is explained as a denial.** When every remaining
  path to the goal needs an action a human declined, the no-plan
  explanation advised raising `max_retries` or attaching an `ActionQos`
  "for transient failures". It now lists the action under
  `denied_actions` with the operator's reason and points at the things
  that apply: an alternative action, a stuck handler that relaxes the
  goal, or running again once the decision changes.
- The `no_plan` explanation distinguishes actions blacklisted after
  failures from `can_rerun=False` actions that already ran: the latter are
  listed under `spent_actions` with advice to add an alternative or allow a
  rerun, instead of the retry and QoS advice that fits transient failures
  (a `NirvanaGoal` run ends this way by design). The log line no longer
  attributes every planning failure to A*.
- A typed-form answer survives the action it approved failing. The
  pre-execution snapshot an executor keeps is taken after the
  human-in-the-loop gate, so a failure's `world_state` (and the parallel
  executor's wave abort) carries the answer, and `_check_form_approval`
  reuses an answer under `human_input_key` that still validates against
  the form model instead of interrupting again on the retry; a missing or
  stale answer still prompts. A valid answer placed in the initial world
  state is used without a prompt (#75).
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
- **Lazy decomposition keeps scores honest (#45).** A truncated plan no
  longer advertises the full plan's CSP resource usage, schedule and
  status: the prefix is re-validated and carries a `HardSoftScore` and
  `CSPMetadata` for its own actions (search statistics stay from the
  inner run), so MultiGoal `any` selection compares like with like.
- **The router's choice reaches the tracer (#46).** After planning with a
  `StrategyRouter`, the planner fires `on_strategy_chosen` /
  `aon_strategy_chosen` with the dispatched strategy's name.
- **Goal satisfaction wins over a cap hit on the same step (#47).** The
  observer checks the goal before termination policies, so a three-step
  plan under `MaxActionsPolicy(3)` reports `goal_achieved`; a cap hit
  while the goal is still open terminates as before.
- **Duplicate action names are rejected (#48).** `GoapGraph`,
  `GoapPlanner`, `GoapExecutor` and `ParallelGoapExecutor` raise a
  `ValueError` listing the duplicates.
- **`GoapGraph(executor="parallel")` (#49)** wires `ParallelGoapExecutor`
  without a hand-built graph; the on-ramps accept the same option.
- **Wall-clock anchor is an epoch timestamp (#50).** The planner stores
  `time.time()` in `wall_clock_started_at` and `MaxWallClockPolicy`
  compares against the same clock, so a budget survives a checkpoint
  resume in another process.
- **`ExecutionRecord` reports what executed (#51).** `plan_actions` lists
  the successful steps in order across replans, `actual_cost` sums their
  costs, `expected_cost` stays the final plan's cost, and `metadata`
  carries `failed_attempts` and `final_plan`.
- **Class-based and decorator actions express every field (#52).**
  `GoapAction` gained `resources`, `duration`, `metadata` and
  `effect_keys` (with callable `effects`), all forwarded by `to_spec`;
  `@goap_action` accepts a callable `effects` plus `effect_keys`.
- **The async path is async all the way down (#53).** `GoapPlanner.acall`
  uses a strategy's `aplan` when it has one and awaits coroutine stuck
  handlers (new `AsyncStuckHandler` protocol; `MulticastStuckHandler`
  dispatches both kinds); an async-only handler under `invoke` is skipped
  with a warning instead of crashing.
- **Renderers evaluate costs against each step's pre-state (#54)** and
  draw no badge when a state-dependent cost cannot be evaluated instead of
  raising; Gantt task ids are positional, so repeated names and names with
  spaces stay valid and unique.
- **Unknown models are not silently free (#55).** `CostAccumulator` warns
  once per model name it has no price for (tokens are still counted) and
  `require_known_pricing=True` raises instead. The bundled price table is
  documented as a convenience snapshot; no new prices were added because
  current list prices could not be verified from a primary source here.
- **Tool descriptions reach the goal interpreter (#56).** `goapify_tool`
  records `description` and `args` in the action's metadata (your metadata
  keys win) and `build_action_catalog` shows both.
- **Compact GOAP reports for agents (#57).** `format_goap_result` renders
  every history entry as one line (with the error for failed steps) and
  clips world-state values and key counts, so a tool result no longer
  carries whole state snapshots.
- **Notebook display is offline by default (#58).** `Plan` displays as
  Markdown with the fenced Mermaid source; set `LANGGOAP_DISPLAY_PNG=1`
  to also render an inline PNG through the mermaid.ink service.
- **Goal interpreter prompt hygiene (#59).** `AzureChatOpenAI` and other
  `BaseChatOpenAI` subclasses get the `function_calling` structured-output
  path; the prompt shows only planning-relevant world-state keys, clipped
  to `max_world_state_chars`, or none with `include_world_state=False`.
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
- CI hygiene (#61): the Claude workflow is pinned to a commit and answers
  only repository owners and members, and no longer requests an OIDC
  token; `actions/checkout` and `astral-sh/setup-uv` versions are
  aligned; Dependabot watches GitHub Actions and uv dependencies; the
  reusable release workflow runs with `contents: read` plus
  `id-token: write` instead of `write-all`; the PR quality report updates
  one sticky comment; the scheduled full matrix runs weekly; the coverage
  badge uses the same 80% floor as the quality gate.
- The README's MFCQI badge shows a measured number. The badge workflow
  (now `Update Badges`) runs `mfcqi badge langgoap` on every push to
  `main`, before the test suite so a red test never leaves a stale
  value, and publishes the result to the gist that serves the coverage
  badge. The hand-maintained `.github/badges/mfcqi.json`, which had read
  0.90 since April 2026 while CI measured 0.81, is removed.
- `@pytest.mark.api` tests skip when `OPENAI_API_KEY` is unset instead of
  erroring, the `MemorySaver` checkpoint tests live in their own module
  so a missing Docker daemon no longer hides them, the Docker probe times
  out after two seconds, and the public-API test compares the version to
  the installed distribution metadata (#43).

### Documentation

- **LangGraph tutorial refactors.** `examples/screencast/llm_compiler/`
  rewrites the LLMCompiler tutorial as GOAP: `$N` dependency references,
  the hand-written scheduler and the `Replan`-emitting joiner become
  preconditions, `executor="parallel"` and the observer's
  blacklist-and-replan loop. `examples/screencast/email_assistant/`
  rewrites `executive-ai-assistant`: three message-shape routers with
  `ValueError` fall-throughs and four Agent Inbox handlers become triage
  facts as preconditions, `require_human_approval=True` and a Pydantic
  form; a denied send or an invalid form is blacklisted and re-planned
  around, and a new process resumes a paused thread from the
  checkpointer. `examples/screencast/weather_alert/` rewrites the
  GenAI_Agents weather-disaster notebook: a router with an unreachable
  branch, an `input()` approval node and a swallowed SMTP failure become a
  sensor's facts as preconditions, `require_human_approval=True` on
  `approve_alert`, an `ActionQos` retry with an SMS fallback, and a
  `FunctionalStuckHandler` that relaxes the goal to "log the decision"
  when the operator denies the alert. Each refactor ships a condensed hermetic `before.py`,
  `after.py`, `after_disrupted.py`, a generated and executed notebook
  rendered in the docs, and an integration test that pins the paths, the
  gates, the disruptions and the async path.
- The API reference is generated from `langgoap.__all__` by
  `docs/generate_api_index.py`, grouped by the comment headers in
  `langgoap/__init__.py`, and a test fails when the page is stale; it
  documented 63 of the public names before (#66).
- The concepts page gains sections for pluggable strategies and MCTS,
  multi-goal decomposition, retries and QoS, human-in-the-loop and typed
  forms, stuck handlers, early termination, sensors / guards / resolvers,
  checkpointing, plan explanations, visualization, the CLI and
  deployment, and Deep Agents / tool interop; `myst.xref_missing` is no
  longer suppressed (#66).
- Catalog and stale-text sweep (#67): every basics notebook is rendered
  in the docs and listed in `examples/README.md`; the notebook counts on
  the docs landing page match the catalog; the incident screencast
  notebook is linked; the research-agent screencast uses the standard
  `python3` kernel; the CHANGELOG dates 0.1.1 and 0.1.2; the removed
  `LANGGOAP_RUN_LIVE_DEMO` gate, the deleted `SECURITY.md` link and the
  `cd langgoap` clone step are corrected; docstrings cite upstream
  repositories instead of the maintainer's private `research/` tree; the
  copyright year is 2026; the Jupyter image no longer installs the unused
  `langchain` package; the unused `bench` pytest marker is gone.
- **Versioned docs document their own release (#60).** The docs workflow
  builds each tag from its own git worktree and virtualenv instead of
  main's checkout, the version switcher's `populate` receives the URL it
  used to fetch `versions.json` (it threw a `ReferenceError` and left the
  menu empty), and the Sphinx toolchain is a uv dependency group
  (`uv sync --group docs`; `docs/requirements.txt` is gone).
- **README claims match the code (#62).** Pareto filtering is described
  as the standalone `pareto_plans` helper, `StoreExecutionHistory` is
  described as using the `BaseStore` `get`/`put` interface and tested with
  `InMemoryStore`, and the flagship screencast is described as three
  agent shapes plus a disruption run.

### Dependencies

- `uv.lock` refreshed to the current releases (#21): langgraph 1.2.22,
  langgraph-checkpoint 4.3.2, langgraph-checkpoint-postgres 3.2.1,
  langgraph-checkpoint-redis 0.6.0, langchain-core 1.8.0, langsmith
  0.14.17. Declared ranges are unchanged. A nightly CI job
  (`test-latest-dependencies`) now runs the suite against the newest
  resolvable versions.

### Deprecated

- `CSPRefinementStrategy` (call `refine_with_csp`) and
  `ActionSpec.is_async_execute()` (the executors pick the async path
  themselves) warn and will be removed in 0.3.0 (#63, #64).
- `langgoap.install_langgoap_serde` and `langgoap.LangGOAPSerializer` are
  no-op shims that emit `DeprecationWarning`;
  `langgoap.LANGGOAP_ALLOWED_MSGPACK_TYPES` is an empty tuple kept for
  import compatibility (a module constant cannot warn on access). All three
  will be removed in 0.3.0.

### Removed

- `LangGOAPRedisSerializer`: the custom Redis serializer imported the Redis
  backend into the core package; the JSON-native state (#17) makes every
  stock checkpointer serializer sufficient.

## [0.1.2] — 2026-05-15

### Fixed

- README images render on PyPI: the six relative image references were
  replaced with absolute `raw.githubusercontent.com` URLs.

### CI

- The TestPyPI reinstall step passes `--index-strategy unsafe-best-match`
  so the freshly uploaded wheel resolves against PyPI's dependencies.

## [0.1.1] — 2026-05-15

### Fixed

- `click` is a core runtime dependency (#6). `0.1.0` was uploaded to
  TestPyPI with it under a `[cli]` extra while the `langgoap` console
  script was always registered, and has been yanked.
- Docker-backed checkpointer tests skip when no daemon is reachable; the
  min-versions release check skips `ortools`.

### Changed

- Public API classes are spelled `LangGOAP*` (`LangGOAPSerializer`, ...)
  instead of `LangGoap*`.

### CI

- Test matrix across Ubuntu, macOS and Windows with a PR quality gate;
  release pipeline split into build, TestPyPI and PyPI stages;
  `actions/checkout`, `astral-sh/setup-uv` and `actions/setup-python`
  bumped to Node 24 releases.

Initial release. LangGOAP is a Goal-Oriented Action Planning framework
for LangGraph with constraint optimization, natural-language goal
interpretation, and a LangChain-first execution model in which the plan
*is* a compiled `StateGraph`.

`0.1.0` was uploaded to TestPyPI with a broken console script (`click`
sat under the `[cli]` extra while the `langgoap` entry point was
always registered) and has been yanked.  `0.1.1` is the first
installable release; no public API or behavior differs from the
`0.1.0` source tree.
