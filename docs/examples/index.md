# Examples

Every example below is a runnable Jupyter notebook committed to the
repository under `examples/`. Each tutorial has a matching integration
test under `tests/integration/` that runs in CI, so the notebooks are
verified end-to-end — they are documentation *of* the test suite, not
a substitute for it.

## Basics

Short, focused walkthroughs of individual mechanics.

- `plan_visualization.ipynb` — Mermaid, DOT, ASCII, and Gantt renderers
  over an A\* → CSP plan.
- `nl_goal_interpreter.ipynb` — converting a plain-English request into
  a `GoalSpec` via any `BaseChatModel`.

## Tutorials

Complete end-to-end walkthroughs across three tiers of complexity.

**Tier 1 — Primers**

1. `directory_handler.ipynb` — GOAP basics (file-system predicates).
2. `robot_navigation.ipynb` — A\* primer on a linear graph.
3. `hungry_agent.ipynb` — natural-language goal + cost-driven action
   selection.

**Tier 2 — Constraint optimization + workflow agents**

4. `cloud_balancing.ipynb` — VM bin-packing with the `create_goap_agent`
   one-liner.
5. `vehicle_routing.ipynb` — capacity-constrained routing with CSP
   temporal scheduling and Gantt visualization.
6. `nurse_rostering.ipynb` — shift assignments, skill matching, and
   hard/soft decomposition via `HardSoftScore`.
7. `project_job_scheduling.ipynb` — RCPSP with precedence constraints
   and makespan minimization.
8. `task_assigning.ipynb` — ticket routing with the fluent
   `ConstraintBuilder`.
9. `sql_query_agent.ipynb` — schema-explore → generate → test → refine
   loop using `effect_validator` and replanning on test failure.
10. `vulnerability_scanner.ipynb` — phased discovery with sensor
    integration and action blacklisting.

**Tier 3 — Full-feature showcases**

11. `deep_research_agent.ipynb` — LLM-heavy research loop with
    `StoreExecutionHistory` and tracing.
12. `hierarchical_product_launch.ipynb` — `MultiGoal` sequential
    decomposition end-to-end.
13. `content_builder_agent.ipynb` — multi-objective CSP with
    conditional format generation.
14. `temporal_match_cellar.ipynb` — durative actions with overlap
    constraints and parallel scheduling.
15. `flexible_job_shop.ipynb` — every v0.1.0 feature in one notebook:
    NL intake, A\* → CSP, hard/soft scoring, temporal scheduling,
    visualization, and tracing.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Basics

basics/plan_visualization
basics/nl_goal_interpreter
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tier 1 — Primers

tutorials/directory_handler
tutorials/robot_navigation
tutorials/hungry_agent
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tier 2 — Constraint optimization + workflow agents

tutorials/cloud_balancing
tutorials/vehicle_routing
tutorials/nurse_rostering
tutorials/project_job_scheduling
tutorials/task_assigning
tutorials/sql_query_agent
tutorials/vulnerability_scanner
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tier 3 — Full-feature showcases

tutorials/deep_research_agent
tutorials/hierarchical_product_launch
tutorials/content_builder_agent
tutorials/temporal_match_cellar
tutorials/flexible_job_shop
```
