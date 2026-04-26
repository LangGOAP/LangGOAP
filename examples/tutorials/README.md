# LangGOAP Tutorial Notebooks

This directory contains runnable tutorial notebooks. The **v0.1.0 featured
catalog** below is the set of notebooks published on the documentation site
and referenced from `README.md`. A handful of earlier-phase demonstrations
live alongside the catalog as **extras** — they have passing integration
tests but are not featured in the docs build.

## v0.1.0 Featured Catalog (15 tutorials)

Each notebook has a corresponding integration test under
`tests/integration/` that is run on every `make check`. The notebook is the
runnable documentation; the integration test is the source of truth.

### Tier 1 — Primers

1. [`directory_handler.ipynb`](directory_handler.ipynb) — basic GOAP loop
   over a file-system state (from GOApy).
2. [`robot_navigation.ipynb`](robot_navigation.ipynb) — A* pathfinding on a
   linear graph (from unified-planning).
3. [`hungry_agent.ipynb`](hungry_agent.ipynb) — natural-language goal intake
   driving cost-based action selection.

### Tier 2 — Intermediate workflows

4. [`cloud_balancing.ipynb`](cloud_balancing.ipynb) — VM bin-packing with
   CSP resource constraints (standard benchmark problem).
5. [`vehicle_routing.ipynb`](vehicle_routing.ipynb) — delivery routing with
   capacity constraints and temporal windows.
6. [`nurse_rostering.ipynb`](nurse_rostering.ipynb) — shift assignment with
   hard/soft constraint decomposition.
7. [`project_job_scheduling.ipynb`](project_job_scheduling.ipynb) — RCPSP
   with precedence + resource constraints and makespan minimisation.
8. [`task_assigning.ipynb`](task_assigning.ipynb) — multi-objective CSP
   with the fluent `ConstraintBuilder`.
9. [`sql_query_agent.ipynb`](sql_query_agent.ipynb) — schema-aware query
   pipeline with `effect_validator` and replanning on test failure.
10. [`vulnerability_scanner.ipynb`](vulnerability_scanner.ipynb) — phased
    sensor-driven planning with action blacklisting.

### Tier 3 — Advanced

11. [`deep_research_agent.ipynb`](deep_research_agent.ipynb) — LLM-heavy
    research loop with `create_goap_agent`, execution history, and tracing.
12. [`hierarchical_product_launch.ipynb`](hierarchical_product_launch.ipynb)
    — `MultiGoal` sequential decomposition.
13. [`content_builder_agent.ipynb`](content_builder_agent.ipynb) —
    multi-format content generation with multi-objective CSP.
14. [`temporal_match_cellar.ipynb`](temporal_match_cellar.ipynb) —
    durative actions with overlap constraints and Gantt rendering.
15. [`flexible_job_shop.ipynb`](flexible_job_shop.ipynb) — full v0.1.0
    feature showcase: NL intake, A*→CSP, hard/soft scoring, `MultiGoal`,
    tracing, visualisation, execution history, and parallel fan-out via
    LangGraph `Send`.

## Extras (pre-v0.1.0 demonstrations)

The notebooks below predate the v0.1.0 featured catalog. They are kept in
the repository because each one is backed by a passing integration test
and exercises real LangGOAP APIs, but they are **not published on the
documentation site** and are not advertised from the project README.
Expect overlap with the featured catalog — some Tier 2/3 notebooks cover
similar ground more thoroughly.

| Notebook | Integration test | Topic |
| --- | --- | --- |
| [`adaptive_rag_goapified.ipynb`](adaptive_rag_goapified.ipynb) | `tests/integration/test_adaptive_rag.py` | GOAPified version of LangGraph's Adaptive RAG tutorial. |
| [`agent_pattern_examples_goapified.ipynb`](agent_pattern_examples_goapified.ipynb) | `tests/integration/test_agent_pattern_examples.py` | Agent-pattern examples translated from the Embabel repository. |
| [`hierarchical_agent_teams_goapified.ipynb`](hierarchical_agent_teams_goapified.ipynb) | `tests/integration/test_hierarchical_teams.py` | GOAPified version of LangGraph's Hierarchical Agent Teams tutorial. |
| [`plan_and_execute_goapified.ipynb`](plan_and_execute_goapified.ipynb) | `tests/integration/test_plan_and_execute.py` | GOAPified version of LangGraph's Plan-and-Execute tutorial. |
| [`nl_goal_tutorial.ipynb`](nl_goal_tutorial.ipynb) | `tests/integration/test_nl_goal_interpreter_api.py` | Earlier walkthrough of `GoalInterpreter`. Superseded by `examples/basics/nl_goal_interpreter.ipynb` and `hungry_agent.ipynb`. |
| [`deepagents_integration.ipynb`](deepagents_integration.ipynb) | `tests/integration/test_deepagents_integration.py` | Embedding GOAP as a LangChain tool or DeepAgents subagent with LLM-powered content analysis. |

### Upstream LangGraph reference notebooks

The three `langgraph_*_original.ipynb` files are unmodified copies of the
upstream LangGraph tutorials that the `*_goapified` notebooks above
translate into GOAP. They are kept alongside the goapified versions so
readers can compare the two side-by-side. They are **not** LangGOAP code
and do not run against LangGOAP APIs.

- `langgraph_adaptive_rag_original.ipynb`
- `langgraph_hierarchical_agent_teams_original.ipynb`
- `langgraph_plan_and_execute_original.ipynb`

## Shared helpers

Shared action specs, execute functions, and domain fixtures live in the
[`tutorial_examples/`](tutorial_examples/) package and are imported by
both the notebooks and the integration tests so notebook output always
reflects tested behaviour.
