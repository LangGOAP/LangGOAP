# Tutorial notebooks

End-to-end walkthroughs that show LangGOAP solving real problems. Each
featured notebook is paired with a passing integration test under
[`tests/integration/`](../../tests/integration/) — the notebook is the
explanation; the test is the source of truth.

Shared action specs, fixtures, and domain helpers live in the
[`tutorial_examples/`](tutorial_examples/) package and are imported by
both the notebooks and the integration tests, so notebook output always
reflects tested behaviour.

## Featured catalog (22 tutorials)

### Tier 1 — Primers

Start here if you have not used GOAP before.

1. [`directory_handler.ipynb`](directory_handler.ipynb) — Minimal GOAP loop over a file-system world state. Adapted from GOApy.
2. [`robot_navigation.ipynb`](robot_navigation.ipynb) — A* pathfinding primer on a linear graph. Adapted from unified-planning.
3. [`hungry_agent.ipynb`](hungry_agent.ipynb) — Natural-language goal intake driving cost-based action selection with `GoalInterpreter`.
4. [`from_routing_graphs_to_goap.ipynb`](from_routing_graphs_to_goap.ipynb) — Side-by-side: a hand-wired LangGraph routing graph rewritten as a GOAP plan that absorbs runtime disruptions. Backed by [`test_screencast_incident.py`](../../tests/integration/test_screencast_incident.py).

### Tier 2 — Workflows and constraint optimization

Substantial problems where GOAP replaces a hand-written routing graph.

5. [`cloud_balancing.ipynb`](cloud_balancing.ipynb) — VM bin-packing under CSP resource constraints.
6. [`vehicle_routing.ipynb`](vehicle_routing.ipynb) — Capacity-constrained delivery routing with temporal windows and a Gantt chart.
7. [`nurse_rostering.ipynb`](nurse_rostering.ipynb) — Shift assignment with `HardSoftScore` and skill matching.
8. [`project_job_scheduling.ipynb`](project_job_scheduling.ipynb) — RCPSP with precedence, resource constraints, and makespan minimization.
9. [`task_assigning.ipynb`](task_assigning.ipynb) — Multi-objective CSP authored with the fluent `ConstraintBuilder`.
10. [`sql_query_agent.ipynb`](sql_query_agent.ipynb) — Schema-aware SQL pipeline with `effect_validator` and replanning on test failure.
11. [`vulnerability_scanner.ipynb`](vulnerability_scanner.ipynb) — Phased sensor-driven planning with action blacklisting.
12. [`cost_bounded_research_agent.ipynb`](cost_bounded_research_agent.ipynb) — Token / USD budget enforced live by `CostAccumulator` + `MaxCostPolicy`. Backed by [`test_early_termination.py`](../../tests/integration/test_early_termination.py).
13. [`personal_shopper_agent.ipynb`](personal_shopper_agent.ipynb) — Utility-maximizing strategy (`UtilityStrategy`) selecting among competing offers. Backed by [`test_utility_planner_loop.py`](../../tests/integration/test_utility_planner_loop.py).
14. [`scheduled_delivery_confirmer.ipynb`](scheduled_delivery_confirmer.ipynb) — Typed-form human-in-the-loop with Pydantic models via `interrupt()`. Backed by [`test_form_hitl.py`](../../tests/integration/test_form_hitl.py).
15. [`deepagents_integration.ipynb`](deepagents_integration.ipynb) — Embed a `GoapGraph` as a `StructuredTool` or Deep Agents subagent with `create_goap_tool` / `create_goap_subagent`. Backed by [`test_deepagents_integration.py`](../../tests/integration/test_deepagents_integration.py).

### Tier 3 — Full-stack showcases

Every primitive in one notebook.

16. [`deep_research_agent.ipynb`](deep_research_agent.ipynb) — LLM-heavy research loop with `create_goap_agent`, `StoreExecutionHistory`, and `LangSmithTracer`.
17. [`hierarchical_product_launch.ipynb`](hierarchical_product_launch.ipynb) — `MultiGoal` sequential decomposition across teams.
18. [`content_builder_agent.ipynb`](content_builder_agent.ipynb) — Multi-format content generation with multi-objective CSP.
19. [`temporal_match_cellar.ipynb`](temporal_match_cellar.ipynb) — Durative actions with overlap constraints and Gantt rendering.
20. [`flexible_job_shop.ipynb`](flexible_job_shop.ipynb) — Full feature showcase: NL intake, A*→CSP, hard/soft scoring, `MultiGoal`, tracing, visualization, execution history, and parallel fan-out via LangGraph `Send`.
21. [`supply_chain_disruption_mediator.ipynb`](supply_chain_disruption_mediator.ipynb) — Stuck handlers (`MulticastStuckHandler`) and goal relaxation when the planner runs out of options. Backed by [`test_stuck_handler_loop.py`](../../tests/integration/test_stuck_handler_loop.py).
22. [`code_review_agent_mcp_deployment.ipynb`](code_review_agent_mcp_deployment.ipynb) — Scaffold a `langgraph dev`-ready deployment with `langgoap deploy-init`; the generated `/mcp` endpoint makes the GOAP graph callable from any MCP client. Backed by [`test_langgraph_deploy_scaffold.py`](../../tests/integration/test_langgraph_deploy_scaffold.py).

## Comparison notebooks

The `*_goapified` notebooks below port well-known LangGraph and Embabel
tutorials onto LangGOAP so the two implementations can be compared
side-by-side. Each has a passing integration test, but they are
secondary to the featured catalog above.

| Notebook | Integration test | Topic |
| --- | --- | --- |
| [`adaptive_rag_goapified.ipynb`](adaptive_rag_goapified.ipynb) | `tests/integration/test_adaptive_rag.py` | LangGraph Adaptive RAG ported to GOAP. |
| [`agent_pattern_examples_goapified.ipynb`](agent_pattern_examples_goapified.ipynb) | `tests/integration/test_agent_pattern_examples.py` | Embabel agent-pattern examples ported to GOAP. |
| [`hierarchical_agent_teams_goapified.ipynb`](hierarchical_agent_teams_goapified.ipynb) | `tests/integration/test_hierarchical_teams.py` | LangGraph Hierarchical Agent Teams ported to GOAP. |
| [`plan_and_execute_goapified.ipynb`](plan_and_execute_goapified.ipynb) | `tests/integration/test_plan_and_execute.py` | LangGraph Plan-and-Execute ported to GOAP. |
| [`nl_goal_tutorial.ipynb`](nl_goal_tutorial.ipynb) | `tests/integration/test_nl_goal_interpreter_api.py` | Earlier `GoalInterpreter` walkthrough. Superseded by `examples/basics/nl_goal_interpreter.ipynb` and `hungry_agent.ipynb`. |
