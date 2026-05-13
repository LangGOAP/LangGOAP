# Examples

Every example below is a runnable Jupyter notebook committed to the
repository under `examples/`.

## Flagship screencast

- `screencast/research_agent/screencast.ipynb` — four-way head-to-head (`create_react_agent` baseline vs. hand-wired `StateGraph` vs. LangGOAP vs. LangGOAP-under-disruption) with real OpenAI + Tavily costs and Tavily-key revocation as the climax.

## Basics

Short primers — each notebook exercises a single mechanic.

- `plan_visualization.ipynb` — Mermaid, DOT, ASCII, and Gantt renderers over an A* → CSP plan.
- `nl_goal_interpreter.ipynb` — Plain-English request → `GoalSpec` via any `BaseChatModel`.
- `create_goap_agent_quickstart.ipynb` — The Layer A one-liner: LangChain tools + goal → compiled `StateGraph`.
- `termination_policies.ipynb` — `MaxCostPolicy`, `MaxActionsPolicy`, and `FirstOfPolicy` composition.
- `stuck_handlers.ipynb` — `FunctionalStuckHandler` and `MulticastStuckHandler` recovering planning failures.
- `typed_form_hitl.ipynb` — Typed-form `interrupt()` with Pydantic validation on resume.
- `action_qos.ipynb` — In-executor retry with `ActionQos` and exponential backoff.

## Tutorials

Complete end-to-end walkthroughs across three tiers of complexity.

**Tier 1 — Primers**

1. `directory_handler.ipynb` — GOAP basics on a file-system world state.
2. `robot_navigation.ipynb` — A* primer on a linear graph.
3. `hungry_agent.ipynb` — Natural-language goal + cost-driven action selection.
4. `from_routing_graphs_to_goap.ipynb` — Hand-wired LangGraph routing graph rewritten as a GOAP plan that absorbs runtime disruptions.

**Tier 2 — Workflows and constraint optimization**

5. `cloud_balancing.ipynb` — VM bin-packing with the `create_goap_agent` one-liner.
6. `vehicle_routing.ipynb` — Capacity-constrained routing with CSP temporal scheduling.
7. `nurse_rostering.ipynb` — Shift assignment with `HardSoftScore` and skill matching.
8. `project_job_scheduling.ipynb` — RCPSP with precedence constraints and makespan minimization.
9. `task_assigning.ipynb` — Ticket routing with the fluent `ConstraintBuilder`.
10. `sql_query_agent.ipynb` — Schema-explore → generate → test → refine loop using `effect_validator`.
11. `vulnerability_scanner.ipynb` — Phased discovery with action blacklisting.
12. `cost_bounded_research_agent.ipynb` — Token / USD budget enforced via `MaxCostPolicy`.
13. `personal_shopper_agent.ipynb` — Utility-maximizing planning with `UtilityStrategy` and `NirvanaGoal`.
14. `scheduled_delivery_confirmer.ipynb` — Typed-form HITL with Pydantic validation.
15. `deepagents_integration.ipynb` — Embed a `GoapGraph` as a `StructuredTool` or Deep Agents subagent.

**Tier 3 — Full-stack showcases**

16. `deep_research_agent.ipynb` — LLM-heavy research loop with `StoreExecutionHistory` and tracing.
17. `hierarchical_product_launch.ipynb` — `MultiGoal` sequential decomposition.
18. `content_builder_agent.ipynb` — Multi-objective CSP with conditional format generation.
19. `temporal_match_cellar.ipynb` — Durative actions with overlap constraints and parallel scheduling.
20. `flexible_job_shop.ipynb` — Every v0.1.0 feature in one notebook.
21. `supply_chain_disruption_mediator.ipynb` — Stuck handlers and goal relaxation under cascading disruption.
22. `code_review_agent_mcp_deployment.ipynb` — `langgoap deploy-init` scaffolds a `langgraph dev`-ready deployment with an `/mcp` endpoint.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Flagship screencast

screencast/research_agent/screencast
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Basics

basics/plan_visualization
basics/nl_goal_interpreter
basics/create_goap_agent_quickstart
basics/termination_policies
basics/stuck_handlers
basics/typed_form_hitl
basics/action_qos
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tier 1 — Primers

tutorials/directory_handler
tutorials/robot_navigation
tutorials/hungry_agent
tutorials/from_routing_graphs_to_goap
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tier 2 — Workflows and constraint optimization

tutorials/cloud_balancing
tutorials/vehicle_routing
tutorials/nurse_rostering
tutorials/project_job_scheduling
tutorials/task_assigning
tutorials/sql_query_agent
tutorials/vulnerability_scanner
tutorials/cost_bounded_research_agent
tutorials/personal_shopper_agent
tutorials/scheduled_delivery_confirmer
tutorials/deepagents_integration
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tier 3 — Full-stack showcases

tutorials/deep_research_agent
tutorials/hierarchical_product_launch
tutorials/content_builder_agent
tutorials/temporal_match_cellar
tutorials/flexible_job_shop
tutorials/supply_chain_disruption_mediator
tutorials/code_review_agent_mcp_deployment
```
