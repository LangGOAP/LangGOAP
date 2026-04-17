"""Integration test for the Cloud Balancing tutorial (Tier 2).

Exercises the full A* → CSP pipeline on a benchmark-derived
bin-packing instance.  The CSP phase is expected to:

- Aggregate per-server cpu/mem/net usage across the chosen plan
- Enforce hard capacity constraints
- Minimize ``cost_usd`` when multiple alternatives are available
- Mark plans as INFEASIBLE when a too-tight budget is added

Helpers live in ``examples/tutorials/tutorial_examples/cloud_balancing.py``
and the instance fixture is at
``examples/tutorials/tutorial_examples/data/cloud_balancing_instance.py``
(derived from standard benchmark data).
"""

from __future__ import annotations

import pytest
from tutorial_examples.cloud_balancing import (
    cloud_balancing_actions,
    cloud_balancing_goal,
    cloud_balancing_start,
)
from tutorial_examples.data.cloud_balancing_instance import PROCESSES, SERVERS

from langgoap import ConstraintSpec, CSPStatus, GoalSpec, GoapGraph
from langgoap.goals import ObjectiveDirection
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.state import PlanningState


class TestCloudBalancingFeasible:
    """Feasible instance: all 4 processes fit across 2 servers."""

    def test_pipeline_finds_feasible_assignment(self) -> None:
        """A* + CSP produces a complete feasible assignment."""
        actions = cloud_balancing_actions()
        start = cloud_balancing_start()
        goal = cloud_balancing_goal()

        plan_obj = pipeline_plan(
            PlanningState.from_dict(start),
            goal,
            actions,
        )

        assert plan_obj is not None, "pipeline must return a plan"
        assert plan_obj.metadata.csp is not None, "CSP metadata required"
        assert plan_obj.metadata.csp.status in (
            CSPStatus.FEASIBLE,
            CSPStatus.OPTIMAL,
        )
        # Every process must be assigned exactly once.
        assigned_processes = {
            a.name.split("_")[1]  # "assign_p0_to_server_big" -> "p0"
            for a in plan_obj.actions
        }
        assert assigned_processes == {p.name for p in PROCESSES}

    def test_all_capacities_respected(self) -> None:
        """No server exceeds its cpu/mem/net capacity."""
        actions = cloud_balancing_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(cloud_balancing_start()),
            cloud_balancing_goal(),
            actions,
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None

        usage_by_key = {u.key: u.total for u in csp.resource_usage}
        for server in SERVERS:
            assert (
                usage_by_key.get(f"cpu_{server.name}", 0.0) <= server.cpu
            ), f"cpu overflow on {server.name}"
            assert (
                usage_by_key.get(f"mem_{server.name}", 0.0) <= server.memory
            ), f"memory overflow on {server.name}"
            assert (
                usage_by_key.get(f"net_{server.name}", 0.0) <= server.network
            ), f"network overflow on {server.name}"

    def test_graph_invocation_executes_plan(self) -> None:
        """End-to-end GoapGraph.invoke() reaches goal_achieved."""
        actions = cloud_balancing_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=cloud_balancing_goal(),
            world_state=cloud_balancing_start(),
        )
        assert result["status"] == "goal_achieved"
        for p in PROCESSES:
            assert result["world_state"][f"assigned_{p.name}"] is True


class TestCloudBalancingForcedAssignment:
    """Process p1 has mem=6 which cannot fit on server_small (mem=4)."""

    def test_p1_forced_to_server_big(self) -> None:
        """The only feasible placement for p1 is server_big."""
        actions = cloud_balancing_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(cloud_balancing_start()),
            cloud_balancing_goal(),
            actions,
        )
        assert plan_obj is not None

        p1_action = next(a for a in plan_obj.actions if "_p1_" in a.name)
        assert p1_action.name == "assign_p1_to_server_big"


class TestCloudBalancingInfeasible:
    """Tight cost budget → CSP must mark the plan INFEASIBLE."""

    def test_impossible_cost_budget(self) -> None:
        """An unreachable cost cap yields INFEASIBLE CSP status."""
        actions = cloud_balancing_actions()
        goal = GoalSpec(
            conditions={f"assigned_{p.name}": True for p in PROCESSES},
            constraints=(
                # Cheapest possible assignment still exceeds $100 — four processes
                # must all be placed, minimum is 3 on server_small (3*66=198) +
                # 1 on server_big (480) = 678.
                ConstraintSpec(key="cost_usd", max=100.0, level="hard"),
            ),
            objectives={"cost_usd": ObjectiveDirection.MINIMIZE},
        )
        plan_obj = pipeline_plan(
            PlanningState.from_dict(cloud_balancing_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.metadata.csp is not None
        assert plan_obj.metadata.csp.status == CSPStatus.INFEASIBLE
