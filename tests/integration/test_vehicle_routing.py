r"""Integration test for the Vehicle Routing tutorial (Tier 2).

Exercises the full A\* → CSP pipeline on a 2-vehicle / 4-customer
subset of a standard CVRP benchmark instance.  The CSP phase is
expected to:

- Enforce per-vehicle ``load_<v>`` hard capacity constraints
- Build a precedence chain per vehicle from shared ``<v>_at_<loc>``
  state and schedule the two chains in parallel
- Return a makespan equal to the longer route's total duration
- Mark plans INFEASIBLE when a capacity budget is tightened below
  the true demand

Helpers live in ``examples/tutorials/tutorial_examples/vehicle_routing.py``
and the instance fixture is at
``examples/tutorials/tutorial_examples/data/vehicle_routing_instance.py``
(derived from standard benchmark data).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tutorial_examples.data.vehicle_routing_instance import (
    CUSTOMERS,
    VEHICLES,
    travel_minutes,
)
from tutorial_examples.vehicle_routing import (
    vehicle_routing_actions,
    vehicle_routing_goal,
    vehicle_routing_start,
)

from langgoap import ConstraintSpec, CSPStatus, GoalSpec, GoapGraph
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.state import PlanningState


class TestVehicleRoutingFeasible:
    """Both vehicles fit their pre-assigned customers within capacity."""

    def test_pipeline_finds_optimal_routes(self) -> None:
        """A\\* picks the shortest route for every vehicle."""
        actions = vehicle_routing_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(vehicle_routing_start()),
            vehicle_routing_goal(),
            actions,
        )

        assert plan_obj is not None
        # The two symmetric optimal routes for v1 both cost 50, plus
        # v2's fixed 50 → total 100.
        assert plan_obj.total_cost == 100.0

        # Every customer must be visited exactly once by exactly one move.
        customer_visits = {c.name: 0 for c in CUSTOMERS}
        for action in plan_obj.actions:
            for c in CUSTOMERS:
                if action.name.endswith(f"_to_{c.name}"):
                    customer_visits[c.name] += 1
        assert all(count == 1 for count in customer_visits.values())

    def test_all_capacities_respected(self) -> None:
        """Per-vehicle load stays within the vehicle's capacity."""
        actions = vehicle_routing_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(vehicle_routing_start()),
            vehicle_routing_goal(),
            actions,
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)

        usage_by_key = {u.key: u.total for u in csp.resource_usage}
        for vehicle in VEHICLES:
            load_key = f"load_{vehicle.name}"
            assert (
                usage_by_key.get(load_key, 0.0) <= vehicle.capacity
            ), f"{vehicle.name} load exceeded capacity"

    def test_schedule_runs_vehicles_in_parallel(self) -> None:
        """Both vehicles start at t=0 and the makespan == longer route."""
        actions = vehicle_routing_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(vehicle_routing_start()),
            vehicle_routing_goal(),
            actions,
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.schedule, "CSP should produce a schedule from durations"

        # Every vehicle's first move (depot_to_*) must start at time 0.
        zero = timedelta()
        first_moves = [e for e in csp.schedule if "_depot_to_" in e.action_name]
        for entry in first_moves:
            assert entry.start == zero, (
                f"{entry.action_name} should start at t=0 " f"(actual: {entry.start})"
            )

        # The longer route determines the makespan.
        # v1 optimal path = depot→c2→c3→c4→depot,
        #                  each leg = travel + service(destination).
        # travel_minutes honors the Manhattan distance matrix.
        v1_leg_minutes = (
            travel_minutes("depot", "c2")
            + 15  # service c2
            + travel_minutes("c2", "c3")
            + 12  # service c3
            + travel_minutes("c3", "c4")
            + 8  # service c4
            + travel_minutes("c4", "depot")  # no service at depot
        )
        v2_leg_minutes = (
            travel_minutes("depot", "c5")
            + 10  # service c5
            + travel_minutes("c5", "depot")
        )
        expected_makespan_min = max(v1_leg_minutes, v2_leg_minutes)

        assert csp.makespan is not None
        assert csp.makespan == timedelta(minutes=expected_makespan_min)

    def test_graph_invocation_executes_plan(self) -> None:
        """End-to-end GoapGraph.invoke() reaches goal_achieved."""
        actions = vehicle_routing_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=vehicle_routing_goal(),
            world_state=vehicle_routing_start(),
        )
        assert result["status"] == "goal_achieved"

        ws = result["world_state"]
        for c in CUSTOMERS:
            assert ws[f"visited_{c.name}"] is True
        for v in VEHICLES:
            assert ws[f"{v.name}_at_depot"] is True


class TestVehicleRoutingInfeasible:
    """Tight capacity budget — CSP must mark the plan INFEASIBLE."""

    def test_impossible_capacity_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        """Dropping v1 capacity below its true demand yields INFEASIBLE."""
        actions = vehicle_routing_actions()
        # v1 must carry 26+17+6 = 49; cap at 40 makes it impossible.
        goal = GoalSpec(
            conditions={
                **{f"visited_{c.name}": True for c in CUSTOMERS},
                **{f"{v.name}_at_depot": True for v in VEHICLES},
            },
            constraints=(
                ConstraintSpec(key="load_v1", max=40.0, level="hard"),
                ConstraintSpec(key="load_v2", max=20.0, level="hard"),
            ),
        )

        plan_obj = pipeline_plan(
            PlanningState.from_dict(vehicle_routing_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.metadata.csp is not None
        assert plan_obj.metadata.csp.status == CSPStatus.INFEASIBLE

        # The reported load exceeds the hard cap.
        usage_by_key = {u.key: u.total for u in plan_obj.metadata.csp.resource_usage}
        assert usage_by_key["load_v1"] > 40.0
