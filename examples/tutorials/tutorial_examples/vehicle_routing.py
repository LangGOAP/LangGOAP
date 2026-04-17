r"""Vehicle Routing — the Capacitated Vehicle Routing Problem (CVRP) in LangGoap.

Models a 2-vehicle / 4-customer CVRP subset as GOAP.  Each vehicle
has a pre-assigned set of customers (clustering happens outside the
planner in this tutorial); A\* searches the ordering within each
vehicle's route, and the CSP phase schedules the two routes in
parallel and renders them as a Gantt.

GOAP modelling
--------------

**World state:**

- ``v<m>_at_<location>`` — per-vehicle position flag.  Exactly one
  location is True at any time; ``depot`` starts True.
- ``visited_<customer>`` — terminal flag flipped to True by the
  move that arrives at a customer.

**Actions:** per vehicle ``m``, one ``move_v<m>_<from>_to_<to>``
action for every ordered pair of legal locations (depot + the
vehicle's assigned customers).  Each move:

- **preconditions**: ``v<m>_at_<from>=True``
- **effects**: ``v<m>_at_<from>=False``, ``v<m>_at_<to>=True``;
  when ``<to>`` is a customer, also ``visited_<to>=True``.
- **cost**: Manhattan distance between the two locations.
- **duration**: ``travel_minutes(from, to) + service_minutes(to)``.
- **resources**:

  - ``load_v<m>``: customer demand at ``<to>`` (zero when
    returning to the depot) — aggregated per vehicle.
  - ``distance_v<m>``: distance of the move — aggregated per
    vehicle for reporting.

**Goal:** every customer visited and every vehicle back at the
depot.  Hard capacity constraints cap ``load_v<m>`` at the
vehicle's carrying capacity; a soft ``distance`` objective is
available via :func:`vehicle_routing_goal` with
``minimize_distance=True`` (the default).

Why per-vehicle position state?
-------------------------------
The position flag couples consecutive moves inside a single
vehicle's route via the dependency graph the CSP scheduler reads
(see ``build_dependency_graph`` in ``planner/csp.py``).  Moves for
different vehicles are independent, so CP-SAT schedules them in
parallel and the resulting makespan equals the longer route's
duration.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from langgoap import ActionSpec, ConstraintSpec, GoalSpec
from langgoap.goals import ObjectiveDirection

from .data.vehicle_routing_instance import (
    CUSTOMERS,
    VEHICLES,
    Customer,
    Vehicle,
    distance,
    travel_minutes,
)


def _at_key(vehicle: str, location: str) -> str:
    return f"{vehicle}_at_{location}"


def _visited_key(customer: str) -> str:
    return f"visited_{customer}"


def _make_execute(effects: dict[str, Any]) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        return dict(effects)

    return execute


def _move_action(
    vehicle: Vehicle,
    from_loc: str,
    to_loc: str,
    customers_by_name: dict[str, Customer],
) -> ActionSpec:
    preconditions = {_at_key(vehicle.name, from_loc): True}
    effects: dict[str, Any] = {
        _at_key(vehicle.name, from_loc): False,
        _at_key(vehicle.name, to_loc): True,
    }
    # Arriving at a customer marks it visited.
    if to_loc in customers_by_name:
        effects[_visited_key(to_loc)] = True

    # Load only consumed when delivering at a customer (not on return).
    demand = 0
    service = 0
    if to_loc in customers_by_name:
        demand = customers_by_name[to_loc].demand
        service = customers_by_name[to_loc].service_minutes

    dist = distance(from_loc, to_loc)
    duration_min = travel_minutes(from_loc, to_loc) + service

    resources: dict[str, float] = {
        f"distance_{vehicle.name}": float(dist),
    }
    if demand > 0:
        resources[f"load_{vehicle.name}"] = float(demand)

    return ActionSpec(
        name=f"move_{vehicle.name}_{from_loc}_to_{to_loc}",
        preconditions=preconditions,
        effects=effects,
        cost=float(dist),
        resources=resources,
        duration=timedelta(minutes=duration_min),
        execute=_make_execute(effects),
    )


def vehicle_routing_actions(
    vehicles: tuple[Vehicle, ...] = VEHICLES,
    customers: tuple[Customer, ...] = CUSTOMERS,
) -> list[ActionSpec]:
    """Return the move action catalog for the whole fleet.

    Each vehicle gets moves between ``depot`` and every customer
    pre-assigned to it, in both directions and between customers.
    """
    by_vehicle: dict[str, list[Customer]] = {v.name: [] for v in vehicles}
    for c in customers:
        by_vehicle[c.assigned_to].append(c)

    customers_by_name = {c.name: c for c in customers}

    actions: list[ActionSpec] = []
    for vehicle in vehicles:
        legal_locations = ["depot"] + [c.name for c in by_vehicle[vehicle.name]]
        for from_loc in legal_locations:
            for to_loc in legal_locations:
                if from_loc == to_loc:
                    continue
                # No depot → depot moves; no depot detours after return.
                actions.append(
                    _move_action(vehicle, from_loc, to_loc, customers_by_name)
                )
    return actions


def vehicle_routing_start(
    vehicles: tuple[Vehicle, ...] = VEHICLES,
    customers: tuple[Customer, ...] = CUSTOMERS,
) -> dict[str, Any]:
    """Every vehicle parked at depot; no customer visited yet."""
    state: dict[str, Any] = {}
    for v in vehicles:
        state[_at_key(v.name, "depot")] = True
        for c in customers:
            if c.assigned_to == v.name:
                state[_at_key(v.name, c.name)] = False
    for c in customers:
        state[_visited_key(c.name)] = False
    return state


def vehicle_routing_goal(
    vehicles: tuple[Vehicle, ...] = VEHICLES,
    customers: tuple[Customer, ...] = CUSTOMERS,
    *,
    minimize_distance: bool = True,
) -> GoalSpec:
    """Every customer visited, every vehicle back at depot, capacity enforced.

    When ``minimize_distance`` is True the goal also carries a
    ``distance_<v>`` minimize objective per vehicle so the CSP
    optimizer prefers shorter routes when multiple are feasible.
    """
    conditions: dict[str, Any] = {
        _visited_key(c.name): True for c in customers
    }
    for v in vehicles:
        conditions[_at_key(v.name, "depot")] = True

    # Hard capacity constraint per vehicle.
    constraints = tuple(
        ConstraintSpec(
            key=f"load_{v.name}",
            max=float(v.capacity),
            level="hard",
        )
        for v in vehicles
    )

    objectives: dict[str, ObjectiveDirection] | None = None
    if minimize_distance:
        objectives = {
            f"distance_{v.name}": ObjectiveDirection.MINIMIZE for v in vehicles
        }

    return GoalSpec(
        conditions=conditions,
        constraints=constraints,
        objectives=objectives,
    )
