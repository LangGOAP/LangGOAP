"""Vehicle routing instance derived from standard CVRP benchmark data.

Provenance
----------
Derived from the CVRPLIB ``A-n32-k5`` benchmark instance.

The original instance has 32 customers and 5 vehicles.  This fixture
keeps the first 4 customers (c2, c3, c4, c5) and 2 vehicles so the
planner and CP-SAT scheduler solve in under a second while still
exercising:

- **Per-vehicle capacity constraints** — ``load_<vehicle>`` resource
  accumulates customer demand and must stay within the vehicle's
  carrying capacity.
- **Route optimization by A\\*** — the planner chooses the ordering
  of customer visits that minimizes total travel distance.
- **Parallel schedule visualization** — each vehicle's route becomes
  a dependency chain (location flows forward in space) and CP-SAT
  schedules both chains in parallel, yielding a real two-lane Gantt.

The full benchmark instance supports time windows (``cvrptw-*.vrp``
fixtures).  LangGOAP's CSP scheduler currently optimizes precedence
+ duration + makespan only, so time windows are deliberately out of
scope for this tutorial.  The demands and capacities below are
proportional to the original CVRP values.

Layout (symbolic Manhattan grid)
--------------------------------
::

             c3 (10, 10)
              |
              |
    depot----c2 (10, 0)       c5 (20, 5)
     (0,0)                  (assigned to v2)
              \\
               c4 (5, 15)

Distances used by the planner are Manhattan distances on this grid.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vehicle:
    name: str
    capacity: int


@dataclass(frozen=True)
class Customer:
    name: str
    demand: int
    service_minutes: int
    assigned_to: str  # vehicle name — pre-assigned in this tutorial


@dataclass(frozen=True)
class Location:
    name: str
    x: int
    y: int


# ---------------------------------------------------------------------------
# 2-vehicle / 4-customer instance
# ---------------------------------------------------------------------------

VEHICLES: tuple[Vehicle, ...] = (
    Vehicle(name="v1", capacity=50),
    Vehicle(name="v2", capacity=20),
)


CUSTOMERS: tuple[Customer, ...] = (
    # v1 carries 26+17+6 = 49 ≤ capacity 50 (tight but feasible).
    Customer(name="c2", demand=26, service_minutes=15, assigned_to="v1"),
    Customer(name="c3", demand=17, service_minutes=12, assigned_to="v1"),
    Customer(name="c4", demand=6, service_minutes=8, assigned_to="v1"),
    # v2 carries 15 ≤ capacity 20.
    Customer(name="c5", demand=15, service_minutes=10, assigned_to="v2"),
)


# Symbolic grid layout.  Manhattan distances drive travel cost and time.
LOCATIONS: tuple[Location, ...] = (
    Location(name="depot", x=0, y=0),
    Location(name="c2", x=10, y=0),
    Location(name="c3", x=10, y=10),
    Location(name="c4", x=5, y=15),
    Location(name="c5", x=20, y=5),
)


# Fast lookup by name.
_LOC_BY_NAME: dict[str, Location] = {loc.name: loc for loc in LOCATIONS}


def distance(a: str, b: str) -> int:
    """Manhattan distance between two named locations."""
    la, lb = _LOC_BY_NAME[a], _LOC_BY_NAME[b]
    return abs(la.x - lb.x) + abs(la.y - lb.y)


def travel_minutes(a: str, b: str) -> int:
    """Travel time = distance (1 grid unit per minute)."""
    return distance(a, b)
