"""Cloud Balancing — the classic cloud-balancing bin-packing problem in LangGOAP.

Models the cloud-balancing assignment problem as a GOAP problem: every
process must be assigned to exactly one computer, no computer may
exceed its cpu / memory / network capacity, and the total USD cost
of the assignment should be minimized.

GOAP modelling
--------------

Each ``(process, computer)`` pair becomes its own ``ActionSpec``:

- **name**: ``assign_<process>_to_<computer>``
- **preconditions**: ``assigned_<process>=False``
- **effects**: ``assigned_<process>=True``
- **resources**:
  - ``cpu_<computer>``  — process cpu demand, consumed on that server
  - ``mem_<computer>``  — process memory demand
  - ``net_<computer>``  — process network demand
  - ``cost_usd``        — server's amortized per-process hourly rate

Hard constraints on the goal cap each per-server resource at the
computer's capacity.  A single soft ``cost_usd`` objective drives
the CSP optimizer toward cheaper assignments (server_small preferred
over server_big when capacity allows).

Two entry points are exposed: :func:`cloud_balancing_actions` builds
the action catalog from the instance, and :func:`cloud_balancing_goal`
builds the corresponding :class:`GoalSpec` with all capacity
constraints plus the cost objective.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, ConstraintSpec, GoalSpec
from langgoap.goals import ObjectiveDirection

from .data.cloud_balancing_instance import PROCESSES, SERVERS, Computer, Process


def _assigned_key(process_name: str) -> str:
    return f"assigned_{process_name}"


def _make_execute(effects: dict[str, Any]) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        return dict(effects)

    return execute


def _assign_action(process: Process, computer: Computer) -> ActionSpec:
    effects = {_assigned_key(process.name): True}
    resources = {
        f"cpu_{computer.name}": float(process.cpu),
        f"mem_{computer.name}": float(process.memory),
        f"net_{computer.name}": float(process.network),
        "cost_usd": computer.cost_usd_per_proc,
    }
    return ActionSpec(
        name=f"assign_{process.name}_to_{computer.name}",
        preconditions={_assigned_key(process.name): False},
        effects=effects,
        cost=1.0,
        resources=resources,
        execute=_make_execute(effects),
    )


def cloud_balancing_actions(
    processes: tuple[Process, ...] = PROCESSES,
    servers: tuple[Computer, ...] = SERVERS,
) -> list[ActionSpec]:
    """Return one ``assign_<process>_to_<computer>`` ActionSpec per pair."""
    return [_assign_action(p, c) for p in processes for c in servers]


def cloud_balancing_start(
    processes: tuple[Process, ...] = PROCESSES,
) -> dict[str, Any]:
    """All processes start unassigned."""
    return {_assigned_key(p.name): False for p in processes}


def cloud_balancing_goal(
    processes: tuple[Process, ...] = PROCESSES,
    servers: tuple[Computer, ...] = SERVERS,
    *,
    minimize_cost: bool = True,
) -> GoalSpec:
    """Goal: every process assigned; hard capacity limits per server.

    When ``minimize_cost`` is True (the default) the goal also carries a
    ``cost_usd → MINIMIZE`` objective so the CSP optimizer prefers
    cheaper assignments.
    """
    conditions = {_assigned_key(p.name): True for p in processes}
    constraints = tuple(
        ConstraintSpec(
            key=f"cpu_{c.name}",
            max=float(c.cpu),
            level="hard",
        )
        for c in servers
    ) + tuple(
        ConstraintSpec(
            key=f"mem_{c.name}",
            max=float(c.memory),
            level="hard",
        )
        for c in servers
    ) + tuple(
        ConstraintSpec(
            key=f"net_{c.name}",
            max=float(c.network),
            level="hard",
        )
        for c in servers
    )
    objectives: dict[str, ObjectiveDirection] | None = None
    if minimize_cost:
        objectives = {"cost_usd": ObjectiveDirection.MINIMIZE}
    return GoalSpec(
        conditions=conditions,
        constraints=constraints,
        objectives=objectives,
    )
