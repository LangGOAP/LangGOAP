r"""Robot Navigation — Tier 1 primer adapted from unified-planning.

Reference: ``research/repos/unified-planning/docs/notebooks/01-basic-example.ipynb``
where a robot moves between ``NLOC`` locations connected in a chain,
starting at ``l0`` and seeking ``l9``.

LangGOAP translation
--------------------
The original uses parameterized fluents (``robot_at(l)``).  LangGOAP's
world state is a plain ``dict[str, Any]`` so we model each location as
its own boolean key: ``at_l0``, ``at_l1``, …, ``at_l9``.  A ``move``
action between two specific locations becomes its own ``ActionSpec``:

>>> ActionSpec(
...     name="move_l3_to_l4",
...     preconditions={"at_l3": True},
...     effects={"at_l3": False, "at_l4": True},
... )

The planner needs to see every edge as a distinct action — which is
exactly how classical GOAP expects the world to be described.  A\* still
finds the optimal plan even though the search space is larger than the
lifted unified-planning version.

Two scenarios are exposed:

``linear_corridor(n)`` — n+1 locations connected in a chain (``l0 → l1
→ … → ln``).  Single path, plan length = n.

``weighted_grid()`` — a small hand-crafted graph where the cheapest
path is non-obvious.  Demonstrates cost-driven planning.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec


def _location_key(name: str) -> str:
    return f"at_{name}"


def _move_execute(effects: dict[str, Any]) -> Any:
    """Build a tiny execute closure that applies the literal effects."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        return dict(effects)

    return execute


def linear_corridor_actions(n_segments: int) -> list[ActionSpec]:
    """Return ``n_segments`` move actions ``l0 → l1 → … → l{n_segments}``.

    Every edge has unit cost.  The planner must chain ``n_segments``
    moves in order to get from ``l0`` to ``l{n_segments}``.
    """
    actions: list[ActionSpec] = []
    for i in range(n_segments):
        src = f"l{i}"
        dst = f"l{i + 1}"
        effects = {_location_key(src): False, _location_key(dst): True}
        actions.append(
            ActionSpec(
                name=f"move_{src}_to_{dst}",
                preconditions={_location_key(src): True},
                effects=effects,
                cost=1.0,
                execute=_move_execute(effects),
            )
        )
    return actions


def linear_corridor_start(n_segments: int) -> dict[str, Any]:
    """Return a world state where the robot is at ``l0``."""
    state: dict[str, Any] = {_location_key(f"l{i}"): False for i in range(n_segments + 1)}
    state[_location_key("l0")] = True
    return state


# ---------------------------------------------------------------------------
# Weighted grid — demonstrates cost-driven planning
# ---------------------------------------------------------------------------

#
#                (4)
#         A ------------ D
#         |              |
#      (1)|           (1)|
#         |              |
#         B --(1)-- C --(1)-- E
#
# Two paths from A to E:
#   1. A → D → E   (cost 4 + 1 = 5)
#   2. A → B → C → E (cost 1 + 1 + 1 = 3)
#
# A\* prefers the longer-step path because total cost is lower.

_GRID_EDGES: tuple[tuple[str, str, float], ...] = (
    ("A", "D", 4.0),
    ("D", "A", 4.0),
    ("A", "B", 1.0),
    ("B", "A", 1.0),
    ("B", "C", 1.0),
    ("C", "B", 1.0),
    ("C", "E", 1.0),
    ("E", "C", 1.0),
    ("D", "E", 1.0),
    ("E", "D", 1.0),
)


def weighted_grid_actions() -> list[ActionSpec]:
    """Return the bidirectional grid action set."""
    actions: list[ActionSpec] = []
    for src, dst, cost in _GRID_EDGES:
        effects = {_location_key(src): False, _location_key(dst): True}
        actions.append(
            ActionSpec(
                name=f"move_{src}_to_{dst}",
                preconditions={_location_key(src): True},
                effects=effects,
                cost=cost,
                execute=_move_execute(effects),
            )
        )
    return actions


def weighted_grid_start(start_location: str = "A") -> dict[str, Any]:
    """Return a world state where the robot sits at ``start_location``."""
    nodes = ("A", "B", "C", "D", "E")
    if start_location not in nodes:
        raise ValueError(f"start_location must be one of {nodes}")
    state: dict[str, Any] = {_location_key(n): False for n in nodes}
    state[_location_key(start_location)] = True
    return state
