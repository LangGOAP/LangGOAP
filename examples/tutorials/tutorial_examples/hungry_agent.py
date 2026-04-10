r"""Hungry Agent — Tier 1 primer for NL goal interpretation.

A classic GOAP demo: an agent is hungry and tired and needs to get back
to baseline.  Multiple actions satisfy "not hungry" with different costs,
and the starting state determines which one A* picks.

The notebook feeds a natural-language request ("I'm tired and hungry,
figure it out") to :class:`langgoap.GoalInterpreter`, which emits a
:class:`GoalSpec` with the two boolean conditions.  A* then selects the
cheapest combination of actions that satisfies both.

Action catalog
--------------

+----------------+----------------------+-------------------------------+------+
| Action         | Preconditions        | Effects                       | Cost |
+================+======================+===============================+======+
| eat_snack      | has_snack=True       | hungry=False, has_snack=False | 1.0  |
+----------------+----------------------+-------------------------------+------+
| cook_meal      | has_ingredients=True | hungry=False,                 | 3.0  |
|                |                      | has_ingredients=False         |      |
+----------------+----------------------+-------------------------------+------+
| order_delivery | (none)               | hungry=False                  | 5.0  |
+----------------+----------------------+-------------------------------+------+
| sleep          | (none)               | tired=False                   | 1.0  |
+----------------+----------------------+-------------------------------+------+

Starting state shapes the plan
------------------------------

- With ``has_snack=True``: the cheapest plan is ``eat_snack + sleep``
  (cost 2.0).
- With ``has_snack=False`` but ``has_ingredients=True``: A* falls back
  to ``cook_meal + sleep`` (cost 4.0).
- With neither snack nor ingredients: the only path is
  ``order_delivery + sleep`` (cost 6.0).

The same natural-language goal produces all three plans — the planner
adapts to the world state, not the request.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec


def _eat_snack_execute(ws: dict[str, Any]) -> dict[str, Any]:
    return {"hungry": False, "has_snack": False}


def _cook_meal_execute(ws: dict[str, Any]) -> dict[str, Any]:
    return {"hungry": False, "has_ingredients": False}


def _order_delivery_execute(ws: dict[str, Any]) -> dict[str, Any]:
    return {"hungry": False}


def _sleep_execute(ws: dict[str, Any]) -> dict[str, Any]:
    return {"tired": False}


def hungry_agent_actions() -> list[ActionSpec]:
    """Return the full action catalog for the hungry agent domain."""
    return [
        ActionSpec(
            name="eat_snack",
            preconditions={"has_snack": True},
            effects={"hungry": False, "has_snack": False},
            cost=1.0,
            execute=_eat_snack_execute,
        ),
        ActionSpec(
            name="cook_meal",
            preconditions={"has_ingredients": True},
            effects={"hungry": False, "has_ingredients": False},
            cost=3.0,
            execute=_cook_meal_execute,
        ),
        ActionSpec(
            name="order_delivery",
            preconditions={},
            effects={"hungry": False},
            cost=5.0,
            execute=_order_delivery_execute,
        ),
        ActionSpec(
            name="sleep",
            preconditions={},
            effects={"tired": False},
            cost=1.0,
            execute=_sleep_execute,
        ),
    ]


def hungry_agent_start(
    *,
    has_snack: bool = False,
    has_ingredients: bool = False,
) -> dict[str, Any]:
    """Return a starting state where the agent is hungry and tired.

    ``has_snack`` and ``has_ingredients`` control which eating actions
    are available, which in turn shapes the A* solution.
    """
    return {
        "hungry": True,
        "tired": True,
        "has_snack": has_snack,
        "has_ingredients": has_ingredients,
    }
