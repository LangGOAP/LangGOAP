"""Per-step cost evaluation shared by the renderers (issue #54).

Renderers used to call ``action.get_cost({})``: a state-dependent cost that
reads a key raised ``KeyError`` and the whole diagram failed.  The cost of
step *i* is evaluated against that step's pre-state (the expected state
after step ``i - 1``), and a cost that cannot be evaluated renders without
a badge instead of crashing.
"""

from __future__ import annotations

from typing import Any

from langgoap.planner.types import Plan


def step_costs(plan: Plan) -> list[float | None]:
    """Return each action's cost in its own pre-state, or ``None`` if unknown."""
    costs: list[float | None] = []
    for i, action in enumerate(plan.actions):
        pre_state: dict[str, Any] = {}
        if i > 0 and i - 1 < len(plan.expected_states):
            pre_state = plan.expected_states[i - 1].as_context()
        try:
            costs.append(float(action.get_cost(pre_state)))
        except Exception:
            costs.append(None)
    return costs


__all__ = ["step_costs"]
