"""Plan infeasibility explanation via greedy MUS approximation.

When the CSP returns ``INFEASIBLE``, this module identifies the minimal
set of conflicting constraints (Minimal Unsatisfiable Subset — MUS) and
produces a human-readable explanation with resource shortfall details
and actionable suggestions.

The greedy MUS algorithm iteratively removes constraints and checks if
the remaining set is still infeasible.  For v0.1.x, this O(n) approach
suffices because constraint counts are typically small (< 10).  A full
QuickXplain (O(n log n) but optimal) is a future enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.planner.csp import (
    CSPMetadata,
    CSPStatus,
    ResourceUsage,
    compute_resource_totals,
)

if TYPE_CHECKING:
    from langgoap.actions import ActionSpec
    from langgoap.planner.types import Plan
    from langgoap.state import PlanningState


@dataclass(frozen=True, slots=True)
class ResourceShortfall:
    """A single resource that exceeds its constraint.

    Attributes:
        key: Resource identifier.
        required: Total consumption in the plan.
        available_max: Upper bound from the constraint (or None if the
            violation is a lower-bound violation).
        available_min: Lower bound from the constraint (or None if the
            violation is an upper-bound violation).
        overrun: Amount by which the constraint is violated. Always >= 0.
    """

    key: str
    required: float
    available_max: float | None = None
    available_min: float | None = None
    overrun: float = 0.0


@dataclass(frozen=True, slots=True)
class InfeasibilityExplanation:
    """Explanation of why a plan is infeasible.

    Attributes:
        conflicting_constraints: The Minimal Unsatisfiable Subset of
            constraints that together cause infeasibility.
        resource_shortfalls: Per-resource breakdown of violations.
        suggestion: Human-readable suggestion for resolving the conflict.
    """

    conflicting_constraints: tuple[ConstraintSpec, ...]
    resource_shortfalls: tuple[ResourceShortfall, ...]
    suggestion: str


def _check_feasibility(
    totals: dict[str, float],
    constraints: list[ConstraintSpec],
) -> bool:
    """Return True if all hard constraints are satisfied given totals."""
    for c in constraints:
        if c.level != "hard":
            continue
        total = totals.get(c.key, 0.0)
        if c.max is not None and total > c.max:
            return False
        if c.min is not None and total < c.min:
            return False
    return True


def _build_shortfalls(
    totals: dict[str, float],
    constraints: tuple[ConstraintSpec, ...],
) -> tuple[ResourceShortfall, ...]:
    """Build shortfall records for violated constraints."""
    shortfalls: list[ResourceShortfall] = []
    for c in constraints:
        total = totals.get(c.key, 0.0)
        if c.max is not None and total > c.max:
            shortfalls.append(
                ResourceShortfall(
                    key=c.key,
                    required=total,
                    available_max=c.max,
                    overrun=total - c.max,
                )
            )
        elif c.min is not None and total < c.min:
            shortfalls.append(
                ResourceShortfall(
                    key=c.key,
                    required=total,
                    available_min=c.min,
                    overrun=c.min - total,
                )
            )
    return tuple(shortfalls)


def _build_suggestion(
    shortfalls: tuple[ResourceShortfall, ...],
    mus: tuple[ConstraintSpec, ...],
) -> str:
    """Generate a human-readable suggestion from shortfalls and MUS."""
    if not shortfalls and not mus:
        return "No constraint violations detected."

    parts: list[str] = []
    for sf in shortfalls:
        if sf.available_max is not None:
            parts.append(
                f"Resource '{sf.key}' uses {sf.required:.4g} but the hard "
                f"limit is {sf.available_max:.4g} (overrun: {sf.overrun:.4g}). "
                f"Consider relaxing the constraint, reducing action costs, "
                f"or removing expensive actions."
            )
        elif sf.available_min is not None:
            parts.append(
                f"Resource '{sf.key}' uses {sf.required:.4g} but the minimum "
                f"required is {sf.available_min:.4g} (shortfall: {sf.overrun:.4g}). "
                f"Consider adding actions that produce more of this resource."
            )

    if not parts:
        keys = [c.key for c in mus]
        parts.append(
            f"Constraints on {', '.join(keys)} are mutually unsatisfiable. "
            f"Relax at least one constraint to make the plan feasible."
        )

    return " ".join(parts)


def _is_violated(c: ConstraintSpec, totals: dict[str, float]) -> bool:
    """Return True when *c* is violated by the observed resource totals."""
    total = totals.get(c.key, 0.0)
    if c.max is not None and total > c.max:
        return True
    if c.min is not None and total < c.min:
        return True
    return False


def _greedy_mus(
    violated: list[ConstraintSpec], totals: dict[str, float]
) -> list[ConstraintSpec]:
    """Greedy MUS: keep constraints whose removal would restore feasibility."""
    mus: list[ConstraintSpec] = []
    remaining = list(violated)
    for candidate in list(violated):
        without = [c for c in remaining if c is not candidate]
        if _check_feasibility(totals, without):
            # Removing this constraint makes it feasible → it's essential
            mus.append(candidate)
        else:
            # Still infeasible without it → it's redundant
            remaining = without
    # If greedy didn't identify any (all are redundant to each other),
    # fall back to the full violated set
    return mus or violated


def explain_infeasibility(
    plan: Plan,
    goal: GoalSpec,
    meta: CSPMetadata,
) -> InfeasibilityExplanation | None:
    """Analyze an INFEASIBLE plan and explain which constraints conflict.

    Uses a greedy MUS approximation: iteratively removes hard constraints
    and checks if the remaining set is still infeasible.  Constraints
    whose removal makes the plan feasible are part of the MUS.

    Args:
        plan: The infeasible plan.
        goal: Goal specification with constraints.
        meta: CSP metadata from the validation run.

    Returns:
        An :class:`InfeasibilityExplanation` describing the conflict,
        or ``None`` if the plan is not infeasible (no hard violations).
    """
    hard_constraints = [c for c in goal.constraints if c.level == "hard"]
    if not hard_constraints:
        return None

    totals = compute_resource_totals(plan.actions)
    violated = [c for c in hard_constraints if _is_violated(c, totals)]
    if not violated:
        return None

    mus_tuple = tuple(_greedy_mus(violated, totals))
    shortfalls = _build_shortfalls(totals, mus_tuple)
    return InfeasibilityExplanation(
        conflicting_constraints=mus_tuple,
        resource_shortfalls=shortfalls,
        suggestion=_build_suggestion(shortfalls, mus_tuple),
    )


@dataclass(frozen=True, slots=True)
class NoPlanExplanation:
    """Explanation of why A* could not find a valid plan.

    Attributes:
        unreachable_conditions: Goal conditions that no available action can
            produce as an effect.  These keys can never be set regardless of
            action ordering.
        missing_preconditions: Preconditions required by available actions
            that neither appear in the start state nor can be produced by any
            other action.  These create unsatisfiable prerequisite chains.
        suggestion: Human-readable explanation and corrective advice.
    """

    unreachable_conditions: tuple[str, ...]
    missing_preconditions: tuple[str, ...]
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict safe for storage in :class:`~langgoap.graph.state.GoapState`."""
        return {
            "unreachable_conditions": list(self.unreachable_conditions),
            "missing_preconditions": list(self.missing_preconditions),
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoPlanExplanation:
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        return cls(
            unreachable_conditions=tuple(data.get("unreachable_conditions", [])),
            missing_preconditions=tuple(data.get("missing_preconditions", [])),
            suggestion=data.get("suggestion", ""),
        )


def explain_no_plan(
    start: PlanningState,
    goal: GoalSpec,
    actions: list[ActionSpec],
) -> NoPlanExplanation:
    """Explain why A* could not find a valid plan.

    Performs a lightweight static analysis of the action set to surface two
    categories of failure:

    1. **Unreachable conditions** — goal conditions that no action can produce
       as an effect.  These keys can *never* become ``True`` regardless of
       ordering, so the root cause is a missing action or a typo in the goal.
    2. **Missing preconditions** — preconditions that actions require but that
       neither appear in the start state nor can be produced by any other
       action.  These create unsatisfiable prerequisite chains.

    Args:
        start: The initial world state from which A* searched.
        goal: Goal specification whose conditions A* failed to satisfy.
        actions: The full set of actions A* had available.

    Returns:
        A :class:`NoPlanExplanation` with categorised unreachable conditions,
        missing preconditions, and a human-readable suggestion.
    """
    # Collect all effects and all preconditions across the available actions.
    all_effects: set[str] = set()
    all_preconditions: set[str] = set()
    for a in actions:
        all_effects.update(str(k) for k in a.effects)
        all_preconditions.update(str(k) for k in a.preconditions)

    start_true: set[str] = {k for k, v in start.to_dict().items() if v}

    # Goal conditions that are neither already satisfied nor producible.
    unreachable: list[str] = sorted(
        str(key)
        for key, required in goal.conditions.items()
        if str(key) not in start_true and str(key) not in all_effects
    )

    # Preconditions that can never be established.
    missing: list[str] = sorted(
        p for p in all_preconditions if p not in start_true and p not in all_effects
    )

    return NoPlanExplanation(
        unreachable_conditions=tuple(unreachable),
        missing_preconditions=tuple(missing),
        suggestion=_build_no_plan_suggestion(unreachable, missing),
    )


def _build_no_plan_suggestion(unreachable: list[str], missing: list[str]) -> str:
    """Assemble the human-readable suggestion for ``explain_no_plan``."""
    parts: list[str] = []
    if unreachable:
        parts.append(
            f"Goal condition(s) {unreachable} cannot be produced by any "
            f"available action. Add an action whose effects include these "
            f"keys, or check for typos in the goal conditions."
        )
    if missing:
        parts.append(
            f"Precondition(s) {missing} are required by available actions "
            f"but cannot be established from the current world state or by "
            f"any other action. Check the action dependency chain."
        )
    if not parts:
        parts.append(
            "The goal conditions appear theoretically reachable but A* found "
            "no valid action ordering. This may indicate circular precondition "
            "dependencies or conflicting action effects. Review action "
            "preconditions and effects for internal consistency."
        )
    return " ".join(parts)


__all__ = [
    "InfeasibilityExplanation",
    "NoPlanExplanation",
    "ResourceShortfall",
    "explain_infeasibility",
    "explain_no_plan",
]
