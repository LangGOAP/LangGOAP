r"""Nurse Rostering — OptaPlanner's NRP scaled to a one-day demo.

Translates a 1-day / 3-shift / 4-nurse subset of OptaPlanner's
``nurserostering`` Sprint instance into GOAP.  Every shift must be
covered by a nurse who holds the required skill and no nurse may
work two shifts in the same day.  Nurses have per-shift preference
costs (lower = happier) and A\* minimizes total unhappiness.

GOAP modelling
--------------

**World state:**

- ``shift_<name>_covered`` — flipped to ``True`` once a nurse is
  assigned to that shift.
- ``nurse_<name>_available`` — starts ``True``; flipped to ``False``
  when a nurse takes a shift so the same nurse cannot double up.

**Actions:** one ``assign_<nurse>_to_<shift>`` per (nurse, shift) pair
**where the nurse holds the shift's required skill**.  Skill filtering
happens at action-build time — unqualified nurses have no action to
take the shift in the first place.

- **preconditions**: ``nurse_<n>_available=True``,
  ``shift_<s>_covered=False``
- **effects**: ``nurse_<n>_available=False``,
  ``shift_<s>_covered=True``
- **cost**: ``PREFERENCE_COST[(nurse, shift)]`` — drives A\* toward
  the lowest-dissatisfaction assignment.
- **resources**: ``unhappiness`` (aggregated across the whole plan)
  so the CSP phase can attach a ``HardSoftScore`` derived from the
  ``unhappiness → MINIMIZE`` objective.

**Goal:** every shift covered; the ``unhappiness`` objective makes
the planner route through CSP so :class:`~langgoap.score.HardSoftScore`
is populated and the NL interpreter has a soft/hard target to
translate natural-language preferences onto.

Why resource-aggregated unhappiness?
------------------------------------
Encoding unhappiness as both the action ``cost`` and an
``unhappiness`` resource lets A\* pick the happiest plan directly
while still giving CP-SAT a scalar to minimize, penalize, or bound.
Soft constraints of the form "it would be nice to keep total
dissatisfaction under 3" become a single
``ConstraintSpec(key="unhappiness", level="soft", max=3.0)`` and the
score decomposition in ``planner/pipeline.py::_score_from_csp`` takes
care of the rest.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, ConstraintSpec, GoalSpec
from langgoap.goals import ObjectiveDirection

from .data.nurse_rostering_instance import (
    NURSES,
    PREFERENCE_COST,
    SHIFTS,
    Nurse,
    Shift,
)


def _covered_key(shift_name: str) -> str:
    return f"shift_{shift_name}_covered"


def _available_key(nurse_name: str) -> str:
    return f"nurse_{nurse_name}_available"


def _make_execute(effects: dict[str, Any]) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        return dict(effects)

    return execute


def _assign_action(nurse: Nurse, shift: Shift) -> ActionSpec:
    unhappiness = float(PREFERENCE_COST[(nurse.name, shift.name)])
    effects = {
        _available_key(nurse.name): False,
        _covered_key(shift.name): True,
    }
    return ActionSpec(
        name=f"assign_{nurse.name}_to_{shift.name}",
        preconditions={
            _available_key(nurse.name): True,
            _covered_key(shift.name): False,
        },
        effects=effects,
        cost=unhappiness,
        resources={"unhappiness": unhappiness},
        execute=_make_execute(effects),
    )


def nurse_rostering_actions(
    nurses: tuple[Nurse, ...] = NURSES,
    shifts: tuple[Shift, ...] = SHIFTS,
) -> list[ActionSpec]:
    """Return one assignment action per qualified (nurse, shift) pair.

    Skill matching is applied at build time: a nurse missing the
    shift's required skill simply has no action to take it.
    """
    actions: list[ActionSpec] = []
    for nurse in nurses:
        for shift in shifts:
            if shift.required_skill not in nurse.skills:
                continue
            actions.append(_assign_action(nurse, shift))
    return actions


def nurse_rostering_start(
    nurses: tuple[Nurse, ...] = NURSES,
    shifts: tuple[Shift, ...] = SHIFTS,
) -> dict[str, Any]:
    """All nurses available; no shift yet covered."""
    state: dict[str, Any] = {}
    for n in nurses:
        state[_available_key(n.name)] = True
    for s in shifts:
        state[_covered_key(s.name)] = False
    return state


def nurse_rostering_goal(
    shifts: tuple[Shift, ...] = SHIFTS,
    *,
    minimize_unhappiness: bool = True,
    max_unhappiness: float | None = None,
    max_unhappiness_level: str = "hard",
) -> GoalSpec:
    """All shifts covered, with optional unhappiness cap and objective.

    Args:
        shifts: Shifts that must be covered.
        minimize_unhappiness: When True (default), attach an
            ``unhappiness → MINIMIZE`` objective so the goal routes
            through the CSP phase and receives a
            :class:`~langgoap.score.HardSoftScore`.
        max_unhappiness: Optional upper bound on the aggregated
            ``unhappiness`` resource.  Omit for no cap.
        max_unhappiness_level: ``"hard"`` (default) marks violation
            as ``INFEASIBLE``; ``"soft"`` records a penalty but keeps
            the plan feasible.
    """
    conditions: dict[str, Any] = {_covered_key(s.name): True for s in shifts}

    constraints: tuple[ConstraintSpec, ...] = ()
    if max_unhappiness is not None:
        constraints = (
            ConstraintSpec(
                key="unhappiness",
                max=float(max_unhappiness),
                level=max_unhappiness_level,  # type: ignore[arg-type]
            ),
        )

    objectives: dict[str, ObjectiveDirection] | None = None
    if minimize_unhappiness:
        objectives = {"unhappiness": ObjectiveDirection.MINIMIZE}

    return GoalSpec(
        conditions=conditions,
        constraints=constraints,
        objectives=objectives,
    )
