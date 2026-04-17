"""Nurse rostering instance derived from standard benchmark data.

Provenance
----------
Derived from the classic INRC Sprint benchmark instance (10 nurses,
4 shift types, 1 week).  This tutorial fixture collapses the problem
to **1 day / 3 shifts / 4 nurses** so the solver runs instantly
while still exercising:

- **Skill matching** — each shift requires a specific skill and only
  nurses possessing that skill can be assigned.
- **Preference scoring** — nurses have varying unhappiness costs per
  shift type, driving A\\* toward the lowest-dissatisfaction plan.
- **Hard coverage constraints** — every shift must be covered, and
  no nurse can work two shifts in the same day (enforced via the
  ``nurse_<name>_available`` precondition).

Standard NRP concepts that are **out of scope** for this tutorial:

- Multi-day patterns and consecutive-work-day limits — the fixture
  is a single day.
- Unwanted shift patterns — encoding these would require action-
  chain history tracking which LangGoap's flat world state can only
  approximate via per-nurse counters.
- Contract-level min/max assignment counts — trivially expressible
  as resource totals but add no pedagogical value for a 1-day demo.

The 4-nurse / 3-shift instance below is deliberately solvable by
hand — the intended optimal preference-weighted assignment is
verified in ``tests/integration/test_nurse_rostering.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Nurse:
    name: str
    skills: frozenset[str]


@dataclass(frozen=True)
class Shift:
    name: str
    required_skill: str
    hours: int


# ---------------------------------------------------------------------------
# 3-shift / 4-nurse single-day instance
# ---------------------------------------------------------------------------

SHIFTS: tuple[Shift, ...] = (
    Shift(name="morning", required_skill="triage", hours=8),
    Shift(name="afternoon", required_skill="general", hours=8),
    Shift(name="night", required_skill="pediatrics", hours=8),
)


NURSES: tuple[Nurse, ...] = (
    Nurse(name="alice", skills=frozenset({"triage", "general", "pediatrics"})),
    Nurse(name="bob", skills=frozenset({"general", "pediatrics"})),
    Nurse(name="carol", skills=frozenset({"triage", "general"})),
    Nurse(name="dave", skills=frozenset({"general", "pediatrics"})),
)


# Unhappiness cost for placing nurse X on shift Y.  Lower is happier.
# Only entries where the nurse has the required skill are consulted
# by :func:`nurse_rostering_actions`; other pairs are never legal
# actions in the first place.
PREFERENCE_COST: dict[tuple[str, str], int] = {
    # Alice is happiest on afternoons; mornings are tolerable;
    # nights are disruptive.
    ("alice", "morning"): 4,
    ("alice", "afternoon"): 0,
    ("alice", "night"): 7,
    # Bob only holds general/pediatrics — skill filter removes morning.
    ("bob", "afternoon"): 3,
    ("bob", "night"): 6,
    # Carol wants mornings; no pediatrics so night is unavailable.
    ("carol", "morning"): 1,
    ("carol", "afternoon"): 5,
    # Dave is a night owl.
    ("dave", "afternoon"): 4,
    ("dave", "night"): 0,
}
