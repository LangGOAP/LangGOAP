"""Score hierarchy for plan evaluation.

LangGOAP uses a tiered score hierarchy for comparing plans:

* :class:`SimpleScore` \u2014 one scalar; used when a plan has no constraint
  context (e.g. after pure A* planning).
* :class:`HardSoftScore` \u2014 two levels, ``hard`` and ``soft``.  A feasible
  plan has ``hard == 0.0``.  Lexicographic comparison: plans with higher
  ``hard`` win first, ties broken by ``soft``.
* :class:`BendableScore` \u2014 variable-width hard/soft levels for problems
  that need multiple tiers.

Sign convention:
  * ``hard`` is ``<= 0``. A feasible plan has ``hard == 0.0`` and every
    hard-constraint violation subtracts the corresponding penalty amount.
  * ``soft`` has **no** sign restriction.  Minimize objectives and soft
    violations contribute negative values; maximize objectives contribute
    positive values.

Rewards can be positive and penalties are negative (penalize/reward
convention).

Comparison is lexicographic within one subclass; comparing across
subclasses raises ``TypeError`` to force callers to stay within a single
score type per planning run.  All four ordering operators are derived
from a single ``_compare_payload`` hook via ``functools.total_ordering``
on :class:`Score` (see :mod:`langgoap._score_base`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langgoap._score_base import Score


@dataclass(frozen=True, order=False, slots=True)
class SimpleScore(Score):
    """A single scalar score.

    ``value`` returns the stored value; ``is_feasible`` is always ``True``
    (A*-produced plans are feasible by construction because A* would not
    return them otherwise).

    Comparison: lower is better.  This matches the existing A* convention
    where ``total_cost`` is minimized.
    """

    scalar: float = 0.0

    @property
    def value(self) -> float:
        return self.scalar

    def is_feasible(self) -> bool:
        return True

    def _compare_payload(self) -> float:
        return self.scalar

    def __repr__(self) -> str:
        return f"SimpleScore({self.scalar:g})"


@dataclass(frozen=True, order=False, slots=True)
class HardSoftScore(Score):
    """A two-level score with hard and soft components.

    ``hard`` is bounded above by 0.  A feasible plan has ``hard == 0.0``;
    every hard-constraint violation subtracts its penalty amount.  ``soft``
    has no sign restriction — maximize objectives push it positive,
    minimize objectives and soft-constraint violations push it negative.

    Comparison is lexicographic: a plan with a greater (less-negative)
    ``hard`` is strictly better; ties break by ``soft`` (greater is
    better).  Using ``>`` is more natural than ``<`` because rewards
    accumulate positively.
    """

    hard: float = 0.0
    soft: float = 0.0

    @property
    def value(self) -> float:
        """Sum of the two levels. Useful for logging only.

        Do **not** use this for plan selection — it discards the
        hard/soft distinction.  Compare ``HardSoftScore`` instances
        directly instead.
        """
        return self.hard + self.soft

    def is_feasible(self) -> bool:
        return self.hard == 0.0

    def _compare_payload(self) -> tuple[float, float]:
        return (self.hard, self.soft)

    def __repr__(self) -> str:
        return f"HardSoftScore(hard={self.hard:g}, soft={self.soft:g})"


@dataclass(frozen=True, order=False, slots=True)
class BendableScore(Score):
    """A variable-width score with any number of hard and soft levels.

    Use this when the problem has multiple tiers of constraints that
    must be respected in strict priority order (e.g. "never violate
    legal limits; then never violate corporate policy; then minimize
    cost").

    Every level in ``hard_levels`` is bounded above by 0.  ``soft_levels``
    are unconstrained in sign.
    """

    hard_levels: tuple[float, ...] = field(default_factory=tuple)
    soft_levels: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.hard_levels, tuple):
            object.__setattr__(self, "hard_levels", tuple(self.hard_levels))
        if not isinstance(self.soft_levels, tuple):
            object.__setattr__(self, "soft_levels", tuple(self.soft_levels))

    @property
    def value(self) -> float:
        """Sum of every level.  Useful for logging; do not use for selection."""
        return sum(self.hard_levels) + sum(self.soft_levels)

    def is_feasible(self) -> bool:
        return all(h == 0.0 for h in self.hard_levels)

    def _compare_check(self, other: Score) -> None:
        assert isinstance(other, BendableScore)  # narrowed by Score.__lt__
        if len(self.hard_levels) != len(other.hard_levels) or len(
            self.soft_levels
        ) != len(other.soft_levels):
            raise TypeError(
                "Cannot compare BendableScore instances with different "
                f"shapes: {len(self.hard_levels)}h/{len(self.soft_levels)}s "
                f"vs {len(other.hard_levels)}h/{len(other.soft_levels)}s."
            )

    def _compare_payload(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (self.hard_levels, self.soft_levels)

    def __repr__(self) -> str:
        return (
            f"BendableScore(hard={list(self.hard_levels)!r}, "
            f"soft={list(self.soft_levels)!r})"
        )


__all__ = [
    "Score",
    "SimpleScore",
    "HardSoftScore",
    "BendableScore",
]
