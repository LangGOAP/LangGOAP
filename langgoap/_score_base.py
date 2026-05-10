"""Internal base class for the :class:`~langgoap.score.Score` hierarchy.

Holds the comparison machinery (``__eq__`` from the dataclass plus a
single ``__lt__`` whose body is shared by every concrete subclass).
Subclasses opt into ordering by defining :meth:`Score._compare_payload`
\u2014 ``functools.total_ordering`` synthesises ``__le__`` / ``__gt__`` /
``__ge__`` from ``__eq__`` + ``__lt__``.

Lives in a private module so the public ``score.py`` stays focused on
the concrete score types and their domain semantics.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any


@functools.total_ordering
@dataclass(frozen=True, order=False, slots=True)
class Score:
    """Abstract base for all score types.

    Subclasses must implement :attr:`value`, :meth:`is_feasible`, and
    :meth:`_compare_payload`.  Cross-subclass comparison raises
    ``TypeError`` so callers stay within a single score type per
    planning run.
    """

    @property
    def value(self) -> float:
        """Return a scalar representation of this score.

        Useful for logging and simple comparisons that do not require
        lexicographic semantics.  Different subclasses use different
        definitions \u2014 read the subclass docstring.
        """
        raise NotImplementedError

    def is_feasible(self) -> bool:
        """Return True if this score represents a feasible plan."""
        raise NotImplementedError

    def _compare_payload(self) -> Any:
        """Return the tuple/scalar used for ordering comparisons.

        Concrete subclasses override this with their lexicographic
        payload (e.g. ``(self.hard, self.soft)`` for
        :class:`~langgoap.score.HardSoftScore`).  The base class never
        compares with itself so it raises rather than returning a
        default.
        """
        raise NotImplementedError

    def _compare_check(self, other: Score) -> None:
        """Hook for subclass-specific pre-comparison validation.

        Default is a no-op.  :class:`~langgoap.score.BendableScore`
        overrides this to assert matching hard/soft level shapes.
        Called from :meth:`__lt__` after the type check passes; only
        receives instances that already passed ``isinstance`` against
        ``type(self)``.
        """

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot compare {type(self).__name__} with "
                f"{type(other).__name__}; scores must be of the same "
                "concrete subclass."
            )
        self._compare_check(other)
        return self._compare_payload() < other._compare_payload()
