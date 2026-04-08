"""Immutable planning state for GOAP world representation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanningState:
    """Immutable, hashable representation of a GOAP world state.

    Internally stores state as a frozenset of (key, value) tuples to ensure
    immutability and hashability. This enables use as dictionary keys and
    in sets — critical for A* closed-set tracking.

    The public attribute :attr:`conditions` exposes the underlying frozenset
    so callers can inspect it directly without an allocation.
    """

    conditions: frozenset[tuple[str, Any]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlanningState:
        """Create a PlanningState from a dictionary."""
        return cls(conditions=frozenset(d.items()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a mutable dictionary snapshot."""
        return dict(self.conditions)

    def satisfies(self, requirements: Mapping[str, Any]) -> bool:
        """Check if this state satisfies all given conditions.

        Returns True if every key-value pair in ``requirements`` is present
        in this state. Missing keys cause failure.

        Uses direct frozenset membership to avoid creating an intermediate
        dict — important for A* hot paths.
        """
        for k, v in requirements.items():
            if (k, v) not in self.conditions:
                return False
        return True

    def apply(self, effects: Mapping[str, Any]) -> PlanningState:
        """Return a new PlanningState with effects applied.

        Existing keys are overwritten; new keys are added.
        The original state is not modified.
        """
        updated = self.to_dict()
        updated.update(effects)
        return PlanningState.from_dict(updated)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, returning default if missing.

        Scans the frozenset directly to avoid a full dict allocation.
        """
        for k, v in self.conditions:
            if k == key:
                return v
        return default

    def __len__(self) -> int:
        return len(self.conditions)

    def __contains__(self, key: str) -> bool:
        return any(k == key for k, _ in self.conditions)

    def __repr__(self) -> str:
        return f"PlanningState({self.to_dict()!r})"
