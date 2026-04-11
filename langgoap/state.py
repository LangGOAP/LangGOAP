"""Immutable planning state for GOAP world representation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlanningState:
    """Immutable, hashable representation of a GOAP world state.

    Internally stores state as a frozenset of (key, value) tuples to ensure
    immutability and hashability. This enables use as dictionary keys and
    in sets — critical for A* closed-set tracking.

    The public attribute :attr:`conditions` exposes the underlying frozenset
    so callers can inspect it directly without an allocation.

    In practice, a GOAP world state dict may contain both **planning flags**
    (hashable booleans/scalars used by the A* planner) and **execution
    context** (rich data like document lists or LLM responses).  Use the
    ``keys`` parameter of :meth:`from_dict` to extract only the
    planning-relevant subset.
    """

    conditions: frozenset[tuple[str, Any]]

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], *, keys: set[str] | None = None
    ) -> PlanningState:
        """Create a PlanningState from a dictionary.

        Args:
            d: Source dictionary (typically the full ``world_state``).
            keys: If provided, only include entries whose key is in this
                set.  Useful for filtering a rich world-state dict down to
                planning-relevant boolean flags.

        Non-hashable values are silently dropped (with a warning log)
        because ``PlanningState`` requires all values to be hashable for
        use as A* closed-set keys.
        """
        if keys is not None:
            d = {k: v for k, v in d.items() if k in keys}

        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            try:
                hash(v)
                items.append((k, v))
            except TypeError:
                logger.warning(
                    "Skipping unhashable value for key %r (type=%s)",
                    k,
                    type(v).__name__,
                )

        return cls(conditions=frozenset(items))

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
