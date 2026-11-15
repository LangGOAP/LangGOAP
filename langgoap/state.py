"""Immutable planning state for GOAP world representation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgoap.actions import ActionSpec

logger = logging.getLogger(__name__)

_EMPTY_CONTEXT: Mapping[str, Any] = MappingProxyType({})


def require_hashable_values(
    mapping: Mapping[str, Any], *, owner: str, field_name: str
) -> None:
    """Raise ``ValueError`` naming the first key whose value is unhashable.

    Planning values live in a ``frozenset`` of ``(key, value)`` pairs, so
    lists and dicts would otherwise surface as ``TypeError: unhashable
    type`` deep inside A* (issue #34).
    """
    for key, value in mapping.items():
        try:
            hash(value)
        except TypeError:
            raise ValueError(
                f"{owner}.{field_name}[{key!r}] must be hashable, got "
                f"{type(value).__name__}. Planning values are scalars, "
                "tuples or frozensets; keep rich data in world_state keys "
                "that no precondition, effect or goal condition reads."
            ) from None


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
    #: Non-planning entries of the world state the search started from
    #: (documents, identifiers, tiers, ...).  Carried along unchanged by
    #: :meth:`apply`, excluded from equality and hashing, and merged back
    #: in by :meth:`as_context` so cost functions, dynamic effects and
    #: transition models see the live world state (issue #30).
    context: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_CONTEXT, compare=False, repr=False
    )
    #: Lazily built ``dict`` view of :attr:`conditions`, so ``get`` and
    #: ``in`` are O(1) instead of a scan of the frozenset (#63).  Excluded
    #: from equality and hashing; filled on first use.
    _dict: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], *, keys: set[str] | None = None
    ) -> PlanningState:
        """Create a PlanningState from a dictionary.

        Args:
            d: Source dictionary (typically the full ``world_state``).
            keys: If provided, only include entries whose key is in this
                set.  Useful for filtering a rich world-state dict down to
                planning-relevant boolean flags.  The excluded entries are
                kept as :attr:`context` (not compared or hashed) so
                :meth:`as_context` can hand callables the full world state.

        Non-hashable values are silently dropped (with a warning log)
        because ``PlanningState`` requires all values to be hashable for
        use as A* closed-set keys.
        """
        context: Mapping[str, Any] = _EMPTY_CONTEXT
        if keys is not None:
            rest = {k: v for k, v in d.items() if k not in keys}
            if rest:
                context = MappingProxyType(rest)
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

        return cls(conditions=frozenset(items), context=context)

    def _view(self) -> dict[str, Any]:
        """The cached read-only dict view (built once; the state is frozen)."""
        view = self._dict
        if view is None:
            view = dict(self.conditions)
            object.__setattr__(self, "_dict", view)
        return view

    def to_dict(self) -> dict[str, Any]:
        """Convert to a mutable dictionary snapshot."""
        return dict(self._view())

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
        new = PlanningState.from_dict(updated)
        if not self.context:
            return new
        return PlanningState(conditions=new.conditions, context=self.context)

    def as_context(self) -> dict[str, Any]:
        """The world state as user callables should see it.

        The planning keys of this state layered over the non-planning
        :attr:`context` captured by :meth:`from_dict`.  Pass this (not
        :meth:`to_dict`) to ``ActionSpec.get_cost`` / ``get_effects`` and to
        ``TransitionModel.expected`` / ``sample``.
        """
        if not self.context:
            return self.to_dict()
        merged = dict(self.context)
        merged.update(self.conditions)
        return merged

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, returning default if missing."""
        return self._view().get(key, default)

    def __len__(self) -> int:
        return len(self.conditions)

    def __contains__(self, key: str) -> bool:
        return key in self._view()

    def __repr__(self) -> str:
        return f"PlanningState({self.to_dict()!r})"


def infer_start_state(actions: list[ActionSpec]) -> dict[str, bool]:
    """Build a clean-slate world state from action preconditions and effects.

    Collects every condition key referenced in preconditions and boolean
    effects, then sets each to ``False``.  Non-boolean effect values are
    skipped.

    This eliminates the hand-written start-state factories common in
    notebooks and tutorial modules::

        start = infer_start_state(actions)
        # → {"has_data": False, "report_complete": False, ...}

    Args:
        actions: List of actions whose preconditions/effects define the
            planning state space.

    Returns:
        A dict mapping every discovered boolean condition key to ``False``.
    """
    keys: set[str] = set()
    for a in actions:
        keys.update(a.preconditions.keys())
        if a.has_dynamic_effects:
            # Dynamic effects don't expose concrete values; skip — callers
            # should pass a real starting state when using them.
            continue
        for k, v in a.effects.items():  # type: ignore[union-attr]
            if isinstance(v, bool):
                keys.add(k)
    return {k: False for k in sorted(keys)}
