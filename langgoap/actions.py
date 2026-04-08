"""Action specifications and decorator for GOAP planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Callable

from langgoap.types import CostFunction


@dataclass(frozen=True)
class ActionSpec:
    """Specification of a GOAP action for planning.

    Attributes:
        name: Unique identifier for the action.
        preconditions: World state conditions that must hold before execution.
            Stored as an immutable MappingProxyType to match the frozen
            contract of this dataclass.
        effects: World state changes produced by execution.
            Stored as an immutable MappingProxyType.
        cost: Static cost or callable that computes cost from world state.
        execute: The function to call when the action is executed.
        effect_validator: Optional runtime postcondition checker.
            Signature: ``(pre_state: dict, post_state: dict) -> bool``.
            Called by the executor after execution to verify actual effects.
            Returning ``False`` signals that the action did not produce its
            declared effects (treated as a soft failure prompting replanning).
        resources: Estimated resource consumption for this action.
            Used by the CSP optimizer (Phase 2).
            Example: ``{"tokens": 500, "cost_usd": 0.02, "api_calls": 1}``.
        duration: Estimated wall-clock duration.
            Used by the CSP optimizer for temporal scheduling.
        metadata: Arbitrary key-value metadata (description, tags, etc.).
    """

    name: str
    # Accept any Mapping at construction; __post_init__ wraps them in
    # MappingProxyType for true immutability.
    preconditions: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    effects: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    cost: float | CostFunction = 1.0
    execute: Callable[..., Any] | None = None
    effect_validator: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None
    # CSP optimizer inputs (ignored by the A* planner)
    resources: Mapping[str, float] | None = None
    duration: timedelta | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # Wrap all Mapping fields in MappingProxyType to enforce immutability.
        for attr in ("preconditions", "effects"):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))
        for attr in ("resources", "metadata"):
            val = getattr(self, attr)
            if val is not None and not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))

    def get_cost(self, world_state: dict[str, Any] | None = None) -> float:
        """Resolve the cost, calling the cost function if dynamic."""
        if callable(self.cost):
            return self.cost(world_state or {})
        return self.cost

    def has_effects(self) -> bool:
        """Return True if this action declares at least one effect."""
        return len(self.effects) > 0

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if self.preconditions:
            parts.append(f"preconditions={dict(self.preconditions)!r}")
        if self.effects:
            parts.append(f"effects={dict(self.effects)!r}")
        if self.cost != 1.0:
            parts.append(f"cost={self.cost!r}")
        return f"ActionSpec({', '.join(parts)})"

    def validate_effects(
        self, pre_state: dict[str, Any], post_state: dict[str, Any]
    ) -> bool:
        """Verify that the action's effects were actually achieved.

        If an ``effect_validator`` callable was provided at construction,
        it is called with the world state snapshots taken before and after
        execution.  The default implementation checks that every declared
        effect key/value is present in ``post_state``.

        Args:
            pre_state: World state snapshot immediately before execution.
            post_state: World state snapshot immediately after execution.

        Returns:
            ``True`` if the effects are satisfied, ``False`` otherwise.
        """
        if self.effect_validator is not None:
            return self.effect_validator(pre_state, post_state)
        # Default: verify each declared effect is present in post_state.
        return all(post_state.get(k) == v for k, v in self.effects.items())


def goap_action(
    *,
    preconditions: dict[str, Any] | None = None,
    effects: dict[str, Any] | None = None,
    cost: float | CostFunction = 1.0,
    name: str | None = None,
    resources: dict[str, float] | None = None,
    duration: timedelta | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], ActionSpec]:
    """Decorator that converts a function into an ActionSpec.

    Usage::

        @goap_action(
            preconditions={"has_data": True},
            effects={"report_ready": True},
            cost=2.0,
            resources={"tokens": 500, "cost_usd": 0.02},
            duration=timedelta(seconds=2),
        )
        def generate_report(state):
            return {"report_ready": True}
    """

    def wrapper(func: Callable[..., Any]) -> ActionSpec:
        return ActionSpec(
            name=name or func.__name__,
            preconditions=preconditions or {},
            effects=effects or {},
            cost=cost,
            execute=func,
            resources=resources,
            duration=duration,
            metadata=metadata,
        )

    return wrapper


class GoapAction:
    """Base class for object-oriented GOAP action definitions.

    Subclass and override ``preconditions``, ``effects``, ``cost()``,
    and ``execute()`` to define an action.

    **Important**: declare ``preconditions`` and ``effects`` as class
    attributes on your subclass, not on ``GoapAction`` itself, to avoid
    the shared-mutable-dict antipattern::

        class GatherData(GoapAction):
            preconditions = {"has_source": True}
            effects = {"has_data": True}

            def cost(self, world_state: dict[str, Any]) -> float:
                return 1.5

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"has_data": True}
    """

    # Class-level declarations — each subclass gets its own copy via
    # __init_subclass__ below, preventing the shared-mutable-dict antipattern.
    preconditions: dict[str, Any] = {}
    effects: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Give each concrete subclass its own independent dict if it didn't
        # declare one explicitly.  This prevents sibling subclasses from
        # sharing the base-class dict and poisoning each other.
        if "preconditions" not in cls.__dict__:
            cls.preconditions = {}
        if "effects" not in cls.__dict__:
            cls.effects = {}

    def cost(self, world_state: dict[str, Any]) -> float:
        """Return the cost of this action. Override for dynamic costs."""
        return 1.0

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the action. Override with actual logic."""
        return {}

    def validate_effects(
        self, pre_state: dict[str, Any], post_state: dict[str, Any]
    ) -> bool:
        """Verify that execution produced the expected effects.

        Override to add custom postcondition checks.  The default
        implementation verifies every declared effect is present in
        ``post_state``.
        """
        return all(post_state.get(k) == v for k, v in self.effects.items())

    def to_spec(self) -> ActionSpec:
        """Convert this action instance to an ActionSpec."""
        return ActionSpec(
            name=type(self).__name__,
            preconditions=dict(self.preconditions),
            effects=dict(self.effects),
            cost=self.cost,
            execute=self.execute,
            effect_validator=self.validate_effects,
        )
