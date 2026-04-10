"""Goal and constraint specifications for GOAP planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from langgoap.types import ObjectiveDirection, ReplanStrategy


@dataclass(frozen=True)
class ConstraintSpec:
    """A hard resource/budget constraint for CSP optimization.

    Attributes:
        key: The resource key this constraint applies to
            (e.g. ``"total_tokens"``, ``"cost_usd"``).
        max: Upper bound.  ``None`` means no upper bound.
        min: Lower bound.  ``None`` means no lower bound.
        weight: Relative importance when used as a soft constraint (default 1.0).
    """

    key: str
    max: float | None = None
    min: float | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"ConstraintSpec(key={self.key!r}): min ({self.min}) must be "
                f"<= max ({self.max})."
            )


@dataclass(frozen=True)
class GoalSpec:
    """Specification of a GOAP goal.

    Attributes:
        conditions: Required world state for the goal to be satisfied.
            Stored as an immutable MappingProxyType.
        replan_strategy: When to trigger replanning during execution.
        objectives: Optional optimization objectives for the CSP phase.
            Maps resource/metric name to an :class:`~langgoap.types.ObjectiveDirection`.
            Example: ``{"cost_usd": Minimize, "quality": Maximize}``.
        constraints: Optional hard resource/budget constraints for the CSP phase.
        priority: Goal priority for multi-goal scenarios (higher = more important).
    """

    conditions: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    replan_strategy: ReplanStrategy = ReplanStrategy.ON_DEVIATION
    objectives: MappingProxyType[str, ObjectiveDirection] | None = None
    constraints: tuple[ConstraintSpec, ...] = field(default_factory=tuple)
    priority: int = 0
    max_replans: int = 10
    """Maximum number of replanning cycles before the observer gives up.

    Guards against infinite replan loops when an action keeps failing or
    the world state never converges.  Set to 0 to disable the limit.
    """

    def __post_init__(self) -> None:
        # Accept plain dicts from callers and silently wrap conditions.
        object.__setattr__(
            self,
            "conditions",
            MappingProxyType(dict(self.conditions)),
        )
        # Normalise objectives: plain dict → MappingProxyType, empty → None.
        if self.objectives is not None:
            obj_dict = dict(self.objectives)
            if not obj_dict:
                # Empty objectives dict is semantically equivalent to no objectives;
                # normalise to None so _needs_csp() and CSP checks behave correctly.
                object.__setattr__(self, "objectives", None)
            elif not isinstance(self.objectives, MappingProxyType):
                object.__setattr__(self, "objectives", MappingProxyType(obj_dict))
        # Normalise constraints: accept list or tuple from callers.
        if not isinstance(self.constraints, tuple):
            object.__setattr__(self, "constraints", tuple(self.constraints))

    def __repr__(self) -> str:
        parts = [f"conditions={dict(self.conditions)!r}"]
        if self.replan_strategy != ReplanStrategy.ON_DEVIATION:
            parts.append(f"replan_strategy={self.replan_strategy!r}")
        if self.objectives:
            parts.append(f"objectives={dict(self.objectives)!r}")
        if self.constraints:
            parts.append(f"constraints={self.constraints!r}")
        if self.priority:
            parts.append(f"priority={self.priority!r}")
        if self.max_replans != 10:
            parts.append(f"max_replans={self.max_replans!r}")
        return f"GoalSpec({', '.join(parts)})"


#: Public alias for :class:`GoalSpec`.  Use ``Goal`` in user-facing code.
Goal = GoalSpec
