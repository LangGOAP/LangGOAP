"""Goal and constraint specifications for GOAP planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from langgoap.types import ObjectiveDirection, ReplanStrategy

if TYPE_CHECKING:
    from langgoap.constraints import BuilderOutput


@dataclass(frozen=True)
class ConstraintSpec:
    """A resource/budget constraint for CSP optimization.

    Constraints can be ``"hard"`` (the default) or ``"soft"``.  A hard
    constraint violation marks the plan ``INFEASIBLE``; a soft violation
    contributes a penalty to the plan's :class:`~langgoap.score.HardSoftScore`
    but does not disqualify the plan.

    Attributes:
        key: The resource key this constraint applies to
            (e.g. ``"total_tokens"``, ``"cost_usd"``).
        max: Upper bound.  ``None`` means no upper bound.
        min: Lower bound.  ``None`` means no lower bound.
        weight: Penalty weight per unit of violation (default 1.0).
            Used by the CSP optimizer and the Score hierarchy.
        level: ``"hard"`` (default) or ``"soft"``.  Hard constraints
            mark the plan infeasible on violation; soft constraints
            subtract ``weight * violation_amount`` from the plan's
            soft score.
    """

    key: str
    max: float | None = None
    min: float | None = None
    weight: float = 1.0
    level: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"ConstraintSpec(key={self.key!r}): min ({self.min}) must be "
                f"<= max ({self.max})."
            )
        if self.level not in ("hard", "soft"):
            raise ValueError(
                f"ConstraintSpec(key={self.key!r}): level must be "
                f"'hard' or 'soft', got {self.level!r}."
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

    @classmethod
    def from_builder(
        cls,
        conditions: dict[str, Any] | None = None,
        builder_output: BuilderOutput | None = None,
        **kwargs: Any,
    ) -> GoalSpec:
        """Construct a GoalSpec from a :class:`~langgoap.constraints.BuilderOutput`.

        Args:
            conditions: Goal conditions.  Forwarded to the ``conditions``
                field.
            builder_output: Output of
                :meth:`~langgoap.constraints.ConstraintBuilder.build`.
                ``None`` produces a GoalSpec with no constraints or
                objectives, equivalent to ``GoalSpec(conditions=...)``.
            **kwargs: Any other ``GoalSpec`` fields (``replan_strategy``,
                ``priority``, ``max_replans``).

        Returns:
            A new ``GoalSpec`` with the builder's constraints and
            objectives wired into the standard fields.
        """
        constraints: tuple[ConstraintSpec, ...] = ()
        objectives: MappingProxyType[str, ObjectiveDirection] | None = None
        if builder_output is not None:
            constraints = builder_output.constraints
            if builder_output.objectives:
                objectives = builder_output.objectives
        return cls(
            conditions=MappingProxyType(dict(conditions or {})),
            constraints=constraints,
            objectives=objectives,
            **kwargs,
        )


# Public alias for GoalSpec. Use ``Goal`` in user-facing code.
Goal = GoalSpec


@dataclass(frozen=True)
class MultiGoal:
    """A composite goal made up of one or more :class:`GoalSpec` children.

    Two execution modes are supported:

    * ``"sequential"`` (default) — the observer plans and executes
      ``goals[0]`` to completion, then uses the resulting world state
      as the starting state for ``goals[1]``, and so on.  This is the
      right model when sub-goals represent successive stages of a
      workflow (e.g. "collect data → analyse → publish").
    * ``"any"`` — the planner plans each sub-goal independently and
      the observer picks the lowest-:class:`~langgoap.score.Score`
      feasible plan.  Useful when several possible goals are
      acceptable and the system should chase the cheapest one.

    Recursive (HTN-style) decomposition where a sub-goal is itself a
    ``MultiGoal`` is **out of scope for v0.1.0** — only flat
    composition is supported and :meth:`__post_init__` rejects
    non-:class:`GoalSpec` children with a clear ``ValueError``.

    In ``"any"`` mode, ties on plan cost are broken by list order:
    the first sub-goal whose plan has the lowest cost wins.  This
    makes the order of ``goals`` semantically meaningful when
    several candidates are equally cheap.

    Args:
        goals: Non-empty sequence of :class:`GoalSpec` children.
            Accepts tuples or lists; a list is coerced to a tuple.
        mode: Either ``"sequential"`` or ``"any"``.
    """

    goals: tuple[GoalSpec, ...]
    mode: Literal["sequential", "any"] = "sequential"

    def __post_init__(self) -> None:
        if not isinstance(self.goals, tuple):
            object.__setattr__(self, "goals", tuple(self.goals))
        if not self.goals:
            raise ValueError("MultiGoal must contain at least one GoalSpec")
        if self.mode not in ("sequential", "any"):
            raise ValueError(
                f"MultiGoal.mode must be 'sequential' or 'any', got {self.mode!r}"
            )
        for i, g in enumerate(self.goals):
            if not isinstance(g, GoalSpec):
                raise ValueError(
                    f"MultiGoal.goals[{i}] must be a GoalSpec, got "
                    f"{type(g).__name__}. Nested MultiGoal (HTN-style "
                    "decomposition) is not supported in v0.1.0."
                )
