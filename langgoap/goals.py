"""Goal and constraint specifications for GOAP planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from langgoap.types import ObjectiveDirection, ReplanStrategy

if TYPE_CHECKING:
    from langgoap.constraints import BuilderOutput
    from langgoap.planner.metrics import PlanQualityMetric


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class SoftGoal:
    """An optional goal condition with a utility weight (oversubscription planning).

    Hard goal conditions (:attr:`GoalSpec.conditions`) *must* be satisfied for
    a plan to be valid.  Soft goals are optional — the planner prefers plans
    that achieve them but will not fail if they cannot be satisfied.

    Attributes:
        conditions: Target conditions for this soft goal.  Accepted as a
            plain ``dict``; silently wrapped in :class:`~types.MappingProxyType`.
        weight: Non-negative utility weight (default ``1.0``).  Higher values
            make this goal more valuable; the planner maximises the total
            achieved weight across all soft goals when selecting among plans.
        name: Optional human-readable identifier used in logging and
            :class:`~langgoap.planner.csp.CSPMetadata` objective keys.
            Defaults to ``str(conditions)`` when empty.
    """

    conditions: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    weight: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, MappingProxyType):
            object.__setattr__(
                self, "conditions", MappingProxyType(dict(self.conditions))
            )
        if self.weight < 0:
            raise ValueError(f"SoftGoal.weight must be >= 0, got {self.weight}")

    @property
    def label(self) -> str:
        """Return name if set, else a compact conditions string."""
        return self.name or str(dict(self.conditions))


_POLICY_KWARG_NAMES = ("replan_strategy", "priority", "max_replans")


def _bundle_policy_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bundle legacy flat policy kwargs into a ``policy=GoalPolicy(...)`` kwarg.

    Used by :meth:`GoalSpec.from_builder` and
    :meth:`GoalSpec.per_entity` so callers that historically passed
    ``replan_strategy=`` / ``priority=`` / ``max_replans=`` directly
    keep working unchanged.  When ``policy=`` is also explicitly
    provided, the explicit value wins and the flat kwargs are ignored
    (tests covering this contract live in
    :mod:`tests.test_per_entity_goals` and
    :mod:`tests.test_constraint_builder`).
    """
    flat = {k: kwargs.pop(k) for k in _POLICY_KWARG_NAMES if k in kwargs}
    if "policy" in kwargs or not flat:
        return kwargs
    kwargs["policy"] = GoalPolicy(**flat)
    return kwargs


@dataclass(frozen=True, slots=True)
class GoalPolicy:
    """Replanning + multi-goal-priority policy for a :class:`GoalSpec`.

    Grouping these three orthogonal-to-the-conditions knobs into a
    sub-dataclass keeps the :class:`GoalSpec` constructor focused on
    *what* the goal is, not *how* the planner should chase it.  The
    defaults match the historical flat-field defaults, so a bare
    ``GoalPolicy()`` is behaviourally identical to the pre-Phase-7
    implicit policy.

    Attributes:
        replan_strategy: When the observer should trigger replanning
            during execution.
        priority: Multi-goal scenarios use this to break ties when
            :class:`MultiGoal` chooses the next sub-goal.  Higher = more
            important.
        max_replans: Maximum number of replanning cycles before the
            observer gives up.  Guards against infinite replan loops
            when an action keeps failing or the world state never
            converges.  Set to ``0`` to disable the limit.
    """

    replan_strategy: ReplanStrategy = ReplanStrategy.ON_DEVIATION
    priority: int = 0
    max_replans: int = 10

    def __post_init__(self) -> None:
        if self.max_replans < 0:
            raise ValueError("max_replans must be >= 0 (0 disables the limit)")


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """Specification of a GOAP goal.

    Attributes:
        conditions: Required world state for the goal to be satisfied.
            Stored as an immutable MappingProxyType.
        policy: :class:`GoalPolicy` bundling the replanning strategy,
            multi-goal priority, and replan budget.  Defaults to
            ``GoalPolicy()`` (ON_DEVIATION, priority 0, max_replans 10).
        objectives: Optional optimization objectives for the CSP phase.
            Maps resource/metric name to an :class:`~langgoap.types.ObjectiveDirection`.
            Example: ``{"cost_usd": Minimize, "quality": Maximize}``.
        constraints: Optional hard resource/budget constraints for the CSP phase.
        soft_goals: Optional soft (penalty-weighted) goal predicates.
        metrics: Optional plan-quality metrics consumed by the CSP phase.
    """

    conditions: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    policy: GoalPolicy = field(default_factory=GoalPolicy)
    objectives: MappingProxyType[str, ObjectiveDirection] | None = None
    constraints: tuple[ConstraintSpec, ...] = field(default_factory=tuple)
    soft_goals: tuple[SoftGoal, ...] = field(default_factory=tuple)
    metrics: tuple["PlanQualityMetric", ...] = field(default_factory=tuple)

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
        # Normalise soft_goals: accept list or tuple from callers.
        if not isinstance(self.soft_goals, tuple):
            object.__setattr__(self, "soft_goals", tuple(self.soft_goals))
        # Normalise metrics: accept list or tuple from callers.
        if not isinstance(self.metrics, tuple):
            object.__setattr__(self, "metrics", tuple(self.metrics))

    def __repr__(self) -> str:
        parts = [f"conditions={dict(self.conditions)!r}"]
        if self.policy != GoalPolicy():
            parts.append(f"policy={self.policy!r}")
        if self.objectives:
            parts.append(f"objectives={dict(self.objectives)!r}")
        if self.constraints:
            parts.append(f"constraints={self.constraints!r}")
        if self.soft_goals:
            parts.append(f"soft_goals={self.soft_goals!r}")
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
            **kwargs: Any other ``GoalSpec`` fields, plus the legacy
                flat policy kwargs (``replan_strategy``, ``priority``,
                ``max_replans``) which are bundled into a
                :class:`GoalPolicy` for the caller's convenience.

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
        kwargs = _bundle_policy_kwargs(kwargs)
        return cls(
            conditions=MappingProxyType(dict(conditions or {})),
            constraints=constraints,
            objectives=objectives,
            **kwargs,
        )

    @classmethod
    def per_entity(
        cls,
        entity_ids: Sequence[str],
        conditions: Mapping[str, Any],
        *,
        mode: Literal["sequential", "any"] = "any",
        **goal_kwargs: Any,
    ) -> "MultiGoal":
        """Build a :class:`MultiGoal` with one :class:`GoalSpec` per entity.

        The ``conditions`` mapping is treated as a template: every
        string key and every string value is formatted with
        :py:meth:`str.format`, substituting the placeholder
        ``{entity}`` with each entity id in turn.  Non-string values
        pass through unchanged so callers can parameterise e.g.
        ``{"safe_from_{entity}": True}`` or
        ``{"target_{entity}": (x, y)}``.

        Args:
            entity_ids: Non-empty sequence of entity identifiers.  The
                returned :class:`MultiGoal` holds one child goal per
                id, in input order.
            conditions: Template mapping.  At least one key or string
                value must contain ``{entity}`` — otherwise every
                generated child would be identical, which is almost
                certainly a caller mistake.
            mode: Forwarded to :class:`MultiGoal`.  Defaults to
                ``"any"`` because the canonical use case (per-
                adversary escape goals) is "satisfy whichever is
                cheapest".  Use ``"sequential"`` when the per-entity
                ordering encodes execution intent.
            **goal_kwargs: Forwarded verbatim to every :class:`GoalSpec`
                child (``priority``, ``max_replans``,
                ``replan_strategy``, etc.).

        Returns:
            A :class:`MultiGoal` whose ``goals`` tuple has one
            :class:`GoalSpec` per entity id, in input order.

        Raises:
            ValueError: If ``entity_ids`` is empty, or if no key /
                string value in ``conditions`` contains the
                ``{entity}`` placeholder.
        """
        ids = list(entity_ids)
        if not ids:
            raise ValueError("GoalSpec.per_entity requires at least one entity id.")
        template_has_placeholder = any(
            "{entity}" in k or (isinstance(v, str) and "{entity}" in v)
            for k, v in conditions.items()
        )
        if not template_has_placeholder:
            raise ValueError(
                "GoalSpec.per_entity requires '{entity}' in at least one "
                "condition key or string value; otherwise every child "
                "goal would be identical.  Got conditions="
                f"{dict(conditions)!r}."
            )
        # Bundle the legacy flat policy kwargs into a GoalPolicy so
        # callers passing ``replan_strategy`` / ``priority`` /
        # ``max_replans`` directly into per_entity (the documented API
        # before Phase 7) keep working.
        goal_kwargs = _bundle_policy_kwargs(goal_kwargs)
        children: list[GoalSpec] = []
        for eid in ids:
            formatted: dict[str, Any] = {}
            for k, v in conditions.items():
                new_k = k.format(entity=eid)
                new_v = v.format(entity=eid) if isinstance(v, str) else v
                formatted[new_k] = new_v
            children.append(cls(conditions=MappingProxyType(formatted), **goal_kwargs))
        return MultiGoal(goals=tuple(children), mode=mode)


# Public alias for GoalSpec. Use ``Goal`` in user-facing code.
Goal = GoalSpec


@dataclass(frozen=True, slots=True)
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
