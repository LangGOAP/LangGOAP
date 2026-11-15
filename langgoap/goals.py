"""Goal and constraint specifications for GOAP planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence

from langgoap.state import require_hashable_values
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-native dict (inverse of :meth:`from_dict`)."""
        return {
            "key": self.key,
            "max": self.max,
            "min": self.min,
            "weight": self.weight,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstraintSpec:
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        return cls(
            key=data["key"],
            max=data.get("max"),
            min=data.get("min"),
            weight=float(data.get("weight", 1.0)),
            level=data.get("level", "hard"),
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

    conditions: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    weight: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, MappingProxyType):
            object.__setattr__(
                self, "conditions", MappingProxyType(dict(self.conditions))
            )
        if self.weight < 0:
            raise ValueError(f"SoftGoal.weight must be >= 0, got {self.weight}")
        require_hashable_values(
            self.conditions, owner="SoftGoal", field_name="conditions"
        )

    @property
    def label(self) -> str:
        """Return name if set, else a compact conditions string."""
        return self.name or str(dict(self.conditions))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-native dict (inverse of :meth:`from_dict`)."""
        return {
            "conditions": dict(self.conditions),
            "weight": self.weight,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SoftGoal:
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        return cls(
            conditions=MappingProxyType(dict(data.get("conditions") or {})),
            weight=float(data.get("weight", 1.0)),
            name=data.get("name", ""),
        )


@dataclass(frozen=True, slots=True)
class GoalPolicy:
    """Replanning + multi-goal-priority policy for a :class:`GoalSpec`.

    Grouping these three orthogonal-to-the-conditions knobs into a
    sub-dataclass keeps the :class:`GoalSpec` constructor focused on
    *what* the goal is, not *how* the planner should chase it.

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
        on_infeasible: What the planner does when the best plan it can
            find violates a *hard* constraint (CSP status
            ``INFEASIBLE``).  ``"fail"`` (default) rejects the plan: the
            run ends with ``status="no_plan"`` and a
            ``no_plan_explanation`` carrying the resource shortfalls, so
            stuck handlers can relax the constraint.  ``"execute"`` keeps
            the pre-0.2 advisory behaviour and runs the plan anyway; the
            violation stays visible in ``plan.metadata.csp``.
        blacklist_fallback: When every path to the goal needs a
            blacklisted action, ``True`` lets the planner retry with the
            full action set (re-admitting the failed action) and clear
            the blacklist.  ``False`` (default) reports ``no_plan`` so a
            permanently failing, human-denied, or guard-blocked action
            is never re-run; use ``ActionSpec.max_retries`` or an
            :class:`~langgoap.qos.ActionQos` for transient failures.
    """

    replan_strategy: ReplanStrategy = ReplanStrategy.ON_DEVIATION
    priority: int = 0
    max_replans: int = 10
    on_infeasible: Literal["fail", "execute"] = "fail"
    blacklist_fallback: bool = False

    def __post_init__(self) -> None:
        if self.max_replans < 0:
            raise ValueError("max_replans must be >= 0 (0 disables the limit)")
        if self.on_infeasible not in ("fail", "execute"):
            raise ValueError(
                "GoalPolicy.on_infeasible must be 'fail' or 'execute', "
                f"got {self.on_infeasible!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-native dict (inverse of :meth:`from_dict`)."""
        return {
            "replan_strategy": self.replan_strategy.value,
            "priority": self.priority,
            "max_replans": self.max_replans,
            "on_infeasible": self.on_infeasible,
            "blacklist_fallback": self.blacklist_fallback,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoalPolicy:
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        return cls(
            replan_strategy=ReplanStrategy(
                data.get("replan_strategy", ReplanStrategy.ON_DEVIATION.value)
            ),
            priority=int(data.get("priority", 0)),
            max_replans=int(data.get("max_replans", 10)),
            on_infeasible=data.get("on_infeasible", "fail"),
            blacklist_fallback=bool(data.get("blacklist_fallback", False)),
        )


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

    conditions: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    policy: GoalPolicy = field(default_factory=GoalPolicy)
    objectives: Mapping[str, ObjectiveDirection] | None = None
    constraints: tuple[ConstraintSpec, ...] = field(default_factory=tuple)
    soft_goals: tuple[SoftGoal, ...] = field(default_factory=tuple)
    metrics: tuple["PlanQualityMetric", ...] = field(default_factory=tuple)
    value: float | Callable[[Mapping[str, Any]], float] = 1.0
    """Utility weight consulted by ``MultiGoal(mode='best_value')``.

    Either a static float or a callable resolved against world state.
    Defaults to ``1.0`` so existing code is unaffected.
    """

    def __post_init__(self) -> None:
        # Accept plain dicts from callers and silently wrap conditions.
        object.__setattr__(
            self,
            "conditions",
            MappingProxyType(dict(self.conditions)),
        )
        require_hashable_values(
            self.conditions, owner="GoalSpec", field_name="conditions"
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise the declarative part of this goal to a JSON-native dict.

        This is the form stored in :class:`~langgoap.graph.state.GoapState`
        so that any LangGraph checkpointer can persist it.  Two fields are
        Python callables and therefore not portable:

        * a callable ``value`` is stored as ``None`` and restored as ``1.0``;
        * ``metrics`` are recorded by ``name`` / ``weight`` / class name only
          and are **not** reconstructed by :meth:`from_dict`.

        Within one process LangGOAP keeps the original object reachable so
        both survive a checkpoint round-trip; see
        :func:`langgoap.graph.state.goal_from_state`.
        """
        objectives = (
            {k: getattr(v, "value", v) for k, v in self.objectives.items()}
            if self.objectives is not None
            else None
        )
        return {
            "kind": "goal",
            "conditions": dict(self.conditions),
            "policy": self.policy.to_dict(),
            "objectives": objectives,
            "constraints": [c.to_dict() for c in self.constraints],
            "soft_goals": [sg.to_dict() for sg in self.soft_goals],
            "metrics": [
                {"name": m.name, "weight": m.weight, "type": type(m).__name__}
                for m in self.metrics
            ],
            "value": None if callable(self.value) else self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoalSpec:
        """Reconstruct from the dict produced by :meth:`to_dict`.

        ``metrics`` are not restored (their expressions are callables) and
        a ``None`` ``value`` becomes the default ``1.0``.
        """
        objectives = data.get("objectives")
        value = data.get("value")
        return cls(
            conditions=MappingProxyType(dict(data.get("conditions") or {})),
            policy=GoalPolicy.from_dict(data.get("policy") or {}),
            objectives=(
                MappingProxyType(
                    {k: ObjectiveDirection(v) for k, v in objectives.items()}
                )
                if objectives
                else None
            ),
            constraints=tuple(
                ConstraintSpec.from_dict(c) for c in data.get("constraints") or ()
            ),
            soft_goals=tuple(
                SoftGoal.from_dict(sg) for sg in data.get("soft_goals") or ()
            ),
            value=1.0 if value is None else float(value),
        )

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
            **kwargs: Any other ``GoalSpec`` fields (e.g.
                ``policy=GoalPolicy(...)``, ``soft_goals=...``,
                ``metrics=...``).

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
                child (e.g. ``policy=GoalPolicy(...)``, ``soft_goals=``,
                ``metrics=``).

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
      A sub-goal that is already satisfied is passed over without
      planning.  A sub-goal that cannot be planned ends the run with
      ``no_plan`` (``on_infeasible="fail"``, the default); with
      ``on_infeasible="skip"`` the planner moves on to the next
      sub-goal, records the index in ``GoapState.skipped_subgoals``,
      and the run finishes with ``status="partial"`` instead of
      ``"goal_achieved"`` (issue #35).
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
        mode: Either ``"sequential"``, ``"any"`` or ``"best_value"``.
        on_infeasible: Sequential mode only.  ``"fail"`` (default) stops
            with ``no_plan`` at the first sub-goal that cannot be planned;
            ``"skip"`` continues with the next one and reports ``partial``.
    """

    goals: tuple[GoalSpec, ...]
    mode: Literal["sequential", "any", "best_value"] = "sequential"
    on_infeasible: Literal["fail", "skip"] = "fail"

    def __post_init__(self) -> None:
        if not isinstance(self.goals, tuple):
            object.__setattr__(self, "goals", tuple(self.goals))
        if not self.goals:
            raise ValueError("MultiGoal must contain at least one GoalSpec")
        if self.mode not in ("sequential", "any", "best_value"):
            raise ValueError(
                "MultiGoal.mode must be 'sequential', 'any', or 'best_value'; "
                f"got {self.mode!r}"
            )
        if self.on_infeasible not in ("fail", "skip"):
            raise ValueError(
                "MultiGoal.on_infeasible must be 'fail' or 'skip'; "
                f"got {self.on_infeasible!r}"
            )
        for i, g in enumerate(self.goals):
            if not isinstance(g, GoalSpec):
                raise ValueError(
                    f"MultiGoal.goals[{i}] must be a GoalSpec, got "
                    f"{type(g).__name__}. Nested MultiGoal (HTN-style "
                    "decomposition) is not supported in v0.1.0."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-native dict (inverse of :meth:`from_dict`)."""
        return {
            "kind": "multi",
            "goals": [g.to_dict() for g in self.goals],
            "mode": self.mode,
            "on_infeasible": self.on_infeasible,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultiGoal:
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        return cls(
            goals=tuple(GoalSpec.from_dict(g) for g in data["goals"]),
            mode=data.get("mode", "sequential"),
            on_infeasible=data.get("on_infeasible", "fail"),
        )


def goal_from_dict(data: Mapping[str, Any]) -> GoalSpec | MultiGoal:
    """Rebuild a :class:`GoalSpec` or :class:`MultiGoal` from its dict form.

    Dispatches on the ``"kind"`` tag written by :meth:`GoalSpec.to_dict`
    (``"goal"``), :meth:`MultiGoal.to_dict` (``"multi"``) and
    :class:`~langgoap.planner.utility.NirvanaGoal` (``"nirvana"``); a dict
    without the tag is treated as a single goal unless it carries a
    ``"goals"`` list.
    """
    kind = data.get("kind")
    if kind == "multi" or (kind is None and "goals" in data):
        return MultiGoal.from_dict(data)
    if kind == "nirvana":
        from langgoap.planner.utility import NirvanaGoal

        return NirvanaGoal.from_dict(data)
    return GoalSpec.from_dict(data)
