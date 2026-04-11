"""Fluent builder for CSP constraints and objectives.

Lets users construct ``ConstraintSpec`` tuples and objective maps in a
readable style inspired by OptaPlanner's ``ConstraintProvider``::

    from langgoap.constraints import ConstraintBuilder

    output = ConstraintBuilder.build(
        ConstraintBuilder.for_plan()
            .sum_resource("gpu_hours")
            .bounded(max=100)
            .penalize(level="hard", weight=1.0)
            .as_constraint("gpu_budget"),
        ConstraintBuilder.for_plan()
            .sum_resource("cost_usd")
            .minimize()
            .weight(2.0)
            .as_objective("cost"),
    )

    goal = GoalSpec.from_builder(
        conditions={"done": True},
        builder_output=output,
    )

The builder is a typed construction helper, not a new field on
``GoalSpec``.  It always terminates in either ``.as_constraint(name)``
(emitting a :class:`~langgoap.goals.ConstraintSpec`) or
``.as_objective(name)`` (emitting an entry in the ``objectives`` map).
:func:`ConstraintBuilder.build` bundles the two lists into a
:class:`BuilderOutput` that ``GoalSpec.from_builder`` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from langgoap.goals import ConstraintSpec
from langgoap.types import ObjectiveDirection

# ---------------------------------------------------------------------------
# Public output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuilderOutput:
    """Tagged bundle returned by :meth:`ConstraintBuilder.build`.

    Attributes:
        constraints: Tuple of :class:`~langgoap.goals.ConstraintSpec`
            instances — hard and soft constraints.
        objectives: Immutable mapping from objective key to direction.
    """

    constraints: tuple[ConstraintSpec, ...] = ()
    objectives: MappingProxyType[str, ObjectiveDirection] = field(
        default_factory=lambda: MappingProxyType({})
    )


# ---------------------------------------------------------------------------
# Internal chain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ChainState:
    """Accumulated state for a single fluent chain.

    Chains are immutable — each builder step returns a new instance.
    Termination via ``as_constraint`` or ``as_objective`` produces a
    :class:`ChainOutput`.
    """

    resource_key: str | None = None
    min_bound: float | None = None
    max_bound: float | None = None
    weight: float = 1.0
    level: Literal["hard", "soft"] = "hard"
    direction: ObjectiveDirection | None = None


@dataclass(frozen=True, slots=True)
class ChainOutput:
    """Output of a single terminated builder chain.

    Exactly one of ``constraint`` or ``objective`` is set.  Consumed by
    :meth:`ConstraintBuilder.build`.
    """

    constraint: ConstraintSpec | None = None
    objective: tuple[str, ObjectiveDirection] | None = None


# ---------------------------------------------------------------------------
# Fluent chain
# ---------------------------------------------------------------------------


class ConstraintChain:
    """A single in-progress fluent chain.

    Users should not instantiate this directly — use the factories on
    :class:`ConstraintBuilder`.
    """

    def __init__(self, state: _ChainState) -> None:
        self._state = state

    def sum_resource(self, key: str) -> ConstraintChain:
        """Sum the given resource key across all matching actions."""
        return ConstraintChain(
            _ChainState(
                resource_key=key,
                min_bound=self._state.min_bound,
                max_bound=self._state.max_bound,
                weight=self._state.weight,
                level=self._state.level,
                direction=self._state.direction,
            )
        )

    def bounded(
        self,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> ConstraintChain:
        """Attach resource bounds to this chain."""
        return ConstraintChain(
            _ChainState(
                resource_key=self._state.resource_key,
                min_bound=min if min is not None else self._state.min_bound,
                max_bound=max if max is not None else self._state.max_bound,
                weight=self._state.weight,
                level=self._state.level,
                direction=self._state.direction,
            )
        )

    def penalize(
        self,
        *,
        level: Literal["hard", "soft"] = "hard",
        weight: float = 1.0,
    ) -> ConstraintChain:
        """Mark this chain as a penalty-emitting constraint."""
        return ConstraintChain(
            _ChainState(
                resource_key=self._state.resource_key,
                min_bound=self._state.min_bound,
                max_bound=self._state.max_bound,
                weight=weight,
                level=level,
                direction=self._state.direction,
            )
        )

    def minimize(self) -> ConstraintChain:
        """Mark this chain as a minimize objective."""
        return ConstraintChain(
            _ChainState(
                resource_key=self._state.resource_key,
                min_bound=self._state.min_bound,
                max_bound=self._state.max_bound,
                weight=self._state.weight,
                level=self._state.level,
                direction=ObjectiveDirection.MINIMIZE,
            )
        )

    def maximize(self) -> ConstraintChain:
        """Mark this chain as a maximize objective."""
        return ConstraintChain(
            _ChainState(
                resource_key=self._state.resource_key,
                min_bound=self._state.min_bound,
                max_bound=self._state.max_bound,
                weight=self._state.weight,
                level=self._state.level,
                direction=ObjectiveDirection.MAXIMIZE,
            )
        )

    def weight(self, w: float) -> ConstraintChain:
        """Attach a weight to this chain."""
        return ConstraintChain(
            _ChainState(
                resource_key=self._state.resource_key,
                min_bound=self._state.min_bound,
                max_bound=self._state.max_bound,
                weight=w,
                level=self._state.level,
                direction=self._state.direction,
            )
        )

    # ------------------------------------------------------------------
    # Terminators
    # ------------------------------------------------------------------

    def as_constraint(self, name: str) -> ChainOutput:
        """Terminate this chain and produce a ConstraintSpec.

        Args:
            name: Name used as the ``ConstraintSpec.key``.  For
                resource-aggregating chains, this is the output key
                (which may differ from the ``sum_resource`` key when
                the user wants a renamed constraint).

        Raises:
            ValueError: If the chain was not configured with bounds.
        """
        if self._state.min_bound is None and self._state.max_bound is None:
            raise ValueError(
                f"Chain '{name}' has no bounds; call .bounded(min=..., max=...) "
                "before .as_constraint()."
            )
        key = self._state.resource_key or name
        spec = ConstraintSpec(
            key=key,
            min=self._state.min_bound,
            max=self._state.max_bound,
            weight=self._state.weight,
            level=self._state.level,
        )
        return ChainOutput(constraint=spec)

    def as_objective(self, name: str) -> ChainOutput:
        """Terminate this chain and produce an objective entry.

        Args:
            name: Objective key.  Usually equal to the underlying
                resource key, but can be any string.

        Raises:
            ValueError: If the chain has no direction set (``.minimize``
                or ``.maximize`` must have been called).
        """
        if self._state.direction is None:
            raise ValueError(
                f"Chain '{name}' has no direction; call .minimize() or "
                ".maximize() before .as_objective()."
            )
        return ChainOutput(objective=(name, self._state.direction))


# ---------------------------------------------------------------------------
# Public builder facade
# ---------------------------------------------------------------------------


class ConstraintBuilder:
    """Factory and aggregator for :class:`ConstraintChain` instances.

    Chains start from :meth:`for_plan` (planet-wide aggregate) or
    :meth:`for_each_action` (placeholder for future per-action filtering;
    currently identical to ``for_plan`` because all constraints are
    plan-level).  Call :meth:`build` with the chain outputs to get a
    :class:`BuilderOutput`.
    """

    @staticmethod
    def for_plan() -> ConstraintChain:
        """Start a new chain aggregating over the whole plan."""
        return ConstraintChain(_ChainState())

    @staticmethod
    def for_each_action() -> ConstraintChain:
        """Start a new chain (alias for :meth:`for_plan` for now).

        Reserved for future per-action filtering; currently produces the
        same plan-level aggregate as :meth:`for_plan`.
        """
        return ConstraintChain(_ChainState())

    @staticmethod
    def build(*outputs: ChainOutput) -> BuilderOutput:
        """Collect chain outputs into a :class:`BuilderOutput`."""
        constraints: list[ConstraintSpec] = []
        objectives: dict[str, ObjectiveDirection] = {}
        for out in outputs:
            if out.constraint is not None:
                constraints.append(out.constraint)
            if out.objective is not None:
                key, direction = out.objective
                if key in objectives:
                    raise ValueError(
                        f"Duplicate objective key {key!r} in builder output."
                    )
                objectives[key] = direction
        return BuilderOutput(
            constraints=tuple(constraints),
            objectives=MappingProxyType(objectives),
        )


__all__ = [
    "BuilderOutput",
    "ChainOutput",
    "ConstraintBuilder",
    "ConstraintChain",
]
