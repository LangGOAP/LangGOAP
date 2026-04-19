"""Action specifications and decorator for GOAP planning."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Callable

from langgoap.types import CostFunction

EffectFunction = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Specification of a GOAP action for planning.

    Attributes:
        name: Unique identifier for the action.
        preconditions: World state conditions that must hold before execution.
            Stored as an immutable MappingProxyType to match the frozen
            contract of this dataclass.
        effects: World state changes produced by execution.  Either a
            static mapping (stored as an immutable MappingProxyType) or a
            callable ``(world_state) -> Mapping[str, Any]`` for
            state-dependent transitions.  Callable effects must declare
            ``effect_keys`` so the planner's static analysis passes can
            reason about which state keys the action may touch.
        effect_keys: Required when ``effects`` is callable; lists the
            state keys the callable may produce.  Must be ``None`` for
            static effects.
        cost: Static cost or callable that computes cost from world state.
        execute: The function to call when the action is executed (sync).
        aexecute: Optional async execute callable. When set, the async
            executor will ``await`` this instead of calling ``execute``.
        effect_validator: Optional runtime postcondition checker.
            Signature: ``(pre_state: dict, post_state: dict) -> bool``.
            Called by the executor after execution to verify actual effects.
            Returning ``False`` signals that the action did not produce its
            declared effects (treated as a soft failure prompting replanning).
        max_retries: Number of allowed retries before blacklisting.
            ``0`` (default) means the action is blacklisted on its first
            failure.  ``N`` allows N retries before blacklisting (i.e.,
            the action is blacklisted after N+1 total failures).
        require_human_approval: When ``True``, the executor calls
            ``interrupt()`` before running this action, pausing the graph
            until a human resumes via ``Command(resume=...)``.  Requires
            a checkpointer.  The resume value is interpreted as approval:
            ``None``, ``True``, or ``{"approved": True}`` continue
            execution; ``False`` or ``{"approved": False}`` deny the
            action and trigger replanning.
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
    effects: Mapping[str, Any] | EffectFunction = field(
        default_factory=lambda: MappingProxyType({})
    )
    effect_keys: frozenset[str] | None = None
    cost: float | CostFunction = 1.0
    execute: Callable[..., Any] | None = None
    aexecute: Callable[..., Any] | None = None
    effect_validator: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None
    max_retries: int = 0
    require_human_approval: bool = False
    # CSP optimizer inputs (ignored by the A* planner)
    resources: Mapping[str, float] | None = None
    duration: timedelta | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # preconditions is always a Mapping; wrap for immutability.
        if not isinstance(self.preconditions, MappingProxyType):
            object.__setattr__(
                self, "preconditions", MappingProxyType(dict(self.preconditions))
            )
        # effects is either a Mapping or a callable.  Only wrap Mappings.
        if callable(self.effects) and not isinstance(self.effects, Mapping):
            if self.effect_keys is None:
                raise ValueError(
                    "ActionSpec with callable effects must declare effect_keys "
                    "(a frozenset of state keys the effect callable may produce)."
                )
        else:
            if self.effect_keys is not None:
                raise ValueError(
                    "ActionSpec.effect_keys is only valid when effects is a "
                    "callable; for static effects the keys are read from the "
                    "mapping itself."
                )
            if not isinstance(self.effects, MappingProxyType):
                object.__setattr__(
                    self, "effects", MappingProxyType(dict(self.effects))
                )
        for attr in ("resources", "metadata"):
            val = getattr(self, attr)
            if val is not None and not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))

    @property
    def has_dynamic_effects(self) -> bool:
        """Return True if effects is resolved from live world state."""
        return callable(self.effects) and not isinstance(self.effects, Mapping)

    def effect_key_set(self) -> frozenset[str]:
        """Return the set of state keys this action may touch.

        For static effects this is the set of mapping keys; for dynamic
        effects it is the declared :attr:`effect_keys` (required at
        construction time).  Used by static-analysis passes that need
        to reason about the action's footprint without executing it.
        """
        if self.has_dynamic_effects:
            return self.effect_keys or frozenset()
        return frozenset(self.effects.keys())  # type: ignore[union-attr]

    def get_effects(self, world_state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Resolve effects against ``world_state``.

        For static effects this returns the stored MappingProxyType
        unchanged.  For callable effects the callable is invoked with
        ``world_state`` and the result is wrapped in a MappingProxyType
        so downstream code can treat both shapes uniformly.
        """
        if self.has_dynamic_effects:
            result = self.effects(world_state)  # type: ignore[operator]
            if isinstance(result, MappingProxyType):
                return result
            return MappingProxyType(dict(result))
        # Static mapping — already a MappingProxyType after __post_init__.
        return self.effects  # type: ignore[return-value]

    def get_cost(self, world_state: dict[str, Any] | None = None) -> float:
        """Resolve the cost, calling the cost function if dynamic."""
        if callable(self.cost):
            return self.cost(world_state or {})
        return self.cost

    def has_effects(self) -> bool:
        """Return True if this action declares at least one effect."""
        if self.has_dynamic_effects:
            return True
        return len(self.effects) > 0  # type: ignore[arg-type]

    def is_async_execute(self) -> bool:
        """Return True if this action has an async execute path.

        Checks ``aexecute`` first, then falls back to inspecting
        ``execute`` for coroutine functions.
        """
        if self.aexecute is not None:
            return True
        if self.execute is not None:
            return inspect.iscoroutinefunction(self.execute)
        return False

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if self.preconditions:
            parts.append(f"preconditions={dict(self.preconditions)!r}")
        if self.has_dynamic_effects:
            parts.append(
                f"effects=<dynamic effect_keys={set(self.effect_keys or ())!r}>"
            )
        elif self.effects:
            parts.append(f"effects={dict(self.effects)!r}")  # type: ignore[arg-type]
        if self.cost != 1.0:
            parts.append(f"cost={self.cost!r}")
        if self.max_retries != 0:
            parts.append(f"max_retries={self.max_retries!r}")
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
        # For dynamic effects, resolve against pre_state to obtain the
        # expected post-condition values.
        expected = self.get_effects(pre_state)
        return all(post_state.get(k) == v for k, v in expected.items())


def goap_action(
    *,
    preconditions: dict[str, Any] | None = None,
    effects: dict[str, Any] | None = None,
    cost: float | CostFunction = 1.0,
    name: str | None = None,
    max_retries: int = 0,
    require_human_approval: bool = False,
    resources: dict[str, float] | None = None,
    duration: timedelta | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], ActionSpec]:
    """Decorator that converts a function into an ActionSpec.

    Works with both sync and async functions.  Async functions are stored
    in ``aexecute`` and must be invoked via :meth:`~GoapGraph.ainvoke` —
    the sync executor will raise a clear error if you accidentally call
    :meth:`~GoapGraph.invoke` with an async-only action.

    **Sync usage**::

        @goap_action(
            preconditions={"has_data": True},
            effects={"report_ready": True},
            cost=2.0,
            resources={"tokens": 500, "cost_usd": 0.02},
            duration=timedelta(seconds=2),
        )
        def generate_report(state: dict) -> dict:
            return {"report_ready": True}

    **Async usage** — requires ``GoapGraph.ainvoke()``::

        @goap_action(
            preconditions={"has_question": True},
            effects={"answer_ready": True},
            cost=3.0,
        )
        async def ask_llm(state: dict) -> dict:
            response = await llm.ainvoke(state["question"])
            return {"answer_ready": True, "answer": response.content}
    """

    def wrapper(func: Callable[..., Any]) -> ActionSpec:
        is_async = inspect.iscoroutinefunction(func)
        return ActionSpec(
            name=name or func.__name__,
            preconditions=preconditions or {},
            effects=effects or {},
            cost=cost,
            # Async functions must NOT be stored in `execute` — the sync
            # executor would call them and receive a coroutine object that
            # is never awaited, silently losing runtime results and leaking
            # the coroutine.  Store async callables only in `aexecute` so
            # the sync executor can detect and reject them at call time.
            execute=None if is_async else func,
            aexecute=func if is_async else None,
            max_retries=max_retries,
            require_human_approval=require_human_approval,
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
    max_retries: int = 0
    require_human_approval: bool = False

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

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Async execute. Override for native async actions.

        The default delegates to :meth:`execute` via ``run_in_executor``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, state)

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
        """Convert this action instance to an ActionSpec.

        Only sets ``effect_validator`` when the subclass overrides
        :meth:`validate_effects`.  This keeps runtime postcondition
        checking opt-in — subclasses that don't override get no
        validation overhead in the executor.

        Sets ``aexecute`` when the subclass overrides :meth:`aexecute`,
        enabling native async execution in the async GOAP loop.
        """
        # Detect whether the subclass provides a custom validator.
        custom_validator = None
        if type(self).validate_effects is not GoapAction.validate_effects:
            custom_validator = self.validate_effects

        # Detect whether the subclass provides a custom async executor.
        custom_aexecute = None
        if type(self).aexecute is not GoapAction.aexecute:
            custom_aexecute = self.aexecute

        return ActionSpec(
            name=type(self).__name__,
            preconditions=dict(self.preconditions),
            effects=dict(self.effects),
            cost=self.cost,
            execute=self.execute,
            aexecute=custom_aexecute,
            effect_validator=custom_validator,
            max_retries=self.max_retries,
            require_human_approval=self.require_human_approval,
        )
