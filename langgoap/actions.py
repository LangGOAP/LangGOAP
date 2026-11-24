"""Action specifications and decorator for GOAP planning."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Callable, Iterable, cast

from langgoap._worker import run_in_worker
from langgoap.qos import ActionQos
from langgoap.state import require_hashable_values
from langgoap.types import CostFunction

EffectFunction = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _unresolved_dynamic_effect(world_state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Placeholder installed when a callable effect fails to deserialize.

    Callable effects cannot cross process boundaries — arbitrary Python
    functions (closures, lambdas, bound methods) are not portable
    through msgpack/JSON checkpoints.  When an :class:`ActionSpec` with
    a callable ``effects`` is restored from a checkpoint, the function
    is lost; this sentinel is installed in its place so the action
    continues to report ``has_dynamic_effects == True`` (keeping
    planner and static-analysis behavior consistent) while raising a
    clear error if the effect is accidentally invoked before the
    caller re-binds the real callable from their action-spec source.
    """
    raise RuntimeError(
        "ActionSpec callable effect was restored from a checkpoint without "
        "being re-bound. Callable effects cannot cross process boundaries; "
        "rebuild your ActionSpec list from source and pass it to the "
        "executor/graph so the real callable is wired back in."
    )


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Specification of a GOAP action for planning.

    Attributes:
        name: Unique identifier for the action.  May be omitted when
            ``execute`` (or ``aexecute``) is a named function or method: the
            action is then named after it (issue #98).
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
        cost_keys: For a cost function, the state keys it reads (``()`` when
            it reads nothing the search changes). The planner then asks it
            once per distinct value of those keys in a planning call, keeps
            the actions that write them, and treats a cost whose keys no
            searched action writes as a constant its heuristic may use.
            ``None`` (the default) declares nothing: the cost is asked at
            every evaluation, an action whose only purpose is to lower it is
            not searched, and the heuristic falls back to zero.
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
            a checkpointer.  ``True`` or ``{"approved": True}`` continues
            execution; ``False`` or ``{"approved": False, "reason": ...}``
            denies the action and triggers replanning; any other resume
            value (a string, a number, a dict without a boolean
            ``approved``) is not an answer and is asked again.  A Pydantic ``BaseModel``
            subclass instead of ``True`` makes the pause a typed form: the
            interrupt payload carries the model's JSON schema, a resume
            payload that validates is stored under ``human_input_key``, one
            that fails validation is asked again with the errors in the next
            payload, and ``False`` / ``{"approved": False}`` declines the
            form like a denied boolean gate.
        resources: Estimated resource consumption for this action.
            Used by the CSP optimizer.
            Example: ``{"tokens": 500, "cost_usd": 0.02, "api_calls": 1}``.
        duration: Estimated wall-clock duration.
            Used by the CSP optimizer for temporal scheduling.
        metadata: Arbitrary key-value metadata (description, tags, etc.).
        observe: Extra state keys to read from the sensors after this action
            runs, besides its effect keys.  Used with
            ``GoapGraph(effects="observed")`` and by waits (issue #96).
        is_wait: ``True`` for an action built with :meth:`wait`: the executor
            pauses the graph until a sensed value of one of the action's keys
            changes.
        on_resume: For a wait, a callable ``(world_state) -> None`` (sync or
            async) run after every resume, before the sensors are asked.
        wait_payload: For a wait, extra entries for its interrupt payload.
    """

    name: str = ""
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
    cost_keys: frozenset[str] | None = None
    execute: Callable[..., Any] | None = None
    aexecute: Callable[..., Any] | None = None
    effect_validator: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None
    max_retries: int = 0
    qos: ActionQos | None = None
    can_rerun: bool = True
    read_only: bool = False
    utility: float | CostFunction | None = None
    require_human_approval: bool | type = False
    human_input_key: str | None = None
    # CSP optimizer inputs (ignored by the A* planner)
    resources: Mapping[str, float] | None = None
    duration: timedelta | None = None
    metadata: Mapping[str, Any] | None = None
    # Observed effects and waits (issue #96)
    observe: tuple[str, ...] = ()
    is_wait: bool = False
    on_resume: Callable[..., Any] | None = None
    wait_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            function = self.execute if self.execute is not None else self.aexecute
            derived = getattr(function, "__name__", None)
            if not derived or derived == "<lambda>":
                raise ValueError(
                    "ActionSpec needs a name: pass name=..., or an execute / "
                    "aexecute that is a named function (a lambda or a "
                    "functools.partial has no name to default to)."
                )
            object.__setattr__(self, "name", derived)
        if isinstance(self.observe, str):
            raise TypeError(
                "ActionSpec.observe must be an iterable of state keys, not a "
                f"string; did you mean observe=({self.observe!r},)?"
            )
        if not isinstance(self.observe, tuple):
            object.__setattr__(self, "observe", tuple(self.observe))
        if isinstance(self.cost_keys, str):
            raise TypeError(
                "ActionSpec.cost_keys must be an iterable of state keys, not a "
                f"string; did you mean cost_keys=({self.cost_keys!r},)?"
            )
        if self.cost_keys is not None:
            if not callable(self.cost):
                raise ValueError(
                    f"ActionSpec {self.name!r} declares cost_keys="
                    f"{sorted(self.cost_keys)!r} but its cost is the number "
                    f"{self.cost!r}; cost_keys says what a cost function reads."
                )
            if not isinstance(self.cost_keys, frozenset):
                object.__setattr__(self, "cost_keys", frozenset(self.cost_keys))
        if not self.is_wait and (
            self.on_resume is not None or self.wait_payload is not None
        ):
            raise ValueError(
                f"ActionSpec({self.name!r}): on_resume and wait_payload only "
                "apply to a wait; build the action with ActionSpec.wait(...)."
            )
        # preconditions is always a Mapping; wrap for immutability.
        if not isinstance(self.preconditions, MappingProxyType):
            object.__setattr__(
                self, "preconditions", MappingProxyType(dict(self.preconditions))
            )
        require_hashable_values(
            self.preconditions,
            owner=f"ActionSpec({self.name!r})",
            field_name="preconditions",
        )
        # Deserialization path: the serializer emits ``effects=None`` for
        # callable effects (functions are not portable).  When
        # ``effect_keys`` is present, we know the original action was
        # dynamic, so install the sentinel callable in place of ``None``
        # before the normal validation runs.
        if self.effects is None and self.effect_keys is not None:
            object.__setattr__(self, "effects", _unresolved_dynamic_effect)
        # effects is either a Mapping or a callable.  Only wrap Mappings.
        if callable(self.effects) and not isinstance(self.effects, Mapping):
            if self.effect_keys is None:
                raise ValueError(
                    "ActionSpec with callable effects must declare effect_keys "
                    "— a frozenset[str] of state keys the callable may "
                    "produce. Example:\n"
                    "    ActionSpec(\n"
                    "        name='eat',\n"
                    "        effects=eat_fn,\n"
                    "        effect_keys=frozenset({'food'}),\n"
                    "    )"
                )
            if isinstance(self.effect_keys, (set, list, tuple)):
                object.__setattr__(self, "effect_keys", frozenset(self.effect_keys))
            elif not isinstance(self.effect_keys, frozenset):
                raise TypeError(
                    "ActionSpec.effect_keys must be a frozenset[str] (or a "
                    f"set/list/tuple coercible to one); got "
                    f"{type(self.effect_keys).__name__}."
                )
            if not self.effect_keys:
                raise ValueError(
                    "ActionSpec.effect_keys must be non-empty — a dynamic "
                    "effect that produces no state changes has no planning "
                    "value."
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
            require_hashable_values(
                self.effects,
                owner=f"ActionSpec({self.name!r})",
                field_name="effects",
            )
        for attr in ("resources", "metadata", "wait_payload"):
            val = getattr(self, attr)
            if val is not None and not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the *declarative* part of this action to a JSON-native dict.

        Callables are not portable, so ``execute``, ``aexecute``,
        ``effect_validator`` and a dynamic ``effects`` / ``cost`` /
        ``utility`` are recorded as ``None`` (dynamic effects keep their
        ``effect_keys``), a typed-form ``require_human_approval`` class is
        recorded as ``True``, and ``qos.retryable_exceptions`` is recorded by
        class name only.  The graph nodes never read callables from a
        checkpoint: they rebind the action by ``name`` from the live action
        list given at compile time.  ``duration`` is stored in seconds.
        """
        return {
            "name": self.name,
            "preconditions": dict(self.preconditions),
            "effects": (
                None
                if self.has_dynamic_effects
                else dict(cast(Mapping[str, Any], self.effects))
            ),
            "effect_keys": (
                sorted(self.effect_keys) if self.effect_keys is not None else None
            ),
            "cost": None if callable(self.cost) else self.cost,
            "cost_keys": sorted(self.cost_keys) if self.cost_keys is not None else None,
            "max_retries": self.max_retries,
            "qos": self.qos.to_dict() if self.qos is not None else None,
            "can_rerun": self.can_rerun,
            "read_only": self.read_only,
            "utility": (
                None if self.utility is None or callable(self.utility) else self.utility
            ),
            "require_human_approval": bool(self.require_human_approval),
            "human_input_key": self.human_input_key,
            "resources": dict(self.resources) if self.resources is not None else None,
            "duration": (
                self.duration.total_seconds() if self.duration is not None else None
            ),
            "metadata": dict(self.metadata) if self.metadata is not None else None,
            "observe": list(self.observe),
            "is_wait": self.is_wait,
            "wait_payload": (
                dict(self.wait_payload) if self.wait_payload is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionSpec:
        """Reconstruct a callable-free action from :meth:`to_dict` output.

        A ``None`` ``cost`` (the original was a cost function) becomes the
        default ``1.0`` and its ``cost_keys`` go with it; dynamic effects come back as the unresolved
        sentinel that :meth:`get_effects` reports clearly.  Prefer binding
        the stored name to the live :class:`ActionSpec` when one exists
        (see :meth:`langgoap.planner.types.Plan.from_dict`).
        """
        effect_keys = data.get("effect_keys")
        cost = data.get("cost")
        duration = data.get("duration")
        qos = data.get("qos")
        return cls(
            name=data["name"],
            preconditions=dict(data.get("preconditions") or {}),
            # ``None`` + ``effect_keys`` is the documented deserialisation path for
            # dynamic effects (see ``__post_init__``).
            effects=(
                cast(Any, None)
                if effect_keys is not None
                else dict(data.get("effects") or {})
            ),
            effect_keys=frozenset(effect_keys) if effect_keys is not None else None,
            cost=1.0 if cost is None else cost,
            max_retries=int(data.get("max_retries", 0)),
            qos=ActionQos.from_dict(qos) if qos is not None else None,
            can_rerun=bool(data.get("can_rerun", True)),
            read_only=bool(data.get("read_only", False)),
            utility=data.get("utility"),
            require_human_approval=bool(data.get("require_human_approval", False)),
            human_input_key=data.get("human_input_key"),
            resources=data.get("resources"),
            duration=timedelta(seconds=duration) if duration is not None else None,
            metadata=data.get("metadata"),
            observe=tuple(data.get("observe") or ()),
            is_wait=bool(data.get("is_wait", False)),
            wait_payload=data.get("wait_payload"),
        )

    @classmethod
    def wait(
        cls,
        name: str,
        preconditions: Mapping[str, Any] | None = None,
        effects: Mapping[str, Any] | EffectFunction | None = None,
        *,
        observe: Iterable[str] = (),
        on_resume: Callable[..., Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> ActionSpec:
        """An action that pauses the graph until the world changes (issue #96).

        The executor interrupts with ``{"type": "goap_action_wait", "action":
        name, **payload}``.  Every resume (``GoapGraph.notify`` or
        ``Command(resume=...)``) makes the wait call ``on_resume`` once, ask
        the graph's sensors once, and complete only when one of its keys (the
        effect keys, ``observe`` and the graph's ``always_observe``) differs
        from the value it had when the wait began; otherwise it pauses again
        in a fresh task, so a resume costs the same however many came before
        it.  It completes with what the sensors report, whatever the plan
        expected; the observer re-plans on a deviation as for any action.

        ``execute`` / ``aexecute`` may be given in ``options`` to prepare the
        wait (send the offer, start the timer).  It runs before the first
        pause and, because LangGraph replays a paused task on resume, once
        more on the first resume; it must be idempotent.  A dict it returns
        is part of the state the wait completes with.

        A graph with a wait needs sensors and, at run time, a checkpointer.

        Example::

            ActionSpec.wait(
                "await_payment",
                {"paid": False},
                {"paid": True},
                observe=("payment_failed",),
                on_resume=confirm_if_paid,
            )
        """
        return cls(
            name,
            preconditions or {},
            effects if effects is not None else {},
            observe=tuple(observe),
            is_wait=True,
            on_resume=on_resume,
            wait_payload=payload,
            **options,
        )

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

        .. deprecated:: 0.2.0
            Nothing in the runtime consults this; the executors resolve
            the async path themselves.  Removed in 0.3.0 (#64).
        """
        warnings.warn(
            "ActionSpec.is_async_execute() is deprecated and will be removed "
            "in 0.3.0; the executors pick the async path themselves.",
            DeprecationWarning,
            stacklevel=2,
        )
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


def ensure_unique_action_names(actions: Iterable[ActionSpec]) -> None:
    """Raise ``ValueError`` when two actions share a name (issue #48).

    Blacklists, failure counts, history records and executor rebinding are
    all keyed by ``ActionSpec.name``, so a duplicate makes the planner plan
    one action and the executor run another.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for action in actions:
        if action.name in seen and action.name not in duplicates:
            duplicates.append(action.name)
        seen.add(action.name)
    if duplicates:
        raise ValueError(
            f"Duplicate action names {duplicates!r}: blacklists, failure counts "
            "and executor rebinding are keyed by name, so every ActionSpec "
            "needs a unique name."
        )


def goap_action(
    *,
    preconditions: dict[str, Any] | None = None,
    effects: Mapping[str, Any] | EffectFunction | None = None,
    effect_keys: frozenset[str] | set[str] | None = None,
    cost: float | CostFunction = 1.0,
    cost_keys: Iterable[str] | None = None,
    name: str | None = None,
    max_retries: int = 0,
    qos: ActionQos | None = None,
    can_rerun: bool = True,
    read_only: bool = False,
    utility: float | CostFunction | None = None,
    require_human_approval: bool | type = False,
    human_input_key: str | None = None,
    resources: dict[str, float] | None = None,
    duration: timedelta | None = None,
    metadata: dict[str, Any] | None = None,
    observe: Iterable[str] = (),
    is_wait: bool = False,
    on_resume: Callable[..., Any] | None = None,
    wait_payload: Mapping[str, Any] | None = None,
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

    **Dynamic effects** — pass a callable and declare the keys it writes::

        @goap_action(
            effects=lambda ws: {"count": ws.get("count", 0) + 1},
            effect_keys={"count"},
        )
        def bump(state: dict) -> dict:
            return {"count": state.get("count", 0) + 1}

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
            effects=effects if effects is not None else {},
            effect_keys=frozenset(effect_keys) if effect_keys else None,
            cost=cost,
            cost_keys=frozenset(cost_keys) if cost_keys is not None else None,
            # Async functions must NOT be stored in `execute` — the sync
            # executor would call them and receive a coroutine object that
            # is never awaited, silently losing runtime results and leaking
            # the coroutine.  Store async callables only in `aexecute` so
            # the sync executor can detect and reject them at call time.
            execute=None if is_async else func,
            aexecute=func if is_async else None,
            max_retries=max_retries,
            qos=qos,
            can_rerun=can_rerun,
            read_only=read_only,
            utility=utility,
            require_human_approval=require_human_approval,
            human_input_key=human_input_key,
            resources=resources,
            duration=duration,
            metadata=metadata,
            observe=tuple(observe),
            is_wait=is_wait,
            on_resume=on_resume,
            wait_payload=wait_payload,
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
    #: Static effects mapping, or a callable ``(world_state) -> effects``
    #: (declare it as a ``@staticmethod`` and set :attr:`effect_keys`).
    effects: dict[str, Any] | EffectFunction = {}
    #: Keys a callable :attr:`effects` may write; required with callable effects.
    effect_keys: frozenset[str] | set[str] | None = None
    #: Keys a :meth:`cost` override reads; see ``ActionSpec.cost_keys``.
    cost_keys: frozenset[str] | set[str] | None = None
    max_retries: int = 0
    qos: ActionQos | None = None
    can_rerun: bool = True
    read_only: bool = False
    utility: float | CostFunction | None = None
    require_human_approval: bool | type = False
    human_input_key: str | None = None
    #: CSP resource consumption per run (e.g. ``{"cost_usd": 0.02}``).
    resources: dict[str, float] | None = None
    #: Wall-clock duration for CP-SAT temporal scheduling.
    duration: timedelta | None = None
    #: Free-form, JSON-serialisable metadata attached to the action.
    metadata: dict[str, Any] | None = None
    #: Extra state keys read from the sensors after the action (observed effects).
    observe: tuple[str, ...] = ()
    #: ``True`` makes the action a wait (see :meth:`ActionSpec.wait`):
    #: :meth:`execute` prepares it and :meth:`on_resume` runs on every resume.
    is_wait: bool = False
    #: Extra entries for a wait's interrupt payload.
    wait_payload: dict[str, Any] | None = None

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
        return await run_in_worker(self.execute, state)

    def on_resume(self, state: dict[str, Any]) -> Any:
        """For a wait (``is_wait = True``): called after every resume, before
        the sensors are asked.  Override it; may be ``async``."""
        return None

    def validate_effects(
        self, pre_state: dict[str, Any], post_state: dict[str, Any]
    ) -> bool:
        """Verify that execution produced the expected effects.

        Override to add custom postcondition checks.  The default
        implementation verifies every declared effect is present in
        ``post_state`` (callable effects are resolved against ``pre_state``).
        """
        declared = (
            self.effects(pre_state)
            if callable(self.effects) and not isinstance(self.effects, Mapping)
            else self.effects
        )
        return all(post_state.get(k) == v for k, v in declared.items())

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

        # A wait's on_resume, only when the subclass provides one.
        custom_on_resume = None
        if self.is_wait and type(self).on_resume is not GoapAction.on_resume:
            custom_on_resume = self.on_resume

        # The default cost() is the constant 1.0: hand the planner the number,
        # not a callable, so class-based actions keep its heuristic (#118).
        cost: float | CostFunction = (
            self.cost if type(self).cost is not GoapAction.cost else 1.0
        )

        effects: Mapping[str, Any] | EffectFunction
        if callable(self.effects) and not isinstance(self.effects, Mapping):
            effects = self.effects
        else:
            effects = dict(self.effects)
        return ActionSpec(
            name=type(self).__name__,
            preconditions=dict(self.preconditions),
            effects=effects,
            effect_keys=frozenset(self.effect_keys) if self.effect_keys else None,
            cost=cost,
            cost_keys=(
                frozenset(self.cost_keys) if self.cost_keys is not None else None
            ),
            execute=self.execute,
            aexecute=custom_aexecute,
            effect_validator=custom_validator,
            max_retries=self.max_retries,
            qos=self.qos,
            can_rerun=self.can_rerun,
            read_only=self.read_only,
            utility=self.utility,
            require_human_approval=self.require_human_approval,
            human_input_key=self.human_input_key,
            resources=self.resources,
            duration=self.duration,
            metadata=self.metadata,
            observe=tuple(self.observe),
            is_wait=self.is_wait,
            on_resume=custom_on_resume,
            wait_payload=self.wait_payload,
        )
