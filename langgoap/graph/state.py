"""State types for GOAP execution.

:class:`GoapState` is the typed view (``GoalSpec`` / ``Plan`` /
``ActionResult`` objects) that nodes and ``GoapGraph.invoke`` work with;
:class:`GoapStateChannels` is the LangGraph schema the graph is built
from.  The channels hold **JSON-native** values only:
dicts, lists, strings, numbers, booleans and ``None``.  LangGraph hands
every channel value and every node write to the checkpointer's
serializer, so a JSON-native state is what lets *any*
``BaseCheckpointSaver`` and *any* ``SerializerProtocol`` persist a GOAP
run without LangGOAP-specific plumbing.

Typed objects live at the edges:

* :meth:`GoapGraph.invoke <langgoap.graph.builder.GoapGraph.invoke>` takes
  a :class:`~langgoap.goals.GoalSpec` / :class:`~langgoap.goals.MultiGoal`
  and returns a *rehydrated* state whose ``goal``, ``plan`` and
  ``execution_history`` are typed objects again.
* The planner, executor and observer nodes rehydrate on entry and
  serialise on exit (:func:`rehydrate_state` / :func:`serialize_updates`),
  so strategies, tracers, guards and stuck handlers keep working with
  :class:`~langgoap.planner.types.Plan` and friends.
* Code that drives a compiled graph directly (``compiled.invoke``,
  ``get_state``) sees the raw JSON state and can call
  :func:`rehydrate_state` (or ``GoapGraph.rehydrate``) to get objects.

The ``goal`` and ``execution_history`` channels carry reducers that
accept typed objects and store their JSON form, so passing a ``GoalSpec``
straight into ``compiled.invoke`` keeps working when no checkpointer is
attached.  With a checkpointer LangGraph serialises the *input* before any
reducer runs, so build the input with
:meth:`GoapGraph.initial_state <langgoap.graph.builder.GoapGraph.initial_state>`
or ``goal.to_dict()`` in that case.

Callables cannot be serialised.  Actions are rebound by name from the
compile-time action list, so their callables are never read from a
checkpoint.  A goal's callable ``value`` and its ``metrics`` expressions
have no such registry; :func:`goal_to_state` therefore keeps the original
goal object reachable for the lifetime of the process (bounded cache) and
:func:`goal_from_state` hands it back when the same process rehydrates the
state.  A checkpoint restored in *another* process gets the declarative
goal only: ``value`` falls back to ``1.0`` and metrics are dropped, with a
warning.
"""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

from typing_extensions import TypedDict

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, MultiGoal, goal_from_dict
from langgoap.planner.types import Plan

logger = logging.getLogger("langgoap.graph.state")


@dataclass
class ActionResult:
    """Result of executing a single GOAP action.

    Attributes:
        action_name: Name of the action that was executed.
        success: Whether the action completed successfully.
        state_before: World state before execution.
        state_after: World state after execution.
        error: Error message if the action failed.
        cost: The action's cost evaluated on the full pre-state at execution
            time (``None`` for failures and for records written before
            0.2.0).  Recorded by the executor so realised-cost accounting
            does not depend on the stored snapshots, which
            ``GoapGraph(history_state="delta")`` compacts (issue #74).
    """

    action_name: str
    success: bool
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON-native form stored in ``execution_history``."""
        return {
            "action_name": self.action_name,
            "success": self.success,
            "state_before": dict(self.state_before),
            "state_after": dict(self.state_after),
            "error": self.error,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionResult:
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        return cls(
            action_name=data["action_name"],
            success=bool(data["success"]),
            state_before=dict(data.get("state_before") or {}),
            state_after=dict(data.get("state_after") or {}),
            error=data.get("error"),
            cost=data.get("cost"),
        )


def _as_action_result(entry: Any) -> ActionResult:
    if isinstance(entry, ActionResult):
        return entry
    if isinstance(entry, Mapping):
        return ActionResult.from_dict(entry)
    raise TypeError(
        f"execution_history entries must be ActionResult or dict, got "
        f"{type(entry).__name__}"
    )


def successful_action_names(result: Mapping[str, Any]) -> list[str]:
    """Return names of successfully executed actions from a GOAP result.

    Accepts both the typed state returned by ``GoapGraph.invoke`` and the
    raw JSON state of a compiled graph.  Replaces the common pattern::

        [h.action_name for h in result["execution_history"] if h.success]

    Args:
        result: A :class:`GoapState` (or compatible dict) containing an
            ``execution_history`` key.

    Returns:
        List of action names that completed successfully, in execution order.
    """
    history = [_as_action_result(h) for h in result.get("execution_history", [])]
    return [h.action_name for h in history if h.success]


# ---------------------------------------------------------------------------
# Goal codec with a process-local registry for callable fields
# ---------------------------------------------------------------------------

_LIVE_KEY = "_live"
_LIVE_GOALS_MAX = 1024
_LIVE_GOALS: OrderedDict[str, GoalSpec | MultiGoal] = OrderedDict()
_LIVE_LOCK = threading.Lock()


def _goal_has_callables(goal: GoalSpec | MultiGoal) -> bool:
    if isinstance(goal, MultiGoal):
        return any(_goal_has_callables(g) for g in goal.goals)
    return callable(goal.value) or bool(goal.metrics)


def goal_to_state(
    goal: GoalSpec | MultiGoal | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the JSON-native form of *goal* for the ``goal`` channel.

    A goal that carries callables (a callable ``value`` or any ``metrics``)
    is additionally kept in a bounded process-local registry and tagged
    with a token so :func:`goal_from_state` can hand the original object
    back within this process.  Dicts pass through unchanged.
    """
    if goal is None:
        return None
    if isinstance(goal, (GoalSpec, MultiGoal)):
        data = goal.to_dict()
        if _goal_has_callables(goal):
            token = uuid.uuid4().hex
            with _LIVE_LOCK:
                _LIVE_GOALS[token] = goal
                while len(_LIVE_GOALS) > _LIVE_GOALS_MAX:
                    _LIVE_GOALS.popitem(last=False)
            data[_LIVE_KEY] = token
        return data
    if isinstance(goal, Mapping):
        return dict(goal)
    raise TypeError(
        f"goal must be a GoalSpec, MultiGoal or dict, got {type(goal).__name__}"
    )


def goal_from_state(data: Any) -> GoalSpec | MultiGoal | None:
    """Rebuild the typed goal from the ``goal`` channel value.

    Returns ``None`` for ``None`` and for the empty dict LangGraph uses as
    the channel's initial value.  Typed objects pass through.  When the
    dict carries a registry token (see :func:`goal_to_state`) and the
    original object is still reachable in this process, that object is
    returned so callable ``value`` / ``metrics`` are intact; otherwise the
    declarative goal is rebuilt and a warning is logged.
    """
    if data is None or isinstance(data, (GoalSpec, MultiGoal)):
        return data
    if not data:
        return None
    token = data.get(_LIVE_KEY)
    if token is not None:
        with _LIVE_LOCK:
            live = _LIVE_GOALS.get(token)
        if live is not None:
            return live
        logger.warning(
            "Goal callables could not be restored from the checkpoint: metric "
            "expressions and a callable `value` are Python objects and are not "
            "persisted. Continuing with the declarative goal only (metrics "
            "ignored, value=1.0)."
        )
    return goal_from_dict(data)


# ---------------------------------------------------------------------------
# Plan / history codecs
# ---------------------------------------------------------------------------

ActionsLike = Mapping[str, ActionSpec] | Iterable[ActionSpec] | None


def plan_to_state(plan: Plan | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the JSON-native form of *plan* for the ``plan`` channel."""
    if plan is None:
        return None
    if isinstance(plan, Plan):
        return plan.to_dict()
    if isinstance(plan, Mapping):
        return dict(plan)
    raise TypeError(f"plan must be a Plan or dict, got {type(plan).__name__}")


def plan_from_state(data: Any, actions: ActionsLike = None) -> Plan | None:
    """Rebuild a :class:`~langgoap.planner.types.Plan` from the channel value.

    *actions* is the live action list used to rebind callables by name.
    """
    if data is None or isinstance(data, Plan):
        return data
    if not data:
        return None
    return Plan.from_dict(data, actions=actions)


def history_to_state(entries: Iterable[Any]) -> list[dict[str, Any]]:
    """Return the JSON-native form of ``execution_history`` entries."""
    return [_as_action_result(e).to_dict() for e in entries]


class LazyHistory(Sequence[ActionResult]):
    """``execution_history`` rebuilt from the channel, one entry at a time, when read.

    Every node rehydrates the state; rebuilding every :class:`ActionResult` on
    every node made a run quadratic in its length.  Reads index, iterate, slice,
    reverse, compare and concatenate like the list this stands in for.
    """

    __slots__ = ("_raw", "_built")

    def __init__(self, raw: Iterable[Any]) -> None:
        self._raw = tuple(raw)
        self._built: list[ActionResult | None] = [None] * len(self._raw)

    def _at(self, index: int) -> ActionResult:
        entry = self._built[index]
        if entry is None:
            entry = _as_action_result(self._raw[index])
            self._built[index] = entry
        return entry

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self._at(i) for i in range(*index.indices(len(self._raw)))]
        if index < 0:
            index += len(self._raw)
        if not 0 <= index < len(self._raw):
            raise IndexError("execution_history index out of range")
        return self._at(index)

    def __iter__(self) -> Iterator[ActionResult]:
        return (self._at(i) for i in range(len(self._raw)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (LazyHistory, list, tuple)):
            return list(self) == list(other)
        return NotImplemented

    def __hash__(self) -> int:  # pragma: no cover - histories are never hashed
        return hash(tuple(self))

    def __add__(self, other: Any) -> list[ActionResult]:
        return list(self) + list(other)

    def __radd__(self, other: Any) -> list[ActionResult]:
        return list(other) + list(self)

    def __repr__(self) -> str:
        return f"LazyHistory({len(self._raw)} entries)"


def history_from_state(entries: Iterable[Any]) -> LazyHistory:
    """Rebuild :class:`ActionResult` objects from ``execution_history``, lazily."""
    if isinstance(entries, LazyHistory):
        return entries
    return LazyHistory(entries)


#: Default value of every bookkeeping channel.  LangGraph returns only the
#: channels a run wrote to, so without these a happy-path result had no
#: ``blacklisted_actions`` key precisely because nothing had gone wrong.
#: :meth:`GoapGraph.initial_state <langgoap.graph.builder.GoapGraph.initial_state>`
#: seeds them, the planner fills any that are missing on its first pass (for
#: inputs built by hand), and :func:`rehydrate_state` fills them on the way
#: out.  ``goal``, ``world_state``, ``plan``, ``status`` and
#: ``wall_clock_started_at`` are owned by the caller or the planner and have
#: no default.  Read-only: use :func:`state_defaults` for fresh copies.
STATE_DEFAULTS: Mapping[str, Any] = {
    "current_step": 0,
    "execution_history": [],
    "replan_count": 0,
    "replan_reason": None,
    "blacklisted_actions": [],
    "action_failure_counts": {},
    "current_subgoal_index": 0,
    "skipped_subgoals": [],
    "no_plan_explanation": None,
    "reflection_context": [],
}


def state_defaults() -> dict[str, Any]:
    """Return fresh copies of :data:`STATE_DEFAULTS` (no shared mutables)."""
    return {key: copy.deepcopy(value) for key, value in STATE_DEFAULTS.items()}


def rehydrate_state(state: Mapping[str, Any], actions: ActionsLike = None) -> GoapState:
    """Return a copy of *state* with typed ``goal``, ``plan`` and history.

    Bookkeeping keys that are absent are filled from
    :data:`STATE_DEFAULTS`; ``goal``, ``world_state``, ``plan`` and
    ``status`` stay absent when absent.  Keys already holding typed objects
    pass through.  *actions* is the live action list (or ``name -> spec``
    mapping) used to rebind plan actions.
    """
    out: dict[str, Any] = dict(state)
    if "goal" in out:
        out["goal"] = goal_from_state(out["goal"])
    if "plan" in out:
        out["plan"] = plan_from_state(out["plan"], actions)
    if "execution_history" in out:
        out["execution_history"] = history_from_state(out["execution_history"] or [])
    for key, default in state_defaults().items():
        out.setdefault(key, default)
    return cast(GoapState, out)


def serialize_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a node's update dict with JSON-native channel values.

    Inverse of :func:`rehydrate_state` for the keys a node may write with
    typed objects: ``goal``, ``plan`` and ``execution_history``.
    """
    out: dict[str, Any] = dict(updates)
    if "goal" in out:
        out["goal"] = goal_to_state(out["goal"])
    if "plan" in out:
        out["plan"] = plan_to_state(out["plan"])
    if "execution_history" in out:
        out["execution_history"] = history_to_state(out["execution_history"] or [])
    return out


# ---------------------------------------------------------------------------
# Channel reducers
# ---------------------------------------------------------------------------


def _goal_channel(existing: Any, new: Any) -> dict[str, Any] | None:
    """Last-value reducer that stores the JSON form of whatever was written."""
    return goal_to_state(new)


def _history_channel(existing: Any, new: Any) -> list[dict[str, Any]]:
    """Append-only reducer that stores every entry in its JSON form."""
    return list(existing or []) + history_to_state(new or [])


class GoapState(TypedDict, total=False):
    """The GOAP execution state as Python objects.

    This is what :meth:`GoapGraph.invoke <langgoap.graph.builder.GoapGraph.invoke>`
    and :func:`rehydrate_state` return and what the planner / executor /
    observer nodes, stuck handlers and termination policies work with.
    LangGraph itself stores the JSON-native form described by
    :class:`GoapStateChannels`; the two differ only in the ``goal``,
    ``plan`` and ``execution_history`` keys.

    Keys
    ----
    ``world_state``
        Current world state as a flat dictionary.
    ``goal``
        The goal to achieve: a :class:`~langgoap.goals.GoalSpec` or a
        :class:`~langgoap.goals.MultiGoal` wrapping several sub-goals.
    ``plan``
        The current :class:`~langgoap.planner.types.Plan`, or ``None``
        when the planner found none (set by the planner node).
    ``current_step``
        Index of the next action to execute.
    ``execution_history``
        Append-only log of :class:`ActionResult` records.
    ``replan_count``
        How many times the planner has been invoked for the current
        (sub-)goal. Reset to zero when a :class:`~langgoap.goals.MultiGoal`
        advances between sequential sub-goals so each sub-goal receives
        its own ``max_replans`` budget.
    ``replan_reason``
        Why the last replan was triggered. One of ``"action_failed"``,
        ``"state_deviation"``, ``"every_action_replan"``,
        ``"plan_exhausted"``, ``"max_replans_exceeded"``, or
        ``"subgoal_achieved"`` (only emitted when a
        :class:`~langgoap.goals.MultiGoal` advances between sequential
        sub-goals).
    ``status``
        Current execution status.
    ``blacklisted_actions``
        Action names the planner must skip.
    ``action_failure_counts``
        Per-action cumulative failure counts.
    ``current_subgoal_index``
        When ``goal`` is a :class:`~langgoap.goals.MultiGoal` running in
        ``sequential`` mode, the 0-based index of the sub-goal being
        worked on.  Defaults to 0.
    ``skipped_subgoals``
        Indices of sequential :class:`~langgoap.goals.MultiGoal` sub-goals
        the planner skipped because they could not be planned
        (``MultiGoal(on_infeasible="skip")``).  Absent or empty otherwise;
        when non-empty the run finishes with ``status="partial"``.
    ``no_plan_explanation``
        Serialised :class:`~langgoap.planner.explain.NoPlanExplanation`
        (via ``.to_dict()``) produced when A* returns ``None``.  ``None``
        when a plan was found.  Use
        :meth:`~langgoap.planner.explain.NoPlanExplanation.from_dict` to
        reconstruct the object.
    ``reflection_context``
        Ordered list of verbal reflection summaries produced by
        :class:`~langgoap.reflexion.ReflexionTracer` after action failures.
        Each entry is a human-readable string of the form
        ``"[action_name] reflection → suggestion"``.  Set by the planner
        node on each planning round so downstream LLM-evaluated conditions
        can read them without coupling directly to the tracer.
    ``wall_clock_started_at``
        Epoch timestamp (``time.time()``) set by the planner on its first
        invocation; consulted by wall-clock termination policies.  Epoch
        seconds stay meaningful after a checkpoint resume in another
        process.
    """

    world_state: dict[str, Any]
    goal: GoalSpec | MultiGoal | None
    plan: Plan | None
    current_step: int
    execution_history: Sequence[ActionResult]
    replan_count: int
    replan_reason: str | None
    status: str
    blacklisted_actions: list[str]
    action_failure_counts: dict[str, int]
    current_subgoal_index: int
    skipped_subgoals: list[int]
    no_plan_explanation: dict[str, Any] | None
    reflection_context: list[str]
    wall_clock_started_at: float | None


class GoapStateChannels(TypedDict, total=False):
    """LangGraph channel schema for the GOAP loop: the JSON-native state.

    ``GoapGraph.compile`` builds its ``StateGraph`` from this schema, so
    every value LangGraph checkpoints is a dict, list, string, number,
    boolean or ``None`` and any checkpointer / serializer works.  It has
    the same keys as :class:`GoapState`; the three that differ are:

    ``goal``
        :meth:`GoalSpec.to_dict <langgoap.goals.GoalSpec.to_dict>` /
        :meth:`MultiGoal.to_dict <langgoap.goals.MultiGoal.to_dict>`
        form.  The reducer also accepts the typed objects and stores
        their dict form, so ``compiled.invoke({"goal": GoalSpec(...)})``
        works when no checkpointer is attached.
    ``plan``
        :meth:`Plan.to_dict <langgoap.planner.types.Plan.to_dict>` form,
        or ``None``.
    ``execution_history``
        :meth:`ActionResult.to_dict` form.  The reducer also accepts
        :class:`ActionResult` objects.

    Use :func:`rehydrate_state` (or ``GoapGraph.rehydrate``) to turn a
    value of this shape into a :class:`GoapState`.
    """

    world_state: dict[str, Any]
    goal: Annotated[dict[str, Any], _goal_channel]
    plan: dict[str, Any] | None
    current_step: int
    execution_history: Annotated[list[dict[str, Any]], _history_channel]
    replan_count: int
    replan_reason: str | None
    status: str
    blacklisted_actions: list[str]
    action_failure_counts: dict[str, int]
    current_subgoal_index: int
    skipped_subgoals: list[int]
    no_plan_explanation: dict[str, Any] | None
    reflection_context: list[str]
    wall_clock_started_at: float | None


__all__ = [
    "ActionResult",
    "GoapState",
    "GoapStateChannels",
    "goal_from_state",
    "goal_to_state",
    "history_from_state",
    "history_to_state",
    "plan_from_state",
    "plan_to_state",
    "rehydrate_state",
    "serialize_updates",
    "STATE_DEFAULTS",
    "state_defaults",
    "successful_action_names",
]
