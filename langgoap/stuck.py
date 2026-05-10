"""Stuck-handler protocol for the GOAP planner.

When :class:`~langgoap.graph.nodes.GoapPlanner` cannot find a plan, it
consults a list of :class:`StuckHandler` instances to give user-supplied
recovery code a chance to mutate the world state, swap in a relaxed
goal, or escalate.  Each handler returns a :class:`StuckHandlerResult`
that either signals ``REPLAN`` (the planner retries with the updated
state / goal) or ``NO_RESOLUTION`` (the next handler is consulted; the
planner emits ``no_plan`` if no handler resolves).

API mirrors Embabel's StuckHandler contract
(``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
com/embabel/agent/api/common/StuckHandler.kt`` and
``MulticastStuckHandlerTest.kt``) with two LangGOAP-specific
extensions:

* The result carries optional ``state_updates`` (merged into world
  state on REPLAN) and ``new_goal`` (replaces the goal on REPLAN).
  These let LangGOAP perform a real second plan in the same
  invocation — Embabel's REPLAN signal expects the handler to mutate
  the agent process out-of-band, but LangGOAP's nodes are pure-ish
  state transformers, so we pass the deltas back explicitly.
* Exceptions thrown by a handler in :class:`MulticastStuckHandler` are
  caught and treated as ``NO_RESOLUTION`` — observability/resilience
  invariant matching the never-raise rule on
  :class:`~langgoap.tracing.PlanningTracer`.

The ``GoapPlanner.max_stuck_iterations`` knob caps how many REPLAN
loops are allowed per planning round so a misbehaving handler cannot
spin the planner forever.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class StuckHandlingResultCode(enum.Enum):
    """Whether a handler resolved a stuck situation."""

    REPLAN = "replan"
    NO_RESOLUTION = "no_resolution"


@dataclass(frozen=True, slots=True)
class StuckHandlerResult:
    """Outcome of a stuck-handler invocation.

    Attributes:
        code: ``REPLAN`` or ``NO_RESOLUTION``.
        message: Human-readable explanation of what happened.
        handler_name: Name of the handler that produced this result;
            ``None`` for aggregated results (e.g. multicast NO_RESOLUTION).
        state_updates: Optional world-state delta merged into
            ``world_state`` before the next planning attempt.  Only
            consulted when ``code == REPLAN``.  Defaults to empty.
        new_goal: Optional :class:`~langgoap.goals.GoalSpec` /
            :class:`~langgoap.goals.MultiGoal` that replaces the goal
            for the next planning attempt.  Only consulted when
            ``code == REPLAN``.  ``None`` keeps the current goal.
        timestamp: When the result was produced (UTC).
    """

    code: StuckHandlingResultCode
    message: str
    handler_name: str | None = None
    state_updates: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    new_goal: Any = (
        None  # GoalSpec | MultiGoal | None — typed loosely to avoid circular import
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not isinstance(self.state_updates, MappingProxyType):
            object.__setattr__(
                self,
                "state_updates",
                MappingProxyType(dict(self.state_updates)),
            )

    @classmethod
    def replan(
        cls,
        *,
        handler_name: str,
        message: str,
        state_updates: Mapping[str, Any] | None = None,
        new_goal: Any = None,
    ) -> StuckHandlerResult:
        """Convenience constructor for the REPLAN branch."""
        return cls(
            code=StuckHandlingResultCode.REPLAN,
            message=message,
            handler_name=handler_name,
            state_updates=MappingProxyType(dict(state_updates or {})),
            new_goal=new_goal,
        )

    @classmethod
    def no_resolution(
        cls, *, handler_name: str | None, message: str
    ) -> StuckHandlerResult:
        """Convenience constructor for the NO_RESOLUTION branch."""
        return cls(
            code=StuckHandlingResultCode.NO_RESOLUTION,
            message=message,
            handler_name=handler_name,
        )


@runtime_checkable
class StuckHandler(Protocol):
    """Recovery hook fired when the planner cannot find a plan.

    Implementations inspect the current ``state`` (typically the
    :class:`~langgoap.graph.state.GoapState` dict at the moment of
    failure) and the optional ``reason`` (a
    :class:`~langgoap.planner.explain.NoPlanExplanation` from
    :func:`~langgoap.planner.explain.explain_no_plan`) and return a
    :class:`StuckHandlerResult` that either resolves the situation
    (``REPLAN``) or declines (``NO_RESOLUTION``).
    """

    @property
    def name(self) -> str:
        """Stable identifier for this handler."""
        ...

    def handle_stuck(self, state: Any, reason: Any) -> StuckHandlerResult:
        """Try to recover from a stuck situation."""
        ...


@dataclass
class FunctionalStuckHandler:
    """Adapter that turns a callable into a :class:`StuckHandler`.

    Useful for one-off handlers, especially in tests and notebooks.
    """

    name: str
    fn: Callable[[Any, Any], StuckHandlerResult]

    def handle_stuck(self, state: Any, reason: Any) -> StuckHandlerResult:
        return self.fn(state, reason)


@dataclass
class MulticastStuckHandler:
    """Composite handler that consults each child in order.

    Returns the first ``REPLAN`` result.  If every child returns
    ``NO_RESOLUTION`` (or raises — exceptions are caught), returns an
    aggregated ``NO_RESOLUTION`` whose message lists every handler tried.

    Mirrors
    ``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
    com/embabel/agent/api/common/MulticastStuckHandler.kt``.
    """

    handlers: Sequence[StuckHandler]
    name: str = "multicast"

    def handle_stuck(self, state: Any, reason: Any) -> StuckHandlerResult:
        tried_names: list[str] = []
        for handler in self.handlers:
            handler_name = getattr(handler, "name", type(handler).__name__)
            tried_names.append(handler_name)
            try:
                result = handler.handle_stuck(state, reason)
            except Exception as exc:
                logger.warning(
                    "Stuck handler %r raised during handle_stuck: %s — "
                    "treating as NO_RESOLUTION",
                    handler_name,
                    exc,
                )
                continue
            if result.code is StuckHandlingResultCode.REPLAN:
                return result
        return StuckHandlerResult.no_resolution(
            handler_name=None,
            message=(
                "No stuck handler could resolve the issue: tried "
                + ", ".join(tried_names)
            ),
        )


__all__ = [
    "FunctionalStuckHandler",
    "MulticastStuckHandler",
    "StuckHandler",
    "StuckHandlerResult",
    "StuckHandlingResultCode",
]
