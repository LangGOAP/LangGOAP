"""Composable early-termination policies for the GOAP loop.

A :class:`TerminationPolicy` is consulted by
:class:`~langgoap.graph.nodes.GoapObserver` between every routing
decision (continue, replan, end).  When ``should_terminate(state)``
returns a non-``None`` :class:`EarlyTermination`, the observer
short-circuits to ``END`` with ``status="terminated"`` and surfaces
the policy's reason as the ``replan_reason`` so downstream consumers
(history, tracer, CLI) see exactly why the run stopped.

API mirrors Embabel's contract on
``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
com/embabel/agent/core/EarlyTerminationPolicy.kt``.  Built-ins:

* :class:`MaxActionsPolicy` — terminate after N executed actions.
* :class:`MaxWallClockPolicy` — terminate after a wall-clock budget
  (LangGOAP-specific; Embabel offloads to Spring's scheduler).
* :class:`MaxCostPolicy` — terminate when ``total_cost_usd`` on
  ``world_state`` exceeds a budget.  Users wire the bookkeeping (via
  their actions or a LangChain ``BaseCallbackHandler``); LangGOAP
  ships only the policy.
* :class:`MaxTokensPolicy` — terminate when ``total_tokens`` exceeds.
* :class:`MaxLLMCallsPolicy` — terminate when ``llm_call_count``
  exceeds.
* :class:`OnStuckPolicy` — terminate (without error) when the prior
  planner emitted ``no_plan``.
* :class:`FirstOfPolicy` — composite: terminate if **any** child
  policy fires.
* :class:`AllOfPolicy` — composite: terminate only when **every**
  child policy fires (LangGOAP-specific symmetry with FirstOf).

Custom policies satisfy the :class:`TerminationPolicy` Protocol — a
``name`` attribute and ``should_terminate(state) -> EarlyTermination |
None``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EarlyTermination:
    """Termination decision returned by a policy.

    Attributes:
        policy_name: The policy that triggered termination.
        reason: Human-readable explanation.
        error: Whether the termination is considered a failure.
            ``OnStuckPolicy`` returns ``False`` (graceful stop, useful
            for utility-style agents).  Cost / action-count policies
            return ``True`` (the run hit a configured ceiling).
    """

    policy_name: str
    reason: str
    error: bool = True


@runtime_checkable
class TerminationPolicy(Protocol):
    """Policy that decides whether the GOAP loop should terminate.

    Implementations must be stateless / thread-safe and **never raise**;
    the observer wraps each call in a try/except as a last line of
    defence, but a well-behaved policy is fire-and-forget.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in logs and termination reasons."""
        ...

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        """Inspect ``state`` (a :class:`~langgoap.graph.state.GoapState`
        dict) and return ``EarlyTermination`` to halt the run, or
        ``None`` to continue."""
        ...


# ---------------------------------------------------------------------------
# Concrete policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaxActionsPolicy:
    """Terminate after ``max_actions`` actions have been executed."""

    max_actions: int
    name: str = "MaxActionsPolicy"

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        executed = len(state.get("execution_history", []))
        if executed >= self.max_actions:
            return EarlyTermination(
                policy_name=self.name,
                reason=f"Max actions of {self.max_actions} reached",
                error=True,
            )
        return None


@dataclass(frozen=True, slots=True)
class MaxWallClockPolicy:
    """Terminate after ``seconds`` of wall-clock time since the run started.

    Reads ``state['wall_clock_started_at']`` if present (set by the
    planner on the first invocation, see
    :class:`~langgoap.graph.nodes.GoapPlanner`).  When absent, the
    policy uses ``time.monotonic`` snapshot lazily on first call;
    construction time is intentionally **not** the start anchor so a
    long-lived ``GoapGraph`` does not bleed wall-clock between runs.
    """

    seconds: float
    name: str = "MaxWallClockPolicy"

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        started_at = state.get("wall_clock_started_at")
        if started_at is None:
            return None
        elapsed = time.monotonic() - float(started_at)
        if elapsed >= self.seconds:
            return EarlyTermination(
                policy_name=self.name,
                reason=(
                    f"Wall-clock budget of {self.seconds:.3f}s exceeded "
                    f"(elapsed={elapsed:.3f}s)"
                ),
                error=True,
            )
        return None


@dataclass(frozen=True, slots=True)
class MaxCostPolicy:
    """Terminate when ``world_state['total_cost_usd']`` exceeds ``usd``.

    Users supply the bookkeeping — typically by emitting
    ``total_cost_usd`` from their action's ``execute`` / ``aexecute``
    return value, or by wiring a LangChain ``BaseCallbackHandler`` that
    accumulates per-call cost into the world state.
    """

    usd: float
    name: str = "MaxCostPolicy"

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        cost = float(state.get("world_state", {}).get("total_cost_usd", 0.0))
        if cost >= self.usd:
            return EarlyTermination(
                policy_name=self.name,
                reason=(
                    f"Cost budget of ${self.usd:.4f} exceeded " f"(spent=${cost:.4f})"
                ),
                error=True,
            )
        return None


@dataclass(frozen=True, slots=True)
class MaxTokensPolicy:
    """Terminate when ``world_state['total_tokens']`` exceeds ``tokens``."""

    tokens: int
    name: str = "MaxTokensPolicy"

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        used = int(state.get("world_state", {}).get("total_tokens", 0))
        if used >= self.tokens:
            return EarlyTermination(
                policy_name=self.name,
                reason=(f"Token budget of {self.tokens} exceeded (used={used})"),
                error=True,
            )
        return None


@dataclass(frozen=True, slots=True)
class MaxLLMCallsPolicy:
    """Terminate when ``world_state['llm_call_count']`` exceeds ``max_calls``."""

    max_calls: int
    name: str = "MaxLLMCallsPolicy"

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        used = int(state.get("world_state", {}).get("llm_call_count", 0))
        if used >= self.max_calls:
            return EarlyTermination(
                policy_name=self.name,
                reason=(
                    f"LLM-call budget of {self.max_calls} exceeded " f"(used={used})"
                ),
                error=True,
            )
        return None


@dataclass(frozen=True, slots=True)
class OnStuckPolicy:
    """Convert a ``no_plan`` status into a graceful termination.

    Useful for utility-style agents where "no productive next action"
    is the natural stopping condition rather than a failure.  ``error``
    is ``False`` so downstream tooling can distinguish from a hit
    cost / action ceiling.
    """

    name: str = "OnStuckPolicy"

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        if state.get("status") == "no_plan":
            return EarlyTermination(
                policy_name=self.name,
                reason="Agent process stuck — no plan available",
                error=False,
            )
        return None


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------


class FirstOfPolicy:
    """Composite: terminate when **any** child policy fires.

    Policies are evaluated in declaration order; the first non-``None``
    result wins.  Mirrors Embabel's
    ``EarlyTerminationPolicy.firstOf(...)`` semantics.
    """

    def __init__(self, *policies: TerminationPolicy) -> None:
        self._policies: tuple[TerminationPolicy, ...] = tuple(policies)
        self.name = "FirstOfPolicy"

    @property
    def policies(self) -> Sequence[TerminationPolicy]:
        return self._policies

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        for policy in self._policies:
            decision = policy.should_terminate(state)
            if decision is not None:
                return decision
        return None


class AllOfPolicy:
    """Composite: terminate only when **every** child policy fires.

    LangGOAP-specific symmetry with :class:`FirstOfPolicy`.  The
    aggregated reason concatenates the inner reasons.
    """

    def __init__(self, *policies: TerminationPolicy) -> None:
        if not policies:
            raise ValueError("AllOfPolicy requires at least one inner policy")
        self._policies: tuple[TerminationPolicy, ...] = tuple(policies)
        self.name = "AllOfPolicy"

    @property
    def policies(self) -> Sequence[TerminationPolicy]:
        return self._policies

    def should_terminate(self, state: Any) -> EarlyTermination | None:
        decisions: list[EarlyTermination] = []
        for policy in self._policies:
            d = policy.should_terminate(state)
            if d is None:
                return None
            decisions.append(d)
        # Any-error wins: if any inner termination is an error, the
        # composite is too.
        any_error = any(d.error for d in decisions)
        return EarlyTermination(
            policy_name=self.name,
            reason="; ".join(d.reason for d in decisions),
            error=any_error,
        )


__all__ = [
    "AllOfPolicy",
    "EarlyTermination",
    "FirstOfPolicy",
    "MaxActionsPolicy",
    "MaxCostPolicy",
    "MaxLLMCallsPolicy",
    "MaxTokensPolicy",
    "MaxWallClockPolicy",
    "OnStuckPolicy",
    "TerminationPolicy",
]
