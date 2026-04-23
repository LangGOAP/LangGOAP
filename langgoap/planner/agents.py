"""Agent-modelling primitives for stochastic planning.

:class:`AgentModel` is the library-level protocol for predicting the
actions of *external* agents — opponents in adversarial games,
other LLM agents in a multi-agent plan, or any entity whose choices
a :class:`~langgoap.planner.transitions.TransitionModel` must
sample to simulate rollouts and chance-node expansions.  The shape
mirrors :class:`~langgoap.planner.transitions.TransitionModel`:

* :meth:`AgentModel.expected` returns a deterministic point
  estimate, consumed by A*, CSP, and MCTS chance-node expansion.
* :meth:`AgentModel.sample` draws one action from the agent's
  distribution, consumed by MCTS rollouts and stochastic
  simulations.

:class:`UniformRandomAgentModel` is the baseline implementation
suitable for adversaries whose policy is unknown or intentionally
random (e.g. CS188 ``RandomGhost``).  Smarter agent models —
shortest-path pursuers, learned policies, per-agent personality
profiles — plug in through the same protocol without touching
the :class:`TransitionModel`.
"""

from __future__ import annotations

from random import Random
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar
from typing import runtime_checkable

A = TypeVar("A")


@runtime_checkable
class AgentModel(Protocol[A]):
    """Predicts an external agent's next action given a world state.

    Generic over the action type ``A``.  Conventional values include
    direction strings (``"North"``, ``"South"``, ...),
    :class:`~langgoap.ActionSpec` references, or any domain-specific
    hashable action token.
    """

    def expected(self, state: Mapping[str, Any]) -> A:
        """Return the deterministic point-estimate action for ``state``."""
        ...

    def sample(self, state: Mapping[str, Any], rng: Random) -> A:
        """Draw one action from this agent's distribution for ``state``."""
        ...


class UniformRandomAgentModel(Generic[A]):
    """Baseline :class:`AgentModel` that draws uniformly from a legal set.

    ``enumerate_actions`` returns the legal actions available to the
    modelled agent in ``state``.  ``sample`` picks one uniformly via
    the supplied RNG; :meth:`expected` returns ``default`` because
    the mode of a uniform distribution is ill-defined — callers who
    want a particular point estimate should plug a smarter
    :class:`AgentModel` implementation.

    The ``default`` action is also returned when
    ``enumerate_actions`` yields an empty sequence (no legal moves).
    """

    __slots__ = ("_enumerate", "_default")

    def __init__(
        self,
        *,
        enumerate_actions: Callable[[Mapping[str, Any]], Sequence[A]],
        default: A,
    ) -> None:
        self._enumerate = enumerate_actions
        self._default = default

    def expected(self, state: Mapping[str, Any]) -> A:
        return self._default

    def sample(self, state: Mapping[str, Any], rng: Random) -> A:
        options = self._enumerate(state)
        if not options:
            return self._default
        return rng.choice(list(options))


__all__ = ["AgentModel", "UniformRandomAgentModel"]
