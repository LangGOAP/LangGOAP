"""Transition models for deterministic and stochastic action dynamics.

LangGOAP planners treat action declarations (:class:`ActionSpec`) as
the specification of *what* an action does.  The ``TransitionModel``
Protocol describes *how the world responds* — separating spec from
dynamics so A* can plan under an expected-value view while MCTS
rollouts and the graph runtime see the actual sampled view.

See ``research/experiments/2026-04-20-mcts-on-stochastic.md`` §8.1
for the committed design.  The default :class:`DeterministicTransitionModel`
is backwards-compatible — callers that do not wire a model observe
the pre-existing bit-identical behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from random import Random
from typing import Any, Literal, Protocol, runtime_checkable

from langgoap.actions import ActionSpec

# Structured taxonomy for the divergence opt-out.  Used by the router's
# ``risk_profile`` feature so the router can dispatch to an MCTS-like
# sampling strategy when ``expected() != sample()`` is by design.
DivergenceKind = Literal["risk-averse", "learned", "hierarchical", "other"]


@dataclass(frozen=True, slots=True)
class DivergencePolicy:
    """Declared divergence between ``expected()`` and declared effects.

    Presence of this policy on a :class:`TransitionModel` instance opts
    the model out of strict-equality enforcement and documents *why*,
    *how far*, and (optionally) *how* the divergence is configured.

    Attributes:
        reason: Short human-readable label, required.  Empty strings
            are rejected at conformance-check time so every opt-out
            carries at least one sentence of documentation.
        kind: Structured taxonomy slot consumed by the router.  See
            :data:`DivergenceKind`.
        max_relative_deviation: Optional bound on
            ``|expected - declared| / max(|declared|, 1.0)`` for each
            numeric effect value.  Boolean and string effects still
            require exact equality regardless of this bound.  ``None``
            means divergence is unbounded (only the ``reason``
            documentation is enforced).
        extra: Implementation-specific configuration — e.g. CVaR
            ``alpha``, pessimism weight, learned-model version hash.
            Opaque to the core library; read by implementations.
    """

    reason: str
    kind: DivergenceKind = "other"
    max_relative_deviation: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TransitionModel(Protocol):
    """Separates action specs from world dynamics.

    Implementations expose two views:

    * :meth:`expected` — deterministic point estimate, used by A*,
      CSP, and MCTS tree expansion.  Must equal
      ``action.get_effects(state)`` unless ``divergence_policy`` opts
      out (see :class:`DivergencePolicy`).
    * :meth:`sample` — one draw from the effect distribution, used by
      MCTS rollouts and the graph action-executor.  Free to diverge
      from :meth:`expected` — that divergence is the whole point of
      a non-deterministic model.

    The optional ``divergence_policy`` attribute declares intentional
    divergence between :meth:`expected` and the action's declared
    effects.  Its absence (or ``None``) means strict equality is
    enforced by :func:`assert_expected_matches_declared`.
    """

    divergence_policy: "DivergencePolicy | None"

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]: ...

    def sample(
        self,
        state: Mapping[str, Any],
        action: ActionSpec,
        rng: Random,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DeterministicTransitionModel:
    """Default model: ``expected`` and ``sample`` both equal the
    action's declared effects, regardless of RNG state.

    This is the zero-configuration path.  Every existing call site
    that does not wire a transition model sees this instance and its
    behaviour is bit-identical to the pre-``TransitionModel`` code.
    """

    # Literal ``None`` — the strict invariant applies to this model.
    divergence_policy: DivergencePolicy | None = None

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(
        self,
        state: Mapping[str, Any],
        action: ActionSpec,
        rng: Random,
    ) -> Mapping[str, Any]:
        # RNG is accepted for Protocol conformance and ignored.
        return action.get_effects(dict(state))


def assert_expected_matches_declared(
    model: TransitionModel,
    *,
    state: Mapping[str, Any],
    action: ActionSpec,
) -> None:
    """Enforce the ``expected()`` / declared-effects contract.

    Dispatches on the model's ``divergence_policy`` attribute:

    * **None / absent (strict path)** — assert ``model.expected(state,
      action) == action.get_effects(state)`` exactly.  Catches silent
      declared-effect typos in user-authored :class:`TransitionModel`
      implementations.
    * **Set, unbounded** (``max_relative_deviation is None``) — skip
      the equality check, but require a non-empty ``reason`` so every
      opt-out carries at least a sentence of documentation.
    * **Set, bounded** — assert
      ``|expected - declared| / max(|declared|, 1.0) ≤
      max_relative_deviation`` for every numeric effect value.
      Boolean and string values still require exact equality; only
      numeric values are bound-relaxed, because flipping a boolean
      flag is a categorical change, not a bounded perturbation.
    """
    policy: DivergencePolicy | None = getattr(model, "divergence_policy", None)
    declared = dict(action.get_effects(dict(state)))
    expected = dict(model.expected(state, action))

    if policy is None:
        assert expected == declared, (
            f"TransitionModel.expected() diverges from declared effects "
            f"but no divergence_policy was set. "
            f"expected={expected!r} declared={declared!r}"
        )
        return

    if not policy.reason.strip():
        raise ValueError(
            "DivergencePolicy.reason must be a non-empty string; opt-out "
            "divergence requires a documented rationale."
        )

    if policy.max_relative_deviation is None:
        # Unbounded opt-out — rationale is documented, no mechanical check.
        return

    bound = float(policy.max_relative_deviation)
    for key, declared_value in declared.items():
        expected_value = expected.get(key, declared_value)
        if isinstance(declared_value, bool) or isinstance(expected_value, bool):
            # Booleans are not bound-relaxed — flipping is categorical.
            assert expected_value == declared_value, (
                f"TransitionModel.expected()[{key!r}] flipped from "
                f"{declared_value!r} to {expected_value!r}; bounded "
                f"divergence does not cover non-numeric effects."
            )
            continue
        if isinstance(declared_value, (int, float)) and isinstance(
            expected_value, (int, float)
        ):
            denom = max(abs(float(declared_value)), 1.0)
            deviation = abs(float(expected_value) - float(declared_value)) / denom
            assert deviation <= bound, (
                f"TransitionModel.expected()[{key!r}] declares bounded "
                f"deviation ≤{bound} but observed {deviation:.3f} "
                f"(declared={declared_value!r}, expected={expected_value!r})."
            )
            continue
        # Non-numeric, non-boolean (strings, frozensets, etc.) require
        # exact equality regardless of the numeric bound.
        assert expected_value == declared_value, (
            f"TransitionModel.expected()[{key!r}] diverges from "
            f"declared={declared_value!r} to expected={expected_value!r}; "
            f"bounded divergence applies only to numeric values."
        )


__all__ = [
    "DeterministicTransitionModel",
    "DivergenceKind",
    "DivergencePolicy",
    "TransitionModel",
    "assert_expected_matches_declared",
]
