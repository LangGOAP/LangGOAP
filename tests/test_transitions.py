"""Failing unit tests for :mod:`langgoap.planner.transitions`.

Validates the ``TransitionModel`` Protocol boundary committed in
``research/experiments/2026-04-20-mcts-on-stochastic.md`` §8.1:

- Default path (``divergence_policy is None``) enforces strict
  ``expected(state, action) == action.get_effects(state)`` equality.
- Opt-out path enforces the declared ``DivergencePolicy`` bounds,
  requires a non-empty ``reason``, and surfaces ``kind`` for the
  router's ``risk_profile`` feature.
- ``DeterministicTransitionModel`` is the backwards-compatible default
  used when callers do not wire a model.
"""

from __future__ import annotations

import random

import pytest

from langgoap.actions import ActionSpec


class TestDivergencePolicy:
    def test_requires_reason(self) -> None:
        from langgoap.planner.transitions import DivergencePolicy

        with pytest.raises(TypeError):
            DivergencePolicy()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        from langgoap.planner.transitions import DivergencePolicy

        p = DivergencePolicy(reason="risk-averse")
        with pytest.raises((AttributeError, TypeError)):
            p.reason = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        from langgoap.planner.transitions import DivergencePolicy

        p = DivergencePolicy(reason="learned from logs")
        assert p.reason == "learned from logs"
        assert p.kind == "other"
        assert p.max_relative_deviation is None
        assert dict(p.extra) == {}

    def test_kind_taxonomy(self) -> None:
        from langgoap.planner.transitions import DivergencePolicy

        # All declared kinds must construct without error.
        for kind in ("risk-averse", "learned", "hierarchical", "other"):
            p = DivergencePolicy(reason="x", kind=kind)  # type: ignore[arg-type]
            assert p.kind == kind


class TestDeterministicTransitionModel:
    def test_satisfies_protocol(self) -> None:
        from langgoap.planner.transitions import (
            DeterministicTransitionModel,
            TransitionModel,
        )

        assert isinstance(DeterministicTransitionModel(), TransitionModel)

    def test_expected_matches_declared_effects(self) -> None:
        from langgoap.planner.transitions import DeterministicTransitionModel

        action = ActionSpec(
            name="a", preconditions={}, effects={"x": 1, "y": True}, cost=1.0
        )
        model = DeterministicTransitionModel()
        assert model.expected({}, action) == {"x": 1, "y": True}

    def test_sample_matches_expected_ignoring_rng(self) -> None:
        from langgoap.planner.transitions import DeterministicTransitionModel

        action = ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0)
        model = DeterministicTransitionModel()
        rng_a = random.Random(1)
        rng_b = random.Random(2)
        # RNG state must not influence a deterministic model.
        assert model.sample({}, action, rng_a) == model.sample({}, action, rng_b)
        assert model.sample({}, action, rng_a) == model.expected({}, action)

    def test_divergence_policy_is_none(self) -> None:
        from langgoap.planner.transitions import DeterministicTransitionModel

        # Absent or None opts into strict enforcement.
        assert getattr(DeterministicTransitionModel(), "divergence_policy", None) is None


class TestConformanceStrict:
    def test_strict_passes_for_default_model(self) -> None:
        from langgoap.planner.transitions import (
            DeterministicTransitionModel,
            assert_expected_matches_declared,
        )

        action = ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0)
        # Must not raise.
        assert_expected_matches_declared(
            DeterministicTransitionModel(), state={}, action=action
        )

    def test_strict_raises_on_silent_divergence(self) -> None:
        from langgoap.planner.transitions import assert_expected_matches_declared

        class BuggyModel:
            divergence_policy = None

            def expected(self, state, action):  # type: ignore[no-untyped-def]
                # Silent bug: shifts the declared +1 to +2.
                declared = dict(action.get_effects(state))
                return {k: (v + 1 if isinstance(v, int) else v) for k, v in declared.items()}

            def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
                return action.get_effects(state)

        action = ActionSpec(name="a", preconditions={}, effects={"x": 1}, cost=1.0)
        with pytest.raises(AssertionError):
            assert_expected_matches_declared(BuggyModel(), state={}, action=action)


class TestConformanceOptOut:
    def test_unbounded_opt_out_skips_equality_but_requires_reason(self) -> None:
        from langgoap.planner.transitions import (
            DivergencePolicy,
            assert_expected_matches_declared,
        )

        class RiskAverseModel:
            divergence_policy = DivergencePolicy(
                reason="cvar planning at 95% confidence", kind="risk-averse"
            )

            def expected(self, state, action):  # type: ignore[no-untyped-def]
                # Deliberately pessimistic — divergence by design.
                declared = dict(action.get_effects(state))
                return {k: (0 if isinstance(v, int) else v) for k, v in declared.items()}

            def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
                return action.get_effects(state)

        action = ActionSpec(name="a", preconditions={}, effects={"x": 10}, cost=1.0)
        # Must not raise: divergence_policy opts out of strict equality.
        assert_expected_matches_declared(RiskAverseModel(), state={}, action=action)

    def test_unbounded_opt_out_rejects_empty_reason(self) -> None:
        from langgoap.planner.transitions import (
            DivergencePolicy,
            assert_expected_matches_declared,
        )

        class BadlyDocumentedModel:
            divergence_policy = DivergencePolicy(reason="", kind="risk-averse")

            def expected(self, state, action):  # type: ignore[no-untyped-def]
                return {"x": 0}

            def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
                return action.get_effects(state)

        action = ActionSpec(name="a", preconditions={}, effects={"x": 10}, cost=1.0)
        with pytest.raises(ValueError, match="reason"):
            assert_expected_matches_declared(
                BadlyDocumentedModel(), state={}, action=action
            )

    def test_bounded_divergence_passes_within_bound(self) -> None:
        from langgoap.planner.transitions import (
            DivergencePolicy,
            assert_expected_matches_declared,
        )

        class BoundedModel:
            divergence_policy = DivergencePolicy(
                reason="learned prior drifts ≤20% from declared",
                kind="learned",
                max_relative_deviation=0.2,
            )

            def expected(self, state, action):  # type: ignore[no-untyped-def]
                return {"x": 11}  # 10% deviation from declared x=10

            def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
                return action.get_effects(state)

        action = ActionSpec(name="a", preconditions={}, effects={"x": 10}, cost=1.0)
        assert_expected_matches_declared(BoundedModel(), state={}, action=action)

    def test_bounded_divergence_raises_when_bound_exceeded(self) -> None:
        from langgoap.planner.transitions import (
            DivergencePolicy,
            assert_expected_matches_declared,
        )

        class BoundedModel:
            divergence_policy = DivergencePolicy(
                reason="claims ≤10% deviation",
                kind="learned",
                max_relative_deviation=0.1,
            )

            def expected(self, state, action):  # type: ignore[no-untyped-def]
                return {"x": 13}  # 30% deviation from declared x=10

            def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
                return action.get_effects(state)

        action = ActionSpec(name="a", preconditions={}, effects={"x": 10}, cost=1.0)
        with pytest.raises(AssertionError, match="deviation"):
            assert_expected_matches_declared(BoundedModel(), state={}, action=action)

    def test_bounded_divergence_still_exact_on_non_numeric(self) -> None:
        from langgoap.planner.transitions import (
            DivergencePolicy,
            assert_expected_matches_declared,
        )

        class BoundedModel:
            divergence_policy = DivergencePolicy(
                reason="≤50% numeric divergence",
                kind="risk-averse",
                max_relative_deviation=0.5,
            )

            def expected(self, state, action):  # type: ignore[no-untyped-def]
                # Flipping a boolean is *never* acceptable under bounded
                # divergence — only numeric effects are bound-relaxed.
                return {"ready": False, "count": 5}

            def sample(self, state, action, rng):  # type: ignore[no-untyped-def]
                return action.get_effects(state)

        action = ActionSpec(
            name="a", preconditions={}, effects={"ready": True, "count": 10}, cost=1.0
        )
        with pytest.raises(AssertionError, match="ready"):
            assert_expected_matches_declared(BoundedModel(), state={}, action=action)
