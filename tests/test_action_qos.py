"""Unit tests for ``langgoap.qos.ActionQos``.

Covers the validator (``__post_init__``), backoff computation
(``compute_backoff_ms``), retry decision (``should_retry``), and the
ready-made ``FIRE_ONCE`` / ``DEFAULT_RETRY_POLICY`` policies.

API reference (Embabel parity):
  research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
    com/embabel/agent/core/ActionQos.kt
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from langgoap.qos import DEFAULT_RETRY_POLICY, FIRE_ONCE, ActionQos

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class TestValidator:
    def test_default_is_constructible(self) -> None:
        ActionQos()

    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_max_attempts_below_one_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            ActionQos(max_attempts=bad)

    def test_negative_backoff_initial_rejected(self) -> None:
        with pytest.raises(ValueError, match="backoff_initial_ms"):
            ActionQos(backoff_initial_ms=-1.0)

    @pytest.mark.parametrize("bad", [0.0, 0.5, 0.999])
    def test_backoff_multiplier_below_one_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="backoff_multiplier"):
            ActionQos(backoff_multiplier=bad)

    def test_backoff_max_below_initial_rejected(self) -> None:
        with pytest.raises(ValueError, match="backoff_max_ms"):
            ActionQos(backoff_initial_ms=100.0, backoff_max_ms=50.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.01, 2.0])
    def test_jitter_outside_unit_range_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="jitter"):
            ActionQos(jitter=bad)


# ---------------------------------------------------------------------------
# Backoff computation
# ---------------------------------------------------------------------------


class TestComputeBackoff:
    def test_attempt_zero_rejected(self) -> None:
        qos = ActionQos(max_attempts=3, jitter=0.0)
        with pytest.raises(ValueError, match="attempt"):
            qos.compute_backoff_ms(0)

    def test_jitter_zero_yields_exact_exponential(self) -> None:
        qos = ActionQos(
            max_attempts=10,
            backoff_initial_ms=4.0,
            backoff_multiplier=2.0,
            backoff_max_ms=1_000_000.0,
            jitter=0.0,
        )
        assert qos.compute_backoff_ms(1) == 4.0
        assert qos.compute_backoff_ms(2) == 8.0
        assert qos.compute_backoff_ms(3) == 16.0
        assert qos.compute_backoff_ms(4) == 32.0

    def test_backoff_capped_at_max(self) -> None:
        qos = ActionQos(
            max_attempts=10,
            backoff_initial_ms=10.0,
            backoff_multiplier=10.0,
            backoff_max_ms=200.0,
            jitter=0.0,
        )
        assert qos.compute_backoff_ms(1) == 10.0
        assert qos.compute_backoff_ms(2) == 100.0
        assert qos.compute_backoff_ms(3) == 200.0  # 1000 capped to 200
        assert qos.compute_backoff_ms(4) == 200.0
        assert qos.compute_backoff_ms(10) == 200.0

    def test_jitter_is_within_declared_bounds(self) -> None:
        qos = ActionQos(
            max_attempts=10,
            backoff_initial_ms=100.0,
            backoff_multiplier=1.0,
            backoff_max_ms=1_000.0,
            jitter=0.25,
        )
        rng = random.Random(0)
        for _ in range(100):
            backoff = qos.compute_backoff_ms(1, rng=rng)
            assert 75.0 <= backoff <= 125.0, backoff

    def test_jitter_uses_supplied_rng_for_determinism(self) -> None:
        qos = ActionQos(
            max_attempts=10,
            backoff_initial_ms=100.0,
            backoff_multiplier=1.0,
            backoff_max_ms=1_000.0,
            jitter=0.5,
        )
        a = [qos.compute_backoff_ms(1, rng=random.Random(42)) for _ in range(5)]
        b = [qos.compute_backoff_ms(1, rng=random.Random(42)) for _ in range(5)]
        assert a == b


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    def test_budget_exhausted_returns_false(self) -> None:
        qos = ActionQos(max_attempts=3, idempotent=True)
        assert qos.should_retry(TimeoutError(), attempt=1) is True
        assert qos.should_retry(TimeoutError(), attempt=2) is True
        assert qos.should_retry(TimeoutError(), attempt=3) is False
        assert qos.should_retry(TimeoutError(), attempt=99) is False

    def test_idempotent_retries_on_any_exception(self) -> None:
        qos = ActionQos(max_attempts=5, idempotent=True)
        assert qos.should_retry(ValueError("anything"), attempt=1) is True
        assert qos.should_retry(KeyError("missing"), attempt=2) is True
        assert qos.should_retry(RuntimeError(), attempt=3) is True

    def test_non_idempotent_retries_only_retryable(self) -> None:
        qos = ActionQos(
            max_attempts=5,
            idempotent=False,
            retryable_exceptions=(TimeoutError, ConnectionError),
        )
        assert qos.should_retry(TimeoutError(), attempt=1) is True
        assert qos.should_retry(ConnectionError(), attempt=1) is True
        assert qos.should_retry(ValueError(), attempt=1) is False

    def test_non_idempotent_retryable_subclass(self) -> None:
        class MyTimeout(TimeoutError):
            pass

        qos = ActionQos(
            max_attempts=5,
            idempotent=False,
            retryable_exceptions=(TimeoutError,),
        )
        assert qos.should_retry(MyTimeout(), attempt=1) is True


# ---------------------------------------------------------------------------
# Standard policies
# ---------------------------------------------------------------------------


class TestStandardPolicies:
    def test_fire_once_does_not_retry(self) -> None:
        assert FIRE_ONCE.max_attempts == 1
        assert FIRE_ONCE.should_retry(TimeoutError(), attempt=1) is False

    def test_default_retry_policy_mirrors_embabel_defaults(self) -> None:
        # Source of truth: research/repos/embabel-agent/embabel-agent-api/
        #   src/main/kotlin/com/embabel/agent/core/ActionQos.kt
        assert DEFAULT_RETRY_POLICY.max_attempts == 5
        assert DEFAULT_RETRY_POLICY.backoff_initial_ms == 10_000.0
        assert DEFAULT_RETRY_POLICY.backoff_multiplier == 5.0
        assert DEFAULT_RETRY_POLICY.backoff_max_ms == 60_000.0
        assert DEFAULT_RETRY_POLICY.idempotent is False


# ---------------------------------------------------------------------------
# Immutability and identity
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_action_qos_is_frozen(self) -> None:
        qos = ActionQos()
        with pytest.raises(dataclasses.FrozenInstanceError):
            qos.max_attempts = 99  # type: ignore[misc]

    def test_action_qos_equality(self) -> None:
        a = ActionQos(max_attempts=3, idempotent=True)
        b = ActionQos(max_attempts=3, idempotent=True)
        c = ActionQos(max_attempts=4, idempotent=True)
        assert a == b
        assert a != c

    def test_action_qos_hashable(self) -> None:
        qos = ActionQos(max_attempts=3)
        # Frozen dataclass; usable in sets / as dict keys.
        assert hash(qos) == hash(ActionQos(max_attempts=3))
