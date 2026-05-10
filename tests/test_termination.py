"""Unit tests for ``langgoap.termination`` policies.

API reference:
  research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
    com/embabel/agent/core/EarlyTerminationPolicy.kt
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from langgoap.termination import (
    AllOfPolicy,
    EarlyTermination,
    FirstOfPolicy,
    MaxActionsPolicy,
    MaxCostPolicy,
    MaxLLMCallsPolicy,
    MaxTokensPolicy,
    MaxWallClockPolicy,
    OnStuckPolicy,
    TerminationPolicy,
)


def _state(
    *,
    history: int = 0,
    cost: float = 0.0,
    tokens: int = 0,
    llm_calls: int = 0,
    status: str = "",
    wall_clock_started_at: float | None = None,
) -> dict[str, Any]:
    """Build a minimal GoapState-shaped dict for unit tests."""
    return {
        "execution_history": [None] * history,
        "world_state": {
            "total_cost_usd": cost,
            "total_tokens": tokens,
            "llm_call_count": llm_calls,
        },
        "status": status,
        "wall_clock_started_at": wall_clock_started_at,
    }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    @pytest.mark.parametrize(
        "policy",
        [
            MaxActionsPolicy(1),
            MaxWallClockPolicy(seconds=1.0),
            MaxCostPolicy(usd=1.0),
            MaxTokensPolicy(tokens=100),
            MaxLLMCallsPolicy(max_calls=1),
            OnStuckPolicy(),
            FirstOfPolicy(MaxActionsPolicy(1)),
            AllOfPolicy(MaxActionsPolicy(1), MaxCostPolicy(usd=1.0)),
        ],
    )
    def test_built_in_policies_satisfy_protocol(
        self, policy: TerminationPolicy
    ) -> None:
        assert isinstance(policy, TerminationPolicy)


# ---------------------------------------------------------------------------
# MaxActionsPolicy
# ---------------------------------------------------------------------------


class TestMaxActionsPolicy:
    def test_below_threshold_does_not_terminate(self) -> None:
        assert MaxActionsPolicy(5).should_terminate(_state(history=4)) is None

    def test_at_threshold_terminates(self) -> None:
        decision = MaxActionsPolicy(5).should_terminate(_state(history=5))
        assert decision is not None
        assert decision.error is True
        assert "5" in decision.reason

    def test_above_threshold_terminates(self) -> None:
        decision = MaxActionsPolicy(5).should_terminate(_state(history=999))
        assert decision is not None


# ---------------------------------------------------------------------------
# MaxWallClockPolicy
# ---------------------------------------------------------------------------


class TestMaxWallClockPolicy:
    def test_no_anchor_means_no_termination(self) -> None:
        assert (
            MaxWallClockPolicy(seconds=0.001).should_terminate(
                _state(wall_clock_started_at=None)
            )
            is None
        )

    def test_below_budget_does_not_terminate(self) -> None:
        anchor = time.monotonic()  # essentially now
        assert (
            MaxWallClockPolicy(seconds=10.0).should_terminate(
                _state(wall_clock_started_at=anchor)
            )
            is None
        )

    def test_above_budget_terminates(self) -> None:
        # Anchor set far in the past — budget already blown.
        anchor = time.monotonic() - 5.0
        decision = MaxWallClockPolicy(seconds=0.5).should_terminate(
            _state(wall_clock_started_at=anchor)
        )
        assert decision is not None
        assert decision.error is True


# ---------------------------------------------------------------------------
# Cost / token / llm-call policies
# ---------------------------------------------------------------------------


class TestAccountingPolicies:
    def test_cost_below_does_not_terminate(self) -> None:
        assert MaxCostPolicy(usd=10.0).should_terminate(_state(cost=5.0)) is None

    def test_cost_at_or_above_terminates(self) -> None:
        d1 = MaxCostPolicy(usd=5.0).should_terminate(_state(cost=5.0))
        d2 = MaxCostPolicy(usd=5.0).should_terminate(_state(cost=10.0))
        assert d1 is not None and d2 is not None

    def test_tokens_terminate_at_threshold(self) -> None:
        assert MaxTokensPolicy(tokens=1000).should_terminate(_state(tokens=999)) is None
        assert (
            MaxTokensPolicy(tokens=1000).should_terminate(_state(tokens=1000))
            is not None
        )

    def test_llm_calls_terminate_at_threshold(self) -> None:
        assert (
            MaxLLMCallsPolicy(max_calls=3).should_terminate(_state(llm_calls=2)) is None
        )
        assert (
            MaxLLMCallsPolicy(max_calls=3).should_terminate(_state(llm_calls=3))
            is not None
        )


# ---------------------------------------------------------------------------
# OnStuckPolicy
# ---------------------------------------------------------------------------


class TestOnStuckPolicy:
    def test_no_plan_status_terminates_without_error(self) -> None:
        decision = OnStuckPolicy().should_terminate(_state(status="no_plan"))
        assert decision is not None
        assert decision.error is False

    def test_other_status_does_not_terminate(self) -> None:
        for status in ("", "executing", "goal_achieved", "action_failed"):
            assert OnStuckPolicy().should_terminate(_state(status=status)) is None


# ---------------------------------------------------------------------------
# FirstOfPolicy
# ---------------------------------------------------------------------------


class TestFirstOfPolicy:
    def test_first_match_short_circuits(self) -> None:
        # Two policies — first fires.
        first = MaxActionsPolicy(1)
        second_called: list[bool] = [False]

        class Recording:
            name = "Recording"

            def should_terminate(self, state: Any) -> EarlyTermination | None:
                second_called[0] = True
                return None

        composite = FirstOfPolicy(first, Recording())
        result = composite.should_terminate(_state(history=5))
        assert result is not None
        assert second_called[0] is False  # short-circuited

    def test_no_inner_match_returns_none(self) -> None:
        composite = FirstOfPolicy(MaxActionsPolicy(1000), MaxCostPolicy(usd=1000.0))
        assert composite.should_terminate(_state(history=1, cost=1.0)) is None


# ---------------------------------------------------------------------------
# AllOfPolicy
# ---------------------------------------------------------------------------


class TestAllOfPolicy:
    def test_requires_at_least_one_inner_policy(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AllOfPolicy()

    def test_returns_none_when_any_inner_does_not_fire(self) -> None:
        composite = AllOfPolicy(MaxActionsPolicy(1), MaxCostPolicy(usd=1000.0))
        assert composite.should_terminate(_state(history=10, cost=0.0)) is None

    def test_terminates_when_every_inner_fires(self) -> None:
        composite = AllOfPolicy(MaxActionsPolicy(1), MaxCostPolicy(usd=0.5))
        decision = composite.should_terminate(_state(history=5, cost=10.0))
        assert decision is not None
        # Aggregated reason includes both.
        assert "Max actions" in decision.reason
        assert "Cost budget" in decision.reason

    def test_aggregated_error_is_any_error(self) -> None:
        # OnStuckPolicy returns error=False; combined with MaxActions
        # (error=True) the AllOf composite's error must be True.
        composite = AllOfPolicy(OnStuckPolicy(), MaxActionsPolicy(1))
        decision = composite.should_terminate(_state(history=5, status="no_plan"))
        assert decision is not None
        assert decision.error is True


# ---------------------------------------------------------------------------
# EarlyTermination dataclass
# ---------------------------------------------------------------------------


class TestEarlyTermination:
    def test_default_error_is_true(self) -> None:
        et = EarlyTermination(policy_name="X", reason="r")
        assert et.error is True

    def test_dataclass_is_frozen(self) -> None:
        import dataclasses as dc

        et = EarlyTermination(policy_name="X", reason="r")
        with pytest.raises(dc.FrozenInstanceError):
            et.reason = "mutated"  # type: ignore[misc]
