"""Integration tests for ``ActionQos`` retry policy in the executor.

Per Rule 3 in research/plans/embabel-gap-closure.md, the scenarios below
mirror Embabel's ActionQos behaviour:
  * research/repos/embabel-agent/embabel-agent-api/src/test/kotlin/
      com/embabel/agent/spi/config/spring/ActionRetryPolicyPropertiesTest.kt
  * research/repos/embabel-agent/embabel-agent-api/src/test/kotlin/
      com/embabel/agent/core/support/ActionQosExtensionsTest.kt

LangGOAP's ``ActionQos`` is in-executor (intra-attempt retries with
exponential backoff) and layers below the existing failure/replan path:
exhausting ``max_attempts`` produces a single ``action_failed`` result
that the planner / observer can then react to (replan, blacklist,
escalate).  These tests focus on the executor surface in isolation —
GoapExecutor invoked directly — so the in-executor retry semantics are
not entangled with the graph-level replan-on-failure machinery.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.graph.nodes import GoapExecutor
from langgoap.graph.state import GoapState
from langgoap.qos import ActionQos
from tests.conftest import make_plan as _make_plan

# ---------------------------------------------------------------------------
# Recording tracer
# ---------------------------------------------------------------------------


class _RetryRecorder:
    """PlanningTracer-like object that records retry + completion events."""

    def __init__(self) -> None:
        self.retries: list[dict[str, Any]] = []
        self.completions: list[dict[str, Any]] = []

    # Required Protocol surface — no-op stubs for hooks the executor
    # may invoke incidentally.
    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None: ...
    def on_plan_complete(self, plan: Any, duration_ms: float) -> None: ...
    def on_plan_failed(self, reason: str, duration_ms: float) -> None: ...
    def on_action_start(self, action: Any, state: Any) -> None: ...
    def on_replan(self, reason: str, new_plan: Any) -> None: ...
    def on_goal_achieved(self, final_state: Any) -> None: ...
    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None: ...

    def on_action_complete(self, result: Any) -> None:
        self.completions.append(
            {
                "status": result.get("status"),
                "history": list(result.get("execution_history", [])),
            }
        )

    def on_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        self.retries.append(
            {
                "action": action.name,
                "attempt": attempt,
                "exc_type": type(exception).__name__,
                "exc_message": str(exception),
                "backoff_ms": backoff_ms,
            }
        )

    # Async siblings
    async def aon_plan_start(
        self, goal: Any, state: Any, strategy_name: str
    ) -> None: ...
    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None: ...
    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None: ...
    async def aon_action_start(self, action: Any, state: Any) -> None: ...
    async def aon_replan(self, reason: str, new_plan: Any) -> None: ...
    async def aon_goal_achieved(self, final_state: Any) -> None: ...
    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None: ...

    async def aon_action_complete(self, result: Any) -> None:
        self.on_action_complete(result)

    async def aon_action_retry(
        self,
        action: Any,
        attempt: int,
        exception: BaseException,
        backoff_ms: float,
    ) -> None:
        self.on_action_retry(action, attempt, exception, backoff_ms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flaky_action(
    name: str,
    *,
    fail_first_n: int,
    exception_factory: Any = TimeoutError,
    qos: ActionQos | None = None,
    use_aexecute: bool = False,
) -> tuple[ActionSpec, list[int]]:
    """Build an action that throws ``fail_first_n`` times then succeeds.

    Returns ``(action, call_log)`` — ``call_log`` is a 1-element list whose
    only entry is the running invocation count, mutated in place by the
    closure so tests can assert exactly how many attempts ran.
    """
    invocations: list[int] = [0]

    if use_aexecute:

        async def aexecute(world_state: dict[str, Any]) -> dict[str, Any] | None:
            invocations[0] += 1
            if invocations[0] <= fail_first_n:
                raise exception_factory(f"flaky failure on attempt {invocations[0]}")
            return {"done": True}

        action = ActionSpec(
            name=name,
            preconditions={"start": True},
            effects={"done": True},
            cost=1.0,
            aexecute=aexecute,
            execute=lambda ws: None,  # placeholder for sync-path validation
            qos=qos,
        )
    else:

        def execute(world_state: dict[str, Any]) -> dict[str, Any] | None:
            invocations[0] += 1
            if invocations[0] <= fail_first_n:
                raise exception_factory(f"flaky failure on attempt {invocations[0]}")
            return {"done": True}

        action = ActionSpec(
            name=name,
            preconditions={"start": True},
            effects={"done": True},
            cost=1.0,
            execute=execute,
            qos=qos,
        )
    return action, invocations


def _state_for(action: ActionSpec) -> GoapState:
    return {
        "world_state": {"start": True},
        "plan": _make_plan(action),
        "current_step": 0,
    }


# ---------------------------------------------------------------------------
# Sync executor
# ---------------------------------------------------------------------------


class TestActionQosSync:
    def test_idempotent_action_succeeds_after_two_retries(self) -> None:
        """Flaky action throws twice, succeeds on third attempt; with
        ``max_attempts=3`` and ``idempotent=True`` the executor must keep
        retrying through both failures and produce a successful result."""
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "flaky",
            fail_first_n=2,
            qos=ActionQos(
                max_attempts=3,
                idempotent=True,
                backoff_initial_ms=1.0,
                backoff_multiplier=1.0,
                backoff_max_ms=5.0,
                jitter=0.0,
            ),
        )
        executor = GoapExecutor(tracer=rec)

        result = executor(_state_for(action))

        assert calls[0] == 3, f"expected 3 invocations, got {calls[0]}"
        # Successful executor result has no explicit status field — it's
        # the failure path that sets ``status="action_failed"``.
        assert result.get("status") != "action_failed"
        assert result["execution_history"][0].success is True
        # Exactly two retries fired — between attempts 1→2 and 2→3.
        assert len(rec.retries) == 2
        assert [r["attempt"] for r in rec.retries] == [1, 2]
        # Effects applied: world state has ``done=True``.
        assert result["world_state"]["done"] is True

    def test_max_attempts_one_means_no_retry(self) -> None:
        """``ActionQos(max_attempts=1)`` (FIRE_ONCE) must fail on first
        attempt with no retry — matches Embabel's
        ``ActionRetryPolicy.FIRE_ONCE``."""
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "fragile",
            fail_first_n=1,
            qos=ActionQos(max_attempts=1, idempotent=True),
        )
        executor = GoapExecutor(tracer=rec)

        result = executor(_state_for(action))

        assert calls[0] == 1, "FIRE_ONCE must not retry"
        assert rec.retries == [], "no retry events must be emitted"
        assert result["status"] == "action_failed"

    def test_non_idempotent_action_does_not_retry_unrelated_exception(self) -> None:
        """``idempotent=False`` and the exception isn't in
        ``retryable_exceptions``: executor must fail on the first attempt
        even with ``max_attempts=10``."""
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "non_idempotent_value_error",
            fail_first_n=10,
            exception_factory=ValueError,
            qos=ActionQos(
                max_attempts=10,
                idempotent=False,
                # TimeoutError is the default retryable; ValueError is not.
            ),
        )
        executor = GoapExecutor(tracer=rec)

        result = executor(_state_for(action))

        assert calls[0] == 1, "non-idempotent + non-retryable must fail fast"
        assert rec.retries == []
        assert result["status"] == "action_failed"

    def test_non_idempotent_action_retries_retryable_exception(self) -> None:
        """``idempotent=False`` but the exception IS in
        ``retryable_exceptions``: executor must retry even though the action
        is not declared idempotent."""
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "retryable_timeout",
            fail_first_n=2,
            exception_factory=TimeoutError,
            qos=ActionQos(
                max_attempts=3,
                idempotent=False,
                backoff_initial_ms=1.0,
                backoff_multiplier=1.0,
                jitter=0.0,
            ),
        )
        executor = GoapExecutor(tracer=rec)

        result = executor(_state_for(action))

        assert calls[0] == 3
        assert len(rec.retries) == 2
        assert result.get("status") != "action_failed"

    def test_attempts_exhausted_emits_single_failure(self) -> None:
        """Action throws on every attempt; after ``max_attempts`` exhaust
        the executor must surface ONE ``action_failed`` result (not N)."""
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "always_fails",
            fail_first_n=99,
            qos=ActionQos(
                max_attempts=3,
                idempotent=True,
                backoff_initial_ms=1.0,
                backoff_multiplier=1.0,
                jitter=0.0,
            ),
        )
        executor = GoapExecutor(tracer=rec)

        result = executor(_state_for(action))

        assert calls[0] == 3, "must use up all attempts"
        # 2 retries (between attempts 1→2 and 2→3); no retry between 3→none.
        assert len(rec.retries) == 2
        # Exactly one completion event with a failure record.
        assert len(rec.completions) == 1
        assert any(not h.success for h in rec.completions[0]["history"])
        assert result["status"] == "action_failed"

    def test_default_qos_none_preserves_legacy_behavior(self) -> None:
        """Action without ``qos`` set must behave exactly as today: one
        attempt, one failure, no retries.  Backwards compatibility test."""
        rec = _RetryRecorder()
        action, calls = _flaky_action("legacy_no_qos", fail_first_n=1, qos=None)
        executor = GoapExecutor(tracer=rec)

        result = executor(_state_for(action))

        assert calls[0] == 1
        assert rec.retries == []
        assert result["status"] == "action_failed"


# ---------------------------------------------------------------------------
# Async executor — same behaviours under acall
# ---------------------------------------------------------------------------


class TestActionQosAsync:
    @pytest.mark.asyncio
    async def test_async_idempotent_action_succeeds_after_two_retries(self) -> None:
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "async_flaky",
            fail_first_n=2,
            qos=ActionQos(
                max_attempts=3,
                idempotent=True,
                backoff_initial_ms=1.0,
                backoff_multiplier=1.0,
                jitter=0.0,
            ),
            use_aexecute=True,
        )
        executor = GoapExecutor(tracer=rec)

        result = await executor.acall(_state_for(action))

        assert calls[0] == 3
        assert result.get("status") != "action_failed"
        assert len(rec.retries) == 2
        assert [r["attempt"] for r in rec.retries] == [1, 2]
        assert result["world_state"]["done"] is True

    @pytest.mark.asyncio
    async def test_async_non_idempotent_does_not_retry_unrelated_exception(
        self,
    ) -> None:
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "async_value_error",
            fail_first_n=10,
            exception_factory=ValueError,
            qos=ActionQos(max_attempts=10, idempotent=False),
            use_aexecute=True,
        )
        executor = GoapExecutor(tracer=rec)

        result = await executor.acall(_state_for(action))

        assert calls[0] == 1
        assert rec.retries == []
        assert result["status"] == "action_failed"

    @pytest.mark.asyncio
    async def test_async_attempts_exhausted_emits_single_failure(self) -> None:
        rec = _RetryRecorder()
        action, calls = _flaky_action(
            "async_always_fails",
            fail_first_n=99,
            qos=ActionQos(
                max_attempts=3,
                idempotent=True,
                backoff_initial_ms=1.0,
                backoff_multiplier=1.0,
                jitter=0.0,
            ),
            use_aexecute=True,
        )
        executor = GoapExecutor(tracer=rec)

        result = await executor.acall(_state_for(action))

        assert calls[0] == 3
        assert len(rec.retries) == 2
        assert len(rec.completions) == 1
        assert result["status"] == "action_failed"


# ---------------------------------------------------------------------------
# Backoff behaviour observable through the tracer
# ---------------------------------------------------------------------------


class TestActionQosBackoff:
    def test_exponential_backoff_observable_through_tracer(self) -> None:
        """Backoff between retries grows by ``backoff_multiplier`` and is
        capped by ``backoff_max_ms``.  jitter=0 makes the values exact."""
        rec = _RetryRecorder()
        action, _ = _flaky_action(
            "exp_backoff",
            fail_first_n=4,
            qos=ActionQos(
                max_attempts=5,
                idempotent=True,
                backoff_initial_ms=2.0,
                backoff_multiplier=3.0,
                backoff_max_ms=20.0,
                jitter=0.0,
            ),
        )
        executor = GoapExecutor(tracer=rec)

        executor(_state_for(action))

        backoffs = [r["backoff_ms"] for r in rec.retries]
        # attempt 1 → backoff initial=2.0
        # attempt 2 → 2*3 = 6.0
        # attempt 3 → 6*3 = 18.0
        # attempt 4 → 18*3 = 54.0 capped to 20.0
        assert backoffs == [2.0, 6.0, 18.0, 20.0], f"got {backoffs!r}"

    def test_jitter_keeps_backoff_within_bounds(self) -> None:
        """``jitter=0.5`` yields backoff in ``[0.5*base, 1.5*base]``."""
        rec = _RetryRecorder()
        action, _ = _flaky_action(
            "jitter",
            fail_first_n=3,
            qos=ActionQos(
                max_attempts=4,
                idempotent=True,
                backoff_initial_ms=10.0,
                backoff_multiplier=1.0,  # constant base for clean bounds
                backoff_max_ms=1_000.0,
                jitter=0.5,
            ),
        )
        executor = GoapExecutor(tracer=rec)

        executor(_state_for(action))

        for retry in rec.retries:
            backoff = retry["backoff_ms"]
            assert 5.0 <= backoff <= 15.0, f"jittered backoff out of bounds: {backoff}"
