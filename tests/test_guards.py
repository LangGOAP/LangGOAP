"""Tests for langgoap.guards — GuardRails / Action Validation Hooks.

Coverage checklist (≥25 tests):
  - GuardSeverity enum values
  - GuardResult frozen dataclass (creation, defaults, immutability)
  - Protocol conformance (isinstance with runtime_checkable)
  - FunctionalGuard: basic usage, name property, WARN vs BLOCK severity
  - run_guards_sync: passing, failing, exception isolation, async-only skip,
    empty list, mixed sync/async list
  - run_guards_async: all async guards concurrent, sync fallback to executor,
    exception isolation, empty list
  - has_blocking_failure: no failures, WARN-only, BLOCK present, mixed
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from langgoap.actions import ActionSpec
from langgoap.guards import (
    ActionGuard,
    AsyncActionGuard,
    FunctionalGuard,
    GuardResult,
    GuardSeverity,
    has_blocking_failure,
    run_guards_async,
    run_guards_sync,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ACTION = ActionSpec(
    name="test_action", preconditions={"ready": True}, effects={"done": True}
)
WORLD: dict[str, Any] = {"ready": True, "budget": 100}


def _passing_result(msg: str = "ok") -> GuardResult:
    return GuardResult(passed=True, message=msg)


def _warn_result(msg: str = "warn") -> GuardResult:
    return GuardResult(passed=False, message=msg, severity=GuardSeverity.WARN)


def _block_result(msg: str = "blocked") -> GuardResult:
    return GuardResult(passed=False, message=msg, severity=GuardSeverity.BLOCK)


# ---------------------------------------------------------------------------
# GuardSeverity
# ---------------------------------------------------------------------------


class TestGuardSeverity:
    def test_warn_value(self) -> None:
        assert GuardSeverity.WARN == "warn"
        assert GuardSeverity.WARN.value == "warn"

    def test_block_value(self) -> None:
        assert GuardSeverity.BLOCK == "block"
        assert GuardSeverity.BLOCK.value == "block"

    def test_is_str_subclass(self) -> None:
        assert isinstance(GuardSeverity.WARN, str)
        assert isinstance(GuardSeverity.BLOCK, str)

    def test_distinct_values(self) -> None:
        assert GuardSeverity.WARN != GuardSeverity.BLOCK


# ---------------------------------------------------------------------------
# GuardResult
# ---------------------------------------------------------------------------


class TestGuardResult:
    def test_creation_with_all_fields(self) -> None:
        r = GuardResult(passed=True, message="all good", severity=GuardSeverity.BLOCK)
        assert r.passed is True
        assert r.message == "all good"
        assert r.severity is GuardSeverity.BLOCK

    def test_default_severity_is_warn(self) -> None:
        r = GuardResult(passed=False, message="warn me")
        assert r.severity is GuardSeverity.WARN

    def test_passed_true_with_default(self) -> None:
        r = GuardResult(passed=True, message="ok")
        assert r.passed is True

    def test_immutable_passed(self) -> None:
        r = GuardResult(passed=True, message="ok")
        with pytest.raises((AttributeError, TypeError)):
            r.passed = False  # type: ignore[misc]

    def test_immutable_message(self) -> None:
        r = GuardResult(passed=True, message="ok")
        with pytest.raises((AttributeError, TypeError)):
            r.message = "changed"  # type: ignore[misc]

    def test_immutable_severity(self) -> None:
        r = GuardResult(passed=False, message="x")
        with pytest.raises((AttributeError, TypeError)):
            r.severity = GuardSeverity.BLOCK  # type: ignore[misc]

    def test_equality(self) -> None:
        r1 = GuardResult(passed=True, message="ok")
        r2 = GuardResult(passed=True, message="ok")
        assert r1 == r2

    def test_inequality(self) -> None:
        r1 = GuardResult(passed=True, message="ok")
        r2 = GuardResult(passed=False, message="ok")
        assert r1 != r2


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class MyGuard:
    """Duck-typed sync guard."""

    @property
    def name(self) -> str:
        return "my_guard"

    def check(self, action: ActionSpec, world_state: dict[str, Any]) -> GuardResult:
        return GuardResult(passed=True, message="passed")


class MyAsyncGuard:
    """Duck-typed async guard."""

    @property
    def name(self) -> str:
        return "my_async_guard"

    async def acheck(
        self, action: ActionSpec, world_state: dict[str, Any]
    ) -> GuardResult:
        return GuardResult(passed=True, message="async passed")


def test_sync_guard_protocol_conformance() -> None:
    g = MyGuard()
    assert isinstance(g, ActionGuard)


def test_async_guard_protocol_conformance() -> None:
    g = MyAsyncGuard()
    assert isinstance(g, AsyncActionGuard)


def test_async_guard_is_not_sync_guard() -> None:
    """Async-only guard must NOT satisfy ActionGuard (no check method)."""
    g = MyAsyncGuard()
    assert not isinstance(g, ActionGuard)


# ---------------------------------------------------------------------------
# FunctionalGuard
# ---------------------------------------------------------------------------


class TestFunctionalGuard:
    def test_name_property(self) -> None:
        g = FunctionalGuard("budget", lambda a, ws: _passing_result())
        assert g.name == "budget"

    def test_check_returns_result(self) -> None:
        g = FunctionalGuard("check", lambda a, ws: _passing_result("ok"))
        r = g.check(ACTION, WORLD)
        assert r.passed is True
        assert r.message == "ok"

    def test_default_severity_warn(self) -> None:
        g = FunctionalGuard("warn_guard", lambda a, ws: _warn_result())
        assert g.severity is GuardSeverity.WARN

    def test_explicit_block_severity(self) -> None:
        g = FunctionalGuard(
            "block_guard", lambda a, ws: _block_result(), GuardSeverity.BLOCK
        )
        assert g.severity is GuardSeverity.BLOCK

    def test_fn_receives_action_and_world_state(self) -> None:
        received: list[Any] = []

        def fn(action: ActionSpec, ws: dict[str, Any]) -> GuardResult:
            received.append((action, ws))
            return _passing_result()

        g = FunctionalGuard("recorder", fn)
        g.check(ACTION, WORLD)
        assert len(received) == 1
        assert received[0][0] is ACTION
        assert received[0][1] is WORLD

    def test_satisfies_action_guard_protocol(self) -> None:
        g = FunctionalGuard("g", lambda a, ws: _passing_result())
        assert isinstance(g, ActionGuard)


# ---------------------------------------------------------------------------
# run_guards_sync
# ---------------------------------------------------------------------------


class TestRunGuardsSync:
    def test_empty_list_returns_empty(self) -> None:
        results = run_guards_sync([], ACTION, WORLD)
        assert results == []

    def test_passing_guard_returns_result(self) -> None:
        g = FunctionalGuard("ok", lambda a, ws: _passing_result())
        results = run_guards_sync([g], ACTION, WORLD)
        assert len(results) == 1
        assert results[0].passed is True

    def test_failing_guard_warn_returns_result(self) -> None:
        g = FunctionalGuard("warn", lambda a, ws: _warn_result("bad"))
        results = run_guards_sync([g], ACTION, WORLD)
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].severity is GuardSeverity.WARN

    def test_failing_guard_block_returns_result(self) -> None:
        g = FunctionalGuard("blk", lambda a, ws: _block_result("stop"))
        results = run_guards_sync([g], ACTION, WORLD)
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].severity is GuardSeverity.BLOCK

    def test_exception_isolation_returns_warn_result(self) -> None:
        def boom(a: ActionSpec, ws: dict[str, Any]) -> GuardResult:
            raise RuntimeError("guard exploded")

        g = FunctionalGuard("bad", boom)
        g2 = FunctionalGuard("good", lambda a, ws: _passing_result())
        results = run_guards_sync([g, g2], ACTION, WORLD)
        assert len(results) == 2
        assert results[0].passed is False
        assert results[0].severity is GuardSeverity.WARN
        assert "exploded" in results[0].message
        assert results[1].passed is True

    def test_async_only_guard_skipped_with_warning(self, caplog: Any) -> None:
        import logging

        g = MyAsyncGuard()
        with caplog.at_level(logging.WARNING):
            results = run_guards_sync([g], ACTION, WORLD)  # type: ignore[arg-type]
        assert results == []
        assert any("async-only" in rec.message for rec in caplog.records)

    def test_mixed_sync_and_async_only_list(self) -> None:
        sync_g = FunctionalGuard("sync", lambda a, ws: _passing_result())
        async_g = MyAsyncGuard()
        results = run_guards_sync([sync_g, async_g], ACTION, WORLD)  # type: ignore[arg-type]
        # Only sync guard produces a result; async-only is skipped
        assert len(results) == 1
        assert results[0].passed is True

    def test_multiple_guards_all_passing(self) -> None:
        guards = [
            FunctionalGuard(f"g{i}", lambda a, ws: _passing_result()) for i in range(3)
        ]
        results = run_guards_sync(guards, ACTION, WORLD)
        assert len(results) == 3
        assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# run_guards_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_guards_async_empty_list() -> None:
    results = await run_guards_async([], ACTION, WORLD)
    assert results == []


@pytest.mark.asyncio
async def test_run_guards_async_with_sync_guard() -> None:
    g = FunctionalGuard("sync_in_async", lambda a, ws: _passing_result("async ok"))
    results = await run_guards_async([g], ACTION, WORLD)
    assert len(results) == 1
    assert results[0].passed is True


@pytest.mark.asyncio
async def test_run_guards_async_with_async_guard() -> None:
    g = MyAsyncGuard()
    results = await run_guards_async([g], ACTION, WORLD)  # type: ignore[arg-type]
    assert len(results) == 1
    assert results[0].passed is True


@pytest.mark.asyncio
async def test_run_guards_async_concurrent_execution() -> None:
    """All async guards run concurrently (gather) rather than serially."""
    order: list[int] = []

    class SlowGuard:
        def __init__(self, idx: int, delay: float) -> None:
            self._idx = idx
            self._delay = delay

        @property
        def name(self) -> str:
            return f"slow_{self._idx}"

        async def acheck(
            self, action: ActionSpec, world_state: dict[str, Any]
        ) -> GuardResult:
            await asyncio.sleep(self._delay)
            order.append(self._idx)
            return GuardResult(passed=True, message="ok")

    g1 = SlowGuard(1, 0.05)
    g2 = SlowGuard(2, 0.01)
    results = await run_guards_async([g1, g2], ACTION, WORLD)  # type: ignore[arg-type]
    # If concurrent, faster guard (g2) appends before slower (g1)
    assert order == [2, 1]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_run_guards_async_exception_isolation() -> None:
    class BrokenGuard:
        @property
        def name(self) -> str:
            return "broken"

        async def acheck(
            self, action: ActionSpec, world_state: dict[str, Any]
        ) -> GuardResult:
            raise ValueError("async boom")

    broken = BrokenGuard()
    good = FunctionalGuard("good", lambda a, ws: _passing_result())
    results = await run_guards_async([broken, good], ACTION, WORLD)  # type: ignore[arg-type]
    assert len(results) == 2
    assert results[0].passed is False
    assert results[0].severity is GuardSeverity.WARN
    assert "boom" in results[0].message
    assert results[1].passed is True


@pytest.mark.asyncio
async def test_run_guards_async_sync_fallback_to_executor() -> None:
    """Sync guards can run inside run_guards_async via executor fallback."""
    executed = []

    def fn(a: ActionSpec, ws: dict[str, Any]) -> GuardResult:
        executed.append(True)
        return _passing_result()

    g = FunctionalGuard("sync_fallback", fn)
    results = await run_guards_async([g], ACTION, WORLD)
    assert len(results) == 1
    assert results[0].passed is True
    assert executed == [True]


# ---------------------------------------------------------------------------
# has_blocking_failure
# ---------------------------------------------------------------------------


class TestHasBlockingFailure:
    def test_empty_results_false(self) -> None:
        assert has_blocking_failure([]) is False

    def test_all_passed_false(self) -> None:
        results = [_passing_result(), _passing_result()]
        assert has_blocking_failure(results) is False

    def test_warn_failure_only_false(self) -> None:
        results = [_warn_result("oops")]
        assert has_blocking_failure(results) is False

    def test_block_failure_true(self) -> None:
        results = [_block_result("halt")]
        assert has_blocking_failure(results) is True

    def test_mixed_passed_and_warn_false(self) -> None:
        results = [_passing_result(), _warn_result()]
        assert has_blocking_failure(results) is False

    def test_mixed_with_block_true(self) -> None:
        results = [_passing_result(), _warn_result(), _block_result()]
        assert has_blocking_failure(results) is True

    def test_multiple_blocks_true(self) -> None:
        results = [_block_result("a"), _block_result("b")]
        assert has_blocking_failure(results) is True

    def test_passed_true_with_block_severity_not_counted(self) -> None:
        """A result that passed=True with BLOCK severity is NOT a blocking failure."""
        r = GuardResult(passed=True, message="ok", severity=GuardSeverity.BLOCK)
        assert has_blocking_failure([r]) is False
