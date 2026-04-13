"""GuardRails — pre/post execution action validation hooks.

Guards are checked before and after each action execution in GoapExecutor.
A WARN-severity failure is logged but execution continues.
A BLOCK-severity failure aborts the action and triggers replanning.

Three flavours are provided:

* :class:`ActionGuard` — synchronous protocol (duck-typed, runtime_checkable).
* :class:`AsyncActionGuard` — asynchronous protocol (duck-typed, runtime_checkable).
* :class:`FunctionalGuard` — convenience wrapper that turns a plain callable
  into an :class:`ActionGuard`.

:func:`run_guards_sync` and :func:`run_guards_async` run all registered guards
with per-guard exception isolation so a single bad guard never crashes the
executor.  :func:`has_blocking_failure` detects whether any result should
abort execution and trigger replanning.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from langgoap.actions import ActionSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class GuardSeverity(str, Enum):
    """Severity of a guard failure.

    * ``WARN``  — failure is logged but execution continues.
    * ``BLOCK`` — failure aborts the action and triggers replanning.
    """

    WARN = "warn"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Immutable result returned by a guard check.

    Attributes:
        passed:   ``True`` if the guard passed (no problem detected).
        message:  Human-readable explanation of the result.
        severity: Severity level applied when ``passed`` is ``False``.
                  Ignored when ``passed`` is ``True``.
    """

    passed: bool
    message: str
    severity: GuardSeverity = GuardSeverity.WARN


# ---------------------------------------------------------------------------
# Guard protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ActionGuard(Protocol):
    """Synchronous action guard protocol."""

    @property
    def name(self) -> str: ...  # pragma: no cover

    def check(
        self, action: ActionSpec, world_state: dict[str, Any]
    ) -> GuardResult: ...  # pragma: no cover


@runtime_checkable
class AsyncActionGuard(Protocol):
    """Asynchronous action guard protocol."""

    @property
    def name(self) -> str: ...  # pragma: no cover

    async def acheck(
        self, action: ActionSpec, world_state: dict[str, Any]
    ) -> GuardResult: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Functional convenience wrapper
# ---------------------------------------------------------------------------


class FunctionalGuard:
    """Wraps a callable into an :class:`ActionGuard`.

    Args:
        name:     Human-readable guard name.
        fn:       A callable ``(action, world_state) -> GuardResult``.
        severity: Default severity applied when constructing failure results
                  inside ``fn``.  This attribute is informational — ``fn``
                  is responsible for embedding severity in the
                  :class:`GuardResult` it returns.

    Example::

        guard = FunctionalGuard(
            "budget_check",
            lambda action, ws: GuardResult(
                passed=ws.get("budget", 0) > 0,
                message="Insufficient budget",
                severity=GuardSeverity.BLOCK,
            ),
        )
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[ActionSpec, dict[str, Any]], GuardResult],
        severity: GuardSeverity = GuardSeverity.WARN,
    ) -> None:
        self._name = name
        self._fn = fn
        self.severity = severity

    @property
    def name(self) -> str:
        return self._name

    def check(self, action: ActionSpec, world_state: dict[str, Any]) -> GuardResult:
        return self._fn(action, world_state)


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


def run_guards_sync(
    guards: list[ActionGuard | AsyncActionGuard],
    action: ActionSpec,
    world_state: dict[str, Any],
) -> list[GuardResult]:
    """Run all synchronous guards and return all :class:`GuardResult` objects.

    Async-only guards are skipped with a warning.  Per-guard exceptions are
    caught and converted into a ``WARN`` result so a bad guard never crashes
    the executor.

    Args:
        guards:      Guards to run.
        action:      The action about to execute.
        world_state: Current world state snapshot.

    Returns:
        :class:`GuardResult` for every guard that ran.
    """
    results: list[GuardResult] = []
    for guard in guards:
        if not hasattr(guard, "check"):
            logger.warning(
                "Guard %r is async-only; skipping in sync path. "
                "Use run_guards_async to evaluate async guards.",
                guard.name,
            )
            continue
        try:
            result = guard.check(action, world_state)
            results.append(result)
            if not result.passed:
                logger.warning(
                    "Guard %r failed (%s): %s",
                    guard.name,
                    result.severity.value,
                    result.message,
                )
        except Exception as exc:
            warn_result = GuardResult(
                passed=False,
                message=f"Guard {guard.name!r} raised: {exc}",
                severity=GuardSeverity.WARN,
            )
            results.append(warn_result)
            logger.warning("Guard %r raised: %s", guard.name, exc)
    return results


async def _run_guard_async(
    guard: ActionGuard | AsyncActionGuard,
    action: ActionSpec,
    world_state: dict[str, Any],
) -> GuardResult:
    try:
        if hasattr(guard, "acheck"):
            # Protocol duck-typing: hasattr confirms acheck exists on the union.
            return await guard.acheck(action, world_state)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, guard.check, action, world_state)
    except Exception as exc:
        logger.warning("Guard %r raised: %s", guard.name, exc)
        return GuardResult(
            passed=False,
            message=f"Guard {guard.name!r} raised: {exc}",
            severity=GuardSeverity.WARN,
        )


async def run_guards_async(
    guards: list[ActionGuard | AsyncActionGuard],
    action: ActionSpec,
    world_state: dict[str, Any],
) -> list[GuardResult]:
    """Run all guards concurrently via :func:`asyncio.gather`.

    Sync guards are dispatched to a thread executor; async guards are awaited
    directly.  Per-guard exceptions are caught and converted into ``WARN``
    results.

    Args:
        guards:      Guards to run.
        action:      The action about to execute.
        world_state: Current world state snapshot.

    Returns:
        :class:`GuardResult` for every guard.
    """
    if not guards:
        return []
    results: list[GuardResult] = list(
        await asyncio.gather(
            *(_run_guard_async(g, action, world_state) for g in guards)
        )
    )
    for guard, result in zip(guards, results):
        if not result.passed:
            logger.warning(
                "Guard %r failed (%s): %s",
                guard.name,
                result.severity.value,
                result.message,
            )
    return results


def has_blocking_failure(results: list[GuardResult]) -> bool:
    """Return ``True`` if any result has ``passed=False`` with ``severity=BLOCK``.

    Args:
        results: Output of :func:`run_guards_sync` or :func:`run_guards_async`.

    Returns:
        ``True`` when at least one BLOCK-severity failure is present.
    """
    return any(not r.passed and r.severity is GuardSeverity.BLOCK for r in results)


__all__ = [
    "ActionGuard",
    "AsyncActionGuard",
    "FunctionalGuard",
    "GuardResult",
    "GuardSeverity",
    "has_blocking_failure",
    "run_guards_async",
    "run_guards_sync",
]
