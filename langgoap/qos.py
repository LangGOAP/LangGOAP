"""Action QoS (retry policy) for the GOAP executor.

``ActionQos`` is the in-executor retry policy: when an action's
``execute`` / ``aexecute`` callable raises an exception, the executor
consults the action's ``qos`` to decide whether to retry the call with
exponential backoff.

This is **layered below** the existing planner-level replan-on-failure
mechanism.  After ``max_attempts`` are exhausted, the executor surfaces
a single ``action_failed`` result — the planner / observer then decide
whether to replan, blacklist, or escalate.

API surface mirrors Embabel's ``ActionQos`` data class
(``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
com/embabel/agent/core/ActionQos.kt``) with two LangGOAP-specific
additions:

* ``jitter`` — ``±jitter%`` random noise on each backoff to avoid
  thundering-herd on shared dependencies.
* ``retryable_exceptions`` — explicit exception allow-list consulted when
  ``idempotent=False``.  Idempotent actions retry on every exception.

Two ready-made policies are exported:

* :data:`FIRE_ONCE` — single attempt, no retry.  LangGOAP's default
  (matches the legacy ``max_retries=0`` behaviour, so adding ``qos`` to
  an existing project is non-breaking).
* :data:`DEFAULT_RETRY_POLICY` — five attempts, 10s initial backoff,
  5x multiplier, 60s cap.  Mirrors Embabel's ``ActionQos()`` defaults.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionQos:
    """Quality-of-service / retry policy for a single action.

    Attributes:
        max_attempts: Total attempts allowed (1 = no retry).
        backoff_initial_ms: Wait between attempt 1 and 2.
        backoff_multiplier: Factor applied each subsequent retry.
        backoff_max_ms: Hard cap on a single backoff interval.
        jitter: Multiplicative noise applied to each backoff in
            ``[1 - jitter, 1 + jitter]``.  ``0.0`` disables jitter.
        idempotent: When ``True``, retry on **any** exception.  When
            ``False``, retry only on exceptions in
            :attr:`retryable_exceptions`.
        retryable_exceptions: Tuple of exception types that are safe to
            retry even for non-idempotent actions (network hiccups,
            timeouts).  Default: ``(TimeoutError,)``.
    """

    max_attempts: int = 1
    backoff_initial_ms: float = 250.0
    backoff_multiplier: float = 2.0
    backoff_max_ms: float = 30_000.0
    jitter: float = 0.1
    idempotent: bool = False
    retryable_exceptions: tuple[type[BaseException], ...] = (TimeoutError,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.backoff_initial_ms < 0:
            raise ValueError(
                f"backoff_initial_ms must be >= 0, got {self.backoff_initial_ms}"
            )
        if self.backoff_multiplier < 1.0:
            raise ValueError(
                f"backoff_multiplier must be >= 1.0, got {self.backoff_multiplier}"
            )
        if self.backoff_max_ms < self.backoff_initial_ms:
            raise ValueError(
                "backoff_max_ms must be >= backoff_initial_ms "
                f"({self.backoff_max_ms} < {self.backoff_initial_ms})"
            )
        if not (0.0 <= self.jitter <= 1.0):
            raise ValueError(f"jitter must be in [0, 1], got {self.jitter}")

    def compute_backoff_ms(
        self, attempt: int, *, rng: random.Random | None = None
    ) -> float:
        """Backoff in milliseconds *before* retrying after ``attempt``."""
        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt}")
        base = self.backoff_initial_ms * (self.backoff_multiplier ** (attempt - 1))
        capped = min(base, self.backoff_max_ms)
        if self.jitter == 0.0:
            return capped
        rng_inst = rng if rng is not None else random.Random()
        factor = 1.0 + rng_inst.uniform(-self.jitter, self.jitter)
        return capped * factor

    def should_retry(self, exception: BaseException, attempt: int) -> bool:
        """Decide whether to retry after ``exception`` on ``attempt``."""
        if attempt >= self.max_attempts:
            return False
        if self.idempotent:
            return True
        return isinstance(exception, self.retryable_exceptions)


FIRE_ONCE: ActionQos = ActionQos(max_attempts=1)
"""Single attempt, no retry."""


DEFAULT_RETRY_POLICY: ActionQos = ActionQos(
    max_attempts=5,
    backoff_initial_ms=10_000.0,
    backoff_multiplier=5.0,
    backoff_max_ms=60_000.0,
    jitter=0.1,
    idempotent=False,
)
"""Mirrors Embabel's ``ActionQos()`` defaults."""


__all__ = [
    "ActionQos",
    "DEFAULT_RETRY_POLICY",
    "FIRE_ONCE",
]
