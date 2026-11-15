r"""Domain data for the flexible job shop tutorial (notebook 15).

This is a Tier 3 showcase instance derived from the classic flexible
job shop scheduling problem (FJSSP) literature and the MINI_FSP example
in ``https://github.com/aiplan4eu/unified-planning/blob/HEAD/unified_planning/test/examples/
scheduling/flexible_jobshop.py``.

The instance is deliberately small so CP-SAT solves it in milliseconds
in CI and every number in the integration test can be asserted by hand:

- **3 jobs** (``alpha``, ``beta``, ``gamma``), each with 2 sequential
  operations (``prepare`` → ``finalize``).
- **2 modes** per operation (``express`` and ``standard``).  Express
  runs on an in-house fast machine that is pricey per hour; standard
  runs on a cloud worker that is cheap but slow.
- **12 actions** total — one per (job, operation, mode) triple.

The instance is tuned so that:

1. The A\* greedy (cost = cost) plan picks every ``*_express`` mode
   and has aggregated ``cost_usd = 150`` and aggregated
   ``duration_hours = 15``.
2. A hard ``cost_usd <= 130`` cap on the goal makes the greedy plan
   ``INFEASIBLE`` and the pipeline enumeration blacklists
   ``prepare_gamma_express``; the resulting alternative (with
   ``prepare_gamma_standard``) has ``cost_usd = 125`` and
   ``duration_hours = 17``.
3. Each job's two operations form a precedence chain
   (``finalize_X`` requires ``prepare_X_done=True``) and the three
   jobs are mutually independent, so CP-SAT schedules them as three
   parallel pipelines with per-job makespans ``3h``/``5h``/``7h``
   (the ``gamma`` chain is the critical path).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    """One execution mode of an operation.

    ``cost`` is the A\\* heuristic cost (wall-clock hours) and
    ``cost_usd`` is the dollar price used as the CSP constraint and
    objective.
    """

    name: str
    cost: float
    duration_hours: float
    cost_usd: float


@dataclass(frozen=True)
class Operation:
    """A single operation on a job, with one or more alternative modes."""

    name: str
    modes: tuple[Mode, ...]


@dataclass(frozen=True)
class Job:
    """A job is a linear chain of operations executed in order."""

    name: str
    operations: tuple[Operation, ...]


# ---------------------------------------------------------------------------
# The canonical instance used by the tutorial and integration test.
# ---------------------------------------------------------------------------

JOBS: tuple[Job, ...] = (
    Job(
        name="alpha",
        operations=(
            Operation(
                name="prepare",
                modes=(
                    Mode("express", cost=2.0, duration_hours=2.0, cost_usd=20.0),
                    Mode("standard", cost=3.0, duration_hours=4.0, cost_usd=8.0),
                ),
            ),
            Operation(
                name="finalize",
                modes=(
                    Mode("express", cost=1.0, duration_hours=1.0, cost_usd=10.0),
                    Mode("standard", cost=2.0, duration_hours=2.0, cost_usd=4.0),
                ),
            ),
        ),
    ),
    Job(
        name="beta",
        operations=(
            Operation(
                name="prepare",
                modes=(
                    Mode("express", cost=3.0, duration_hours=3.0, cost_usd=30.0),
                    Mode("standard", cost=4.0, duration_hours=5.0, cost_usd=12.0),
                ),
            ),
            Operation(
                name="finalize",
                modes=(
                    Mode("express", cost=2.0, duration_hours=2.0, cost_usd=20.0),
                    Mode("standard", cost=3.0, duration_hours=3.0, cost_usd=8.0),
                ),
            ),
        ),
    ),
    Job(
        name="gamma",
        operations=(
            Operation(
                name="prepare",
                modes=(
                    Mode("express", cost=4.0, duration_hours=4.0, cost_usd=40.0),
                    Mode("standard", cost=5.0, duration_hours=6.0, cost_usd=15.0),
                ),
            ),
            Operation(
                name="finalize",
                modes=(
                    Mode("express", cost=3.0, duration_hours=3.0, cost_usd=30.0),
                    Mode("standard", cost=4.0, duration_hours=4.0, cost_usd=10.0),
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Derived constants — the test module imports these verbatim so any
# change to the instance surfaces in a single place.
# ---------------------------------------------------------------------------

# Greedy (all *_express) aggregates.
GREEDY_TOTAL_COST = 15.0  # 2+1+3+2+4+3
GREEDY_COST_USD = 150.0  # 20+10+30+20+40+30
GREEDY_DURATION_HOURS = 15.0  # 2+1+3+2+4+3

# Critical-path (all express) per-job end times:
#   alpha: 2h + 1h = 3h
#   beta:  3h + 2h = 5h
#   gamma: 4h + 3h = 7h
# CP-SAT runs the three chains in parallel, so makespan is max() = 7h.
GREEDY_MAKESPAN_HOURS = 7.0

# Alternative the pipeline selects for ``cost_usd -> MINIMIZE`` with the
# default ``max_alternatives`` (8): alternatives are enumerated breadth-first
# over excluded actions, so the search reaches plans two swaps away from
# the greedy plan.  The cheapest of those swaps the two prepares whose
# standard mode saves the most without stretching the critical path:
#   prepare_alpha_standard + prepare_beta_standard, everything else express.
ALT_PLAN_COST = 17.0  # 3+1+4+2+4+3
ALT_PLAN_COST_USD = 120.0  # 8+10+12+20+40+30
ALT_PLAN_DURATION_HOURS = 19.0  # 4+1+5+2+4+3
# alpha 4h + 1h = 5h, beta 5h + 2h = 7h, gamma still 4h + 3h = 7h.
ALT_PLAN_MAKESPAN_HOURS = 7.0

# The global minimum of ``cost_usd`` is every mode standard.  It lies six
# swaps away from the greedy plan, beyond the default enumeration budget;
# the notebook reaches it by widening ``max_alternatives`` or by encoding
# ``cost_usd`` as the A* cost (see ``TestMinimizeObjectiveSelection``).
ALL_STANDARD_TOTAL_COST = 21.0  # 3+2+4+3+5+4
ALL_STANDARD_COST_USD = 57.0  # 8+4+12+8+15+10
ALL_STANDARD_MAKESPAN_HOURS = 10.0  # gamma: 6h + 4h

# Hard cap the greedy plan ($150) breaks; several enumerated alternatives
# pass it and selection picks the cheapest (``ALT_PLAN_COST_USD``).
HARD_COST_CAP = 130.0

# Cap strictly below every one of the 64 mode combinations (the cheapest
# costs $57).  No alternative can rescue the plan → pipeline returns INFEASIBLE.
IMPOSSIBLE_HARD_CAP = 50.0

# Cap below the greedy aggregate but declared ``soft``.  The plan stays
# feasible; the overflow (``GREEDY_COST_USD - SOFT_CAP``) contributes a
# penalty to ``score.soft``.
SOFT_CAP = 100.0

__all__ = [
    "JOBS",
    "Mode",
    "Operation",
    "Job",
    "GREEDY_TOTAL_COST",
    "GREEDY_COST_USD",
    "GREEDY_DURATION_HOURS",
    "GREEDY_MAKESPAN_HOURS",
    "ALL_STANDARD_COST_USD",
    "ALL_STANDARD_MAKESPAN_HOURS",
    "ALL_STANDARD_TOTAL_COST",
    "ALT_PLAN_COST",
    "ALT_PLAN_COST_USD",
    "ALT_PLAN_DURATION_HOURS",
    "ALT_PLAN_MAKESPAN_HOURS",
    "HARD_COST_CAP",
    "IMPOSSIBLE_HARD_CAP",
    "SOFT_CAP",
]
