r"""Multi-mode RCPSP instance derived from standard benchmark data.

Provenance
----------
Derived from the PSPLIB / RCPSP benchmark suite (specifically a
10-job multi-mode instance, ``j1011_7.mm``).

The original instance has 12 jobs (10 real + source + sink) with 3
modes each and 2 renewable + 2 non-renewable resources.  This
tutorial fixture keeps the **precedence DAG and non-renewable
budget** but collapses the problem to **5 jobs × 2 modes × 1
budget** so the solver runs instantly while still exercising:

- **Precedence scheduling** — the dependency graph is built from
  effect→precondition chains, exactly as ``planner/csp.py::
  build_dependency_graph`` computes it for every other tutorial.
- **Mode selection** — each job has a *fast but expensive* and a
  *slow but cheap* mode.  A\* picks whichever mode combination
  minimizes the primary objective.
- **Non-renewable resource budget** — aggregated ``budget`` cap
  becomes a ``ConstraintSpec(key="budget", level="hard")`` that can
  mark the cheapest plan INFEASIBLE and force the pipeline to
  enumerate alternatives.
- **Makespan minimization** — per-job durations feed CP-SAT's
  ``IntervalVar`` scheduler; makespan falls out as the longest
  path respecting precedence.

Out of scope for this tutorial
------------------------------

- **Renewable resource capacities** (machines/labor constrained per
  time slice).  LangGoap's CSP supports *total* resource
  aggregation but not per-period capacity — exactly the same
  limitation documented in the vehicle routing fixture.  A
  post-v0.1.0 extension could push these into CP-SAT directly via
  ``add_cumulative``.
- **Multiple projects, activity release dates, tardiness costs.**
  Single-project horizon keeps the example short.

Job DAG
-------
::

    design (j1, source)
        │
        ├── frontend (j2)  ──┐
        │                    ├── integrate (j4) ── deploy (j5, sink)
        └── backend  (j3)  ──┘

Every job has two modes.  ``fast`` has shorter duration but larger
budget consumption; ``slow`` is the opposite.  The source and sink
jobs are trivial (zero duration, zero budget) and kept as single
actions so the precedence structure is visible in the rendered
plan without cluttering the action catalog.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobMode:
    name: str  # "fast" | "slow"
    duration_hours: int
    budget: int


@dataclass(frozen=True)
class Job:
    name: str
    modes: tuple[JobMode, ...]
    predecessors: tuple[str, ...]  # names of prerequisite jobs


# ---------------------------------------------------------------------------
# 5-job / 2-mode / single-budget instance
# ---------------------------------------------------------------------------

JOBS: tuple[Job, ...] = (
    Job(
        name="design",
        modes=(JobMode(name="single", duration_hours=1, budget=0),),
        predecessors=(),
    ),
    Job(
        name="frontend",
        modes=(
            JobMode(name="fast", duration_hours=3, budget=10),
            JobMode(name="slow", duration_hours=6, budget=4),
        ),
        predecessors=("design",),
    ),
    Job(
        name="backend",
        modes=(
            JobMode(name="fast", duration_hours=4, budget=12),
            JobMode(name="slow", duration_hours=8, budget=6),
        ),
        predecessors=("design",),
    ),
    Job(
        name="integrate",
        modes=(
            JobMode(name="fast", duration_hours=2, budget=8),
            JobMode(name="slow", duration_hours=5, budget=3),
        ),
        predecessors=("frontend", "backend"),
    ),
    Job(
        name="deploy",
        modes=(JobMode(name="single", duration_hours=1, budget=2),),
        predecessors=("integrate",),
    ),
)


# Budget cap that keeps both "all slow" (15) feasible and blocks
# "all fast" (32) — the feasibility boundary used by the hard-budget
# tutorial scenario.
DEFAULT_BUDGET: int = 20
