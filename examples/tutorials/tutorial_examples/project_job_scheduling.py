r"""Project Job Scheduling — the Multi-mode RCPSP in LangGOAP.

Models the multi-mode Resource-Constrained Project Scheduling Problem
(MRCPSP) as GOAP.  Each job has one or more modes
(fast-expensive vs slow-cheap), A\* picks the mode combination that
minimizes total work, and the CSP phase:

- Builds a precedence chain from effect→precondition matching so the
  scheduler knows which jobs must run before which.
- Computes per-plan makespan respecting precedence but running
  independent jobs in parallel via CP-SAT ``IntervalVar``.
- Validates a ``budget`` resource cap (hard or soft) and enumerates
  alternative mode combinations when the cheapest-by-duration plan
  violates it.

GOAP modelling
--------------

**World state:**

- ``<job>_done`` — flipped to ``True`` by the action that executes
  that job (in whatever mode).

**Actions:** one ``do_<job>_<mode>`` per (job, mode) pair.

- **preconditions**: all predecessors' ``<pred>_done=True`` plus the
  job's own ``<job>_done=False``.
- **effects**: ``<job>_done=True``.
- **cost**: the mode's ``duration_hours`` — A\* minimizes total
  work so the primary plan is the fastest by wall-clock-if-sequential.
- **resources**: ``duration_hours`` (aggregate) and ``budget``.
- **duration**: ``timedelta(hours=duration_hours)`` — fed to
  CP-SAT's scheduler so makespan reflects real parallelism.

**Goal:** every job done, with an optional hard ``budget`` cap and
a ``duration_hours → MINIMIZE`` objective.  The objective routes
the plan through CSP so the returned plan carries a
``HardSoftScore`` and a full schedule.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from langgoap import ActionSpec, ConstraintSpec, GoalSpec
from langgoap.goals import ObjectiveDirection

from .data.project_job_scheduling_instance import (
    DEFAULT_BUDGET,
    JOBS,
    Job,
    JobMode,
)


def _done_key(job_name: str) -> str:
    return f"{job_name}_done"


def _make_execute(effects: dict[str, Any]) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        return dict(effects)

    return execute


def _mode_action(job: Job, mode: JobMode) -> ActionSpec:
    preconditions: dict[str, Any] = {_done_key(job.name): False}
    for pred in job.predecessors:
        preconditions[_done_key(pred)] = True
    effects = {_done_key(job.name): True}
    return ActionSpec(
        name=f"do_{job.name}_{mode.name}",
        preconditions=preconditions,
        effects=effects,
        cost=float(mode.duration_hours),
        resources={
            "duration_hours": float(mode.duration_hours),
            "budget": float(mode.budget),
        },
        duration=timedelta(hours=mode.duration_hours),
        execute=_make_execute(effects),
    )


def project_job_scheduling_actions(
    jobs: tuple[Job, ...] = JOBS,
) -> list[ActionSpec]:
    """Return one ``do_<job>_<mode>`` action per (job, mode) pair."""
    actions: list[ActionSpec] = []
    for job in jobs:
        for mode in job.modes:
            actions.append(_mode_action(job, mode))
    return actions


def project_job_scheduling_start(
    jobs: tuple[Job, ...] = JOBS,
) -> dict[str, Any]:
    """Every job starts undone."""
    return {_done_key(j.name): False for j in jobs}


def project_job_scheduling_goal(
    jobs: tuple[Job, ...] = JOBS,
    *,
    minimize_duration: bool = True,
    max_budget: float | None = None,
    max_budget_level: str = "hard",
) -> GoalSpec:
    """Goal: every job done, with optional budget cap.

    Args:
        jobs: Jobs that must be completed.
        minimize_duration: When True (default), attach a
            ``duration_hours → MINIMIZE`` objective so the pipeline
            routes through CSP and reports a
            :class:`~langgoap.score.HardSoftScore`.
        max_budget: Optional upper bound on aggregated ``budget``.
            Omit for no cap.
        max_budget_level: ``"hard"`` (default) rejects violating
            plans as ``INFEASIBLE``; ``"soft"`` keeps the plan
            feasible but subtracts the violation amount from
            ``score.soft``.
    """
    conditions: dict[str, Any] = {_done_key(j.name): True for j in jobs}

    constraints: tuple[ConstraintSpec, ...] = ()
    if max_budget is not None:
        constraints = (
            ConstraintSpec(
                key="budget",
                max=float(max_budget),
                level=max_budget_level,  # type: ignore[arg-type]
            ),
        )

    objectives: dict[str, ObjectiveDirection] | None = None
    if minimize_duration:
        objectives = {"duration_hours": ObjectiveDirection.MINIMIZE}

    return GoalSpec(
        conditions=conditions,
        constraints=constraints,
        objectives=objectives,
    )


# Re-export for convenience in notebooks / tests.
__all__ = [
    "DEFAULT_BUDGET",
    "project_job_scheduling_actions",
    "project_job_scheduling_goal",
    "project_job_scheduling_start",
]
