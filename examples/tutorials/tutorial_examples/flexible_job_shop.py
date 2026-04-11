r"""Flexible Job Shop — Tier 3 showcase tutorial for notebook 15.

This is the final notebook in the Tier 3 set and deliberately exercises
every major v0.1.0 subsystem at once:

- **A\*** finds the greedy-by-cost plan (all ``*_express`` modes).
- **CSP validation** rejects it when a ``cost_usd`` hard cap is below
  the greedy aggregate, triggering the pipeline's enumeration path.
- **CSP optimization** picks the cheapest single-blacklist alternative
  via CP-SAT multi-plan selection with a ``cost_usd → MINIMIZE``
  objective.
- **CSP temporal scheduling** places the three independent jobs on
  parallel tracks with per-job precedence chains, giving a makespan
  dominated by the critical-path job (``gamma``).
- **Fluent :class:`~langgoap.constraints.ConstraintBuilder`** reproduces
  the hand-rolled goal structurally — the
  ``ConstraintProvider`` analogue.
- **GoapGraph tracer + history** capture the planning telemetry and
  persist an :class:`~langgoap.history.ExecutionRecord` in a
  :class:`~langgraph.store.memory.InMemoryStore`.
- **Visualization** renders the resulting schedule as an ASCII Gantt
  chart via :func:`~langgoap.viz.render_ascii_gantt`.

Flexible job shop — mode flexibility, not machine flexibility
-------------------------------------------------------------

The classic Flexible Job Shop Scheduling Problem (FJSSP) lets each
operation run on one of several eligible machines with different
processing times.  LangGoap's CP-SAT scheduler does not model machine
mutex natively (no ``NoOverlap`` per-machine resource), so the FJSSP
"choose a machine" decision is recast here as a
**"choose a mode"** decision on each operation: ``express`` runs on
an in-house fast machine that is pricey per hour; ``standard`` runs on
a cloud worker that is cheap but slow.  This is the same recasting
that ``project_job_scheduling.py`` (notebook 7) uses for MRCPSP, and it
is the idiomatic LangGoap pattern for "flexibility" problems.

GOAP modelling
--------------

**World state flags** (all ``False`` at start):

- ``prepare_<job>_done`` — the ``prepare`` operation on a job has
  finished in whatever mode was chosen.
- ``finalize_<job>_done`` — the final operation on a job has finished.

**Action catalog** — 12 actions, one per (job, operation, mode) triple:

+-------------------------+--------------------+----------------------+------+-----+
| Action                  | Pre                | Effects              | Dur  | Cst |
+=========================+====================+======================+======+=====+
| prepare_alpha_express   | (none)             | prepare_alpha_done   | 2h   | 2   |
+-------------------------+--------------------+----------------------+------+-----+
| prepare_alpha_standard  | (none)             | prepare_alpha_done   | 4h   | 3   |
+-------------------------+--------------------+----------------------+------+-----+
| finalize_alpha_express  | prepare_alpha_done | finalize_alpha_done  | 1h   | 1   |
+-------------------------+--------------------+----------------------+------+-----+
| finalize_alpha_standard | prepare_alpha_done | finalize_alpha_done  | 2h   | 2   |
+-------------------------+--------------------+----------------------+------+-----+
| prepare_beta_express    | (none)             | prepare_beta_done    | 3h   | 3   |
+-------------------------+--------------------+----------------------+------+-----+
| prepare_beta_standard   | (none)             | prepare_beta_done    | 5h   | 4   |
+-------------------------+--------------------+----------------------+------+-----+
| finalize_beta_express   | prepare_beta_done  | finalize_beta_done   | 2h   | 2   |
+-------------------------+--------------------+----------------------+------+-----+
| finalize_beta_standard  | prepare_beta_done  | finalize_beta_done   | 3h   | 3   |
+-------------------------+--------------------+----------------------+------+-----+
| prepare_gamma_express   | (none)             | prepare_gamma_done   | 4h   | 4   |
+-------------------------+--------------------+----------------------+------+-----+
| prepare_gamma_standard  | (none)             | prepare_gamma_done   | 6h   | 5   |
+-------------------------+--------------------+----------------------+------+-----+
| finalize_gamma_express  | prepare_gamma_done | finalize_gamma_done  | 3h   | 3   |
+-------------------------+--------------------+----------------------+------+-----+
| finalize_gamma_standard | prepare_gamma_done | finalize_gamma_done  | 4h   | 4   |
+-------------------------+--------------------+----------------------+------+-----+

Every action aggregates two resources: ``cost_usd`` (the CSP
constraint and objective) and ``duration_hours`` (informational).
Pinned totals and the reasoning behind the
:data:`~.data.flexible_job_shop_instance.HARD_COST_CAP` live in the
data-fixture module alongside the :data:`~.data.flexible_job_shop_instance.JOBS`
tuple.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any, Literal

from langgoap import ActionSpec, ConstraintSpec, GoalSpec
from langgoap.constraints import ConstraintBuilder
from langgoap.goals import ObjectiveDirection

from .data.flexible_job_shop_instance import JOBS, Job, Mode, Operation


# ---------------------------------------------------------------------------
# Naming helpers — pinned so tests and notebooks can cross-reference.
# ---------------------------------------------------------------------------


def action_name(job: Job, operation: Operation, mode: Mode) -> str:
    """Canonical ``<operation>_<job>_<mode>`` action name."""
    return f"{operation.name}_{job.name}_{mode.name}"


def done_key(job: Job, operation: Operation) -> str:
    """Canonical ``<operation>_<job>_done`` world-state flag."""
    return f"{operation.name}_{job.name}_done"


# ---------------------------------------------------------------------------
# Execute helpers — deterministic, world-state-only mutation.
# ---------------------------------------------------------------------------


def _make_execute(
    effects: dict[str, Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        del ws  # effects are static; starting state is not read
        return dict(effects)

    return execute


# ---------------------------------------------------------------------------
# Action factory
# ---------------------------------------------------------------------------


def _mode_action(
    job: Job,
    operation: Operation,
    mode: Mode,
    prior: Operation | None,
) -> ActionSpec:
    """Build one ``ActionSpec`` for a (job, operation, mode) triple.

    ``prior`` is the preceding operation in the job's linear chain, or
    ``None`` for the first operation.  Its ``done_key`` becomes the
    precondition that serializes the operations within the job.
    """
    preconditions: dict[str, Any] = {done_key(job, operation): False}
    if prior is not None:
        preconditions[done_key(job, prior)] = True
    effects = {done_key(job, operation): True}
    return ActionSpec(
        name=action_name(job, operation, mode),
        preconditions=preconditions,
        effects=effects,
        cost=float(mode.cost),
        resources={
            "cost_usd": float(mode.cost_usd),
            "duration_hours": float(mode.duration_hours),
        },
        duration=timedelta(hours=mode.duration_hours),
        execute=_make_execute(effects),
    )


def flexible_job_shop_actions(
    jobs: tuple[Job, ...] = JOBS,
) -> list[ActionSpec]:
    """Return one ``ActionSpec`` per (job, operation, mode) triple."""
    actions: list[ActionSpec] = []
    for job in jobs:
        prior: Operation | None = None
        for operation in job.operations:
            for mode in operation.modes:
                actions.append(_mode_action(job, operation, mode, prior))
            prior = operation
    return actions


def flexible_job_shop_start(
    jobs: tuple[Job, ...] = JOBS,
) -> dict[str, Any]:
    """Every operation starts undone."""
    state: dict[str, Any] = {}
    for job in jobs:
        for operation in job.operations:
            state[done_key(job, operation)] = False
    return state


# ---------------------------------------------------------------------------
# Goal factories — hand-rolled and fluent-builder variants.
# ---------------------------------------------------------------------------


def _goal_conditions(jobs: tuple[Job, ...]) -> dict[str, Any]:
    """Every job's last operation must be done."""
    conditions: dict[str, Any] = {}
    for job in jobs:
        last = job.operations[-1]
        conditions[done_key(job, last)] = True
    return conditions


def flexible_job_shop_goal(
    jobs: tuple[Job, ...] = JOBS,
    *,
    minimize_cost_usd: bool = True,
    max_cost_usd: float | None = None,
    max_cost_level: Literal["hard", "soft"] = "hard",
) -> GoalSpec:
    """Finish every job, with an optional ``cost_usd`` cap and objective.

    Args:
        jobs: Jobs to complete.
        minimize_cost_usd: Attach a ``cost_usd → MINIMIZE`` objective so
            the pipeline routes through CSP and
            :class:`~langgoap.score.HardSoftScore` is populated.  This
            also gives CP-SAT a deterministic tie-breaker when several
            single-blacklist alternatives satisfy the hard cap.
        max_cost_usd: Optional aggregated ``cost_usd`` ceiling.  When
            ``level="hard"`` and the greedy plan exceeds the cap the
            pipeline enumerates alternatives; CP-SAT picks the
            alternative with the lowest ``cost_usd``.
        max_cost_level: ``"hard"`` (default) or ``"soft"``.
    """
    conditions = _goal_conditions(jobs)

    constraints: tuple[ConstraintSpec, ...] = ()
    if max_cost_usd is not None:
        constraints = (
            ConstraintSpec(
                key="cost_usd",
                max=float(max_cost_usd),
                level=max_cost_level,
            ),
        )

    objectives: dict[str, ObjectiveDirection] | None = None
    if minimize_cost_usd:
        objectives = {"cost_usd": ObjectiveDirection.MINIMIZE}

    return GoalSpec(
        conditions=conditions,
        constraints=constraints,
        objectives=objectives,
    )


def flexible_job_shop_goal_fluent(
    jobs: tuple[Job, ...] = JOBS,
    *,
    max_cost_usd: float,
    max_cost_level: Literal["hard", "soft"] = "hard",
) -> GoalSpec:
    """Fluent-builder twin of :func:`flexible_job_shop_goal`.

    Exercises :class:`~langgoap.constraints.ConstraintBuilder` as the
    ``ConstraintProvider`` analogue.  The integration test pins that
    this produces a :class:`GoalSpec` structurally equivalent to the
    hand-rolled form with the same ``max_cost_usd`` and
    ``minimize_cost_usd=True``.
    """
    conditions = _goal_conditions(jobs)
    output = ConstraintBuilder.build(
        ConstraintBuilder.for_plan()
        .sum_resource("cost_usd")
        .bounded(max=float(max_cost_usd))
        .penalize(level=max_cost_level, weight=1.0)
        .as_constraint("cost_usd"),
        ConstraintBuilder.for_plan()
        .sum_resource("cost_usd")
        .minimize()
        .as_objective("cost_usd"),
    )
    return GoalSpec.from_builder(conditions=conditions, builder_output=output)


__all__ = [
    "JOBS",
    "action_name",
    "done_key",
    "flexible_job_shop_actions",
    "flexible_job_shop_goal",
    "flexible_job_shop_goal_fluent",
    "flexible_job_shop_start",
]
