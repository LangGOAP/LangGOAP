r"""Task Assigning — OptaPlanner's ticket routing in LangGoap.

Translates a compact subset of OptaPlanner's
[TaskAssigning](https://github.com/apache/incubator-kie-optaplanner/tree/main/optaplanner-examples/src/main/java/org/optaplanner/examples/taskassigning)
example into GOAP.  Each task must be assigned to exactly one
employee; skill matching is a hard constraint (filtered at
action-build time) and the soft objective is a flattened weighted
delay that folds OptaPlanner's four bendable-score soft levels into
a single scalar the CSP phase can minimize.

GOAP modelling
--------------

**World state:**

- ``task_<name>_done`` — flipped to ``True`` when an employee is
  assigned to that task.

**Actions:** one ``assign_<employee>_to_<task>`` per qualified pair.
Skill filtering happens at action-build time so unqualified
employees simply have no action to take the task.

- **preconditions**: ``task_<t>_done=False``
- **effects**: ``task_<t>_done=True``
- **cost**:
  ``priority_weight * base_duration * affinity_multiplier`` — A\*
  naturally front-loads critical work and prefers high-affinity
  employees.
- **resources**:
    - ``weighted_delay`` (plan-wide) — same number as cost, used by
      CSP to attach a ``HardSoftScore``.
    - ``workload_<employee>`` — sum of *effective* (not weighted)
      hours the employee is carrying.  A per-employee soft cap
      turns this into a load-balance signal.

**Goal:** every task assigned; optional per-employee workload cap
plus a ``weighted_delay → MINIMIZE`` objective.

ConstraintBuilder integration
-----------------------------

:func:`task_assigning_goal_fluent` demonstrates the fluent
:class:`~langgoap.constraints.ConstraintBuilder` — the OptaPlanner
``ConstraintProvider`` analogue — for constructing the same goal in
a readable, chained style.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, ConstraintSpec, GoalSpec
from langgoap.constraints import ConstraintBuilder
from langgoap.goals import ObjectiveDirection

from .data.task_assigning_instance import (
    AFFINITY,
    EMPLOYEES,
    PRIORITY_WEIGHT,
    TASK_TYPE_BY_NAME,
    TASKS,
    Employee,
    Task,
)


def _done_key(task_name: str) -> str:
    return f"task_{task_name}_done"


def _workload_key(employee_name: str) -> str:
    return f"workload_{employee_name}"


def _make_execute(effects: dict[str, Any]) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        return dict(effects)

    return execute


def _effective_duration(employee: Employee, task: Task) -> int:
    """Base duration scaled by the employee/task-type affinity multiplier."""
    tt = TASK_TYPE_BY_NAME[task.task_type]
    multiplier = AFFINITY[(employee.name, task.task_type)]
    return tt.base_duration_hours * multiplier


def _weighted_delay(employee: Employee, task: Task) -> int:
    """Cost = priority_weight × base_duration × affinity_multiplier."""
    return PRIORITY_WEIGHT[task.priority] * _effective_duration(employee, task)


def _assign_action(employee: Employee, task: Task) -> ActionSpec:
    effective_duration = _effective_duration(employee, task)
    weighted_delay = _weighted_delay(employee, task)
    effects = {_done_key(task.name): True}
    return ActionSpec(
        name=f"assign_{employee.name}_to_{task.name}",
        preconditions={_done_key(task.name): False},
        effects=effects,
        cost=float(weighted_delay),
        resources={
            "weighted_delay": float(weighted_delay),
            _workload_key(employee.name): float(effective_duration),
        },
        execute=_make_execute(effects),
    )


def task_assigning_actions(
    employees: tuple[Employee, ...] = EMPLOYEES,
    tasks: tuple[Task, ...] = TASKS,
) -> list[ActionSpec]:
    """Return one assignment action per qualified (employee, task) pair.

    Skill matching is applied at build time: an employee missing the
    task type's required skill has no action to take the task.
    """
    actions: list[ActionSpec] = []
    for employee in employees:
        for task in tasks:
            tt = TASK_TYPE_BY_NAME[task.task_type]
            if tt.required_skill not in employee.skills:
                continue
            if (employee.name, task.task_type) not in AFFINITY:
                # No affinity configured → treat as unreachable for this
                # tutorial (matches OptaPlanner's Affinity.NONE semantics,
                # which simply makes the multiplier huge).
                continue
            actions.append(_assign_action(employee, task))
    return actions


def task_assigning_start(
    tasks: tuple[Task, ...] = TASKS,
) -> dict[str, Any]:
    """All tasks pending."""
    return {_done_key(t.name): False for t in tasks}


def task_assigning_goal(
    tasks: tuple[Task, ...] = TASKS,
    *,
    minimize_weighted_delay: bool = True,
    max_workload_per_employee: float | None = None,
    max_workload_level: str = "soft",
    employees: tuple[Employee, ...] = EMPLOYEES,
) -> GoalSpec:
    """Build the default goal by hand.

    Args:
        tasks: Tasks that must be assigned.
        minimize_weighted_delay: When True (default), attach a
            ``weighted_delay → MINIMIZE`` objective so the pipeline
            routes through CSP and the plan receives a
            :class:`~langgoap.score.HardSoftScore`.
        max_workload_per_employee: Optional per-employee effective-hour
            cap.  One :class:`~langgoap.goals.ConstraintSpec` is emitted
            per employee so CSP can enforce or penalize them
            independently.  Omit for no cap.
        max_workload_level: ``"soft"`` (default) keeps the plan
            feasible and penalizes overflow; ``"hard"`` marks the plan
            ``INFEASIBLE`` when any employee is overloaded.
        employees: Employees that receive workload caps when
            ``max_workload_per_employee`` is set.
    """
    conditions: dict[str, Any] = {_done_key(t.name): True for t in tasks}

    constraints: tuple[ConstraintSpec, ...] = ()
    if max_workload_per_employee is not None:
        constraints = tuple(
            ConstraintSpec(
                key=_workload_key(e.name),
                max=float(max_workload_per_employee),
                level=max_workload_level,  # type: ignore[arg-type]
            )
            for e in employees
        )

    objectives: dict[str, ObjectiveDirection] | None = None
    if minimize_weighted_delay:
        objectives = {"weighted_delay": ObjectiveDirection.MINIMIZE}

    return GoalSpec(
        conditions=conditions,
        constraints=constraints,
        objectives=objectives,
    )


def task_assigning_goal_fluent(
    tasks: tuple[Task, ...] = TASKS,
    *,
    max_workload_per_employee: float | None = None,
    max_workload_level: str = "soft",
    employees: tuple[Employee, ...] = EMPLOYEES,
) -> GoalSpec:
    """Build the same goal via :class:`ConstraintBuilder`.

    Demonstrates the fluent alternative to the hand-rolled
    :func:`task_assigning_goal` constructor.  Both return functionally
    identical :class:`~langgoap.goals.GoalSpec` instances.
    """
    chains: list[Any] = [
        ConstraintBuilder.for_plan()
        .sum_resource("weighted_delay")
        .minimize()
        .as_objective("weighted_delay"),
    ]
    if max_workload_per_employee is not None:
        for e in employees:
            chains.append(
                ConstraintBuilder.for_plan()
                .sum_resource(_workload_key(e.name))
                .bounded(max=float(max_workload_per_employee))
                .penalize(level=max_workload_level, weight=1.0)  # type: ignore[arg-type]
                .as_constraint(_workload_key(e.name))
            )

    output = ConstraintBuilder.build(*chains)
    return GoalSpec.from_builder(
        conditions={_done_key(t.name): True for t in tasks},
        builder_output=output,
    )
