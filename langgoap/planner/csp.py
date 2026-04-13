"""OR-Tools CP-SAT constraint optimizer for GOAP plans.

Runs after A* to validate resource constraints, optimize multi-objective
trade-offs, and compute temporal schedules. When no constraints or objectives
exist, the CSP is skipped entirely (zero overhead).

Architecture:
  A* planner → CSP optimizer → Optimized Plan

The pure-Python fast path handles simple resource validation without importing
ortools. CP-SAT is only loaded for temporal scheduling and multi-plan
optimization.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langgoap.planner.types import Plan

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec, SoftGoal
from langgoap.types import ObjectiveDirection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class CSPStatus(str, enum.Enum):
    """Status of a CSP solve attempt."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Per-key resource breakdown for a plan.

    Attributes:
        key: Resource identifier (e.g. ``"tokens"``, ``"cost_usd"``).
        total: Total consumption across all actions.
        constraint_min: Lower bound from the constraint, or ``None``.
        constraint_max: Upper bound from the constraint, or ``None``.
        satisfied: Whether the total falls within the constraint bounds.
        level: Constraint level.

            * ``"hard"`` — the usage corresponds to a
              :class:`~langgoap.goals.ConstraintSpec` with
              ``level="hard"``.  Violation marks the plan
              :attr:`CSPStatus.INFEASIBLE`.
            * ``"soft"`` — the usage corresponds to a soft constraint.
              Violation contributes to the plan's soft score.
            * ``"info"`` (default) — informational only; the resource
              key has no matching ``ConstraintSpec``.  Violation is
              impossible because there is no bound to violate.
    """

    key: str
    total: float
    constraint_min: float | None = None
    constraint_max: float | None = None
    satisfied: bool = True
    level: Literal["hard", "soft", "info"] = "info"


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    """Per-action temporal slot in a schedule.

    Attributes:
        action_name: Name of the scheduled action.
        start: Start time offset from plan begin.
        duration: Duration of the action.
        end: End time (start + duration).
    """

    action_name: str
    start: timedelta
    duration: timedelta
    end: timedelta


@dataclass(frozen=True, slots=True)
class CSPMetadata:
    """Results from the CSP optimizer.

    Attributes:
        status: Solve outcome.
        solver_time_ms: Wall-clock time spent in the solver.
        resource_usage: Per-key resource breakdown.
        objective_values: Objective function values for the selected plan.
        schedule: Per-action temporal schedule.
        makespan: Total schedule duration (max end time), or ``None``
            if no scheduling was performed.
        plans_evaluated: Number of candidate plans evaluated.
        scale_factor: Integer scaling factor used for CP-SAT (float → int).
        explanation: Explanation of why the plan is infeasible, or ``None``
            when the plan is feasible or when no explanation has been
            computed.
    """

    status: CSPStatus
    solver_time_ms: float = 0.0
    resource_usage: tuple[ResourceUsage, ...] = ()
    objective_values: MappingProxyType[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schedule: tuple[ScheduleEntry, ...] = ()
    makespan: timedelta | None = None
    plans_evaluated: int = 0
    scale_factor: int = 1000
    explanation: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.objective_values, MappingProxyType):
            object.__setattr__(
                self,
                "objective_values",
                MappingProxyType(dict(self.objective_values)),
            )

    def __repr__(self) -> str:
        parts = [f"status={self.status.value!r}"]
        if self.solver_time_ms > 0:
            parts.append(f"solver_time_ms={self.solver_time_ms:.2f}")
        if self.resource_usage:
            parts.append(f"resources={len(self.resource_usage)}")
        if self.objective_values:
            parts.append(f"objectives={dict(self.objective_values)!r}")
        if self.schedule:
            parts.append(f"schedule_entries={len(self.schedule)}")
        if self.makespan is not None:
            parts.append(f"makespan={self.makespan}")
        if self.plans_evaluated > 0:
            parts.append(f"plans_evaluated={self.plans_evaluated}")
        return f"CSPMetadata({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_ortools() -> Any:
    """Lazy-import ortools CP-SAT solver.

    Raises ``ImportError`` with a clear message pointing to the
    ``langgoap[optimization]`` extra.
    """
    try:
        from ortools.sat.python import cp_model

        return cp_model
    except ImportError:
        raise ImportError(
            "OR-Tools is required for CSP optimization. "
            "Install it with: pip install langgoap[optimization]"
        ) from None


def compute_resource_totals(
    actions: tuple[ActionSpec, ...],
) -> dict[str, float]:
    """Sum ``action.resources`` across all actions in a plan.

    Actions without resources are silently skipped.
    """
    totals: dict[str, float] = {}
    for action in actions:
        if action.resources is not None:
            for key, value in action.resources.items():
                totals[key] = totals.get(key, 0.0) + value
    return totals


def build_dependency_graph(
    actions: tuple[ActionSpec, ...],
) -> dict[int, list[int]]:
    """Build a precedence DAG from precondition/effect chains.

    Action j depends on action i (i < j) if j's preconditions include a
    condition that is in i's effects and no closer producer exists between
    them.

    Returns:
        Mapping from action index to list of predecessor indices.
    """
    deps: dict[int, list[int]] = {i: [] for i in range(len(actions))}

    for j in range(len(actions)):
        if not actions[j].preconditions:
            continue
        for key, value in actions[j].preconditions.items():
            # Find the closest producer before j
            for i in range(j - 1, -1, -1):
                if actions[i].effects.get(key) == value:
                    if i not in deps[j]:
                        deps[j].append(i)
                    break  # closest producer found
    return deps


# Back-compat aliases; prefer the public names in new code.
_compute_resource_totals = compute_resource_totals
_build_dependency_graph = build_dependency_graph


# ---------------------------------------------------------------------------
# Public API — validate_plan (pure-Python fast path)
# ---------------------------------------------------------------------------


def validate_plan(
    plan: Plan,
    goal: GoalSpec,
    *,
    scale: int = 1000,
) -> CSPMetadata:
    """Validate a plan against goal constraints and compute a temporal schedule.

    Pure-Python path for resource-only constraints (no ortools needed).
    Invokes CP-SAT only when temporal scheduling is required (actions have
    durations).

    Args:
        plan: The plan to validate.
        goal: Goal specification with constraints and objectives.
        scale: Integer scaling factor for CP-SAT (float → int conversion).

    Returns:
        CSPMetadata with validation results.
    """
    t0 = time.monotonic()

    # No constraints → skip entirely
    if not goal.constraints and goal.objectives is None:
        return CSPMetadata(
            status=CSPStatus.SKIPPED,
            solver_time_ms=(time.monotonic() - t0) * 1000,
            scale_factor=scale,
        )

    # Compute resource totals
    totals = compute_resource_totals(plan.actions)

    # Build resource usage and check constraints.  Hard violations mark
    # the plan INFEASIBLE; soft violations are recorded but do not.
    constraint_map: dict[str, ConstraintSpec] = {c.key: c for c in goal.constraints}
    usage_list: list[ResourceUsage] = []
    all_hard_satisfied = True

    # Process all constrained keys
    for key, constraint in constraint_map.items():
        total = totals.get(key, 0.0)
        satisfied = True
        if constraint.max is not None and total > constraint.max:
            satisfied = False
        if constraint.min is not None and total < constraint.min:
            satisfied = False
        if not satisfied and constraint.level == "hard":
            all_hard_satisfied = False
        usage_list.append(
            ResourceUsage(
                key=key,
                total=total,
                constraint_min=constraint.min,
                constraint_max=constraint.max,
                satisfied=satisfied,
                level=constraint.level,
            )
        )

    # Also include resource keys that have no constraints (informational)
    for key, total in totals.items():
        if key not in constraint_map:
            usage_list.append(ResourceUsage(key=key, total=total, level="info"))

    # Compute objective values
    obj_values: dict[str, float] = {}
    if goal.objectives is not None:
        for obj_key in goal.objectives:
            obj_values[obj_key] = totals.get(obj_key, 0.0)

    # Evaluate soft goals against the plan's final expected state
    if goal.soft_goals and plan.expected_states:
        final_state = plan.expected_states[-1]
        for sg in goal.soft_goals:
            achieved = final_state.satisfies(sg.conditions)
            obj_values[f"soft_goal:{sg.label}"] = sg.weight if achieved else 0.0

    status = CSPStatus.FEASIBLE if all_hard_satisfied else CSPStatus.INFEASIBLE

    # Check if any actions have durations → schedule
    schedule_meta: CSPMetadata | None = None
    has_durations = any(a.duration is not None for a in plan.actions)
    if has_durations and all_hard_satisfied:
        schedule_meta = schedule_plan(plan, scale=scale)

    # Merge resource and scheduling statuses: scheduling failure overrides feasible.
    final_status = status
    if schedule_meta is not None and schedule_meta.status == CSPStatus.INFEASIBLE:
        final_status = CSPStatus.INFEASIBLE

    # Compute infeasibility explanation when INFEASIBLE
    explanation = None
    if final_status == CSPStatus.INFEASIBLE:
        from langgoap.planner.explain import explain_infeasibility

        meta_for_explain = CSPMetadata(
            status=final_status,
            resource_usage=tuple(usage_list),
        )
        explanation = explain_infeasibility(plan, goal, meta_for_explain)

    elapsed = (time.monotonic() - t0) * 1000
    return CSPMetadata(
        status=final_status,
        solver_time_ms=elapsed,
        resource_usage=tuple(usage_list),
        objective_values=MappingProxyType(obj_values),
        schedule=schedule_meta.schedule if schedule_meta else (),
        makespan=schedule_meta.makespan if schedule_meta else None,
        plans_evaluated=1,
        scale_factor=scale,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Public API — schedule_plan (CP-SAT temporal scheduling)
# ---------------------------------------------------------------------------


def schedule_plan(
    plan: Plan,
    *,
    scale: int = 1000,
) -> CSPMetadata:
    """Compute a temporal schedule for a plan using CP-SAT.

    Creates an ``IntervalVar`` per action with precedence constraints from
    the dependency graph and minimizes makespan.

    Args:
        plan: The plan to schedule.
        scale: Temporal resolution in ticks per second.  With the default
            value of 1000 each tick represents 1 millisecond.  Lower values
            give coarser resolution; higher values give finer resolution.

    Returns:
        CSPMetadata with schedule entries and makespan.
        Returns ``SKIPPED`` if no actions have durations.
    """
    t0 = time.monotonic()

    # No durations → skip
    if not any(a.duration is not None for a in plan.actions):
        return CSPMetadata(
            status=CSPStatus.SKIPPED,
            solver_time_ms=(time.monotonic() - t0) * 1000,
            scale_factor=scale,
        )

    cp_model = _require_ortools()

    model = cp_model.CpModel()
    n = len(plan.actions)

    # Scale durations to integer milliseconds
    durations_ms: list[int] = []
    for a in plan.actions:
        if a.duration is not None:
            ms = int(a.duration.total_seconds() * scale)
            durations_ms.append(max(ms, 0))
        else:
            durations_ms.append(0)  # instantaneous

    horizon = sum(durations_ms) + 1

    # Decision variables
    starts: list[Any] = []
    intervals: list[Any] = []
    for i in range(n):
        start = model.new_int_var(0, horizon, f"start_{i}")
        interval = model.new_interval_var(
            start, durations_ms[i], start + durations_ms[i], f"interval_{i}"
        )
        starts.append(start)
        intervals.append(interval)

    # Precedence constraints from dependency graph
    deps = build_dependency_graph(plan.actions)
    for j, predecessors in deps.items():
        for i in predecessors:
            model.add(starts[j] >= starts[i] + durations_ms[i])

    # Objective: minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    for i in range(n):
        model.add(makespan >= starts[i] + durations_ms[i])
    model.minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    solver_status = solver.solve(model)

    elapsed = (time.monotonic() - t0) * 1000

    if solver_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule_entries: list[ScheduleEntry] = []
        for i in range(n):
            start_val = solver.value(starts[i])
            dur_val = durations_ms[i]
            schedule_entries.append(
                ScheduleEntry(
                    action_name=plan.actions[i].name,
                    start=timedelta(milliseconds=start_val * 1000 / scale),
                    duration=timedelta(milliseconds=dur_val * 1000 / scale),
                    end=timedelta(milliseconds=(start_val + dur_val) * 1000 / scale),
                )
            )
        makespan_val = solver.value(makespan)
        csp_status = (
            CSPStatus.OPTIMAL
            if solver_status == cp_model.OPTIMAL
            else CSPStatus.FEASIBLE
        )
        return CSPMetadata(
            status=csp_status,
            solver_time_ms=elapsed,
            schedule=tuple(schedule_entries),
            makespan=timedelta(milliseconds=makespan_val * 1000 / scale),
            scale_factor=scale,
        )

    return CSPMetadata(
        status=CSPStatus.INFEASIBLE,
        solver_time_ms=elapsed,
        scale_factor=scale,
    )


# ---------------------------------------------------------------------------
# Public API — optimize_plans (CP-SAT multi-plan selection)
# ---------------------------------------------------------------------------


def optimize_plans(
    plans: list[Plan],
    goal: GoalSpec,
    *,
    scale: int = 1000,
) -> tuple[Plan, CSPMetadata]:
    """Select the best plan from candidates using CP-SAT.

    Creates a boolean selection variable per plan, applies resource constraints
    on the selected plan, and optimizes a weighted objective combining
    ``GoalSpec.objectives``.

    Args:
        plans: Candidate plans to evaluate.
        goal: Goal with constraints and objectives.
        scale: Scaling factor for float → int conversion.

    Returns:
        Tuple of (selected plan, CSPMetadata).

    Raises:
        ValueError: If plans list is empty.
    """
    if not plans:
        raise ValueError("No plans to optimize")

    t0 = time.monotonic()
    cp_model = _require_ortools()

    model = cp_model.CpModel()
    n = len(plans)

    # Boolean selection: exactly one plan is chosen
    selected = [model.new_bool_var(f"plan_{i}") for i in range(n)]
    model.add_exactly_one(selected)

    # Precompute resource totals for each plan
    plan_resources: list[dict[str, float]] = [
        compute_resource_totals(p.actions) for p in plans
    ]

    # Collect all resource keys
    all_keys: set[str] = set()
    for res in plan_resources:
        all_keys.update(res.keys())

    # Resource variables: IntVar per resource key = weighted sum over plans
    resource_vars: dict[str, Any] = {}
    for key in all_keys:
        scaled_values = [int(res.get(key, 0.0) * scale) for res in plan_resources]
        rv = model.new_int_var(0, max(scaled_values) + 1, f"resource_{key}")
        model.add(rv == sum(selected[i] * scaled_values[i] for i in range(n)))
        resource_vars[key] = rv

    # Constraints from ConstraintSpec.  Hard constraints are enforced
    # directly on the solver; soft constraints introduce a non-negative
    # violation variable that contributes ``viol * weight`` to the
    # objective (R1).
    constraint_map: dict[str, ConstraintSpec] = {c.key: c for c in goal.constraints}
    soft_violation_terms: list[Any] = []
    # Big-M for soft-violation variables. The maximum possible violation
    # is bounded by the largest scaled resource total across all plans.
    max_scaled = 1
    for res in plan_resources:
        for v in res.values():
            scaled = int(abs(v) * scale) + 1
            if scaled > max_scaled:
                max_scaled = scaled
    # Also account for explicit bounds when larger than observed totals.
    for constraint in constraint_map.values():
        if constraint.max is not None:
            max_scaled = max(max_scaled, int(abs(constraint.max) * scale) + 1)
        if constraint.min is not None:
            max_scaled = max(max_scaled, int(abs(constraint.min) * scale) + 1)
    big_m = max_scaled * 2 + 1

    for key, constraint in constraint_map.items():
        if key not in resource_vars:
            # Create a zero variable for constrained keys with no resources
            rv = model.new_int_var(0, 0, f"resource_{key}")
            resource_vars[key] = rv
        rv = resource_vars[key]
        if constraint.level == "hard":
            if constraint.max is not None:
                model.add(rv <= int(constraint.max * scale))
            if constraint.min is not None:
                model.add(rv >= int(constraint.min * scale))
        else:
            # Soft bound: violations allowed but penalized.
            weight_scaled = max(int(constraint.weight * scale), 1)
            if constraint.max is not None:
                viol_max = model.new_int_var(0, big_m, f"soft_viol_max_{key}")
                model.add(viol_max >= rv - int(constraint.max * scale))
                soft_violation_terms.append(viol_max * weight_scaled)
            if constraint.min is not None:
                viol_min = model.new_int_var(0, big_m, f"soft_viol_min_{key}")
                model.add(viol_min >= int(constraint.min * scale) - rv)
                soft_violation_terms.append(viol_min * weight_scaled)

    # Objective: weighted sum aligned with ObjectiveDirection,
    # plus soft-constraint violation penalties.
    objective_terms: list[Any] = []
    if goal.objectives:
        for obj_key, direction in goal.objectives.items():
            if obj_key in resource_vars:
                if direction == ObjectiveDirection.MINIMIZE:
                    # Minimizing → positive coefficient (solver minimizes)
                    objective_terms.append(resource_vars[obj_key])
                else:
                    # Maximizing → negative coefficient (solver minimizes -value)
                    objective_terms.append(-resource_vars[obj_key])

    # Soft goals: maximise weighted achievement (binary per plan × weight).
    # A soft goal is satisfied by plan i if its conditions are all true in the
    # plan's final expected state.
    soft_goal_terms: list[Any] = []
    if goal.soft_goals:
        for sg in goal.soft_goals:
            weight_scaled = max(int(sg.weight * scale), 1)
            for i, p in enumerate(plans):
                achieved = bool(
                    p.expected_states and p.expected_states[-1].satisfies(sg.conditions)
                )
                if achieved:
                    # Subtract (negative in minimise) to reward achieving this goal
                    soft_goal_terms.append(-selected[i] * weight_scaled)

    combined_terms: list[Any] = (
        list(objective_terms) + list(soft_violation_terms) + list(soft_goal_terms)
    )
    if combined_terms:
        model.minimize(sum(combined_terms))
    else:
        # No objectives or soft violations — prefer lower total cost
        cost_terms = [selected[i] * int(plans[i].total_cost * scale) for i in range(n)]
        model.minimize(sum(cost_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver_status = solver.solve(model)

    elapsed = (time.monotonic() - t0) * 1000

    if solver_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Find which plan was selected
        chosen_idx = 0
        for i in range(n):
            if solver.value(selected[i]):
                chosen_idx = i
                break

        chosen_plan = plans[chosen_idx]
        chosen_resources = plan_resources[chosen_idx]

        # Build resource usage.  satisfied=True for hard constraints is
        # guaranteed by the solver; for soft constraints it depends on
        # the actual chosen totals vs. bounds.
        usage_list: list[ResourceUsage] = []
        for key in sorted(all_keys):
            cspec = constraint_map.get(key)
            total = chosen_resources.get(key, 0.0)
            satisfied = True
            if cspec is not None:
                if cspec.max is not None and total > cspec.max:
                    satisfied = False
                if cspec.min is not None and total < cspec.min:
                    satisfied = False
            usage_list.append(
                ResourceUsage(
                    key=key,
                    total=total,
                    constraint_min=cspec.min if cspec else None,
                    constraint_max=cspec.max if cspec else None,
                    satisfied=satisfied,
                    level=cspec.level if cspec else "info",
                )
            )
        # Include constrained keys that don't appear in any plan's resources.
        for key, cspec in constraint_map.items():
            if key not in all_keys:
                usage_list.append(
                    ResourceUsage(
                        key=key,
                        total=0.0,
                        constraint_min=cspec.min,
                        constraint_max=cspec.max,
                        satisfied=(cspec.min is None or cspec.min <= 0),
                        level=cspec.level,
                    )
                )

        # Compute objective values
        obj_values: dict[str, float] = {}
        if goal.objectives:
            for obj_key in goal.objectives:
                obj_values[obj_key] = chosen_resources.get(obj_key, 0.0)

        csp_status = (
            CSPStatus.OPTIMAL
            if solver_status == cp_model.OPTIMAL
            else CSPStatus.FEASIBLE
        )
        meta = CSPMetadata(
            status=csp_status,
            solver_time_ms=elapsed,
            resource_usage=tuple(usage_list),
            objective_values=MappingProxyType(obj_values),
            plans_evaluated=n,
            scale_factor=scale,
        )
        return chosen_plan, meta

    # All infeasible
    meta = CSPMetadata(
        status=CSPStatus.INFEASIBLE,
        solver_time_ms=elapsed,
        plans_evaluated=n,
        scale_factor=scale,
    )
    # Return the first plan with infeasible metadata
    return plans[0], meta
