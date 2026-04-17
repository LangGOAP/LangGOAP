"""OR-Tools CP-SAT constraint optimizer for GOAP plans.

Runs after A* to validate resource constraints, optimize multi-objective
trade-offs, and compute temporal schedules. When no constraints or objectives
exist, the CSP is skipped entirely (zero overhead).

Architecture:
  A* planner → CSP optimizer → Optimized Plan

Resource validation uses a pure-Python fast path (no solver invocation).
CP-SAT is used for temporal scheduling and multi-plan optimization.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from ortools.sat.python import cp_model as _cp_model

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

    totals = compute_resource_totals(plan.actions)
    usage_list, all_hard_satisfied = _compute_resource_usage(totals, goal)
    obj_values = _compute_objective_values(totals, plan, goal)

    status = CSPStatus.FEASIBLE if all_hard_satisfied else CSPStatus.INFEASIBLE

    # Schedule only when actions carry durations and no hard resource
    # constraint is already violated (scheduling a known-infeasible plan
    # wastes solver time).
    schedule_meta: CSPMetadata | None = None
    if all_hard_satisfied and any(a.duration is not None for a in plan.actions):
        schedule_meta = schedule_plan(plan, scale=scale)

    # Scheduling failure overrides a feasible resource check.
    final_status = status
    if schedule_meta is not None and schedule_meta.status == CSPStatus.INFEASIBLE:
        final_status = CSPStatus.INFEASIBLE

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


def _compute_resource_usage(
    totals: dict[str, float], goal: GoalSpec
) -> tuple[list[ResourceUsage], bool]:
    """Classify every resource against its constraint, returning usage + hard-sat flag.

    Hard violations mark the plan INFEASIBLE; soft violations are recorded
    but do not.  Resource keys that have no declared constraint are returned
    with ``level="info"`` for downstream reporting.
    """
    constraint_map: dict[str, ConstraintSpec] = {c.key: c for c in goal.constraints}
    usage_list: list[ResourceUsage] = []
    all_hard_satisfied = True
    for key, constraint in constraint_map.items():
        total = totals.get(key, 0.0)
        satisfied = not (
            (constraint.max is not None and total > constraint.max)
            or (constraint.min is not None and total < constraint.min)
        )
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
    for key, total in totals.items():
        if key not in constraint_map:
            usage_list.append(ResourceUsage(key=key, total=total, level="info"))
    return usage_list, all_hard_satisfied


def _compute_objective_values(
    totals: dict[str, float], plan: Plan, goal: GoalSpec
) -> dict[str, float]:
    """Collect objective resource totals and evaluate soft goals."""
    obj_values: dict[str, float] = {}
    if goal.objectives is not None:
        for obj_key in goal.objectives:
            obj_values[obj_key] = totals.get(obj_key, 0.0)
    if goal.soft_goals and plan.expected_states:
        final_state = plan.expected_states[-1]
        for sg in goal.soft_goals:
            achieved = final_state.satisfies(sg.conditions)
            obj_values[f"soft_goal:{sg.label}"] = sg.weight if achieved else 0.0
    return obj_values


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

    if not any(a.duration is not None for a in plan.actions):
        return CSPMetadata(
            status=CSPStatus.SKIPPED,
            solver_time_ms=(time.monotonic() - t0) * 1000,
            scale_factor=scale,
        )

    cp_model = _cp_model
    model = cp_model.CpModel()
    durations_ms = _scaled_durations_ms(plan, scale)
    starts, makespan = _build_schedule_vars(model, plan, durations_ms)

    solver = cp_model.CpSolver()
    solver_status = solver.solve(model)
    elapsed = (time.monotonic() - t0) * 1000

    if solver_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return CSPMetadata(
            status=CSPStatus.INFEASIBLE,
            solver_time_ms=elapsed,
            scale_factor=scale,
        )

    schedule_entries = _decode_schedule(plan, solver, starts, durations_ms, scale)
    makespan_val = solver.value(makespan)
    csp_status = (
        CSPStatus.OPTIMAL if solver_status == cp_model.OPTIMAL else CSPStatus.FEASIBLE
    )
    return CSPMetadata(
        status=csp_status,
        solver_time_ms=elapsed,
        schedule=tuple(schedule_entries),
        makespan=timedelta(milliseconds=makespan_val * 1000 / scale),
        scale_factor=scale,
    )


def _scaled_durations_ms(plan: Plan, scale: int) -> list[int]:
    """Return each action's scaled integer duration; 0 for durationless actions."""
    out: list[int] = []
    for a in plan.actions:
        if a.duration is not None:
            out.append(max(int(a.duration.total_seconds() * scale), 0))
        else:
            out.append(0)
    return out


def _build_schedule_vars(
    model: Any, plan: Plan, durations_ms: list[int]
) -> tuple[list[Any], Any]:
    """Create start/interval/makespan variables and attach precedence constraints."""
    n = len(plan.actions)
    horizon = sum(durations_ms) + 1
    starts: list[Any] = []
    for i in range(n):
        start = model.new_int_var(0, horizon, f"start_{i}")
        model.new_interval_var(
            start, durations_ms[i], start + durations_ms[i], f"interval_{i}"
        )
        starts.append(start)
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
    return starts, makespan


def _decode_schedule(
    plan: Plan,
    solver: Any,
    starts: list[Any],
    durations_ms: list[int],
    scale: int,
) -> list[ScheduleEntry]:
    """Materialize ``ScheduleEntry`` instances from solved CP-SAT variables."""
    entries: list[ScheduleEntry] = []
    for i in range(len(plan.actions)):
        start_val = solver.value(starts[i])
        dur_val = durations_ms[i]
        entries.append(
            ScheduleEntry(
                action_name=plan.actions[i].name,
                start=timedelta(milliseconds=start_val * 1000 / scale),
                duration=timedelta(milliseconds=dur_val * 1000 / scale),
                end=timedelta(milliseconds=(start_val + dur_val) * 1000 / scale),
            )
        )
    return entries


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
    cp_model = _cp_model
    model = cp_model.CpModel()
    n = len(plans)

    plan_resources: list[dict[str, float]] = [
        compute_resource_totals(p.actions) for p in plans
    ]
    all_keys: set[str] = set()
    for res in plan_resources:
        all_keys.update(res.keys())

    selected, resource_vars = _build_selection_vars(
        model, n, plan_resources, all_keys, scale
    )
    constraint_map: dict[str, ConstraintSpec] = {c.key: c for c in goal.constraints}
    soft_violation_terms = _wire_constraints(
        model, resource_vars, constraint_map, plan_resources, scale
    )
    _set_optimization_objective(
        model,
        selected,
        resource_vars,
        soft_violation_terms,
        plans,
        goal,
        scale,
    )

    solver = cp_model.CpSolver()
    solver_status = solver.solve(model)
    elapsed = (time.monotonic() - t0) * 1000

    if solver_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return plans[0], CSPMetadata(
            status=CSPStatus.INFEASIBLE,
            solver_time_ms=elapsed,
            plans_evaluated=n,
            scale_factor=scale,
        )

    chosen_idx = _decode_selection(solver, selected, n)
    chosen_plan = plans[chosen_idx]
    chosen_resources = plan_resources[chosen_idx]
    usage_list = _build_usage_list(all_keys, constraint_map, chosen_resources)
    obj_values: dict[str, float] = {}
    if goal.objectives:
        for obj_key in goal.objectives:
            obj_values[obj_key] = chosen_resources.get(obj_key, 0.0)

    csp_status = (
        CSPStatus.OPTIMAL if solver_status == cp_model.OPTIMAL else CSPStatus.FEASIBLE
    )
    return chosen_plan, CSPMetadata(
        status=csp_status,
        solver_time_ms=elapsed,
        resource_usage=tuple(usage_list),
        objective_values=MappingProxyType(obj_values),
        plans_evaluated=n,
        scale_factor=scale,
    )


def _build_selection_vars(
    model: Any,
    n: int,
    plan_resources: list[dict[str, float]],
    all_keys: set[str],
    scale: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Create boolean plan selectors plus one IntVar per resource key.

    The resource IntVars are constrained so each one equals the weighted
    sum of ``selected[i] * scaled_value[i]`` — effectively a dispatch
    table that yields the chosen plan's resource total.
    """
    selected = [model.new_bool_var(f"plan_{i}") for i in range(n)]
    model.add_exactly_one(selected)
    resource_vars: dict[str, Any] = {}
    for key in all_keys:
        scaled_values = [int(res.get(key, 0.0) * scale) for res in plan_resources]
        rv = model.new_int_var(0, max(scaled_values) + 1, f"resource_{key}")
        model.add(rv == sum(selected[i] * scaled_values[i] for i in range(n)))
        resource_vars[key] = rv
    return selected, resource_vars


def _compute_big_m(
    plan_resources: list[dict[str, float]],
    constraint_map: dict[str, ConstraintSpec],
    scale: int,
) -> int:
    """Return a safe Big-M bound for soft-violation slack variables."""
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
    return max_scaled * 2 + 1


def _wire_constraints(
    model: Any,
    resource_vars: dict[str, Any],
    constraint_map: dict[str, ConstraintSpec],
    plan_resources: list[dict[str, float]],
    scale: int,
) -> list[Any]:
    """Attach hard bounds directly; emit soft-violation slack terms.

    Hard constraints become solver assertions.  Soft constraints introduce
    a non-negative slack variable (Big-M bounded) whose weighted value is
    returned so the caller can fold it into the objective (R1).
    """
    soft_violation_terms: list[Any] = []
    big_m = _compute_big_m(plan_resources, constraint_map, scale)
    for key, constraint in constraint_map.items():
        if key not in resource_vars:
            # Constrained key not present in any plan — model it as a zero var.
            resource_vars[key] = model.new_int_var(0, 0, f"resource_{key}")
        rv = resource_vars[key]
        if constraint.level == "hard":
            if constraint.max is not None:
                model.add(rv <= int(constraint.max * scale))
            if constraint.min is not None:
                model.add(rv >= int(constraint.min * scale))
            continue
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
    return soft_violation_terms


def _set_optimization_objective(
    model: Any,
    selected: list[Any],
    resource_vars: dict[str, Any],
    soft_violation_terms: list[Any],
    plans: list[Plan],
    goal: GoalSpec,
    scale: int,
) -> None:
    """Combine hard objectives, soft-constraint penalties, and soft-goal rewards.

    Falls back to plain total-cost minimisation when no objectives, soft
    constraints, or soft goals are declared.
    """
    objective_terms: list[Any] = []
    if goal.objectives:
        for obj_key, direction in goal.objectives.items():
            if obj_key not in resource_vars:
                continue
            if direction == ObjectiveDirection.MINIMIZE:
                # Minimizing → positive coefficient (solver minimizes)
                objective_terms.append(resource_vars[obj_key])
            else:
                # Maximizing → negative coefficient (solver minimizes -value)
                objective_terms.append(-resource_vars[obj_key])

    # Soft goals: reward weighted achievement (binary per plan × weight).
    soft_goal_terms: list[Any] = []
    if goal.soft_goals:
        for sg in goal.soft_goals:
            weight_scaled = max(int(sg.weight * scale), 1)
            for i, p in enumerate(plans):
                achieved = bool(
                    p.expected_states and p.expected_states[-1].satisfies(sg.conditions)
                )
                if achieved:
                    # Subtract (negative in minimise) to reward achievement
                    soft_goal_terms.append(-selected[i] * weight_scaled)

    combined_terms = objective_terms + soft_violation_terms + soft_goal_terms
    if combined_terms:
        model.minimize(sum(combined_terms))
    else:
        # No objectives or soft violations — prefer lower total cost.
        cost_terms = [
            selected[i] * int(plans[i].total_cost * scale) for i in range(len(plans))
        ]
        model.minimize(sum(cost_terms))


def _decode_selection(solver: Any, selected: list[Any], n: int) -> int:
    """Return the index of the plan the solver picked (defaults to 0)."""
    for i in range(n):
        if solver.value(selected[i]):
            return i
    return 0


def _build_usage_list(
    all_keys: set[str],
    constraint_map: dict[str, ConstraintSpec],
    chosen_resources: dict[str, float],
) -> list[ResourceUsage]:
    """Materialize ResourceUsage entries from the chosen plan's totals.

    Hard constraints are satisfied by the solver; for soft constraints
    the ``satisfied`` flag reflects the actual chosen totals vs. bounds.
    Keys declared in constraints but absent from every plan are appended
    with ``total=0`` so downstream consumers see a complete picture.
    """
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
    return usage_list


# ---------------------------------------------------------------------------
# Public API — pareto_plans (Pareto-optimal plan enumeration)
# ---------------------------------------------------------------------------


def _dominates(
    a_obj: dict[str, float],
    b_obj: dict[str, float],
    objectives: "MappingProxyType[str, Any]",
) -> bool:
    """Return True if plan-A dominates plan-B on all objectives.

    A dominates B when A is at least as good on every objective AND strictly
    better on at least one.  "Better" is direction-dependent:
    MINIMIZE → lower is better, MAXIMIZE → higher is better.
    """
    from langgoap.types import ObjectiveDirection

    at_least_as_good_count = 0
    strictly_better_count = 0
    for key, direction in objectives.items():
        a_val = a_obj.get(key, 0.0)
        b_val = b_obj.get(key, 0.0)
        if direction == ObjectiveDirection.MINIMIZE:
            if a_val > b_val:
                return False
            if a_val < b_val:
                strictly_better_count += 1
        else:  # MAXIMIZE
            if a_val < b_val:
                return False
            if a_val > b_val:
                strictly_better_count += 1
        at_least_as_good_count += 1
    return at_least_as_good_count > 0 and strictly_better_count > 0


def pareto_plans(
    plans: list["Plan"],
    goal: GoalSpec,
    *,
    scale: int = 1000,
    max_frontier: int | None = None,
) -> list[tuple["Plan", CSPMetadata]]:
    """Enumerate the Pareto-optimal subset of candidate plans.

    From a finite set of candidate plans, returns the non-dominated subset
    (Pareto frontier) ordered by primary objective value.  A plan A
    *dominates* plan B when A is at least as good on **all** objectives and
    strictly better on **at least one**.

    Hard constraints are respected: plans violating any hard
    :class:`~langgoap.goals.ConstraintSpec` are excluded from the frontier
    entirely before dominance is computed.

    When ``goal.objectives`` is empty or ``None``, every feasible plan is
    non-dominated (no objective can distinguish them) and the full feasible
    set is returned (subject to *max_frontier*) sorted by plan cost.

    Args:
        plans:         Candidate plans.  Typically produced by running A*
                       with different starting states, action subsets, or by
                       :class:`~langgoap.planner.strategy.LazyDecompositionStrategy`.
        goal:          Goal spec with constraints and objectives.
        scale:         Integer scaling factor for resource computations
                       (forwarded to :func:`validate_plan`).
        max_frontier:  Cap on the number of returned frontier plans.  The
                       ``max_frontier`` plans with the best primary-objective
                       value are kept.  ``None`` returns all non-dominated
                       plans.

    Returns:
        A list of ``(Plan, CSPMetadata)`` tuples for non-dominated plans,
        ordered by the primary (first) objective value.  Empty when all
        plans are infeasible or the input list is empty.

    Raises:
        ValueError: When *plans* is empty.
    """
    if not plans:
        raise ValueError("pareto_plans: plans list must not be empty")

    t0 = time.monotonic()

    feasible = _collect_feasible_plans(plans, goal, scale=scale)
    if not feasible:
        return []

    plan_obj_values = _collect_objective_values(feasible, goal)

    if not goal.objectives:
        # No objectives — all feasible plans are non-dominated; sort by cost.
        frontier = sorted(feasible, key=lambda t: t[0].total_cost)
        if max_frontier is not None:
            frontier = frontier[:max_frontier]
        return frontier

    frontier_items = _pareto_frontier(feasible, plan_obj_values, goal.objectives)
    frontier = _sort_by_primary_objective(frontier_items, goal.objectives)
    if max_frontier is not None:
        frontier = frontier[:max_frontier]

    elapsed_ms = (time.monotonic() - t0) * 1000
    return _annotate_metadata(frontier, len(plans), elapsed_ms, scale)


def _collect_feasible_plans(
    plans: list["Plan"], goal: GoalSpec, *, scale: int
) -> list[tuple["Plan", CSPMetadata]]:
    """Validate each plan and keep only the ones whose hard constraints hold."""
    feasible: list[tuple["Plan", CSPMetadata]] = []
    for p in plans:
        meta = validate_plan(p, goal, scale=scale)
        if meta.status != CSPStatus.INFEASIBLE:
            feasible.append((p, meta))
    return feasible


def _collect_objective_values(
    feasible: list[tuple["Plan", CSPMetadata]], goal: GoalSpec
) -> list[dict[str, float]]:
    """Return per-plan objective dicts, filling in missing keys from resource totals."""
    plan_obj_values: list[dict[str, float]] = []
    for p, meta in feasible:
        obj: dict[str, float] = dict(meta.objective_values)
        # Also include resource totals that match objective keys but weren't
        # in meta.objective_values (e.g. from the pure-Python fast path).
        if goal.objectives:
            totals = compute_resource_totals(p.actions)
            for key in goal.objectives:
                if key not in obj:
                    obj[key] = totals.get(key, 0.0)
        plan_obj_values.append(obj)
    return plan_obj_values


def _pareto_frontier(
    feasible: list[tuple["Plan", CSPMetadata]],
    plan_obj_values: list[dict[str, float]],
    objectives: "MappingProxyType[str, Any]",
) -> list[tuple[tuple["Plan", CSPMetadata], dict[str, float]]]:
    """Return the non-dominated ``(item, objective_values)`` pairs."""
    n = len(feasible)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if _dominates(plan_obj_values[j], plan_obj_values[i], objectives):
                dominated[i] = True
                break
    return [(feasible[i], plan_obj_values[i]) for i in range(n) if not dominated[i]]


def _sort_by_primary_objective(
    frontier_items: list[tuple[tuple["Plan", CSPMetadata], dict[str, float]]],
    objectives: "MappingProxyType[str, Any]",
) -> list[tuple["Plan", CSPMetadata]]:
    """Sort frontier items by the first declared objective, honoring its direction."""
    from langgoap.types import ObjectiveDirection

    primary_key, primary_dir = next(iter(objectives.items()))
    reverse_sort = primary_dir == ObjectiveDirection.MAXIMIZE
    frontier_items.sort(key=lambda t: t[1].get(primary_key, 0.0), reverse=reverse_sort)
    return [item for item, _ in frontier_items]


def _annotate_metadata(
    frontier: list[tuple["Plan", CSPMetadata]],
    plans_evaluated: int,
    elapsed_ms: float,
    scale: int,
) -> list[tuple["Plan", CSPMetadata]]:
    """Stamp each metadata with aggregate solver time and plan count."""
    result: list[tuple["Plan", CSPMetadata]] = []
    for p, meta in frontier:
        annotated = CSPMetadata(
            status=meta.status,
            solver_time_ms=elapsed_ms,
            resource_usage=meta.resource_usage,
            objective_values=meta.objective_values,
            schedule=meta.schedule,
            makespan=meta.makespan,
            plans_evaluated=plans_evaluated,
            scale_factor=scale,
            explanation=meta.explanation,
        )
        result.append((p, annotated))
    return result
