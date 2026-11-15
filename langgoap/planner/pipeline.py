"""A* → CSP two-phase planning pipeline.

Orchestrates the A* planner and CSP optimizer:
1. If no constraints/objectives → delegate directly to A* (zero overhead).
2. Run A* for a primary plan.
3. Validate with CSP → if feasible, return with CSP metadata attached.
4. If rejected, generate alternatives and select the best via CP-SAT.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.csp import (
    CSPMetadata,
    CSPStatus,
    optimize_plans,
    validate_plan,
)
from langgoap.planner.types import Plan
from langgoap.score import HardSoftScore
from langgoap.state import PlanningState
from langgoap.types import ObjectiveDirection

logger = logging.getLogger(__name__)


def _score_from_csp(plan: Plan, goal: GoalSpec, meta: CSPMetadata) -> HardSoftScore:
    """Build a :class:`HardSoftScore` from CSP metadata.

    The sign convention follows a penalize/reward pattern:

    * ``hard`` starts at ``0.0`` and each hard-constraint violation
      subtracts ``violation_amount * weight``.  A feasible plan has
      ``hard == 0.0``.
    * ``soft`` starts at ``0.0``.  Soft-constraint violations subtract
      the weighted amount.  Objective values contribute with a sign
      aligned to their direction — ``MINIMIZE`` subtracts, ``MAXIMIZE``
      adds.
    """
    constraint_by_key = {c.key: c for c in goal.constraints}
    hard = 0.0
    soft = 0.0

    for usage in meta.resource_usage:
        c = constraint_by_key.get(usage.key)
        if c is None:
            continue
        penalty = _constraint_penalty(c, usage.total)
        if penalty == 0.0:
            continue
        if c.level == "hard":
            hard -= penalty
        else:
            soft -= penalty

    # Objective contributions go to soft, aligned with direction.
    if goal.objectives:
        for key, direction in goal.objectives.items():
            v = meta.objective_values.get(key, 0.0)
            soft += -v if direction == ObjectiveDirection.MINIMIZE else v

    # Achieved soft goals are rewards: ``soft += weight`` per satisfied
    # soft goal, evaluated on the plan's expected end state.
    if goal.soft_goals and plan.expected_states:
        final_state = plan.expected_states[-1]
        for sg in goal.soft_goals:
            if final_state.satisfies(sg.conditions):
                soft += sg.weight

    # Plan-quality metrics contribute to soft as weighted minimisation
    # penalties: ``soft -= weight * metric.evaluate(plan)``.  Metrics
    # are declarative minimise-semantics; clients wanting a maximise
    # metric set ``weight=-1.0`` or invert the expression.
    for metric in goal.metrics:
        soft -= metric.weight * metric.evaluate(plan)

    return HardSoftScore(hard=hard, soft=soft)


def _constraint_penalty(c: ConstraintSpec, total: float) -> float:
    """Return ``violation_amount * weight`` for a constraint, or ``0`` if satisfied."""
    if c.max is not None and total > c.max:
        return (total - c.max) * c.weight
    if c.min is not None and total < c.min:
        return (c.min - total) * c.weight
    return 0.0


def needs_csp(goal: GoalSpec) -> bool:
    """Return True if the goal has constraints, objectives, soft goals, or metrics."""
    return (
        bool(goal.constraints)
        or goal.objectives is not None
        or bool(goal.metrics)
        or bool(goal.soft_goals)
    )


def needs_selection(goal: GoalSpec) -> bool:
    """Return True when the goal states *preferences* rather than only limits.

    Objectives, soft constraints, and soft goals can only be honoured by
    comparing candidate plans against each other; a goal that carries
    only hard constraints is satisfied by any plan that respects them, so
    the first feasible A* plan is final.

    Plan-quality ``metrics`` deliberately do *not* trigger selection: they
    are arbitrary Python callables that CP-SAT cannot express, so
    enumerating alternatives for them would compare plans on cost alone
    while paying for an unbounded name-blacklist A* search (which hung the
    trajectory-cost benchmark on long-horizon problems).  Metrics are
    folded into the chosen plan's soft score by :func:`_score_from_csp`.
    """
    return (
        goal.objectives is not None
        or bool(goal.soft_goals)
        or any(c.level == "soft" for c in goal.constraints)
    )


# Back-compat alias; prefer needs_csp in new code.
_needs_csp = needs_csp


def _augment_plan(plan: Plan, goal: GoalSpec, csp_meta: CSPMetadata) -> Plan:
    """Create a new Plan with CSP metadata and a HardSoftScore attached.

    After CSP evaluation the plan's pure-A* :class:`~langgoap.score.SimpleScore`
    is replaced with a :class:`~langgoap.score.HardSoftScore` computed from
    ``csp_meta`` via :func:`_score_from_csp`.  The original plan is left
    unchanged (frozen dataclass → ``dataclasses.replace``).
    """
    new_metadata = replace(plan.metadata, csp=csp_meta)
    new_score = _score_from_csp(plan, goal, csp_meta)
    return replace(plan, metadata=new_metadata, score=new_score)


def enumerate_alternatives(
    start: PlanningState | dict[str, Any],
    goal: GoalSpec,
    actions: list[ActionSpec],
    blacklisted: list[str] | None,
    exclude_plan: Plan,
    max_count: int,
) -> list[Plan]:
    """Generate up to ``max_count`` alternative plans, nearest first.

    Each found plan is re-planned with one more of its actions excluded,
    breadth-first over the exclusion sets: the first ``len(exclude_plan)``
    alternatives differ from the rejected plan by one action, the next ring
    by two, and so on.  Excluding one action at a time only ever reached
    ``len(exclude_plan)`` neighbours, so with uniform action costs the
    objective-driven selection (issue #12) could not see a cheaper plan two
    swaps away whatever ``max_alternatives`` said; the cloud-balancing
    tutorial settled on a $1506 placement while a $1092 one existed.
    Plans are deduplicated by action-name sequence.
    """
    from collections import deque

    seen: set[tuple[str, ...]] = {tuple(exclude_plan.action_names)}
    alternatives: list[Plan] = []
    base_blacklist = list(blacklisted) if blacklisted else []
    frontier: deque[tuple[Plan, tuple[str, ...]]] = deque([(exclude_plan, ())])

    while frontier and len(alternatives) < max_count:
        plan_obj, excluded = frontier.popleft()
        for action in plan_obj.actions:
            if len(alternatives) >= max_count:
                break
            if action.name in excluded:
                continue
            trial_excluded = (*excluded, action.name)
            # Never fall back here: the whole point is to plan *without* the
            # excluded actions, so re-admitting them would just reproduce a
            # plan already seen.
            result = astar_plan(
                start,
                goal,
                actions,
                blacklisted_actions=base_blacklist + list(trial_excluded),
                blacklist_fallback=False,
            )
            if result is None:
                continue
            key = tuple(result.action_names)
            if key in seen:
                continue
            seen.add(key)
            alternatives.append(result)
            frontier.append((result, trial_excluded))

    return alternatives


# Back-compat alias; prefer enumerate_alternatives in new code.
_enumerate_alternatives = enumerate_alternatives


def plan(
    start: PlanningState | dict[str, Any],
    goal: GoalSpec,
    actions: list[ActionSpec],
    blacklisted_actions: list[str] | None = None,
    *,
    max_alternatives: int = 8,
    tracer: Any = None,
    record_expansions: bool = False,
    blacklist_fallback: bool | None = None,
) -> Plan | None:
    """Two-phase planning: A* finds sequences, CSP validates/optimizes.

    Args:
        start: Current world state.  Accepts a plain ``dict`` for
            convenience; it will be coerced to ``PlanningState`` internally.
        goal: Goal specification with target conditions, constraints, objectives.
        actions: Available actions.
        blacklisted_actions: Action names to exclude.
        max_alternatives: Maximum number of alternative plans to compare
            against the primary plan, when it fails CSP validation or when
            the goal states preferences.  Alternatives are enumerated
            breadth-first over excluded actions (one swap away, then two,
            ...), so this bounds how far the search widens; the default of 8
            reaches two swaps on a four-step plan.
        tracer: Optional :class:`~langgoap.tracing.PlanningTracer` whose
            ``on_search_expand`` / ``on_search_dead_end`` hooks receive
            primary-search expansions and pipeline-level dead-end events
            (e.g. ``csp_infeasible``).  The pipeline owns a single
            ``on_search_complete`` emission for the logical plan call —
            alternative-enumeration A* runs do **not** re-emit completions.
        record_expansions: Opt-in gate for the per-expansion firehose.
            Must be ``True`` *and* a tracer supplied for any hook to fire.
        blacklist_fallback: Forwarded to :func:`langgoap.planner.astar.plan`
            for the primary search; ``None`` reads the goal policy.

    Returns:
        A ``Plan`` with ``metadata.csp`` attached.  When CSP finds the
        primary plan feasible it is returned directly.  When all alternatives
        are also infeasible the *primary* plan is returned with
        ``CSPStatus.INFEASIBLE`` metadata — the CSP layer is advisory and
        does not block graph execution; inspect ``plan.metadata.csp.status``
        to decide whether to proceed.  Returns ``None`` only when A* cannot
        find any plan at all.
    """
    if isinstance(start, dict):
        start = PlanningState.from_dict(start)
    if not needs_csp(goal):
        # No constraints/objectives → pure A* (zero overhead)
        return astar_plan(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            tracer=tracer,
            record_expansions=record_expansions,
            blacklist_fallback=blacklist_fallback,
        )

    # Resolve the CSP-level dead-end hook up-front, matching the
    # astar.plan pattern: missing attributes on legacy tracers are
    # tolerated and the opt-in gate must be honoured.
    on_dead_end = (
        getattr(tracer, "on_search_dead_end", None)
        if tracer is not None and record_expansions
        else None
    )

    # Phase 1: A* search for primary plan — the sole owner of
    # ``on_search_complete`` for this pipeline invocation.
    primary = astar_plan(
        start,
        goal,
        actions,
        blacklisted_actions=blacklisted_actions,
        tracer=tracer,
        record_expansions=record_expansions,
        blacklist_fallback=blacklist_fallback,
    )
    if primary is None:
        logger.warning("A* found no plan; CSP pipeline cannot proceed")
        return None

    # Goal already satisfied (empty plan) → skip CSP
    if len(primary) == 0:
        return primary

    # Phase 2: CSP validation
    csp_meta = validate_plan(primary, goal)
    logger.info(
        "CSP validation: status=%s for plan %s",
        csp_meta.status.value,
        primary.action_names,
    )

    # ``SKIPPED`` means ``validate_plan`` had nothing to bound-check
    # (a goal with soft goals / metrics but no constraints or
    # objectives); it is a feasible outcome.
    primary_feasible = csp_meta.status in (
        CSPStatus.FEASIBLE,
        CSPStatus.OPTIMAL,
        CSPStatus.SKIPPED,
    )

    # A goal that only carries hard limits is satisfied by the cheapest
    # A* plan that respects them: no comparison against alternatives is
    # needed.  Goals that state *preferences* (objectives, soft
    # constraints, soft goals, metrics) must look at alternatives even
    # when the primary plan is feasible, otherwise the preference never
    # influences which plan runs (issue #12).
    if primary_feasible and not needs_selection(goal):
        return _augment_plan(primary, goal, csp_meta)

    if not primary_feasible:
        # Primary plan rejected → emit a pipeline-level dead-end event so
        # dashboards can render the rejection, then generate alternatives.
        # Alternatives intentionally run without the tracer: they are
        # internal retries that belong to this single logical plan() call
        # and must not re-emit ``on_search_complete``.
        if on_dead_end is not None:
            on_dead_end(
                "csp_infeasible",
                {
                    "plan": list(primary.action_names),
                    "status": csp_meta.status.value,
                },
            )
        logger.info(
            "Primary plan infeasible; generating up to %d alternatives",
            max_alternatives,
        )
    else:
        logger.info(
            "Goal states preferences; comparing primary plan against up to %d "
            "alternatives",
            max_alternatives,
        )
    alternatives = enumerate_alternatives(
        start,
        goal,
        actions,
        blacklisted_actions,
        primary,
        max_alternatives,
    )

    if not alternatives:
        if not primary_feasible:
            logger.warning("No alternative plans found")
        return _augment_plan(primary, goal, csp_meta)

    # Phase 3: CP-SAT multi-plan selection.  A feasible primary competes
    # on equal footing with the alternatives; an infeasible one is left
    # out so the solver cannot pick it.
    candidates = ([primary] if primary_feasible else []) + alternatives
    best_plan, opt_meta = optimize_plans(candidates, goal)

    if opt_meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL):
        logger.info(
            "CSP selected plan %s (status=%s, candidates=%d)",
            best_plan.action_names,
            opt_meta.status.value,
            len(candidates),
        )
        return _augment_plan(
            best_plan,
            goal,
            _selection_metadata(best_plan, goal, opt_meta, len(candidates)),
        )

    if primary_feasible:
        return _augment_plan(primary, goal, csp_meta)
    # All alternatives infeasible — return primary with infeasible metadata
    logger.warning("All alternative plans infeasible")
    return _augment_plan(primary, goal, csp_meta)


def _selection_metadata(
    chosen: Plan, goal: GoalSpec, opt_meta: CSPMetadata, candidates: int
) -> CSPMetadata:
    """Re-validate the selected plan so its metadata carries the full picture.

    :func:`~langgoap.planner.csp.optimize_plans` reports the selection
    outcome but not the temporal schedule, makespan, or infeasibility
    explanation that :func:`~langgoap.planner.csp.validate_plan` computes
    for a single plan.  Merge the two: validation supplies schedule and
    resource usage, selection supplies objective values, status, and the
    number of candidates compared.
    """
    full = validate_plan(chosen, goal)
    # Keep the solver's verdict (OPTIMAL vs FEASIBLE) when the plan passed
    # re-validation; only a scheduling failure may downgrade it.
    status = full.status if full.status == CSPStatus.INFEASIBLE else opt_meta.status
    objective_values = {
        **dict(opt_meta.objective_values),
        **dict(full.objective_values),
    }
    return replace(
        full,
        status=status,
        solver_time_ms=full.solver_time_ms + opt_meta.solver_time_ms,
        objective_values=MappingProxyType(objective_values),
        resource_usage=full.resource_usage or opt_meta.resource_usage,
        plans_evaluated=candidates,
    )
