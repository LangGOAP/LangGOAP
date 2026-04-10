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

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
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

    The sign convention matches OptaPlanner's ``penalize(amount)``:

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

    # Constraint violations → hard or soft depending on level.
    for usage in meta.resource_usage:
        c = constraint_by_key.get(usage.key)
        if c is None:
            continue
        violation = 0.0
        if c.max is not None and usage.total > c.max:
            violation = usage.total - c.max
        elif c.min is not None and usage.total < c.min:
            violation = c.min - usage.total
        if violation == 0.0:
            continue
        penalty = violation * c.weight
        if c.level == "hard":
            hard -= penalty
        else:
            soft -= penalty

    # Objective contributions go to soft, aligned with direction.
    if goal.objectives:
        for key, direction in goal.objectives.items():
            v = meta.objective_values.get(key, 0.0)
            soft += -v if direction == ObjectiveDirection.MINIMIZE else v

    return HardSoftScore(hard=hard, soft=soft)


def needs_csp(goal: GoalSpec) -> bool:
    """Return True if the goal has constraints or objectives requiring CSP."""
    return bool(goal.constraints) or goal.objectives is not None


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
    start: PlanningState,
    goal: GoalSpec,
    actions: list[ActionSpec],
    blacklisted: list[str] | None,
    exclude_plan: Plan,
    max_count: int,
) -> list[Plan]:
    """Generate alternative plans by blacklisting each action in the rejected plan.

    For each action in ``exclude_plan``, temporarily add it to the blacklist
    and re-run A*. Deduplicate by action name sequence.
    """
    seen: set[tuple[str, ...]] = {tuple(exclude_plan.action_names)}
    alternatives: list[Plan] = []
    base_blacklist = list(blacklisted) if blacklisted else []

    for action in exclude_plan.actions:
        if len(alternatives) >= max_count:
            break
        trial_blacklist = base_blacklist + [action.name]
        result = astar_plan(start, goal, actions, blacklisted_actions=trial_blacklist)
        if result is not None:
            key = tuple(result.action_names)
            if key not in seen:
                seen.add(key)
                alternatives.append(result)

    return alternatives


# Back-compat alias; prefer enumerate_alternatives in new code.
_enumerate_alternatives = enumerate_alternatives


def plan(
    start: PlanningState,
    goal: GoalSpec,
    actions: list[ActionSpec],
    blacklisted_actions: list[str] | None = None,
    *,
    max_alternatives: int = 5,
) -> Plan | None:
    """Two-phase planning: A* finds sequences, CSP validates/optimizes.

    Args:
        start: Current world state.
        goal: Goal specification with target conditions, constraints, objectives.
        actions: Available actions.
        blacklisted_actions: Action names to exclude.
        max_alternatives: Maximum number of alternative plans to generate
            when the primary plan fails CSP validation.

    Returns:
        A ``Plan`` with ``metadata.csp`` attached.  When CSP finds the
        primary plan feasible it is returned directly.  When all alternatives
        are also infeasible the *primary* plan is returned with
        ``CSPStatus.INFEASIBLE`` metadata — the CSP layer is advisory and
        does not block graph execution; inspect ``plan.metadata.csp.status``
        to decide whether to proceed.  Returns ``None`` only when A* cannot
        find any plan at all.
    """
    if not needs_csp(goal):
        # No constraints/objectives → pure A* (zero overhead)
        return astar_plan(start, goal, actions, blacklisted_actions=blacklisted_actions)

    # Phase 1: A* search for primary plan
    primary = astar_plan(start, goal, actions, blacklisted_actions=blacklisted_actions)
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

    if csp_meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL):
        return _augment_plan(primary, goal, csp_meta)

    # Primary plan rejected → generate alternatives
    logger.info(
        "Primary plan infeasible; generating up to %d alternatives",
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
        logger.warning("No alternative plans found")
        return _augment_plan(primary, goal, csp_meta)

    # Phase 3: CP-SAT multi-plan optimization
    try:
        best_plan, opt_meta = optimize_plans(alternatives, goal)
    except ImportError:
        # ortools not installed — return first alternative with basic validation
        logger.warning("ortools not available for multi-plan optimization")
        for alt in alternatives:
            alt_meta = validate_plan(alt, goal)
            if alt_meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL):
                return _augment_plan(alt, goal, alt_meta)
        return _augment_plan(primary, goal, csp_meta)

    if opt_meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL):
        logger.info(
            "CSP selected plan %s (status=%s)",
            best_plan.action_names,
            opt_meta.status.value,
        )
        return _augment_plan(best_plan, goal, opt_meta)

    # All alternatives infeasible — return primary with infeasible metadata
    logger.warning("All alternative plans infeasible")
    return _augment_plan(primary, goal, csp_meta)
