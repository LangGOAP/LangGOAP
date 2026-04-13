"""Tests for plan infeasibility explanation (langgoap.planner.explain)."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.planner.csp import CSPMetadata, CSPStatus, ResourceUsage, validate_plan
from langgoap.planner.explain import (
    InfeasibilityExplanation,
    NoPlanExplanation,
    ResourceShortfall,
    _check_feasibility,
    explain_infeasibility,
    explain_no_plan,
)
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.state import PlanningState


def _make_action(
    name: str,
    resources: dict[str, float] | None = None,
) -> ActionSpec:
    return ActionSpec(name=name, effects={f"{name}_done": True}, resources=resources)


def _make_plan(*actions: ActionSpec) -> Plan:
    states = []
    state = PlanningState.from_dict({})
    for a in actions:
        state = state.apply(a.effects)
        states.append(state)
    return Plan(
        actions=actions,
        expected_states=tuple(states),
        total_cost=sum(a.get_cost() for a in actions),
    )


# ---------------------------------------------------------------------------
# ResourceShortfall
# ---------------------------------------------------------------------------


def test_resource_shortfall_creation() -> None:
    sf = ResourceShortfall(key="cost", required=10.0, available_max=5.0, overrun=5.0)
    assert sf.key == "cost"
    assert sf.overrun == 5.0


def test_resource_shortfall_min_violation() -> None:
    sf = ResourceShortfall(key="quality", required=3.0, available_min=5.0, overrun=2.0)
    assert sf.available_min == 5.0
    assert sf.overrun == 2.0


# ---------------------------------------------------------------------------
# Single violated hard constraint → MUS = {that constraint}
# ---------------------------------------------------------------------------


def test_single_violated_hard_constraint() -> None:
    a1 = _make_action("expensive", resources={"cost_usd": 10.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"expensive_done": True},
        constraints=(ConstraintSpec(key="cost_usd", max=5.0, level="hard"),),
    )
    meta = CSPMetadata(
        status=CSPStatus.INFEASIBLE,
        resource_usage=(
            ResourceUsage(
                key="cost_usd",
                total=10.0,
                constraint_max=5.0,
                satisfied=False,
                level="hard",
            ),
        ),
    )

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    assert len(explanation.conflicting_constraints) == 1
    assert explanation.conflicting_constraints[0].key == "cost_usd"
    assert len(explanation.resource_shortfalls) == 1
    assert explanation.resource_shortfalls[0].overrun == 5.0


# ---------------------------------------------------------------------------
# Two violated constraints, one redundant → MUS identifies minimal subset
# ---------------------------------------------------------------------------


def test_subsumed_constraint_excluded_from_mus() -> None:
    """The tighter of two overlapping constraints is excluded from the MUS.

    With cost=20 and constraints max=5 (tighter) and max=10 (looser), both are
    individually violated.  The greedy MUS algorithm iterates in declaration
    order: it checks whether removing max=5 still leaves the set infeasible
    (it does, because max=10 alone is still violated at cost=20), so max=5 is
    marked *redundant* and dropped from the working set.  Then removing max=10
    from the remaining singleton {max=10} yields an empty feasible set, so
    max=10 is marked *essential*.  The returned MUS is exactly {max=10}.
    """
    a1 = _make_action("act", resources={"cost": 20.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(
            ConstraintSpec(key="cost", max=5.0, level="hard"),  # tighter — redundant
            ConstraintSpec(key="cost", max=10.0, level="hard"),  # looser  — essential
        ),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    # Greedy MUS: exactly one constraint — the looser max=10 (essential).
    assert len(explanation.conflicting_constraints) == 1
    assert explanation.conflicting_constraints[0].max == 10.0


# ---------------------------------------------------------------------------
# No violated constraints → None
# ---------------------------------------------------------------------------


def test_no_violated_constraints_returns_none() -> None:
    a1 = _make_action("cheap", resources={"cost": 3.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"cheap_done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="hard"),),
    )
    meta = CSPMetadata(status=CSPStatus.FEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is None


# ---------------------------------------------------------------------------
# Soft-only violations → no explanation
# ---------------------------------------------------------------------------


def test_soft_only_violations_return_none() -> None:
    a1 = _make_action("act", resources={"cost": 10.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="soft"),),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is None  # soft constraints don't cause infeasibility


# ---------------------------------------------------------------------------
# ResourceShortfall computed correctly
# ---------------------------------------------------------------------------


def test_shortfall_overrun_correct() -> None:
    a1 = _make_action("act", resources={"tokens": 1500})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="tokens", max=1000, level="hard"),),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    sf = explanation.resource_shortfalls[0]
    assert sf.key == "tokens"
    assert sf.required == 1500.0
    assert sf.available_max == 1000.0
    assert sf.overrun == 500.0


# ---------------------------------------------------------------------------
# Min-bound violation
# ---------------------------------------------------------------------------


def test_min_bound_violation() -> None:
    a1 = _make_action("act", resources={"quality": 2.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="quality", min=5.0, level="hard"),),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    sf = explanation.resource_shortfalls[0]
    assert sf.available_min == 5.0
    assert sf.overrun == 3.0


# ---------------------------------------------------------------------------
# Suggestion text
# ---------------------------------------------------------------------------


def test_suggestion_text_generated() -> None:
    a1 = _make_action("act", resources={"cost": 10.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="hard"),),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    assert "cost" in explanation.suggestion
    assert "overrun" in explanation.suggestion


# ---------------------------------------------------------------------------
# Empty constraints → trivial
# ---------------------------------------------------------------------------


def test_empty_constraints_returns_none() -> None:
    a1 = _make_action("act")
    plan = _make_plan(a1)
    goal = GoalSpec(conditions={"act_done": True})
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is None


# ---------------------------------------------------------------------------
# Integration with validate_plan
# ---------------------------------------------------------------------------


def test_validate_plan_populates_explanation_on_infeasible() -> None:
    """validate_plan attaches explanation when status is INFEASIBLE."""
    a1 = _make_action("expensive_action", resources={"cost_usd": 10.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"expensive_action_done": True},
        constraints=(ConstraintSpec(key="cost_usd", max=5.0, level="hard"),),
    )

    meta = validate_plan(plan, goal)
    assert meta.status == CSPStatus.INFEASIBLE
    assert meta.explanation is not None
    assert isinstance(meta.explanation, InfeasibilityExplanation)
    assert len(meta.explanation.conflicting_constraints) == 1


def test_validate_plan_no_explanation_on_feasible() -> None:
    """validate_plan does not attach explanation when feasible."""
    a1 = _make_action("cheap_action", resources={"cost_usd": 3.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"cheap_action_done": True},
        constraints=(ConstraintSpec(key="cost_usd", max=5.0, level="hard"),),
    )

    meta = validate_plan(plan, goal)
    assert meta.status == CSPStatus.FEASIBLE
    assert meta.explanation is None


# ---------------------------------------------------------------------------
# Viz: ASCII output includes explanation section
# ---------------------------------------------------------------------------


def test_ascii_includes_explanation() -> None:
    a1 = _make_action("act", resources={"cost": 10.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="hard"),),
    )
    meta = validate_plan(plan, goal)
    from dataclasses import replace

    augmented = replace(plan, metadata=replace(plan.metadata, csp=meta))
    ascii_output = augmented.to_ascii()

    assert "Infeasibility Explanation" in ascii_output
    assert "cost" in ascii_output
    assert "overrun" in ascii_output


# ---------------------------------------------------------------------------
# Viz: Mermaid output includes note
# ---------------------------------------------------------------------------


def test_mermaid_includes_explanation_note() -> None:
    a1 = _make_action("act", resources={"cost": 10.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="cost", max=5.0, level="hard"),),
    )
    meta = validate_plan(plan, goal)
    from dataclasses import replace

    augmented = replace(plan, metadata=replace(plan.metadata, csp=meta))
    mermaid_output = augmented.to_mermaid()

    assert "INFEASIBLE" in mermaid_output
    assert "note right of" in mermaid_output


# ---------------------------------------------------------------------------
# Multiple hard violations
# ---------------------------------------------------------------------------


def test_multiple_hard_violations() -> None:
    """Two independently violated hard constraints: MUS only needs one."""
    a1 = _make_action("act", resources={"cost": 10.0, "tokens": 2000})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(
            ConstraintSpec(key="cost", max=5.0, level="hard"),
            ConstraintSpec(key="tokens", max=1000, level="hard"),
        ),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    # MUS is minimal — with two independent violations, greedy finds one
    assert len(explanation.conflicting_constraints) >= 1
    assert len(explanation.resource_shortfalls) >= 1
    # All shortfall keys should be from violated constraints
    violated_keys = {c.key for c in explanation.conflicting_constraints}
    for sf in explanation.resource_shortfalls:
        assert sf.key in violated_keys


# ---------------------------------------------------------------------------
# F7 — MUS minimality verified explicitly
# ---------------------------------------------------------------------------


def test_mus_every_constraint_is_essential() -> None:
    """Every constraint returned in the MUS is essential (minimality invariant).

    For every constraint C in the MUS, the set MUS \\ {C} must be feasible.
    If removing C still leaves the set infeasible, C is redundant and should
    not appear in a true minimal unsatisfiable subset.
    """
    a1 = _make_action("act", resources={"cost": 15.0})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="cost", max=10.0, level="hard"),),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    mus = list(explanation.conflicting_constraints)
    assert len(mus) == 1  # single violated constraint → MUS has exactly one member

    # Minimality: removing the sole constraint yields an empty (feasible) set.
    totals = {"cost": 15.0}
    for c in mus:
        without = [cc for cc in mus if cc is not c]
        assert _check_feasibility(totals, without), (
            f"Constraint {c} is not essential — removing it should make the "
            f"remaining set feasible, violating the MUS minimality invariant."
        )


def test_mus_single_constraint_is_its_own_mus() -> None:
    """A single violated constraint is trivially its own MUS."""
    a1 = _make_action("act", resources={"tokens": 5000})
    plan = _make_plan(a1)
    goal = GoalSpec(
        conditions={"act_done": True},
        constraints=(ConstraintSpec(key="tokens", max=1000, level="hard"),),
    )
    meta = CSPMetadata(status=CSPStatus.INFEASIBLE)

    explanation = explain_infeasibility(plan, goal, meta)
    assert explanation is not None
    assert len(explanation.conflicting_constraints) == 1
    assert explanation.conflicting_constraints[0].key == "tokens"
    assert explanation.conflicting_constraints[0].max == 1000

    # Minimality: the empty set is feasible.
    totals = {"tokens": 5000.0}
    assert _check_feasibility(totals, [])


# ---------------------------------------------------------------------------
# F8 — explain_no_plan: diagnostic when A* returns None
# ---------------------------------------------------------------------------


def test_explain_no_plan_unreachable_goal_condition() -> None:
    """Goal condition that no action can produce is flagged as unreachable."""
    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.state import PlanningState

    action = ActionSpec(name="a", effects={"done": True})
    goal = GoalSpec(conditions={"done": True, "impossible": True})
    start = PlanningState.from_dict({})

    explanation = explain_no_plan(start, goal, [action])

    assert "impossible" in explanation.unreachable_conditions
    assert "done" not in explanation.unreachable_conditions
    assert "impossible" in explanation.suggestion


def test_explain_no_plan_all_conditions_reachable_but_no_path() -> None:
    """When every goal condition is producible, explanation is cycle/ordering advice."""
    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.state import PlanningState

    # a0 produces 'x' but requires 'y'; a1 produces 'y' but requires 'x' → deadlock
    a0 = ActionSpec(name="a0", preconditions={"y": True}, effects={"x": True})
    a1 = ActionSpec(name="a1", preconditions={"x": True}, effects={"y": True})
    goal = GoalSpec(conditions={"x": True})
    start = PlanningState.from_dict({})

    explanation = explain_no_plan(start, goal, [a0, a1])

    # Both x and y are producible by some action — no unreachable conditions.
    assert len(explanation.unreachable_conditions) == 0
    # Suggestion acknowledges the circular / ordering failure.
    assert explanation.suggestion  # non-empty


def test_explain_no_plan_to_dict_round_trip() -> None:
    """NoPlanExplanation serialises and deserialises cleanly."""
    original = NoPlanExplanation(
        unreachable_conditions=("impossible",),
        missing_preconditions=("never_set",),
        suggestion="Fix it.",
    )
    restored = NoPlanExplanation.from_dict(original.to_dict())
    assert restored == original


def test_explain_no_plan_stored_in_goapstate() -> None:
    """GoapPlanner puts no_plan_explanation into GoapState when A* returns None."""
    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.graph.nodes import GoapPlanner

    action = ActionSpec(name="a", effects={"done": True})
    goal = GoalSpec(conditions={"impossible": True})
    planner = GoapPlanner([action])

    result = planner({"goal": goal, "world_state": {}})

    assert result.get("status") == "no_plan"
    expl_dict = result.get("no_plan_explanation")
    assert expl_dict is not None
    assert "impossible" in expl_dict.get("unreachable_conditions", [])
    assert "impossible" in expl_dict.get("suggestion", "")
