"""Unit tests for the A* → CSP pipeline."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import patch

import pytest

from langgoap import ConstraintSpec, CSPMetadata, CSPStatus, GoalSpec, Minimize
from langgoap.planner.pipeline import (
    enumerate_alternatives,
    needs_csp,
    plan,
)
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from tests.conftest import make_action, make_plan

# ---------------------------------------------------------------------------
# needs_csp
# ---------------------------------------------------------------------------


class TestNeedsCSP:
    def test_with_constraints(self) -> None:
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        assert needs_csp(goal) is True

    def test_with_objectives(self) -> None:
        goal = GoalSpec(
            conditions={"done": True},
            objectives={"cost": Minimize},
        )
        assert needs_csp(goal) is True

    def test_plain_goal(self) -> None:
        goal = GoalSpec(conditions={"done": True})
        assert needs_csp(goal) is False


# ---------------------------------------------------------------------------
# plan (pipeline)
# ---------------------------------------------------------------------------


class TestPipelinePlan:
    def test_plain_goal_delegates_to_astar(self) -> None:
        """Without constraints/objectives, delegates directly to A*."""
        a = make_action("a", eff={"done": True})
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})
        result = plan(start, goal, [a])
        assert result is not None
        assert result.action_names == ["a"]
        # No CSP metadata on the plan
        assert result.metadata.csp is None

    def test_feasible_plan_passthrough(self) -> None:
        """Plan within resource constraints passes CSP validation."""
        a = make_action("a", eff={"done": True}, resources={"cost": 10})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        result = plan(start, goal, [a])
        assert result is not None
        assert result.action_names == ["a"]
        assert result.metadata.csp is not None
        assert result.metadata.csp.status == CSPStatus.FEASIBLE

    def test_infeasible_finds_alternative(self) -> None:
        """Primary plan exceeds budget → pipeline finds a cheaper alternative."""
        expensive = make_action(
            "expensive",
            eff={"done": True},
            cost=5,
            resources={"cost": 200},
        )
        cheap = make_action(
            "cheap",
            eff={"done": True},
            cost=1,
            resources={"cost": 50},
        )
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
            objectives={"cost": Minimize},
        )
        result = plan(start, goal, [expensive, cheap])
        assert result is not None
        assert result.metadata.csp is not None
        # The optimizer should select the cheap plan
        assert result.metadata.csp.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)

    def test_no_astar_plan_returns_none(self) -> None:
        """When A* finds no plan, pipeline returns None."""
        a = make_action("a", pre={"impossible": True}, eff={"done": True})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        result = plan(start, goal, [a])
        assert result is None

    def test_all_alternatives_infeasible_returns_primary(self) -> None:
        """When all alternatives are infeasible, returns primary with infeasible metadata."""
        a = make_action("a", eff={"done": True}, resources={"cost": 500})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        result = plan(start, goal, [a])
        assert result is not None
        assert result.metadata.csp is not None
        assert result.metadata.csp.status == CSPStatus.INFEASIBLE

    def test_metadata_includes_csp(self) -> None:
        """Plan metadata includes CSP results."""
        a = make_action("a", eff={"done": True}, resources={"cost": 10})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        result = plan(start, goal, [a])
        assert result is not None
        assert result.metadata.csp is not None
        assert result.metadata.csp.solver_time_ms >= 0
        assert result.metadata.csp.scale_factor == 1000

    def test_already_satisfied_goal(self) -> None:
        """Goal already satisfied → empty plan returned without CSP metadata.

        The pipeline short-circuits before CSP validation for empty plans.
        """
        start = PlanningState.from_dict({"done": True})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        result = plan(start, goal, [])
        assert result is not None
        assert len(result) == 0
        # CSP is skipped; the plan carries no CSP metadata
        assert result.metadata.csp is None


# ---------------------------------------------------------------------------
# enumerate_alternatives
# ---------------------------------------------------------------------------


class TestEnumerateAlternatives:
    def test_generates_distinct_plans(self) -> None:
        """Blacklisting each action in the rejected plan yields distinct alternatives."""
        a1 = make_action("step1", eff={"x": True})
        a2 = make_action("step2", pre={"x": True}, eff={"done": True})
        alt_path = make_action("shortcut", eff={"x": True, "done": True})
        all_actions = [a1, a2, alt_path]

        primary = make_plan(a1, a2)
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})

        alts = enumerate_alternatives(
            start, goal, all_actions, None, primary, max_count=5
        )
        # Blacklisting step1 or step2 should find the shortcut path
        assert len(alts) >= 1
        alt_names = [tuple(a.action_names) for a in alts]
        # All alternatives must differ from the primary
        assert tuple(primary.action_names) not in alt_names

    def test_respects_max_count(self) -> None:
        """Stops generating alternatives after max_count."""
        actions = [
            make_action("a", eff={"x": True}),
            make_action("b", pre={"x": True}, eff={"y": True}),
            make_action("c", pre={"y": True}, eff={"done": True}),
            make_action("alt_a", eff={"x": True}),
            make_action("alt_b", pre={"x": True}, eff={"y": True}),
        ]
        primary = make_plan(actions[0], actions[1], actions[2])
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})

        alts = enumerate_alternatives(start, goal, actions, None, primary, max_count=1)
        assert len(alts) <= 1

    def test_deduplication(self) -> None:
        """Duplicate plans (same action sequence) are deduplicated."""
        a = make_action("a", eff={"x": True, "done": True})
        # Both blacklist attempts would find the same single-action plan
        primary_actions = (
            make_action("p1", eff={"x": True}),
            make_action("p2", pre={"x": True}, eff={"done": True}),
        )
        primary = make_plan(*primary_actions)
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"done": True})

        alts = enumerate_alternatives(
            start, goal, [a, *primary_actions], None, primary, max_count=10
        )
        # Even if multiple blacklist attempts find the same alt, only 1 unique
        names = [tuple(alt.action_names) for alt in alts]
        assert len(names) == len(set(names))
