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

    def test_pipeline_plan_accepts_dict_start(self) -> None:
        """pipeline plan() accepts a plain dict and coerces to PlanningState."""
        a = make_action("a", eff={"done": True})
        result = plan({"x": True}, GoalSpec(conditions={"done": True}), [a])
        assert result is not None
        assert result.action_names == ["a"]

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


# ---------------------------------------------------------------------------
# Observability: pipeline forwards search hooks into A* primary search
# ---------------------------------------------------------------------------


class _PipelineRecorder:
    """Captures pipeline-level search events for assertions."""

    def __init__(self) -> None:
        self.expansions: list[dict[str, Any]] = []
        self.dead_ends: list[tuple[str, dict[str, Any]]] = []
        self.completions: list[tuple[int, float, bool]] = []

    def on_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.expansions.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "action_name": action_name,
                "g": g,
                "h": h,
                "f": f,
            }
        )

    def on_search_dead_end(self, reason: str, detail: dict[str, Any]) -> None:
        self.dead_ends.append((reason, detail))

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.completions.append((nodes_explored, duration_ms, found))


class TestPipelineSearchHooks:
    """CSP pipeline forwards ``tracer`` + ``record_expansions`` into A*.

    Landing 3 of the observability hardening plan: the two-phase
    pipeline must surface the same per-expansion / dead-end /
    completion telemetry that pure A* already does, while remaining
    zero-overhead when the gate is off.
    """

    def test_primary_expansions_reach_tracer(self) -> None:
        """Pipeline plan() with a feasible primary forwards expansions."""
        a = make_action("a", eff={"done": True}, resources={"cost": 10})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        rec = _PipelineRecorder()
        result = plan(start, goal, [a], tracer=rec, record_expansions=True)
        assert result is not None
        assert result.metadata.csp is not None
        assert len(rec.expansions) >= 1
        # Primary A* succeeded → exactly one completion with found=True.
        assert len(rec.completions) == 1
        _, _, found = rec.completions[0]
        assert found is True

    def test_gate_off_silences_all_pipeline_hooks(self) -> None:
        """``record_expansions=False`` must silence per-expansion hooks."""
        a = make_action("a", eff={"done": True}, resources={"cost": 10})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        rec = _PipelineRecorder()
        plan(start, goal, [a], tracer=rec, record_expansions=False)
        assert rec.expansions == []
        assert rec.dead_ends == []
        assert rec.completions == []

    def test_single_completion_across_csp_alternatives(self) -> None:
        """Pipeline emits exactly one ``on_search_complete`` per call.

        Even when the primary plan is CSP-infeasible and alternatives
        must be generated (each involving internal A* calls), only the
        primary search fires the lifecycle completion event. Internal
        alternative-generation A* calls must not re-emit completions —
        they are part of a single logical ``plan()`` operation.
        """
        expensive = make_action(
            "expensive", eff={"done": True}, cost=5, resources={"cost": 200}
        )
        cheap = make_action("cheap", eff={"done": True}, cost=1, resources={"cost": 50})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
            objectives={"cost": Minimize},
        )
        rec = _PipelineRecorder()
        result = plan(
            start, goal, [expensive, cheap], tracer=rec, record_expansions=True
        )
        assert result is not None
        assert len(rec.completions) == 1

    def test_csp_rejection_emits_dead_end(self) -> None:
        """Primary plan rejected by CSP → ``on_search_dead_end`` event.

        Recorded with ``reason='csp_infeasible'`` and a detail payload
        containing the rejected action sequence so dashboards can show
        *why* the pipeline had to fall back to alternative enumeration.
        """
        a = make_action("a", eff={"done": True}, resources={"cost": 500})
        start = PlanningState.from_dict({})
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        rec = _PipelineRecorder()
        plan(start, goal, [a], tracer=rec, record_expansions=True)
        csp_dead_ends = [
            (reason, detail)
            for reason, detail in rec.dead_ends
            if reason == "csp_infeasible"
        ]
        assert len(csp_dead_ends) == 1
        _, detail = csp_dead_ends[0]
        assert detail.get("plan") == ["a"]
