"""End-to-end integration tests for plan visualization.

Runs the real A*→CSP pipeline to produce a Plan with CSP metadata and
verifies every renderer can serialize it without raising and produces
the features (parallel clusters, resource bars, schedule Gantt) the
Epic 1 plan calls out.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from langgoap import (
    ActionSpec,
    ConstraintSpec,
    CSPStatus,
    GoalSpec,
    Minimize,
    PlanningState,
)
from langgoap.planner.pipeline import plan as pipeline_plan

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _three_step_linear_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="collect_specs",
            effects={"specs_collected": True},
            cost=1.0,
            resources={"tokens": 100.0},
        ),
        ActionSpec(
            name="design_layout",
            preconditions={"specs_collected": True},
            effects={"layout_designed": True},
            cost=1.0,
            resources={"tokens": 200.0},
        ),
        ActionSpec(
            name="compile_report",
            preconditions={"layout_designed": True},
            effects={"report_ready": True},
            cost=1.0,
            resources={"tokens": 150.0},
        ),
    ]


def _parallel_fetch_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="fetch_source_a",
            effects={"source_a": True},
            cost=1.0,
            duration=timedelta(seconds=2),
            resources={"cost_usd": 0.10},
        ),
        ActionSpec(
            name="fetch_source_b",
            effects={"source_b": True},
            cost=1.0,
            duration=timedelta(seconds=3),
            resources={"cost_usd": 0.15},
        ),
        ActionSpec(
            name="merge_sources",
            preconditions={"source_a": True, "source_b": True},
            effects={"merged": True},
            cost=1.0,
            duration=timedelta(seconds=1),
            resources={"cost_usd": 0.05},
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLinearPlanRendering:
    def test_linear_plan_mermaid_has_all_actions_and_edges(self) -> None:
        actions = _three_step_linear_actions()
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"report_ready": True}),
            actions,
        )
        assert plan is not None
        src = plan.to_mermaid()
        assert "flowchart TD" in src
        for name in ("collect_specs", "design_layout", "compile_report"):
            assert name in src
        # Two precondition-derived edges
        assert src.count("-->") == 2

    def test_linear_plan_ascii(self) -> None:
        actions = _three_step_linear_actions()
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"report_ready": True}),
            actions,
        )
        assert plan is not None
        text = plan.to_ascii()
        assert "Plan (3 steps" in text

    def test_linear_plan_dot(self) -> None:
        actions = _three_step_linear_actions()
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"report_ready": True}),
            actions,
        )
        assert plan is not None
        src = plan.to_dot()
        assert "digraph Plan {" in src
        assert "->" in src


class TestScheduledPlanRendering:
    def test_scheduled_plan_has_schedule_metadata(self) -> None:
        """Prereq: pipeline must produce a schedule for durative actions."""
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(
                conditions={"merged": True},
                constraints=(ConstraintSpec(key="cost_usd", max=1.0),),
            ),
            _parallel_fetch_actions(),
        )
        assert plan is not None
        assert plan.metadata.csp is not None
        assert plan.metadata.csp.status in (
            CSPStatus.OPTIMAL,
            CSPStatus.FEASIBLE,
        )
        assert plan.metadata.csp.schedule  # non-empty
        assert plan.metadata.csp.makespan is not None

    def test_scheduled_plan_mermaid_has_parallel_subgraph(self) -> None:
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(
                conditions={"merged": True},
                constraints=(ConstraintSpec(key="cost_usd", max=1.0),),
            ),
            _parallel_fetch_actions(),
        )
        assert plan is not None
        src = plan.to_mermaid()
        # The two fetches share start=0, so a subgraph should exist.
        assert "subgraph" in src
        # Resource summary visible
        assert "cost_usd" in src
        assert "[OK]" in src

    def test_scheduled_plan_mermaid_gantt(self) -> None:
        from langgoap.viz import render_mermaid_gantt

        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(
                conditions={"merged": True},
                constraints=(ConstraintSpec(key="cost_usd", max=1.0),),
            ),
            _parallel_fetch_actions(),
        )
        assert plan is not None
        src = render_mermaid_gantt(plan)
        assert src.startswith("gantt")
        for name in ("fetch_source_a", "fetch_source_b", "merge_sources"):
            assert name in src

    def test_scheduled_plan_ascii_gantt(self) -> None:
        from langgoap.viz import render_ascii_gantt

        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(
                conditions={"merged": True},
                constraints=(ConstraintSpec(key="cost_usd", max=1.0),),
            ),
            _parallel_fetch_actions(),
        )
        assert plan is not None
        bars = render_ascii_gantt(plan)
        assert "#" in bars

    def test_scheduled_plan_dot_has_cluster(self) -> None:
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(
                conditions={"merged": True},
                constraints=(ConstraintSpec(key="cost_usd", max=1.0),),
            ),
            _parallel_fetch_actions(),
        )
        assert plan is not None
        src = plan.to_dot()
        assert "subgraph cluster_" in src
        assert "legend" in src


class TestConstraintViolationRendering:
    def test_violation_is_rendered_as_violated(self) -> None:
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(
                conditions={"report_ready": True},
                # Tight budget that all resource-heavy plans will violate.
                constraints=(ConstraintSpec(key="tokens", max=100.0),),
            ),
            _three_step_linear_actions(),
        )
        assert plan is not None
        assert plan.metadata.csp is not None
        assert plan.metadata.csp.status == CSPStatus.INFEASIBLE
        src = plan.to_mermaid()
        assert "[VIOLATED]" in src
        ascii_src = plan.to_ascii()
        assert "VIOLATED" in ascii_src


class TestSaveRoundTrip:
    def test_save_to_tmp_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        plan = pipeline_plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"report_ready": True}),
            _three_step_linear_actions(),
        )
        assert plan is not None
        mmd = plan.save(tmp_path / "plan.mmd")
        dot = plan.save(tmp_path / "plan.dot")
        txt = plan.save(tmp_path / "plan.txt")
        assert "flowchart TD" in mmd.read_text()
        assert "digraph" in dot.read_text()
        assert "Plan (" in txt.read_text()
