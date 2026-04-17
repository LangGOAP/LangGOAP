"""Temporal scheduling benchmarks for the A* → CSP pipeline.

Exercises the full two-phase pipeline (A* planning + CP-SAT temporal
scheduling via IntervalVar) across 10, 50, and 100-action linear chains
with resource constraints and duration annotations.

These benchmarks answer "does the CSP optimizer scale linearly with
action count" and give a regression baseline that ``make benchmark-compare``
enforces going forward.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan as astar_plan
from langgoap.state import PlanningState


def _run_pipeline(
    actions: list[ActionSpec],
    goal: GoalSpec,
    start: PlanningState,
) -> Any:
    """Run the full A* → CSP pipeline and return the plan."""
    from langgoap.planner.pipeline import plan as pipeline_plan

    result = pipeline_plan(start, goal, actions)
    assert result is not None, "Pipeline returned None"
    return result


def _run_astar_only(
    actions: list[ActionSpec],
    goal: GoalSpec,
    start: PlanningState,
) -> Any:
    """Run pure A* (no CSP) and return the plan."""
    result = astar_plan(start, goal, actions)
    assert result is not None, "A* returned None"
    return result


# ---------------------------------------------------------------------------
# A* only (baseline)
# ---------------------------------------------------------------------------


@pytest.mark.bench
class TestAStarPlanning:
    """Benchmark A* planning without CSP for temporal chains."""

    def test_astar_10(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        temporal_10: tuple[list[ActionSpec], GoalSpec, PlanningState],
    ) -> None:
        actions, goal, start = temporal_10
        result = benchmark(_run_astar_only, actions, goal, start)
        assert len(result) == 10

    def test_astar_50(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        temporal_50: tuple[list[ActionSpec], GoalSpec, PlanningState],
    ) -> None:
        actions, goal, start = temporal_50
        result = benchmark(_run_astar_only, actions, goal, start)
        assert len(result) == 50

    def test_astar_100(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        temporal_100: tuple[list[ActionSpec], GoalSpec, PlanningState],
    ) -> None:
        actions, goal, start = temporal_100
        result = benchmark(_run_astar_only, actions, goal, start)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# Full pipeline (A* + CSP temporal scheduling)
# ---------------------------------------------------------------------------


@pytest.mark.bench
class TestPipelinePlanning:
    """Benchmark the full A* → CSP pipeline with temporal scheduling."""

    def test_pipeline_10(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        temporal_10: tuple[list[ActionSpec], GoalSpec, PlanningState],
    ) -> None:
        actions, goal, start = temporal_10
        result = benchmark(_run_pipeline, actions, goal, start)
        assert len(result) == 10
        assert result.metadata.csp is not None

    def test_pipeline_50(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        temporal_50: tuple[list[ActionSpec], GoalSpec, PlanningState],
    ) -> None:
        actions, goal, start = temporal_50
        result = benchmark(_run_pipeline, actions, goal, start)
        assert len(result) == 50
        assert result.metadata.csp is not None

    def test_pipeline_100(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        temporal_100: tuple[list[ActionSpec], GoalSpec, PlanningState],
    ) -> None:
        actions, goal, start = temporal_100
        result = benchmark(_run_pipeline, actions, goal, start)
        assert len(result) == 100
        assert result.metadata.csp is not None
