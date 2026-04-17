r"""Integration test for the Project Job Scheduling tutorial (Tier 2).

Exercises the full A\* → CSP pipeline on an MRCPSP instance derived
from standard benchmark data.  The CSP phase is expected to:

- Respect the precedence DAG extracted from effect→precondition
  matching in ``planner/csp.py::build_dependency_graph``.
- Run independent jobs in parallel via CP-SAT ``IntervalVar`` and
  report a true makespan (< sum of durations).
- Aggregate the non-renewable ``budget`` resource across modes and
  mark plans INFEASIBLE when a hard cap cannot be met.
- Enumerate alternative mode combinations when the cheapest-by-cost
  plan violates a hard budget cap.
- Penalize soft budget violations without flipping the status.

Helpers live in
``examples/tutorials/tutorial_examples/project_job_scheduling.py`` and
the instance fixture is at
``examples/tutorials/tutorial_examples/data/project_job_scheduling_instance.py``
(derived from reference benchmark data).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tutorial_examples.data.project_job_scheduling_instance import JOBS
from tutorial_examples.project_job_scheduling import (
    project_job_scheduling_actions,
    project_job_scheduling_goal,
    project_job_scheduling_start,
)

from langgoap import CSPStatus, GoapGraph
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.score import HardSoftScore
from langgoap.state import PlanningState

pytest.importorskip(
    "ortools", reason="project job scheduling tutorial exercises CP-SAT"
)


# Primary (no-cap) plan: A* picks the fastest mode for every job.
#   design_single(1) + frontend_fast(3) + backend_fast(4)
#   + integrate_fast(2) + deploy_single(1) = 11 duration_hours
# Makespan respects precedence but parallelizes frontend ∥ backend:
#   design(0-1) → {frontend(1-4) ∥ backend(1-5)} → integrate(5-7) → deploy(7-8)
PRIMARY_ACTIONS = {
    "do_design_single",
    "do_frontend_fast",
    "do_backend_fast",
    "do_integrate_fast",
    "do_deploy_single",
}
PRIMARY_DURATION_HOURS = 11.0
PRIMARY_BUDGET = 32.0  # 0 + 10 + 12 + 8 + 2
PRIMARY_MAKESPAN = timedelta(hours=8)


class TestProjectJobSchedulingPrimary:
    """Default goal: minimize duration, no budget cap → all-fast modes."""

    def test_pipeline_picks_fastest_modes(self) -> None:
        """A* selects the fast mode for every multi-mode job."""
        actions = project_job_scheduling_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            project_job_scheduling_goal(),
            actions,
        )
        assert plan_obj is not None
        assert set(plan_obj.action_names) == PRIMARY_ACTIONS
        assert plan_obj.total_cost == PRIMARY_DURATION_HOURS

    def test_precedence_respected_in_plan(self) -> None:
        """Design runs first and deploy runs last in the action list."""
        actions = project_job_scheduling_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            project_job_scheduling_goal(),
            actions,
        )
        assert plan_obj is not None
        names = plan_obj.action_names
        assert names[0] == "do_design_single"
        assert names[-1] == "do_deploy_single"
        # integrate must follow frontend and backend.
        integrate_idx = names.index("do_integrate_fast")
        frontend_idx = names.index("do_frontend_fast")
        backend_idx = names.index("do_backend_fast")
        assert frontend_idx < integrate_idx
        assert backend_idx < integrate_idx

    def test_csp_schedule_parallelizes_frontend_and_backend(self) -> None:
        """Frontend and backend run concurrently → makespan < sum of durations."""
        actions = project_job_scheduling_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            project_job_scheduling_goal(),
            actions,
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        # Parallel execution: 8h makespan vs 11h serial duration sum.
        assert csp.makespan == PRIMARY_MAKESPAN

        schedule_by_name = {s.action_name: s for s in csp.schedule}
        frontend = schedule_by_name["do_frontend_fast"]
        backend = schedule_by_name["do_backend_fast"]
        # Overlap window exists: max(start) < min(end).
        overlap_start = max(frontend.start, backend.start)
        overlap_end = min(frontend.end, backend.end)
        assert overlap_start < overlap_end

        # Design ends before frontend/backend start; integrate ends before deploy.
        design = schedule_by_name["do_design_single"]
        integrate = schedule_by_name["do_integrate_fast"]
        deploy = schedule_by_name["do_deploy_single"]
        assert design.end <= frontend.start
        assert design.end <= backend.start
        assert frontend.end <= integrate.start
        assert backend.end <= integrate.start
        assert integrate.end <= deploy.start

    def test_hardsoft_score_reflects_duration_objective(self) -> None:
        """Soft score mirrors the negated duration_hours objective."""
        actions = project_job_scheduling_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            project_job_scheduling_goal(),
            actions,
        )
        assert plan_obj is not None
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -PRIMARY_DURATION_HOURS

    def test_graph_invocation_executes_plan(self) -> None:
        """End-to-end GoapGraph.invoke() reaches goal_achieved."""
        actions = project_job_scheduling_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=project_job_scheduling_goal(),
            world_state=project_job_scheduling_start(),
        )
        assert result["status"] == "goal_achieved"
        for job in JOBS:
            assert result["world_state"][f"{job.name}_done"] is True


class TestProjectJobSchedulingHardBudgetAlternative:
    """A hard budget cap rules out all-fast → pipeline enumerates alternatives.

    With ``max_budget=26`` the primary plan (budget=32) is infeasible
    so the pipeline falls back on alternative mode combinations.  The
    best feasible alternative swaps ``frontend_fast`` (10) for
    ``frontend_slow`` (4), landing at exactly 26.
    """

    def test_alternative_plan_respects_hard_budget(self) -> None:
        actions = project_job_scheduling_actions()
        goal = project_job_scheduling_goal(max_budget=26, max_budget_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)

        # Alternative: frontend slow keeps backend+integrate fast.
        assert "do_frontend_slow" in plan_obj.action_names
        assert "do_backend_fast" in plan_obj.action_names
        assert "do_integrate_fast" in plan_obj.action_names

        usage_by_key = {u.key: u for u in csp.resource_usage}
        budget_usage = usage_by_key["budget"]
        assert budget_usage.level == "hard"
        assert budget_usage.total <= 26.0
        assert budget_usage.satisfied is True

    def test_hardsoft_score_feasible_under_hard_cap(self) -> None:
        """Feasible alternative → hard==0, soft tracks the objective."""
        actions = project_job_scheduling_actions()
        goal = project_job_scheduling_goal(max_budget=26, max_budget_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        # The chosen alternative has duration_hours=14 (1+6+4+2+1).
        assert plan_obj.score.soft == -14.0


class TestProjectJobSchedulingHardInfeasible:
    """A budget below every achievable alternative → INFEASIBLE status."""

    def test_impossible_hard_cap(self) -> None:
        actions = project_job_scheduling_actions()
        goal = project_job_scheduling_goal(max_budget=10, max_budget_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.metadata.csp is not None
        assert plan_obj.metadata.csp.status == CSPStatus.INFEASIBLE
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard < 0.0


class TestProjectJobSchedulingSoftBudget:
    """Soft cap records a penalty but keeps the plan feasible."""

    def test_soft_budget_violation_penalizes_soft_score(self) -> None:
        actions = project_job_scheduling_actions()
        goal = project_job_scheduling_goal(max_budget=20, max_budget_level="soft")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(project_job_scheduling_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)

        budget_usage = next(u for u in csp.resource_usage if u.key == "budget")
        assert budget_usage.level == "soft"
        assert budget_usage.satisfied is False
        assert budget_usage.total == PRIMARY_BUDGET

        # soft = -duration(11) - overflow(32 - 20 = 12) = -23
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -(PRIMARY_DURATION_HOURS + 12.0)
