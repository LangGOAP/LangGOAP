r"""Integration test for the Task Assigning tutorial (Tier 2).

Exercises the full A\* → CSP pipeline on a ticket-routing instance
derived from OptaPlanner's ``taskassigning`` example.  The test
battery verifies:

- **Skill matching** — unqualified employees have no action for a
  task whose required skill they lack.
- **Weighted delay minimization** — A\* picks the cheapest
  assignment using ``priority × base_duration × affinity``.
- **HardSoftScore population** — the
  ``weighted_delay → MINIMIZE`` objective routes through CSP and
  the score has ``hard == 0`` and ``soft == -57``.
- **ConstraintBuilder parity** — :func:`task_assigning_goal_fluent`
  and :func:`task_assigning_goal` produce functionally equivalent
  goals.
- **Soft workload cap** — overloaded employees subtract overflow
  from the soft score without flipping status.
- **Hard workload cap** — the same cap at ``level="hard"`` flips
  CSP status to ``INFEASIBLE``.
- **NL intake** — a ``FakeStructuredModel`` round-trips the NL
  request through :class:`GoalInterpreter` and the resulting plan
  still assigns every task.

Helpers live in
``examples/tutorials/tutorial_examples/task_assigning.py`` and the
instance fixture is at
``examples/tutorials/tutorial_examples/data/task_assigning_instance.py``
(provenance: OptaPlanner ``24tasks-8employees.json`` compacted to
6 tasks / 3 employees / 4 skills).
"""

from __future__ import annotations

import pytest
from tutorial_examples.data.task_assigning_instance import (
    EMPLOYEES,
    TASKS,
)
from tutorial_examples.task_assigning import (
    task_assigning_actions,
    task_assigning_goal,
    task_assigning_goal_fluent,
    task_assigning_start,
)

from langgoap import (
    CSPStatus,
    GoalInterpreter,
    GoapGraph,
    InterpretedConstraint,
    InterpretedGoal,
    InterpretedObjective,
)
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.score import HardSoftScore
from langgoap.state import PlanningState
from tests.conftest import FakeStructuredModel

pytest.importorskip("ortools", reason="task assigning tutorial exercises CP-SAT")


# Optimal assignment by weighted delay:
#   alice → t1 (sales_strategy CRIT, HIGH, 4*4*1 = 16)
#   bob   → t3 (brand_story  CRIT, HIGH, 5*4*1 = 20)
#   bob   → t4 (root_cause   MAJR, HIGH, 4*2*1 =  8)
#   bob   → t6 (root_cause   MINR, HIGH, 4*1*1 =  4)
#   carol → t2 (compliance   MAJR, HIGH, 3*2*1 =  6)
#   carol → t5 (compliance   MINR, HIGH, 3*1*1 =  3)
#   total weighted_delay = 57
OPTIMAL_ASSIGNMENT = {
    "assign_alice_to_t1",
    "assign_bob_to_t3",
    "assign_bob_to_t4",
    "assign_bob_to_t6",
    "assign_carol_to_t2",
    "assign_carol_to_t5",
}
OPTIMAL_WEIGHTED_DELAY = 57.0
OPTIMAL_WORKLOAD = {"alice": 4.0, "bob": 13.0, "carol": 6.0}


class TestTaskAssigningSkillFilter:
    """Skill matching drops unqualified pairs at action-build time."""

    def test_skill_filter_removes_invalid_pairs(self) -> None:
        """12 legal actions for 3 employees × 4 task types mapped by skill."""
        actions = task_assigning_actions()
        names = {a.name for a in actions}

        # alice has no creative_thinking → cannot do brand_story (t3).
        assert "assign_alice_to_t3" not in names
        # bob has no strategic_planning → cannot do sales_strategy (t1).
        assert "assign_bob_to_t1" not in names
        # bob has no risk_management → cannot do compliance (t2, t5).
        assert "assign_bob_to_t2" not in names
        assert "assign_bob_to_t5" not in names
        # carol has no problem_solving → cannot do root_cause (t4, t6).
        assert "assign_carol_to_t4" not in names
        assert "assign_carol_to_t6" not in names

        assert len(actions) == 12


class TestTaskAssigningOptimal:
    """Happy-path pipeline: A* + CSP find the lowest-weighted-delay assignment."""

    def test_pipeline_finds_optimal_assignment(self) -> None:
        """A\\* picks the minimum-weighted-delay plan (total 57)."""
        actions = task_assigning_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            task_assigning_goal(),
            actions,
        )
        assert plan_obj is not None
        assert set(plan_obj.action_names) == OPTIMAL_ASSIGNMENT
        assert plan_obj.total_cost == OPTIMAL_WEIGHTED_DELAY

    def test_every_task_assigned_exactly_once(self) -> None:
        actions = task_assigning_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            task_assigning_goal(),
            actions,
        )
        assert plan_obj is not None
        assigned = {a.name.rsplit("_to_", 1)[1] for a in plan_obj.actions}
        assert assigned == {t.name for t in TASKS}

    def test_hardsoft_score_reflects_objective(self) -> None:
        actions = task_assigning_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            task_assigning_goal(),
            actions,
        )
        assert plan_obj is not None
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -OPTIMAL_WEIGHTED_DELAY

    def test_per_employee_workload_matches_expectations(self) -> None:
        """Resource aggregation mirrors the hand-computed loads."""
        actions = task_assigning_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            task_assigning_goal(),
            actions,
        )
        assert plan_obj is not None
        usage_by_key = {u.key: u.total for u in plan_obj.metadata.csp.resource_usage}
        for employee_name, expected in OPTIMAL_WORKLOAD.items():
            assert usage_by_key[f"workload_{employee_name}"] == expected

    def test_graph_invocation_executes_plan(self) -> None:
        actions = task_assigning_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=task_assigning_goal(),
            world_state=task_assigning_start(),
        )
        assert result["status"] == "goal_achieved"
        for task in TASKS:
            assert result["world_state"][f"task_{task.name}_done"] is True


class TestTaskAssigningConstraintBuilder:
    """The fluent builder produces a functionally identical goal."""

    def test_fluent_goal_matches_hand_built(self) -> None:
        hand = task_assigning_goal()
        fluent = task_assigning_goal_fluent()

        assert hand.conditions == fluent.conditions
        assert hand.constraints == fluent.constraints
        # objectives dicts are MappingProxy — compare as plain dicts.
        assert dict(hand.objectives or {}) == dict(fluent.objectives or {})

    def test_fluent_goal_with_workload_cap_produces_one_constraint_per_employee(
        self,
    ) -> None:
        fluent = task_assigning_goal_fluent(max_workload_per_employee=8.0)
        keys = {c.key for c in fluent.constraints}
        assert keys == {f"workload_{e.name}" for e in EMPLOYEES}
        assert all(c.level == "soft" for c in fluent.constraints)
        assert all(c.max == 8.0 for c in fluent.constraints)

    def test_fluent_goal_solves_to_optimal(self) -> None:
        actions = task_assigning_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            task_assigning_goal_fluent(),
            actions,
        )
        assert plan_obj is not None
        assert set(plan_obj.action_names) == OPTIMAL_ASSIGNMENT
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.soft == -OPTIMAL_WEIGHTED_DELAY


class TestTaskAssigningSoftWorkloadCap:
    """Per-employee soft cap records a penalty without flipping status."""

    def test_soft_workload_cap_penalizes_overflow(self) -> None:
        """bob's 13h plan stays feasible but overflow is subtracted from soft."""
        actions = task_assigning_actions()
        goal = task_assigning_goal(
            max_workload_per_employee=8.0,
            max_workload_level="soft",
        )
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)

        # bob overflows the cap; alice and carol fit under.
        usage_by_key = {u.key: u for u in csp.resource_usage}
        assert usage_by_key["workload_alice"].satisfied is True
        assert usage_by_key["workload_bob"].satisfied is False
        assert usage_by_key["workload_carol"].satisfied is True

        # soft = -objective(57) - bob_overflow(13 - 8 = 5) = -62
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -62.0


class TestTaskAssigningHardWorkloadCap:
    """Per-employee hard cap the pipeline cannot meet → INFEASIBLE."""

    def test_hard_workload_cap_flips_status(self) -> None:
        actions = task_assigning_actions()
        goal = task_assigning_goal(
            max_workload_per_employee=8.0,
            max_workload_level="hard",
        )
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.metadata.csp is not None
        assert plan_obj.metadata.csp.status == CSPStatus.INFEASIBLE

        # hard records the total overflow (bob: 13 - 8 = 5).
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == -5.0


def _nl_interpreted_goal() -> InterpretedGoal:
    """NL goal shape used by both NL-intake tests below."""
    return InterpretedGoal(
        conditions={f"task_{t.name}_done": True for t in TASKS},
        constraints=[
            InterpretedConstraint(
                key="workload_alice",
                max=10.0,
                level="soft",
            ),
            InterpretedConstraint(
                key="workload_bob",
                max=10.0,
                level="soft",
            ),
            InterpretedConstraint(
                key="workload_carol",
                max=10.0,
                level="soft",
            ),
        ],
        objectives=[
            InterpretedObjective(metric="weighted_delay", direction="minimize"),
        ],
        reasoning=(
            "Assign every task; prefer the lowest weighted delay; "
            "'ideally' no one works more than 10 hours → soft cap."
        ),
    )


class TestTaskAssigningNaturalLanguage:
    """NL intake via FakeStructuredModel + GoalInterpreter."""

    def test_nl_request_produces_feasible_assignment(self) -> None:
        actions = task_assigning_actions()
        llm = FakeStructuredModel(response=_nl_interpreted_goal())
        interpreter = GoalInterpreter(llm=llm, actions=actions)

        goal = interpreter.interpret(
            "Route every ticket to the best-fit team member; "
            "ideally nobody carries more than ten hours of work."
        )
        plan_obj = pipeline_plan(
            PlanningState.from_dict(task_assigning_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert set(plan_obj.action_names) == OPTIMAL_ASSIGNMENT

    def test_invoke_nl_reaches_goal(self) -> None:
        actions = task_assigning_actions()
        llm = FakeStructuredModel(response=_nl_interpreted_goal())
        result = GoapGraph(actions=actions).invoke_nl(
            "Route every ticket to the best-fit team member; "
            "ideally nobody carries more than ten hours of work.",
            llm=llm,
            world_state=task_assigning_start(),
        )
        assert result["status"] == "goal_achieved"
        for task in TASKS:
            assert result["world_state"][f"task_{task.name}_done"] is True
