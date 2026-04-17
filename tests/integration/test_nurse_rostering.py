r"""Integration test for the Nurse Rostering tutorial (Tier 2).

Exercises the full NL → GoalInterpreter → A\* → CSP → scoring
pipeline on a 4-nurse / 3-shift / 1-day instance derived from
standard benchmark data.

The test battery verifies:

- **Skill matching** — unqualified nurses have no action to take
  a shift (bob cannot take the triage-only morning).
- **Preference optimization** — A\* picks the happiest legal plan
  (total unhappiness 1 out of a possible 24).
- **HardSoftScore population** — the ``unhappiness → MINIMIZE``
  objective routes through CSP and the resulting score has
  ``hard == 0`` and ``soft == -1``.
- **Soft constraints** — lowering the ``unhappiness`` cap to 0 with
  ``level="soft"`` keeps the plan feasible but decreases the soft
  score by the violation amount.
- **Hard infeasibility** — the same cap at ``level="hard"`` flips
  CSP status to INFEASIBLE.
- **NL intake** — a ``FakeStructuredModel`` round-trips the NL
  request through the interpreter and the resulting plan covers
  every shift.

Helpers live in
``examples/tutorials/tutorial_examples/nurse_rostering.py`` and the
instance fixture is at
``examples/tutorials/tutorial_examples/data/nurse_rostering_instance.py``
(derived from standard benchmark data).
"""

from __future__ import annotations

import pytest
from tutorial_examples.data.nurse_rostering_instance import (
    NURSES,
    SHIFTS,
)
from tutorial_examples.nurse_rostering import (
    nurse_rostering_actions,
    nurse_rostering_goal,
    nurse_rostering_start,
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

# The optimal assignment by preference cost:
#   carol→morning(1)  +  alice→afternoon(0)  +  dave→night(0)
# total unhappiness = 1
OPTIMAL_ASSIGNMENTS = {
    "assign_carol_to_morning",
    "assign_alice_to_afternoon",
    "assign_dave_to_night",
}
OPTIMAL_UNHAPPINESS = 1.0


class TestNurseRosteringSkillFilter:
    """Skill matching drops unqualified nurses from the action catalog."""

    def test_skill_filter_removes_unqualified_assignments(self) -> None:
        """Bob / carol / dave have no triage → no morning action.

        The expected legal action count is 9: alice has all three
        skills (3 actions), bob has general+pediatrics (2), carol has
        triage+general (2), dave has general+pediatrics (2).
        """
        actions = nurse_rostering_actions()
        names = {a.name for a in actions}

        # Bob cannot do morning (no triage).
        assert "assign_bob_to_morning" not in names
        # Carol cannot do night (no pediatrics).
        assert "assign_carol_to_night" not in names
        # Dave cannot do morning (no triage).
        assert "assign_dave_to_morning" not in names
        # Alice is fully qualified.
        assert "assign_alice_to_morning" in names
        assert "assign_alice_to_afternoon" in names
        assert "assign_alice_to_night" in names

        assert len(actions) == 9


class TestNurseRosteringOptimal:
    """Happy-path pipeline: A* + CSP find the minimum-unhappiness roster."""

    def test_pipeline_finds_happiest_assignment(self) -> None:
        """A\\* picks the 3 actions with the lowest aggregate preference cost."""
        actions = nurse_rostering_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(nurse_rostering_start()),
            nurse_rostering_goal(),
            actions,
        )

        assert plan_obj is not None
        assert set(plan_obj.action_names) == OPTIMAL_ASSIGNMENTS
        assert plan_obj.total_cost == OPTIMAL_UNHAPPINESS

    def test_hardsoft_score_reflects_objective(self) -> None:
        """HardSoftScore.soft == -unhappiness when no constraints are violated."""
        actions = nurse_rostering_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(nurse_rostering_start()),
            nurse_rostering_goal(),
            actions,
        )
        assert plan_obj is not None
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -OPTIMAL_UNHAPPINESS

        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        assert csp.objective_values["unhappiness"] == OPTIMAL_UNHAPPINESS

    def test_every_shift_covered_exactly_once(self) -> None:
        """Each of the 3 shifts is assigned to exactly one nurse."""
        actions = nurse_rostering_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(nurse_rostering_start()),
            nurse_rostering_goal(),
            actions,
        )
        assert plan_obj is not None

        covered = {s.name: 0 for s in SHIFTS}
        for action in plan_obj.actions:
            for shift in SHIFTS:
                if action.name.endswith(f"_to_{shift.name}"):
                    covered[shift.name] += 1
        assert all(count == 1 for count in covered.values())

    def test_no_nurse_works_two_shifts(self) -> None:
        """The ``nurse_<n>_available`` precondition prevents double-booking."""
        actions = nurse_rostering_actions()
        plan_obj = pipeline_plan(
            PlanningState.from_dict(nurse_rostering_start()),
            nurse_rostering_goal(),
            actions,
        )
        assert plan_obj is not None

        assigned_nurses: list[str] = []
        for action in plan_obj.actions:
            for nurse in NURSES:
                if action.name.startswith(f"assign_{nurse.name}_"):
                    assigned_nurses.append(nurse.name)
        assert len(assigned_nurses) == len(set(assigned_nurses))

    def test_graph_invocation_executes_plan(self) -> None:
        """End-to-end GoapGraph.invoke() reaches goal_achieved."""
        actions = nurse_rostering_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=nurse_rostering_goal(),
            world_state=nurse_rostering_start(),
        )
        assert result["status"] == "goal_achieved"

        ws = result["world_state"]
        for shift in SHIFTS:
            assert ws[f"shift_{shift.name}_covered"] is True


class TestNurseRosteringSoftConstraint:
    """Soft cap on unhappiness stays feasible but subtracts a penalty."""

    def test_soft_violation_is_feasible_with_penalty(self) -> None:
        """Tightening the cap to 0 with level='soft' keeps the plan feasible."""
        actions = nurse_rostering_actions()
        goal = nurse_rostering_goal(max_unhappiness=0.0, max_unhappiness_level="soft")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(nurse_rostering_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.metadata.csp is not None
        assert plan_obj.metadata.csp.status in (
            CSPStatus.FEASIBLE,
            CSPStatus.OPTIMAL,
        )

        # hard == 0 (feasible); soft = -objective(1) - penalty(1) = -2
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -2.0

        # The usage entry should carry level='soft'.
        usage = next(
            u for u in plan_obj.metadata.csp.resource_usage if u.key == "unhappiness"
        )
        assert usage.level == "soft"
        assert usage.satisfied is False
        assert usage.total == OPTIMAL_UNHAPPINESS


class TestNurseRosteringHardInfeasible:
    """Hard cap the planner cannot meet → INFEASIBLE status."""

    def test_impossible_hard_cap(self) -> None:
        """Capping unhappiness at 0 with level='hard' yields INFEASIBLE."""
        actions = nurse_rostering_actions()
        goal = nurse_rostering_goal(max_unhappiness=0.0, max_unhappiness_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(nurse_rostering_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.metadata.csp is not None
        assert plan_obj.metadata.csp.status == CSPStatus.INFEASIBLE

        # hard records the shortfall; soft still tracks the objective.
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == -OPTIMAL_UNHAPPINESS


def _nl_interpreted_goal() -> InterpretedGoal:
    """NL goal shape used by both NL-intake tests below."""
    return InterpretedGoal(
        conditions={
            "shift_morning_covered": True,
            "shift_afternoon_covered": True,
            "shift_night_covered": True,
        },
        constraints=[
            InterpretedConstraint(
                key="unhappiness",
                max=3.0,
                level="soft",
            )
        ],
        objectives=[InterpretedObjective(metric="unhappiness", direction="minimize")],
        reasoning=("Cover every shift, prefer low unhappiness; 'ideally' → soft cap."),
    )


class TestNurseRosteringNaturalLanguage:
    """NL intake via FakeStructuredModel + GoalInterpreter."""

    def test_nl_request_produces_feasible_roster(self) -> None:
        """A fake LLM maps the NL request to a feasible GoalSpec."""
        actions = nurse_rostering_actions()
        llm = FakeStructuredModel(response=_nl_interpreted_goal())
        interpreter = GoalInterpreter(llm=llm, actions=actions)

        goal = interpreter.interpret(
            "Cover every shift for the day; ideally keep " "total unhappiness under 3."
        )

        # The NL constraint (max=3 soft) does not flip status: the
        # optimal plan has unhappiness=1 which is under the cap.
        plan_obj = pipeline_plan(
            PlanningState.from_dict(nurse_rostering_start()),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.metadata.csp is not None
        assert plan_obj.metadata.csp.status in (
            CSPStatus.FEASIBLE,
            CSPStatus.OPTIMAL,
        )
        assert set(plan_obj.action_names) == OPTIMAL_ASSIGNMENTS

    def test_invoke_nl_reaches_goal(self) -> None:
        """GoapGraph.invoke_nl() executes the NL-derived plan."""
        actions = nurse_rostering_actions()
        llm = FakeStructuredModel(response=_nl_interpreted_goal())

        result = GoapGraph(actions=actions).invoke_nl(
            "Cover every shift for the day; ideally keep " "total unhappiness under 3.",
            llm=llm,
            world_state=nurse_rostering_start(),
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        for shift in SHIFTS:
            assert ws[f"shift_{shift.name}_covered"] is True
