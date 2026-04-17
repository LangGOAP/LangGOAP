r"""Integration test for the Flexible Job Shop tutorial (Tier 3, notebook 15).

Final Tier 3 showcase test — exercises every major v0.1.0 subsystem
against a small, deterministic flexible job shop instance:

- A\* greedy-by-cost plan (all ``*_express`` modes).
- CSP validation: primary plan feasible under no cap, infeasible under
  a ``cost_usd=130`` hard cap.
- CSP enumeration + multi-plan optimization: with the hard cap the
  pipeline blacklists each action in the rejected plan and picks the
  single-blacklist alternative with the lowest ``cost_usd``.
- CSP temporal scheduling: 7-hour critical-path makespan for the
  greedy plan (three jobs run in parallel; ``gamma`` is the bottleneck).
- Hard / soft / infeasible score decomposition.
- Fluent :class:`~langgoap.constraints.ConstraintBuilder` structural
  parity with the hand-rolled goal factory.
- End-to-end :class:`~langgoap.graph.builder.GoapGraph` invocation
  with a :class:`~langgoap.tracing.LoggingTracer` and a
  :class:`~langgoap.history.StoreExecutionHistory` backed by
  :class:`~langgraph.store.memory.InMemoryStore`.
- Gantt visualization via :func:`~langgoap.viz.render_ascii_gantt`.

Expected values live in
``examples/tutorials/tutorial_examples/data/flexible_job_shop_instance.py``
so a single change to the instance surfaces here.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from langgraph.store.memory import InMemoryStore
from tutorial_examples.data.flexible_job_shop_instance import (
    ALT_PLAN_COST,
    ALT_PLAN_COST_USD,
    ALT_PLAN_DURATION_HOURS,
    ALT_PLAN_MAKESPAN_HOURS,
    GREEDY_COST_USD,
    GREEDY_DURATION_HOURS,
    GREEDY_MAKESPAN_HOURS,
    GREEDY_TOTAL_COST,
    HARD_COST_CAP,
    IMPOSSIBLE_HARD_CAP,
    JOBS,
    SOFT_CAP,
)
from tutorial_examples.flexible_job_shop import (
    action_name,
    done_key,
    flexible_job_shop_actions,
    flexible_job_shop_goal,
    flexible_job_shop_goal_fluent,
    flexible_job_shop_start,
)

from langgoap import (
    CSPStatus,
    GoapGraph,
    InterpretedConstraint,
    InterpretedGoal,
    InterpretedObjective,
    pipeline_plan,
)
from langgoap.history import StoreExecutionHistory
from langgoap.score import HardSoftScore
from langgoap.state import PlanningState
from langgoap.testing import FakeStructuredModel
from langgoap.tracing import LoggingTracer
from langgoap.viz import render_ascii_gantt

# ---------------------------------------------------------------------------
# Expected action-name constants — pinned so any rename of the factory's
# naming scheme surfaces in a single place.
# ---------------------------------------------------------------------------

GREEDY_ACTION_NAMES = {
    "prepare_alpha_express",
    "finalize_alpha_express",
    "prepare_beta_express",
    "finalize_beta_express",
    "prepare_gamma_express",
    "finalize_gamma_express",
}
ALT_ACTION_NAMES = {
    "prepare_alpha_express",
    "finalize_alpha_express",
    "prepare_beta_express",
    "finalize_beta_express",
    "prepare_gamma_standard",
    "finalize_gamma_express",
}
EXPECTED_ACTION_COUNT = 12  # 3 jobs x 2 ops x 2 modes
EXPECTED_STATE_FLAG_COUNT = 6  # 3 jobs x 2 ops


# ---------------------------------------------------------------------------
# Smoke tests on the factories
# ---------------------------------------------------------------------------


class TestTutorialHelpers:
    def test_action_catalog_has_twelve_actions(self) -> None:
        actions = flexible_job_shop_actions()
        assert len(actions) == EXPECTED_ACTION_COUNT

    def test_every_mode_combination_is_present(self) -> None:
        names = {a.name for a in flexible_job_shop_actions()}
        expected: set[str] = set()
        for job in JOBS:
            for operation in job.operations:
                for mode in operation.modes:
                    expected.add(action_name(job, operation, mode))
        assert names == expected

    def test_start_state_has_every_flag_false(self) -> None:
        start = flexible_job_shop_start()
        assert len(start) == EXPECTED_STATE_FLAG_COUNT
        assert all(v is False for v in start.values())

    def test_prepare_actions_have_no_preconditions(self) -> None:
        actions = {a.name: a for a in flexible_job_shop_actions()}
        for job in JOBS:
            prepare = job.operations[0]
            for mode in prepare.modes:
                spec = actions[action_name(job, prepare, mode)]
                assert done_key(job, prepare) in spec.preconditions
                # Only the "not yet done" self-flag — no prior-op guard.
                assert dict(spec.preconditions) == {done_key(job, prepare): False}

    def test_finalize_actions_require_prepare_done(self) -> None:
        actions = {a.name: a for a in flexible_job_shop_actions()}
        for job in JOBS:
            prepare = job.operations[0]
            finalize = job.operations[1]
            for mode in finalize.modes:
                spec = actions[action_name(job, finalize, mode)]
                assert dict(spec.preconditions) == {
                    done_key(job, finalize): False,
                    done_key(job, prepare): True,
                }

    def test_every_action_carries_cost_usd_and_duration_hours(self) -> None:
        for spec in flexible_job_shop_actions():
            assert spec.resources is not None
            assert "cost_usd" in spec.resources
            assert "duration_hours" in spec.resources
            assert spec.duration is not None
            # duration timedelta and duration_hours resource must agree.
            hours_from_timedelta = spec.duration.total_seconds() / 3600.0
            assert hours_from_timedelta == spec.resources["duration_hours"]


# ---------------------------------------------------------------------------
# Greedy (no-cap) pipeline: A* picks every *_express mode
# ---------------------------------------------------------------------------


class TestGreedyPlan:
    """With no ``cost_usd`` cap the pipeline routes through CSP for the
    MINIMIZE objective but finds the A*-optimal greedy plan feasible."""

    def test_greedy_plan_picks_every_express_mode(self) -> None:
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        assert set(plan_obj.action_names) == GREEDY_ACTION_NAMES
        assert plan_obj.total_cost == GREEDY_TOTAL_COST

    def test_greedy_csp_is_feasible_with_full_schedule(self) -> None:
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        # Six actions, six schedule entries.
        assert len(csp.schedule) == len(GREEDY_ACTION_NAMES)

    def test_greedy_makespan_matches_gamma_critical_path(self) -> None:
        """Three parallel pipelines → makespan = max per-job duration.

        alpha: 2+1 = 3h
        beta:  3+2 = 5h
        gamma: 4+3 = 7h  ← critical path
        """
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.makespan == timedelta(hours=GREEDY_MAKESPAN_HOURS)

    def test_greedy_schedule_respects_per_job_precedence(self) -> None:
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        by_name = {s.action_name: s for s in csp.schedule}
        for job in JOBS:
            prep_entry = by_name[f"prepare_{job.name}_express"]
            fin_entry = by_name[f"finalize_{job.name}_express"]
            # Finalize can start only once prepare is done.
            assert prep_entry.end <= fin_entry.start
            # Same job on a single track → prepare starts at t=0.
            assert prep_entry.start == timedelta(0)

    def test_greedy_jobs_run_in_parallel(self) -> None:
        """All three prepare_*_express actions start at t=0 because they
        have no shared precondition — this is the parallel payoff.
        """
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        by_name = {s.action_name: s for s in csp.schedule}
        for job in JOBS:
            entry = by_name[f"prepare_{job.name}_express"]
            assert entry.start == timedelta(0)

    def test_greedy_resource_totals_are_pinned(self) -> None:
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        usage = {u.key: u for u in csp.resource_usage}
        assert usage["cost_usd"].total == GREEDY_COST_USD
        assert usage["duration_hours"].total == GREEDY_DURATION_HOURS
        # No declared constraint → level is "info".
        assert usage["cost_usd"].level == "info"
        assert usage["duration_hours"].level == "info"

    def test_greedy_score_reflects_cost_usd_minimize(self) -> None:
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -GREEDY_COST_USD
        assert plan_obj.score.is_feasible() is True


# ---------------------------------------------------------------------------
# Hard cap → enumeration selects prepare_gamma_standard alternative
# ---------------------------------------------------------------------------


class TestHardCapEnumeration:
    """The ``HARD_COST_CAP`` is tuned so the greedy plan fails CSP
    validation and the pipeline's single-blacklist enumeration finds
    exactly one unambiguous best alternative: blacklist
    ``prepare_gamma_express`` → use ``prepare_gamma_standard``.

    This is the cleanest demonstration in the tutorial catalog of the
    ``planner/pipeline.py::enumerate_alternatives`` → ``optimize_plans``
    path running together.
    """

    def test_hard_cap_picks_prepare_gamma_standard_alternative(self) -> None:
        goal = flexible_job_shop_goal(max_cost_usd=HARD_COST_CAP, max_cost_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        assert set(plan_obj.action_names) == ALT_ACTION_NAMES
        # A* cost (sum of Mode.cost) for the alternative plan.
        assert plan_obj.total_cost == ALT_PLAN_COST

    def test_hard_cap_alternative_is_optimal(self) -> None:
        goal = flexible_job_shop_goal(max_cost_usd=HARD_COST_CAP, max_cost_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        # Multi-plan CP-SAT returns OPTIMAL when it finds the minimum.
        assert csp.status == CSPStatus.OPTIMAL

    def test_hard_cap_alternative_resource_totals_and_levels(self) -> None:
        goal = flexible_job_shop_goal(max_cost_usd=HARD_COST_CAP, max_cost_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        usage = {u.key: u for u in csp.resource_usage}
        # cost_usd is below the cap and declared hard.
        assert usage["cost_usd"].total == ALT_PLAN_COST_USD
        assert usage["cost_usd"].level == "hard"
        assert usage["cost_usd"].satisfied is True
        # duration_hours has no declared constraint → info level.
        assert usage["duration_hours"].total == ALT_PLAN_DURATION_HOURS
        assert usage["duration_hours"].level == "info"

    def test_hard_cap_alternative_score_is_feasible(self) -> None:
        goal = flexible_job_shop_goal(max_cost_usd=HARD_COST_CAP, max_cost_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -ALT_PLAN_COST_USD
        assert plan_obj.score.is_feasible() is True

    def test_alternative_plan_makespan_matches_gamma_critical_path(self) -> None:
        """The alternative plan's critical path is ``gamma_standard``.

        ``optimize_plans`` does not compute a temporal schedule — it
        only selects between candidate plans.  Re-plan with
        ``prepare_gamma_express`` blacklisted and the default
        (no-cap) goal so the pipeline routes through
        ``validate_plan()`` which runs CP-SAT temporal scheduling:

        - alpha: 2+1 = 3h
        - beta:  3+2 = 5h
        - gamma: 6+3 = 9h  ← new critical path (prepare_gamma_standard)

        Three parallel pipelines → makespan = max() = 9h.
        """
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
            blacklisted_actions=["prepare_gamma_express"],
        )
        assert plan_obj is not None
        assert set(plan_obj.action_names) == ALT_ACTION_NAMES
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.makespan == timedelta(hours=ALT_PLAN_MAKESPAN_HOURS)

    def test_hard_cap_alternative_beats_greedy_on_cost_usd(self) -> None:
        """CP-SAT's ``cost_usd → MINIMIZE`` objective breaks ties across
        feasible single-blacklist alternatives by picking the cheapest.

        At ``HARD_COST_CAP=130`` more than one single-blacklist
        alternative satisfies the cap (e.g. blacklisting
        ``finalize_gamma_express`` yields a $130 alternative that also
        clears the cap).  The ``optimize_plans`` multi-plan path uses
        the MINIMIZE objective as the tie-breaker and therefore
        deterministically returns the $125 plan produced by blacklisting
        ``prepare_gamma_express``.
        """
        goal = flexible_job_shop_goal(max_cost_usd=HARD_COST_CAP, max_cost_level="hard")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        usage_cost = next(u for u in csp.resource_usage if u.key == "cost_usd")
        assert usage_cost.total < GREEDY_COST_USD
        assert usage_cost.total == ALT_PLAN_COST_USD


# ---------------------------------------------------------------------------
# Infeasible hard cap — no single-blacklist rescues the plan
# ---------------------------------------------------------------------------


class TestInfeasibleHardCap:
    """A cap below every alternative's ``cost_usd`` aggregate → the
    pipeline returns the primary plan with ``INFEASIBLE`` status and
    a negative hard score equal to the greedy violation amount.
    """

    def test_impossible_cap_marks_plan_infeasible(self) -> None:
        goal = flexible_job_shop_goal(
            max_cost_usd=IMPOSSIBLE_HARD_CAP, max_cost_level="hard"
        )
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status == CSPStatus.INFEASIBLE

    def test_impossible_cap_hard_score_equals_violation_amount(self) -> None:
        goal = flexible_job_shop_goal(
            max_cost_usd=IMPOSSIBLE_HARD_CAP, max_cost_level="hard"
        )
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        assert isinstance(plan_obj.score, HardSoftScore)
        # Violation = GREEDY_COST_USD - IMPOSSIBLE_HARD_CAP → hard = -violation.
        assert plan_obj.score.hard == -(GREEDY_COST_USD - IMPOSSIBLE_HARD_CAP)
        assert plan_obj.score.is_feasible() is False


# ---------------------------------------------------------------------------
# Soft cap — violation stays, plan remains feasible
# ---------------------------------------------------------------------------


class TestSoftCap:
    """A soft cap below the greedy aggregate records a penalty in
    ``score.soft`` but leaves ``score.hard == 0`` and the plan feasible.
    This demonstrates the hard/soft split added in Epic 3.
    """

    def test_soft_cap_violation_penalizes_soft_score(self) -> None:
        goal = flexible_job_shop_goal(max_cost_usd=SOFT_CAP, max_cost_level="soft")
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            goal,
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        csp = plan_obj.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)

        usage_cost = next(u for u in csp.resource_usage if u.key == "cost_usd")
        assert usage_cost.level == "soft"
        assert usage_cost.satisfied is False
        assert usage_cost.total == GREEDY_COST_USD

        # soft = -objective(GREEDY_COST_USD) - overflow(GREEDY_COST_USD - SOFT_CAP)
        overflow = GREEDY_COST_USD - SOFT_CAP
        assert isinstance(plan_obj.score, HardSoftScore)
        assert plan_obj.score.hard == 0.0
        assert plan_obj.score.soft == -(GREEDY_COST_USD + overflow)
        assert plan_obj.score.is_feasible() is True


# ---------------------------------------------------------------------------
# Fluent ConstraintBuilder structural parity
# ---------------------------------------------------------------------------


class TestFluentBuilderParity:
    def test_hand_rolled_and_fluent_goals_are_structurally_equivalent(
        self,
    ) -> None:
        hand = flexible_job_shop_goal(max_cost_usd=HARD_COST_CAP, max_cost_level="hard")
        fluent = flexible_job_shop_goal_fluent(
            max_cost_usd=HARD_COST_CAP, max_cost_level="hard"
        )
        assert dict(hand.conditions) == dict(fluent.conditions)
        assert hand.constraints == fluent.constraints
        assert dict(hand.objectives or {}) == dict(fluent.objectives or {})

    def test_fluent_goal_produces_identical_plan(self) -> None:
        actions = flexible_job_shop_actions()
        start = PlanningState.from_dict(flexible_job_shop_start())

        hand_plan = pipeline_plan(
            start,
            flexible_job_shop_goal(max_cost_usd=HARD_COST_CAP, max_cost_level="hard"),
            actions,
        )
        fluent_plan = pipeline_plan(
            start,
            flexible_job_shop_goal_fluent(
                max_cost_usd=HARD_COST_CAP, max_cost_level="hard"
            ),
            actions,
        )
        assert hand_plan is not None and fluent_plan is not None
        assert set(hand_plan.action_names) == set(fluent_plan.action_names)
        assert hand_plan.total_cost == fluent_plan.total_cost
        assert hand_plan.score == fluent_plan.score


# ---------------------------------------------------------------------------
# ASCII Gantt visualization — the notebook's headline output
# ---------------------------------------------------------------------------


class TestGanttVisualization:
    def test_gantt_renders_every_action_bar(self) -> None:
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        chart = render_ascii_gantt(plan_obj, width=80)
        for name in GREEDY_ACTION_NAMES:
            assert name in chart

    def test_gantt_via_plan_method_matches_module_helper(self) -> None:
        plan_obj = pipeline_plan(
            PlanningState.from_dict(flexible_job_shop_start()),
            flexible_job_shop_goal(),
            flexible_job_shop_actions(),
        )
        assert plan_obj is not None
        module_chart = render_ascii_gantt(plan_obj)
        method_chart = plan_obj.visualize(format="ascii_gantt")
        assert isinstance(method_chart, str)
        assert module_chart == method_chart


# ---------------------------------------------------------------------------
# End-to-end execution via GoapGraph with tracer and history
# ---------------------------------------------------------------------------


class TestEndToEndExecution:
    def test_graph_invoke_reaches_goal_with_greedy_plan(self) -> None:
        graph = GoapGraph(actions=flexible_job_shop_actions())
        result = graph.invoke(
            goal=flexible_job_shop_goal(),
            world_state=flexible_job_shop_start(),
        )
        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        for job in JOBS:
            for operation in job.operations:
                assert ws[done_key(job, operation)] is True

    def test_graph_invoke_under_hard_cap_uses_alternative(self) -> None:
        graph = GoapGraph(actions=flexible_job_shop_actions())
        result = graph.invoke(
            goal=flexible_job_shop_goal(
                max_cost_usd=HARD_COST_CAP, max_cost_level="hard"
            ),
            world_state=flexible_job_shop_start(),
        )
        assert result["status"] == "goal_achieved"
        executed = [r.action_name for r in result["execution_history"]]
        # The alternative plan swaps in prepare_gamma_standard and
        # therefore prepare_gamma_express must NOT appear in the
        # executed history.
        assert "prepare_gamma_standard" in executed
        assert "prepare_gamma_express" not in executed

    def test_graph_invoke_with_tracer_and_history_records_one_execution(
        self,
    ) -> None:
        """Exercise tracer + history wiring end-to-end.

        The :class:`LoggingTracer` records domain events (plan start,
        plan complete, action complete, goal achieved) to the stdlib
        logger.  The :class:`StoreExecutionHistory` records a single
        :class:`ExecutionRecord` in the store's ``executions`` namespace
        once the loop finishes and the reverse index under the goal
        hash contains its UUID.
        """
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        tracer = LoggingTracer()
        graph = GoapGraph(
            actions=flexible_job_shop_actions(),
            tracer=tracer,
            history=history,
        )
        goal = flexible_job_shop_goal()
        result = graph.invoke(
            goal=goal,
            world_state=flexible_job_shop_start(),
        )
        assert result["status"] == "goal_achieved"

        # Reverse-index lookup — the goal hash must carry exactly one ID.
        goal_hash = history.goal_hash_for(goal)
        records = history.query_by_goal(goal_hash)
        assert len(records) == 1
        record = records[0]
        assert record.outcome == "success"
        assert set(record.plan_actions) == GREEDY_ACTION_NAMES
        # Expected cost is the A* total cost; actual matches because
        # every action in this tutorial is deterministic and succeeds
        # on the first try.
        assert record.expected_cost == GREEDY_TOTAL_COST
        assert record.actual_cost == GREEDY_TOTAL_COST
        assert record.replan_count == 0

    async def test_graph_ainvoke_reaches_goal_under_hard_cap(self) -> None:
        graph = GoapGraph(actions=flexible_job_shop_actions())
        result = await graph.ainvoke(
            goal=flexible_job_shop_goal(
                max_cost_usd=HARD_COST_CAP, max_cost_level="hard"
            ),
            world_state=flexible_job_shop_start(),
        )
        assert result["status"] == "goal_achieved"
        for job in JOBS:
            for operation in job.operations:
                assert result["world_state"][done_key(job, operation)] is True


# ---------------------------------------------------------------------------
# Natural-language goal intake via GoalInterpreter + invoke_nl
# ---------------------------------------------------------------------------


class TestNaturalLanguageIntake:
    """Exercise :meth:`GoapGraph.invoke_nl` with a
    :class:`FakeStructuredModel` so the interpreted goal matches the
    hand-rolled capped goal and the graph executes the same alternative
    plan.  This covers the ``NL intake`` capability the v0.1.0 plan
    requires the notebook to spotlight.
    """

    def _capped_interpreted_goal(self) -> InterpretedGoal:
        return InterpretedGoal(
            conditions={done_key(job, job.operations[-1]): True for job in JOBS},
            constraints=[
                InterpretedConstraint(
                    key="cost_usd",
                    max=HARD_COST_CAP,
                    min=None,
                    weight=1.0,
                    level="hard",
                ),
            ],
            objectives=[
                InterpretedObjective(metric="cost_usd", direction="minimize"),
            ],
            reasoning=(
                "The user wants every job finished at the lowest cost "
                "subject to a hard dollar budget."
            ),
        )

    def test_nl_goal_matches_hand_rolled_capped_goal(self) -> None:
        from langgoap import GoalInterpreter

        llm = FakeStructuredModel(response=self._capped_interpreted_goal())
        interpreter = GoalInterpreter(llm=llm, actions=flexible_job_shop_actions())
        nl_goal = interpreter.interpret(
            "Finish every job with cost_usd at or under $130 and minimize total cost."
        )
        hand_goal = flexible_job_shop_goal(
            max_cost_usd=HARD_COST_CAP, max_cost_level="hard"
        )
        assert dict(nl_goal.conditions) == dict(hand_goal.conditions)
        assert nl_goal.constraints == hand_goal.constraints
        assert dict(nl_goal.objectives or {}) == dict(hand_goal.objectives or {})

    def test_invoke_nl_runs_the_capped_alternative_plan(self) -> None:
        llm = FakeStructuredModel(response=self._capped_interpreted_goal())
        graph = GoapGraph(actions=flexible_job_shop_actions())
        result = graph.invoke_nl(
            "Finish every job with cost_usd at or under $130 and minimize total cost.",
            llm=llm,
            world_state=flexible_job_shop_start(),
        )
        assert result["status"] == "goal_achieved"
        executed = [r.action_name for r in result["execution_history"]]
        assert set(executed) == ALT_ACTION_NAMES
        assert "prepare_gamma_standard" in executed
        assert "prepare_gamma_express" not in executed
