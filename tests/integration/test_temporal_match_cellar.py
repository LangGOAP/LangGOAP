r"""Integration test for the temporal match cellar tutorial (Tier 3).

Exercises the A\* → CSP pipeline on a durative-action domain and pins
every schedule entry, makespan, and score contribution.  Unlike the
unified-planning original this domain does **not** use "over-all"
interval containment — the hand mutex is expressed as a precondition
chain (``mend_fuse_N`` depends on ``fuse_(N-1)_mended``), which is the
shape LangGoap's dependency-graph-based scheduler can honor.

The headline assertions are structural:

- ``plan.metadata.csp.schedule`` contains exactly six entries with the
  exact ``start`` and ``end`` timedeltas the CP-SAT scheduler returns
  for this deterministic domain.
- ``plan.metadata.csp.makespan == timedelta(seconds=21)``.
- The three ``light_match_*`` actions share the exact same start and
  end times (parallel execution at ``t=0..6``).
- The three ``mend_fuse_*`` actions form a strict sequence with zero
  gaps (``t=6..11``, ``t=11..16``, ``t=16..21``).
- The ``HardSoftScore`` encodes ``hard=0.0`` and
  ``soft=-33.0`` — the negated ``duration_seconds`` total (6+6+6+5+5+5).
- Hand-rolled and fluent-builder goal variants yield identical
  :class:`GoalSpec` instances and identical plans.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tutorial_examples.temporal_match_cellar import (
    LIGHT_DURATION_SECONDS,
    MEND_DURATION_SECONDS,
    NUM_PAIRS,
    match_cellar_actions,
    match_cellar_goal,
    match_cellar_goal_fluent,
    match_cellar_start,
)

from langgoap import CSPStatus, GoapGraph
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.score import HardSoftScore
from langgoap.state import PlanningState
from langgoap.viz import render_ascii_gantt

# ---------------------------------------------------------------------------
# Pinned totals — derived from the domain constants, *not* duplicated.
# ---------------------------------------------------------------------------

EXPECTED_ACTION_COUNT = 2 * NUM_PAIRS  # 3 lights + 3 mends = 6
EXPECTED_TOTAL_COST = float(EXPECTED_ACTION_COUNT)  # cost=1.0 per action
EXPECTED_DURATION_TOTAL_SECONDS = NUM_PAIRS * LIGHT_DURATION_SECONDS + NUM_PAIRS * (
    MEND_DURATION_SECONDS
)  # 33.0
EXPECTED_MAKESPAN = timedelta(
    seconds=LIGHT_DURATION_SECONDS + NUM_PAIRS * MEND_DURATION_SECONDS
)  # 6 + 15 = 21s

LIGHT_START = timedelta(seconds=0)
LIGHT_END = timedelta(seconds=LIGHT_DURATION_SECONDS)  # 6s
MEND_1_START = LIGHT_END  # 6s
MEND_1_END = MEND_1_START + timedelta(seconds=MEND_DURATION_SECONDS)  # 11s
MEND_2_START = MEND_1_END  # 11s
MEND_2_END = MEND_2_START + timedelta(seconds=MEND_DURATION_SECONDS)  # 16s
MEND_3_START = MEND_2_END  # 16s
MEND_3_END = MEND_3_START + timedelta(seconds=MEND_DURATION_SECONDS)  # 21s


# ---------------------------------------------------------------------------
# Smoke tests on the pure factories
# ---------------------------------------------------------------------------


class TestTutorialHelpers:
    def test_start_state_has_every_flag_false(self) -> None:
        start = match_cellar_start()
        assert len(start) == 3 * NUM_PAIRS  # light_ready, match_used, fuse_mended
        assert all(v is False for v in start.values())

    def test_action_catalog_size_and_names(self) -> None:
        actions = match_cellar_actions()
        assert len(actions) == EXPECTED_ACTION_COUNT
        names = [a.name for a in actions]
        assert names == [
            "light_match_1",
            "light_match_2",
            "light_match_3",
            "mend_fuse_1",
            "mend_fuse_2",
            "mend_fuse_3",
        ]

    def test_light_actions_have_no_preconditions(self) -> None:
        actions = {a.name: a for a in match_cellar_actions()}
        for i in range(1, NUM_PAIRS + 1):
            light = actions[f"light_match_{i}"]
            assert dict(light.preconditions) == {}
            assert dict(light.effects) == {
                f"light_{i}_ready": True,
                f"match_{i}_used": True,
            }
            assert light.duration == timedelta(seconds=LIGHT_DURATION_SECONDS)

    def test_mend_chain_wires_hand_mutex_via_preconditions(self) -> None:
        actions = {a.name: a for a in match_cellar_actions()}
        mend_1 = actions["mend_fuse_1"]
        mend_2 = actions["mend_fuse_2"]
        mend_3 = actions["mend_fuse_3"]

        assert dict(mend_1.preconditions) == {"light_1_ready": True}
        assert dict(mend_2.preconditions) == {
            "light_2_ready": True,
            "fuse_1_mended": True,
        }
        assert dict(mend_3.preconditions) == {
            "light_3_ready": True,
            "fuse_2_mended": True,
        }
        for m in (mend_1, mend_2, mend_3):
            assert m.duration == timedelta(seconds=MEND_DURATION_SECONDS)

    def test_every_action_carries_a_duration_seconds_resource(self) -> None:
        # ``duration_seconds`` is the resource the goal's MINIMIZE
        # objective aggregates; it must be populated on every action
        # or the soft score drifts.
        total = 0.0
        for a in match_cellar_actions():
            assert a.resources is not None
            assert "duration_seconds" in a.resources
            total += a.resources["duration_seconds"]
        assert total == EXPECTED_DURATION_TOTAL_SECONDS


# ---------------------------------------------------------------------------
# Core pipeline: A* + CSP schedule
# ---------------------------------------------------------------------------


class TestPipelineSchedule:
    """The pipeline runs A\\*, routes through CSP because the goal has
    a ``duration_seconds`` MINIMIZE objective, and the scheduler
    computes the exact staircase schedule the docstring table pins.
    """

    def test_primary_plan_has_all_six_actions(self) -> None:
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        assert len(plan.actions) == EXPECTED_ACTION_COUNT
        assert set(plan.action_names) == {
            "light_match_1",
            "light_match_2",
            "light_match_3",
            "mend_fuse_1",
            "mend_fuse_2",
            "mend_fuse_3",
        }
        assert plan.total_cost == EXPECTED_TOTAL_COST

    def test_csp_metadata_is_feasible_with_full_schedule(self) -> None:
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        csp = plan.metadata.csp
        assert csp is not None
        assert csp.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        # Schedule must cover every action in the plan.
        assert len(csp.schedule) == EXPECTED_ACTION_COUNT
        assert csp.makespan == EXPECTED_MAKESPAN

    def test_all_three_lights_run_in_parallel_at_time_zero(self) -> None:
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        csp = plan.metadata.csp
        assert csp is not None
        by_name = {s.action_name: s for s in csp.schedule}
        for i in range(1, NUM_PAIRS + 1):
            entry = by_name[f"light_match_{i}"]
            assert entry.start == LIGHT_START
            assert entry.end == LIGHT_END
            assert entry.duration == timedelta(seconds=LIGHT_DURATION_SECONDS)

    def test_mends_form_sequential_chain_with_zero_gaps(self) -> None:
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        csp = plan.metadata.csp
        assert csp is not None
        by_name = {s.action_name: s for s in csp.schedule}

        mend_1 = by_name["mend_fuse_1"]
        mend_2 = by_name["mend_fuse_2"]
        mend_3 = by_name["mend_fuse_3"]

        assert mend_1.start == MEND_1_START
        assert mend_1.end == MEND_1_END
        assert mend_2.start == MEND_2_START
        assert mend_2.end == MEND_2_END
        assert mend_3.start == MEND_3_START
        assert mend_3.end == MEND_3_END
        # Zero gap between consecutive mends.
        assert mend_1.end == mend_2.start
        assert mend_2.end == mend_3.start

    def test_makespan_beats_serial_execution_by_twelve_seconds(self) -> None:
        # The core demonstration: parallel lights save exactly the time
        # the two redundant lights would have cost if they'd run serially.
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        csp = plan.metadata.csp
        assert csp is not None
        serial = timedelta(seconds=EXPECTED_DURATION_TOTAL_SECONDS)  # 33s
        assert csp.makespan is not None
        saved = serial - csp.makespan
        assert saved == timedelta(seconds=12)


# ---------------------------------------------------------------------------
# Score decomposition: the MINIMIZE objective becomes -duration_seconds
# ---------------------------------------------------------------------------


class TestScoreDecomposition:
    def test_hardsoft_score_matches_minimize_formula(self) -> None:
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        assert isinstance(plan.score, HardSoftScore)
        # No constraints → hard is exactly zero.
        assert plan.score.hard == 0.0
        # MINIMIZE duration_seconds → soft = -total_duration_seconds.
        assert plan.score.soft == -EXPECTED_DURATION_TOTAL_SECONDS
        assert plan.score.is_feasible() is True

    def test_objective_values_map_reports_total_duration(self) -> None:
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        csp = plan.metadata.csp
        assert csp is not None
        assert csp.objective_values.get("duration_seconds") == (
            EXPECTED_DURATION_TOTAL_SECONDS
        )


# ---------------------------------------------------------------------------
# Fluent builder parity
# ---------------------------------------------------------------------------


class TestFluentBuilderParity:
    def test_hand_rolled_and_fluent_goals_are_structurally_equivalent(
        self,
    ) -> None:
        hand = match_cellar_goal()
        fluent = match_cellar_goal_fluent()
        assert dict(hand.conditions) == dict(fluent.conditions)
        assert hand.constraints == fluent.constraints  # both empty tuples
        assert dict(hand.objectives or {}) == dict(fluent.objectives or {})

    def test_fluent_goal_produces_identical_plan(self) -> None:
        actions = match_cellar_actions()
        start = PlanningState.from_dict(match_cellar_start())

        hand_plan = pipeline_plan(start, match_cellar_goal(), actions)
        fluent_plan = pipeline_plan(start, match_cellar_goal_fluent(), actions)

        assert hand_plan is not None and fluent_plan is not None
        assert set(hand_plan.action_names) == set(fluent_plan.action_names)
        assert hand_plan.total_cost == fluent_plan.total_cost
        assert hand_plan.score == fluent_plan.score
        # Both plans must produce the same makespan.
        assert hand_plan.metadata.csp is not None
        assert fluent_plan.metadata.csp is not None
        assert hand_plan.metadata.csp.makespan == fluent_plan.metadata.csp.makespan


# ---------------------------------------------------------------------------
# Variable pair count — the domain scales
# ---------------------------------------------------------------------------


class TestTwoPairInstance:
    """Reducing the instance to two pairs must shrink the makespan
    proportionally: ``6 + 2*5 = 16`` seconds.  This guards against a
    silent regression where the scheduler ignores ``num_pairs``.
    """

    def test_two_pair_makespan_is_sixteen_seconds(self) -> None:
        actions = match_cellar_actions(num_pairs=2)
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start(num_pairs=2)),
            match_cellar_goal(num_pairs=2),
            actions,
        )
        assert plan is not None
        csp = plan.metadata.csp
        assert csp is not None
        assert len(csp.schedule) == 4
        assert csp.makespan == timedelta(seconds=16)

    def test_two_pair_plan_has_exactly_four_actions(self) -> None:
        actions = match_cellar_actions(num_pairs=2)
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start(num_pairs=2)),
            match_cellar_goal(num_pairs=2),
            actions,
        )
        assert plan is not None
        assert set(plan.action_names) == {
            "light_match_1",
            "light_match_2",
            "mend_fuse_1",
            "mend_fuse_2",
        }
        # Two actions each for light+mend → total cost 4.0.
        assert plan.total_cost == 4.0


# ---------------------------------------------------------------------------
# ASCII Gantt viz — the notebook's headline output must not crash
# ---------------------------------------------------------------------------


class TestGanttVisualization:
    def test_render_ascii_gantt_produces_every_action_bar(self) -> None:
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        chart = render_ascii_gantt(plan, width=60)
        # Every action name appears on its own row.
        for name in (
            "light_match_1",
            "light_match_2",
            "light_match_3",
            "mend_fuse_1",
            "mend_fuse_2",
            "mend_fuse_3",
        ):
            assert name in chart
        # The 21s makespan is labelled on the header row.
        assert "0..21s" in chart

    def test_render_ascii_gantt_from_plan_method_matches_module_function(
        self,
    ) -> None:
        # The ``Plan.visualize(format="ascii_gantt")`` method is the
        # public surface the notebook uses.  It must agree with the
        # module-level helper byte-for-byte.
        actions = match_cellar_actions()
        plan = pipeline_plan(
            PlanningState.from_dict(match_cellar_start()),
            match_cellar_goal(),
            actions,
        )
        assert plan is not None
        module_chart = render_ascii_gantt(plan)
        method_chart = plan.visualize(format="ascii_gantt")
        assert isinstance(method_chart, str)
        assert module_chart == method_chart


# ---------------------------------------------------------------------------
# End-to-end execution via GoapGraph
# ---------------------------------------------------------------------------


class TestEndToEndExecution:
    def test_graph_invoke_mends_every_fuse(self) -> None:
        graph = GoapGraph(match_cellar_actions())
        result = graph.invoke(
            goal=match_cellar_goal(), world_state=match_cellar_start()
        )
        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        for i in range(1, NUM_PAIRS + 1):
            assert ws[f"fuse_{i}_mended"] is True
            assert ws[f"light_{i}_ready"] is True
            assert ws[f"match_{i}_used"] is True

    def test_execution_sequence_respects_mend_chain_order(self) -> None:
        graph = GoapGraph(match_cellar_actions())
        result = graph.invoke(
            goal=match_cellar_goal(), world_state=match_cellar_start()
        )
        sequence = [r.action_name for r in result["execution_history"]]
        # mend_fuse_1 must precede mend_fuse_2 which must precede mend_fuse_3.
        idx1 = sequence.index("mend_fuse_1")
        idx2 = sequence.index("mend_fuse_2")
        idx3 = sequence.index("mend_fuse_3")
        assert idx1 < idx2 < idx3
        # Each mend must be preceded by its matching light.
        for i in range(1, NUM_PAIRS + 1):
            assert sequence.index(f"light_match_{i}") < sequence.index(f"mend_fuse_{i}")

    @pytest.mark.asyncio
    async def test_graph_ainvoke_mends_every_fuse(self) -> None:
        graph = GoapGraph(match_cellar_actions())
        result = await graph.ainvoke(
            goal=match_cellar_goal(), world_state=match_cellar_start()
        )
        assert result["status"] == "goal_achieved"
        for i in range(1, NUM_PAIRS + 1):
            assert result["world_state"][f"fuse_{i}_mended"] is True
