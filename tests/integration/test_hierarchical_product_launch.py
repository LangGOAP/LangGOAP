r"""Integration test for the hierarchical product launch tutorial (Tier 3).

Exercises :class:`~langgoap.MultiGoal` sequential decomposition through
:class:`~langgoap.GoapGraph`.  The test battery pins the four features
the notebook showcases:

- **Sequential chaining** — three sub-goals execute in order in a
  single ``graph.invoke`` call and the world state hands off between
  stages.
- **Per-stage planning isolation** — each sub-goal's plan contains only
  the actions needed to satisfy *that* stage, not the actions from
  later stages.
- **Pre-satisfied sub-goal advance** — starting with a stage already
  satisfied causes the observer to skip straight to the next stage.
- **Per-sub-goal accounting reset** — a transient failure in stage 2
  does not reduce stage 3's replan budget or leak its blacklist.

All assertions use exact equality wherever possible — weakened ``>=``
comparisons are flagged in the audit checklist as a structural test
smell.  The tracer recordings pin the exact sub-goal keys, plan
action tuples, and hook counts.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import (
    ActionSpec,
    GoalPolicy,
    GoalSpec,
    GoapGraph,
    MultiGoal,
    NullTracer,
)

# ---------------------------------------------------------------------------
# Deterministic stubs for GOAP mechanics tests.
# These mirror the original tutorial_examples functions but are self-contained
# so the tutorial module can evolve to require a real LLM without breaking
# planning-focused tests.
# ---------------------------------------------------------------------------


def _research_market(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"market_data": True}


def _write_prd(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"prd_approved": True}


def _implement_features(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"code_written": True}


def _qa_test(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"qa_passed": True}


def _prepare_marketing(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"marketing_ready": True}


def _announce_launch(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"launched": True}


def product_launch_actions() -> list[ActionSpec]:
    """Six-action catalog spanning all three launch stages (deterministic stubs)."""
    return [
        ActionSpec(
            name="research_market",
            preconditions={},
            effects={"market_data": True},
            cost=2.0,
            execute=_research_market,
        ),
        ActionSpec(
            name="write_prd",
            preconditions={"market_data": True},
            effects={"prd_approved": True},
            cost=3.0,
            execute=_write_prd,
        ),
        ActionSpec(
            name="implement_features",
            preconditions={"prd_approved": True},
            effects={"code_written": True},
            cost=5.0,
            execute=_implement_features,
        ),
        ActionSpec(
            name="qa_test",
            preconditions={"code_written": True},
            effects={"qa_passed": True},
            cost=2.0,
            execute=_qa_test,
        ),
        ActionSpec(
            name="prepare_marketing",
            preconditions={"qa_passed": True},
            effects={"marketing_ready": True},
            cost=2.0,
            execute=_prepare_marketing,
        ),
        ActionSpec(
            name="announce_launch",
            preconditions={"marketing_ready": True, "qa_passed": True},
            effects={"launched": True},
            cost=1.0,
            execute=_announce_launch,
        ),
    ]


def product_launch_start() -> dict[str, Any]:
    """Clean-slate world state: no milestones reached yet."""
    return {
        "market_data": False,
        "prd_approved": False,
        "code_written": False,
        "qa_passed": False,
        "marketing_ready": False,
        "launched": False,
    }


def product_launch_goal() -> MultiGoal:
    """Three-stage sequential MultiGoal for the launch."""
    return MultiGoal(
        goals=(
            GoalSpec(conditions={"prd_approved": True}),
            GoalSpec(conditions={"qa_passed": True}),
            GoalSpec(conditions={"launched": True}),
        ),
        mode="sequential",
    )


# ---------------------------------------------------------------------------
# Recording tracer — used by TestTracerObservability
# ---------------------------------------------------------------------------


class _RecordingTracer(NullTracer):
    """Capture plan/replan/goal-achieved events for the launch loop.

    Each recorded entry pins *what* the planner saw (sub-goal keys,
    plan action tuples) rather than *how many* events fired.  This is
    the structural-test shape the notebook 11 audit demanded: assert
    on semantics, not on counts alone.
    """

    def __init__(self) -> None:
        self.plan_starts: list[tuple[str, ...]] = []
        self.plan_completes: list[tuple[str, ...]] = []
        self.replans: list[tuple[str, tuple[str, ...]]] = []
        self.goal_achieved_states: list[dict[str, Any]] = []

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        # Key-set of the effective (sub-)goal being planned.
        self.plan_starts.append(tuple(sorted(goal.conditions.keys())))

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.plan_completes.append(tuple(plan.action_names))

    def on_replan(self, reason: str, new_plan: Any) -> None:
        names = tuple(new_plan.action_names) if new_plan is not None else ()
        self.replans.append((reason, names))

    def on_goal_achieved(self, final_state: Any) -> None:
        self.goal_achieved_states.append(dict(final_state))


# ---------------------------------------------------------------------------
# Tutorial helpers: smoke-test the pure factories before the GOAP loop
# ---------------------------------------------------------------------------


class TestTutorialHelpers:
    def test_start_state_has_all_flags_false(self) -> None:
        start = product_launch_start()
        assert start == {
            "market_data": False,
            "prd_approved": False,
            "code_written": False,
            "qa_passed": False,
            "marketing_ready": False,
            "launched": False,
        }

    def test_action_catalog_has_expected_names(self) -> None:
        names = tuple(a.name for a in product_launch_actions())
        assert names == (
            "research_market",
            "write_prd",
            "implement_features",
            "qa_test",
            "prepare_marketing",
            "announce_launch",
        )

    def test_goal_is_three_stage_sequential_multigoal(self) -> None:
        goal = product_launch_goal()
        assert isinstance(goal, MultiGoal)
        assert goal.mode == "sequential"
        assert len(goal.goals) == 3
        assert tuple(dict(g.conditions) for g in goal.goals) == (
            {"prd_approved": True},
            {"qa_passed": True},
            {"launched": True},
        )


# ---------------------------------------------------------------------------
# Happy path: all three stages run in order
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_three_stage_sequential_launch(self) -> None:
        graph = GoapGraph(product_launch_actions())
        result = graph.invoke(
            goal=product_launch_goal(),
            world_state=product_launch_start(),
        )
        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        # Every stage's effects present in the final world state:
        assert ws["market_data"] is True  # stage 1 intermediate
        assert ws["prd_approved"] is True  # stage 1 gate
        assert ws["code_written"] is True  # stage 2 intermediate
        assert ws["qa_passed"] is True  # stage 2 gate
        assert ws["marketing_ready"] is True  # stage 3 intermediate
        assert ws["launched"] is True  # stage 3 gate

    def test_full_action_sequence_in_history(self) -> None:
        graph = GoapGraph(product_launch_actions())
        result = graph.invoke(
            goal=product_launch_goal(),
            world_state=product_launch_start(),
        )
        sequence = [r.action_name for r in result["execution_history"]]
        assert sequence == [
            "research_market",
            "write_prd",
            "implement_features",
            "qa_test",
            "prepare_marketing",
            "announce_launch",
        ]

    def test_final_subgoal_index_points_past_last_stage(self) -> None:
        # After the third and final sub-goal completes the observer
        # routes to END; ``current_subgoal_index`` stays at the last
        # stage index (2) because the advance branch is never taken.
        graph = GoapGraph(product_launch_actions())
        result = graph.invoke(
            goal=product_launch_goal(),
            world_state=product_launch_start(),
        )
        assert result["current_subgoal_index"] == 2

    @pytest.mark.asyncio
    async def test_async_three_stage_launch(self) -> None:
        graph = GoapGraph(product_launch_actions())
        result = await graph.ainvoke(
            goal=product_launch_goal(),
            world_state=product_launch_start(),
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["launched"] is True
        # Execution history contains every action.
        sequence = [r.action_name for r in result["execution_history"]]
        assert sequence == [
            "research_market",
            "write_prd",
            "implement_features",
            "qa_test",
            "prepare_marketing",
            "announce_launch",
        ]


# ---------------------------------------------------------------------------
# Stage advance: pre-satisfied stages skip
# ---------------------------------------------------------------------------


class TestStageAdvance:
    def test_pre_satisfied_first_stage_skips_to_second(self) -> None:
        graph = GoapGraph(product_launch_actions())
        start = product_launch_start()
        start["prd_approved"] = True  # stage 1 already satisfied
        result = graph.invoke(goal=product_launch_goal(), world_state=start)
        assert result["status"] == "goal_achieved"
        history = [r.action_name for r in result["execution_history"]]
        # Neither of stage 1's actions should run.
        assert "research_market" not in history
        assert "write_prd" not in history
        # Stages 2 and 3 still execute in full.
        assert history == [
            "implement_features",
            "qa_test",
            "prepare_marketing",
            "announce_launch",
        ]

    def test_pre_satisfied_first_two_stages_only_runs_stage_three(self) -> None:
        graph = GoapGraph(product_launch_actions())
        start = product_launch_start()
        start["prd_approved"] = True
        start["qa_passed"] = True
        result = graph.invoke(goal=product_launch_goal(), world_state=start)
        assert result["status"] == "goal_achieved"
        history = [r.action_name for r in result["execution_history"]]
        assert history == ["prepare_marketing", "announce_launch"]


# ---------------------------------------------------------------------------
# Tracer observability: one plan_start/plan_complete per sub-goal
# ---------------------------------------------------------------------------


class TestTracerObservability:
    def test_tracer_fires_one_plan_start_per_subgoal(self) -> None:
        tracer = _RecordingTracer()
        graph = GoapGraph(product_launch_actions(), tracer=tracer)
        result = graph.invoke(
            goal=product_launch_goal(),
            world_state=product_launch_start(),
        )
        assert result["status"] == "goal_achieved"
        # Exactly one plan_start per sub-goal, in stage order.
        assert tracer.plan_starts == [
            ("prd_approved",),
            ("qa_passed",),
            ("launched",),
        ]
        # The happy path never replans — no transient failures.
        assert tracer.replans == []
        # on_goal_achieved fires exactly once at the very end.
        assert len(tracer.goal_achieved_states) == 1
        assert tracer.goal_achieved_states[0].get("launched") is True

    def test_each_subgoal_plan_only_contains_its_stage_actions(self) -> None:
        """Each sub-goal's plan contains only the actions that stage
        requires — the planner literally never sees later stages
        during an earlier sub-goal because ``GoapPlanner`` resolves
        ``MultiGoal`` to the current sub-goal before calling A*."""
        tracer = _RecordingTracer()
        graph = GoapGraph(product_launch_actions(), tracer=tracer)
        graph.invoke(
            goal=product_launch_goal(),
            world_state=product_launch_start(),
        )
        # One plan_complete per sub-goal, with the exact stage actions.
        assert tracer.plan_completes == [
            ("research_market", "write_prd"),
            ("implement_features", "qa_test"),
            ("prepare_marketing", "announce_launch"),
        ]


# ---------------------------------------------------------------------------
# Per-sub-goal replan budget: a transient failure in stage 2 does not
# leak into stage 3's budget or blacklist.
# ---------------------------------------------------------------------------


class TestPerSubgoalReplanBudget:
    def test_stage_two_replan_does_not_starve_stage_three(self) -> None:
        """The QA action fails once then succeeds.  Stage 2 consumes
        one replan.  Stage 3's ``replan_count`` must still start at
        zero — otherwise a cumulative budget would poison the launch
        stage whenever an earlier stage hiccuped.

        In LangGOAP, ``max_retries`` is the *blacklist threshold*, not
        an in-step retry counter.  Every action failure routes the
        observer back to the planner, incrementing ``replan_count``
        and firing ``on_replan``.  The per-sub-goal reset in
        ``GoapObserver._route`` is what keeps stage 3's budget intact.
        """
        qa_attempts = {"n": 0}
        replan_events: list[str] = []

        def flaky_qa(ws: dict[str, Any]) -> dict[str, Any]:
            del ws
            qa_attempts["n"] += 1
            if qa_attempts["n"] == 1:
                raise RuntimeError("transient flake on first QA attempt")
            return {"qa_passed": True}

        class ReplanRecorder(NullTracer):
            def on_replan(self, reason: str, new_plan: Any) -> None:
                replan_events.append(reason)

        # Replace the stock qa_test execute with the flaky version.
        actions: list[ActionSpec] = []
        for a in product_launch_actions():
            if a.name == "qa_test":
                actions.append(
                    ActionSpec(
                        name=a.name,
                        preconditions=dict(a.preconditions),
                        effects=dict(a.effects),
                        cost=a.cost,
                        execute=flaky_qa,
                        max_retries=1,
                    )
                )
            else:
                actions.append(a)

        # Tight per-sub-goal budget: 1 replan each.  Stage 2 needs it,
        # stage 3 does not — but stage 3 must still *have* it.
        mg = MultiGoal(
            goals=(
                GoalSpec(
                    conditions={"prd_approved": True}, policy=GoalPolicy(max_replans=1)
                ),
                GoalSpec(
                    conditions={"qa_passed": True}, policy=GoalPolicy(max_replans=1)
                ),
                GoalSpec(
                    conditions={"launched": True}, policy=GoalPolicy(max_replans=1)
                ),
            ),
            mode="sequential",
        )
        graph = GoapGraph(actions, tracer=ReplanRecorder())
        result = graph.invoke(goal=mg, world_state=product_launch_start())

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["launched"] is True
        # Stage 2's flaky action fired exactly twice: one failure, one
        # recovery (the second plan also routes through qa_test because
        # max_retries=1 stops it from being blacklisted).
        assert qa_attempts["n"] == 2
        # Exactly one replan event fired during the whole launch and
        # its reason was the stage-2 action failure.  If the per-
        # sub-goal reset were missing, stage 3 would have started its
        # plan with ``replan_count=1`` and any future hiccup would
        # immediately exhaust its budget.
        assert replan_events == ["action_failed"]
        # Final ``replan_count`` is zero because the observer reset it
        # when it advanced from stage 2 to stage 3.
        assert result["replan_count"] == 0


# ---------------------------------------------------------------------------
# Real LLM integration tests — run with ``uv run pytest -m api``
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestProductLaunchWithLLM:
    """Exercises the LLM-powered tutorial_examples factories end-to-end."""

    @pytest.fixture
    def llm(self) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def test_full_three_stage_launch_with_llm(self, llm: Any) -> None:
        """LLM-powered three-stage sequential launch reaches goal_achieved."""
        from tutorial_examples.hierarchical_product_launch import (
            product_launch_actions as llm_actions,
        )
        from tutorial_examples.hierarchical_product_launch import (
            product_launch_goal as llm_goal,
        )
        from tutorial_examples.hierarchical_product_launch import (
            product_launch_start as llm_start,
        )

        actions = llm_actions(llm)
        result = GoapGraph(actions=actions).invoke(
            goal=llm_goal(),
            world_state={**llm_start(), "product_name": "LangGoap SaaS"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["launched"] is True
        # LLM-generated content exists in the world state
        assert isinstance(ws.get("market_research"), str)
        assert isinstance(ws.get("prd_content"), str)
        assert isinstance(ws.get("marketing_content"), str)
        assert isinstance(ws.get("launch_announcement"), str)
