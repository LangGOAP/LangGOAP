r"""Integration test for the content builder agent tutorial (Tier 3).

Exercises the three features the notebook showcases:

- **Conditional format generation via CSP enumeration** — two blog
  writers (``write_blog_fast``, ``write_blog_deep``) produce the same
  effect with different quality / cost profiles.  A\* alone always
  picks the fast writer; a hard ``quality_score >= 8`` constraint
  forces the pipeline's ``enumerate_alternatives`` path to kick in and
  the deep writer is selected.
- **Multi-objective CSP** — ``premium_campaign_goal`` mixes a hard
  cost cap, a soft writer_hours cap, and both a MINIMIZE and MAXIMIZE
  objective.  The plan's :class:`~langgoap.score.HardSoftScore` pins
  every contribution exactly (no ``>=`` comparisons).
- **Fluent ConstraintBuilder** — every goal variant has a twin built
  with :class:`~langgoap.constraints.ConstraintBuilder` and the test
  battery pins that the hand-rolled and fluent forms yield
  structurally identical :class:`~langgoap.goals.GoalSpec` instances.

All resource totals are computed ahead of time in the tutorial module's
docstring table and pinned exactly in every assertion.
"""

from __future__ import annotations

from typing import Any

import pytest
from tutorial_examples.content_builder_agent import (
    blog_only_goal,
    content_builder_start,
    multi_channel_goal,
    premium_campaign_goal,
    premium_campaign_goal_fluent,
    quality_blog_goal,
    quality_blog_goal_fluent,
)

from langgoap import ActionSpec, GoapGraph

# ---------------------------------------------------------------------------
# Deterministic action factory for GOAP mechanics / CSP tests.
# The tutorial module's content_builder_actions() will evolve to require a
# real LLM; this local stub keeps the 15+ resource-pinning tests stable.
# ---------------------------------------------------------------------------


def _make_execute(effects: dict[str, Any]) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        del ws
        return dict(effects)

    return execute


def content_builder_actions() -> list[ActionSpec]:
    """12-action content marketing catalog (deterministic stubs)."""
    return [
        ActionSpec(
            name="research_topic",
            preconditions={},
            effects={"research_done": True},
            cost=1.0,
            resources={"writer_hours": 1.0, "cost_usd": 5.0},
            execute=_make_execute({"research_done": True}),
        ),
        ActionSpec(
            name="draft_outline",
            preconditions={"research_done": True},
            effects={"outline_ready": True},
            cost=1.0,
            resources={"writer_hours": 2.0, "cost_usd": 10.0},
            execute=_make_execute({"outline_ready": True}),
        ),
        ActionSpec(
            name="write_blog_fast",
            preconditions={"outline_ready": True},
            effects={"blog_drafted": True},
            cost=3.0,
            resources={"writer_hours": 2.0, "cost_usd": 30.0, "quality_score": 5.0},
            execute=_make_execute({"blog_drafted": True}),
        ),
        ActionSpec(
            name="write_blog_deep",
            preconditions={"outline_ready": True},
            effects={"blog_drafted": True},
            cost=6.0,
            resources={"writer_hours": 6.0, "cost_usd": 80.0, "quality_score": 10.0},
            execute=_make_execute({"blog_drafted": True}),
        ),
        ActionSpec(
            name="generate_blog_cover",
            preconditions={"blog_drafted": True},
            effects={"blog_cover_ready": True},
            cost=2.0,
            resources={"gpu_minutes": 5.0, "cost_usd": 15.0},
            execute=_make_execute({"blog_cover_ready": True}),
        ),
        ActionSpec(
            name="publish_blog",
            preconditions={"blog_drafted": True, "blog_cover_ready": True},
            effects={"blog_live": True},
            cost=1.0,
            execute=_make_execute({"blog_live": True}),
        ),
        ActionSpec(
            name="write_linkedin_post",
            preconditions={"outline_ready": True},
            effects={"linkedin_drafted": True},
            cost=2.0,
            resources={"writer_hours": 1.0, "cost_usd": 10.0, "quality_score": 4.0},
            execute=_make_execute({"linkedin_drafted": True}),
        ),
        ActionSpec(
            name="generate_linkedin_image",
            preconditions={"linkedin_drafted": True},
            effects={"linkedin_image_ready": True},
            cost=1.0,
            resources={"gpu_minutes": 2.0, "cost_usd": 5.0},
            execute=_make_execute({"linkedin_image_ready": True}),
        ),
        ActionSpec(
            name="publish_linkedin",
            preconditions={"linkedin_drafted": True, "linkedin_image_ready": True},
            effects={"linkedin_live": True},
            cost=1.0,
            execute=_make_execute({"linkedin_live": True}),
        ),
        ActionSpec(
            name="write_twitter_thread",
            preconditions={"outline_ready": True},
            effects={"twitter_drafted": True},
            cost=1.0,
            resources={"writer_hours": 0.5, "cost_usd": 4.0, "quality_score": 2.0},
            execute=_make_execute({"twitter_drafted": True}),
        ),
        ActionSpec(
            name="generate_twitter_image",
            preconditions={"twitter_drafted": True},
            effects={"twitter_image_ready": True},
            cost=1.0,
            resources={"gpu_minutes": 1.0, "cost_usd": 2.0},
            execute=_make_execute({"twitter_image_ready": True}),
        ),
        ActionSpec(
            name="publish_twitter",
            preconditions={"twitter_drafted": True, "twitter_image_ready": True},
            effects={"twitter_live": True},
            cost=1.0,
            execute=_make_execute({"twitter_live": True}),
        ),
    ]


from langgoap.goals import ObjectiveDirection
from langgoap.planner.csp import CSPStatus
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.score import HardSoftScore, SimpleScore
from langgoap.state import PlanningState

# ---------------------------------------------------------------------------
# Resource totals — pinned once here so every test can reference them.
# The docstring table in tutorial_examples/content_builder_agent.py is
# the source of truth; these numbers are derived from that table by hand.
# ---------------------------------------------------------------------------

# Fast blog-only plan: research → outline → write_blog_fast → cover → publish.
FAST_BLOG_ACTIONS = (
    "research_topic",
    "draft_outline",
    "write_blog_fast",
    "generate_blog_cover",
    "publish_blog",
)
FAST_BLOG_COST = 1.0 + 1.0 + 3.0 + 2.0 + 1.0  # 8.0
FAST_BLOG_WRITER_HOURS = 1.0 + 2.0 + 2.0  # 5.0
FAST_BLOG_COST_USD = 5.0 + 10.0 + 30.0 + 15.0  # 60.0
FAST_BLOG_QUALITY = 5.0
FAST_BLOG_GPU_MINUTES = 5.0

# Deep blog-only plan: same shape but with write_blog_deep instead.
DEEP_BLOG_ACTIONS = (
    "research_topic",
    "draft_outline",
    "write_blog_deep",
    "generate_blog_cover",
    "publish_blog",
)
DEEP_BLOG_COST = 1.0 + 1.0 + 6.0 + 2.0 + 1.0  # 11.0
DEEP_BLOG_WRITER_HOURS = 1.0 + 2.0 + 6.0  # 9.0
DEEP_BLOG_COST_USD = 5.0 + 10.0 + 80.0 + 15.0  # 110.0
DEEP_BLOG_QUALITY = 10.0
DEEP_BLOG_GPU_MINUTES = 5.0

# Fast multi-channel plan (A* picks fast blog, all three channels live).
FAST_MULTI_ACTION_SET = {
    "research_topic",
    "draft_outline",
    "write_blog_fast",
    "generate_blog_cover",
    "publish_blog",
    "write_linkedin_post",
    "generate_linkedin_image",
    "publish_linkedin",
    "write_twitter_thread",
    "generate_twitter_image",
    "publish_twitter",
}
FAST_MULTI_COST = (
    1.0 + 1.0 + 3.0 + 2.0 + 1.0 + 2.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0
)  # 15.0
FAST_MULTI_WRITER_HOURS = 1.0 + 2.0 + 2.0 + 1.0 + 0.5  # 6.5
FAST_MULTI_COST_USD = 5.0 + 10.0 + 30.0 + 15.0 + 10.0 + 5.0 + 4.0 + 2.0  # 81.0
FAST_MULTI_QUALITY = 5.0 + 4.0 + 2.0  # 11.0
FAST_MULTI_GPU_MINUTES = 5.0 + 2.0 + 1.0  # 8.0


# ---------------------------------------------------------------------------
# Tutorial helper smoke tests
# ---------------------------------------------------------------------------


class TestTutorialHelpers:
    def test_action_catalog_has_twelve_actions(self) -> None:
        actions = content_builder_actions()
        assert len(actions) == 12
        assert {a.name for a in actions} == {
            "research_topic",
            "draft_outline",
            "write_blog_fast",
            "write_blog_deep",
            "generate_blog_cover",
            "publish_blog",
            "write_linkedin_post",
            "generate_linkedin_image",
            "publish_linkedin",
            "write_twitter_thread",
            "generate_twitter_image",
            "publish_twitter",
        }

    def test_start_state_has_every_flag_false(self) -> None:
        start = content_builder_start()
        assert set(start.keys()) == {
            "research_done",
            "outline_ready",
            "blog_drafted",
            "blog_cover_ready",
            "blog_live",
            "linkedin_drafted",
            "linkedin_image_ready",
            "linkedin_live",
            "twitter_drafted",
            "twitter_image_ready",
            "twitter_live",
        }
        assert all(v is False for v in start.values())

    def test_two_blog_writers_share_effect_and_precondition(self) -> None:
        actions = {a.name: a for a in content_builder_actions()}
        fast = actions["write_blog_fast"]
        deep = actions["write_blog_deep"]
        # Identical precondition and effect — only the numeric profile
        # differs, which is what lets CSP enumeration swap one for the
        # other.
        assert dict(fast.preconditions) == dict(deep.preconditions)
        assert dict(fast.effects) == dict(deep.effects)
        assert fast.cost < deep.cost
        assert fast.resources is not None and deep.resources is not None
        assert fast.resources["quality_score"] < deep.resources["quality_score"]


# ---------------------------------------------------------------------------
# Blog-only goal — plain A*, no CSP routing
# ---------------------------------------------------------------------------


class TestBlogOnlyPlan:
    def test_pipeline_skips_csp_for_unconstrained_blog_goal(self) -> None:
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(start, blog_only_goal(), actions)
        assert plan is not None
        # A* picks the fast writer.
        assert tuple(plan.action_names) == FAST_BLOG_ACTIONS
        assert plan.total_cost == FAST_BLOG_COST
        # No constraints → pipeline skips CSP and returns the raw A* plan
        # with a SimpleScore equal to total_cost.
        assert plan.metadata.csp is None
        assert isinstance(plan.score, SimpleScore)
        assert plan.score.value == FAST_BLOG_COST


# ---------------------------------------------------------------------------
# Quality enforcement — the canonical enumeration demo
# ---------------------------------------------------------------------------


class TestQualityEnforcementTriggersEnumeration:
    """With a hard ``quality_score >= 8`` floor, the fast-writer primary
    plan fails CSP validation and the pipeline enumerates alternatives
    by blacklisting each action in the rejected plan.  The blacklist
    on ``write_blog_fast`` yields the deep-writer alternative, which
    CSP then selects.  This test pins that the final plan is the deep
    one and that the CSPMetadata reflects a feasible selection.
    """

    def test_hard_quality_floor_swaps_fast_writer_for_deep(self) -> None:
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(start, quality_blog_goal(min_quality=8.0), actions)
        assert plan is not None
        # CSP selected the deep writer plan.
        assert tuple(plan.action_names) == DEEP_BLOG_ACTIONS
        assert plan.total_cost == DEEP_BLOG_COST
        # CSP metadata is attached and feasible.
        meta = plan.metadata.csp
        assert meta is not None
        assert meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        # The quality_score resource usage should now clear the min bound.
        quality_usage = next(u for u in meta.resource_usage if u.key == "quality_score")
        assert quality_usage.total == DEEP_BLOG_QUALITY
        assert quality_usage.constraint_min == 8.0
        assert quality_usage.satisfied is True
        assert quality_usage.level == "hard"

    def test_score_is_hardsoft_with_zero_hard_and_positive_soft(self) -> None:
        # The deep plan clears the hard floor (hard=0.0) and the
        # MAXIMIZE quality objective contributes +quality_total to soft.
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(start, quality_blog_goal(min_quality=8.0), actions)
        assert plan is not None
        assert isinstance(plan.score, HardSoftScore)
        assert plan.score.hard == 0.0
        # MAXIMIZE quality_score → soft += 10.0 (deep blog quality total).
        assert plan.score.soft == DEEP_BLOG_QUALITY
        assert plan.score.is_feasible() is True

    def test_primary_plan_is_infeasible_under_lower_floor(self) -> None:
        # Sanity check: a quality floor of 5 is exactly met by the fast
        # writer, so enumeration should *not* fire and the fast plan is
        # returned feasible.
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(start, quality_blog_goal(min_quality=5.0), actions)
        assert plan is not None
        assert tuple(plan.action_names) == FAST_BLOG_ACTIONS
        meta = plan.metadata.csp
        assert meta is not None
        assert meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)


# ---------------------------------------------------------------------------
# Fluent ConstraintBuilder parity with hand-rolled goals
# ---------------------------------------------------------------------------


class TestFluentBuilderParity:
    """The fluent-builder twin of each goal must be structurally
    equivalent to the hand-rolled version.  Structural equivalence
    means: same conditions, same constraint tuple (as a set), same
    objectives map.  Tuple order is not pinned because the builder
    assembles constraints in the order chains are passed to
    ``build()``, which is an implementation detail.
    """

    def test_quality_goal_hand_rolled_matches_fluent(self) -> None:
        hand = quality_blog_goal(min_quality=8.0)
        fluent = quality_blog_goal_fluent(min_quality=8.0)
        assert dict(hand.conditions) == dict(fluent.conditions)
        assert set(hand.constraints) == set(fluent.constraints)
        # objectives is MappingProxyType | None; normalize to dict.
        assert dict(hand.objectives or {}) == dict(fluent.objectives or {})

    def test_quality_goal_fluent_plan_matches_hand_rolled_plan(self) -> None:
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        hand_plan = pipeline_plan(start, quality_blog_goal(min_quality=8.0), actions)
        fluent_plan = pipeline_plan(
            start, quality_blog_goal_fluent(min_quality=8.0), actions
        )
        assert hand_plan is not None and fluent_plan is not None
        assert tuple(hand_plan.action_names) == tuple(fluent_plan.action_names)
        assert hand_plan.total_cost == fluent_plan.total_cost
        assert hand_plan.score == fluent_plan.score

    def test_premium_campaign_hand_rolled_matches_fluent(self) -> None:
        hand = premium_campaign_goal(hard_cost_cap=100.0, soft_writer_hours_cap=5.0)
        fluent = premium_campaign_goal_fluent(
            hard_cost_cap=100.0, soft_writer_hours_cap=5.0
        )
        assert dict(hand.conditions) == dict(fluent.conditions)
        assert set(hand.constraints) == set(fluent.constraints)
        assert dict(hand.objectives or {}) == dict(fluent.objectives or {})


# ---------------------------------------------------------------------------
# Multi-channel — hard cap triggers INFEASIBLE, soft cap stays feasible
# ---------------------------------------------------------------------------


class TestMultiChannelConstraints:
    def test_unconstrained_multi_channel_plan_covers_all_three_channels(
        self,
    ) -> None:
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(start, multi_channel_goal(), actions)
        assert plan is not None
        # A* picks the fast blog writer + single LI + single Twitter path.
        assert set(plan.action_names) == FAST_MULTI_ACTION_SET
        assert plan.total_cost == FAST_MULTI_COST
        # MINIMIZE cost objective routes through CSP → HardSoftScore.
        assert isinstance(plan.score, HardSoftScore)
        meta = plan.metadata.csp
        assert meta is not None
        assert meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        assert meta.objective_values.get("cost_usd") == FAST_MULTI_COST_USD

    def test_hard_cost_cap_below_fast_plan_total_is_infeasible(self) -> None:
        # Fast multi-channel plan costs 81.0; a hard cap of 50 forces
        # INFEASIBLE because every action in the plan is load-bearing
        # (LI and Twitter have single writers, so blacklist-based
        # enumeration cannot produce an alternative that still reaches
        # all three channels).
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(
            start,
            multi_channel_goal(max_cost_usd=50.0, max_cost_level="hard"),
            actions,
        )
        assert plan is not None
        meta = plan.metadata.csp
        assert meta is not None
        assert meta.status == CSPStatus.INFEASIBLE
        # The hard-level cost_usd resource usage records the violation.
        cost_usage = next(u for u in meta.resource_usage if u.key == "cost_usd")
        assert cost_usage.total == FAST_MULTI_COST_USD
        assert cost_usage.constraint_max == 50.0
        assert cost_usage.satisfied is False
        assert cost_usage.level == "hard"
        # Hard violation shows up in the HardSoftScore.
        assert isinstance(plan.score, HardSoftScore)
        assert plan.score.hard < 0.0
        assert plan.score.is_feasible() is False

    def test_soft_cost_cap_records_penalty_without_infeasibility(self) -> None:
        # Same 50.0 cap but at level="soft" — violation records a
        # HardSoftScore.soft penalty but the plan stays feasible.
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(
            start,
            multi_channel_goal(max_cost_usd=50.0, max_cost_level="soft"),
            actions,
        )
        assert plan is not None
        meta = plan.metadata.csp
        assert meta is not None
        assert meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        assert isinstance(plan.score, HardSoftScore)
        assert plan.score.hard == 0.0
        # Score breakdown:
        #   soft_violation = (81 - 50) * weight(1) = -31
        #   MINIMIZE cost_usd contribution = -81
        #   total soft = -31 - 81 = -112
        expected_soft = -(FAST_MULTI_COST_USD - 50.0) - FAST_MULTI_COST_USD
        assert plan.score.soft == expected_soft
        assert plan.score.is_feasible() is True


# ---------------------------------------------------------------------------
# Premium campaign — hard + soft + MIN + MAX in one goal
# ---------------------------------------------------------------------------


class TestPremiumCampaignScoreBreakdown:
    """The premium campaign goal is the multi-objective showcase.  The
    assertions decompose the final :class:`HardSoftScore` into the
    exact contributions the pipeline formula produces so a drifted
    score is obvious.
    """

    def test_feasible_plan_soft_score_is_sum_of_all_contributions(self) -> None:
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        plan = pipeline_plan(
            start,
            premium_campaign_goal(hard_cost_cap=100.0, soft_writer_hours_cap=5.0),
            actions,
        )
        assert plan is not None
        meta = plan.metadata.csp
        assert meta is not None
        assert meta.status in (CSPStatus.FEASIBLE, CSPStatus.OPTIMAL)
        assert set(plan.action_names) == FAST_MULTI_ACTION_SET
        # cost_usd = 81 ≤ 100 (hard OK, hard == 0.0).
        # writer_hours = 6.5 > 5 → soft penalty = -(6.5 - 5) = -1.5.
        # MINIMIZE cost_usd → soft += -81.
        # MAXIMIZE quality_score → soft += +11.
        # Final: hard=0.0, soft = -1.5 - 81 + 11 = -71.5.
        assert isinstance(plan.score, HardSoftScore)
        assert plan.score.hard == 0.0
        assert plan.score.soft == pytest.approx(
            -(FAST_MULTI_WRITER_HOURS - 5.0) - FAST_MULTI_COST_USD + FAST_MULTI_QUALITY
        )
        assert plan.score.soft == pytest.approx(-71.5)
        # Resource usages must flag writer_hours as a soft violation.
        hours_usage = next(u for u in meta.resource_usage if u.key == "writer_hours")
        assert hours_usage.level == "soft"
        assert hours_usage.satisfied is False
        assert hours_usage.total == FAST_MULTI_WRITER_HOURS

    def test_both_objectives_populate_objective_values_map(self) -> None:
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        goal = premium_campaign_goal(hard_cost_cap=100.0, soft_writer_hours_cap=5.0)
        plan = pipeline_plan(start, goal, actions)
        assert plan is not None
        meta = plan.metadata.csp
        assert meta is not None
        # Both objectives exposed in the objective_values map.
        assert dict(meta.objective_values) == {
            "cost_usd": FAST_MULTI_COST_USD,
            "quality_score": FAST_MULTI_QUALITY,
        }
        # Goal's objectives field pins direction.
        assert goal.objectives is not None
        assert goal.objectives["cost_usd"] == ObjectiveDirection.MINIMIZE
        assert goal.objectives["quality_score"] == ObjectiveDirection.MAXIMIZE

    def test_premium_goal_fluent_and_hand_rolled_produce_same_score(
        self,
    ) -> None:
        actions = content_builder_actions()
        start = PlanningState.from_dict(content_builder_start())
        hand_plan = pipeline_plan(
            start,
            premium_campaign_goal(hard_cost_cap=100.0, soft_writer_hours_cap=5.0),
            actions,
        )
        fluent_plan = pipeline_plan(
            start,
            premium_campaign_goal_fluent(
                hard_cost_cap=100.0, soft_writer_hours_cap=5.0
            ),
            actions,
        )
        assert hand_plan is not None and fluent_plan is not None
        assert hand_plan.score == fluent_plan.score


# ---------------------------------------------------------------------------
# End-to-end execution via GoapGraph
# ---------------------------------------------------------------------------


class TestEndToEndExecution:
    def test_blog_only_goal_executes_cleanly(self) -> None:
        graph = GoapGraph(content_builder_actions())
        result = graph.invoke(
            goal=blog_only_goal(), world_state=content_builder_start()
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["blog_live"] is True
        sequence = tuple(r.action_name for r in result["execution_history"])
        assert sequence == FAST_BLOG_ACTIONS

    def test_quality_goal_executes_the_deep_writer_plan(self) -> None:
        graph = GoapGraph(content_builder_actions())
        result = graph.invoke(
            goal=quality_blog_goal(min_quality=8.0),
            world_state=content_builder_start(),
        )
        assert result["status"] == "goal_achieved"
        sequence = tuple(r.action_name for r in result["execution_history"])
        # Enumeration selected the deep writer plan end-to-end.
        assert sequence == DEEP_BLOG_ACTIONS

    @pytest.mark.asyncio
    async def test_premium_campaign_async_execution(self) -> None:
        graph = GoapGraph(content_builder_actions())
        result = await graph.ainvoke(
            goal=premium_campaign_goal(hard_cost_cap=100.0, soft_writer_hours_cap=5.0),
            world_state=content_builder_start(),
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["blog_live"] is True
        assert result["world_state"]["linkedin_live"] is True
        assert result["world_state"]["twitter_live"] is True
        sequence = {r.action_name for r in result["execution_history"]}
        assert sequence == FAST_MULTI_ACTION_SET


# ---------------------------------------------------------------------------
# Real LLM integration tests — run with ``uv run pytest -m api``
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestContentBuilderWithLLM:
    """Exercises the LLM-powered tutorial_examples factories end-to-end."""

    @pytest.fixture
    def llm(self) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def test_blog_pipeline_with_llm(self, llm: Any) -> None:
        """LLM-powered research → outline → blog → cover → publish."""
        from tutorial_examples.content_builder_agent import (
            content_builder_actions as llm_actions,
        )

        actions = llm_actions(llm)
        graph = GoapGraph(actions)
        result = graph.invoke(
            goal=blog_only_goal(),
            world_state={**content_builder_start(), "topic": "AI agent planning"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["blog_live"] is True
        # LLM generated real content
        assert isinstance(ws.get("research_findings"), str)
        assert len(ws["research_findings"]) > 20
        assert isinstance(ws.get("content_outline"), str)
        assert isinstance(ws.get("blog_content"), str)
        assert len(ws["blog_content"]) > 100

    def test_multi_channel_with_llm(self, llm: Any) -> None:
        """LLM-powered full campaign: blog + LinkedIn + Twitter."""
        from tutorial_examples.content_builder_agent import (
            content_builder_actions as llm_actions,
        )

        actions = llm_actions(llm)
        graph = GoapGraph(actions)
        result = graph.invoke(
            goal=multi_channel_goal(),
            world_state={**content_builder_start(), "topic": "LLM-based planning"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["blog_live"] is True
        assert ws["linkedin_live"] is True
        assert ws["twitter_live"] is True
        # Each channel produced LLM content
        assert isinstance(ws.get("blog_content"), str)
        assert isinstance(ws.get("linkedin_content"), str)
        assert isinstance(ws.get("twitter_content"), str)
