"""Integration tests for canonical agent workflow patterns in LangGoap.

Each test exercises a common agent pattern end-to-end through
:class:`GoapGraph`:

1. Star News Finder — multi-step LLM pipeline with web search.
2. Meal Preparation — parallel preconditions merging at a single goal.
3. Write and Review — iterative refinement via replanning on deviation.
4. Fact Checker — multi-step verification pipeline.
5. Cost-Based Selection — cheapest satisfying action wins.

The shared execute functions live in
``examples/tutorials/tutorial_examples/agent_pattern_examples.py`` so
the notebook and the tests stay in sync.
"""

from __future__ import annotations

from typing import Any

import pytest
from tutorial_examples.agent_pattern_examples import (
    assess_story,
    assess_story_strict,
    check_facts,
    choose_cook,
    craft_story,
    extract_assertions,
    extract_person,
    extract_star_sign,
    finalize_review,
    find_news_stories,
    meal_prep_actions,
    prepare_meal,
    rationalize_assertions,
    retrieve_horoscope,
    revise_story,
    star_news_actions,
    star_news_writeup,
    take_order,
    write_review_actions,
)

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy


class TestStarNewsFinder:
    """Star News Finder: multi-step LLM pipeline with web search."""

    def test_full_pipeline(self) -> None:
        """Planner discovers full extraction → retrieval → writeup path."""
        actions = star_news_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"writeup_complete": True}),
            world_state={
                "has_user_input": True,
                "user_input": "Tell me about Alice who is an Aries",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["writeup_complete"] is True
        assert "Alice" in ws["writeup"]
        assert "Aries" in ws["writeup"]

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "extract_person" in successful
        assert "extract_star_sign" in successful
        assert "retrieve_horoscope" in successful
        assert "find_news_stories" in successful
        assert "star_news_writeup" in successful

    def test_extraction_chain_order(self) -> None:
        """Person extraction must precede star sign extraction."""
        actions = star_news_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_star_person": True}),
            world_state={"has_user_input": True, "user_input": "Bob, Sagittarius"},
        )

        assert result["status"] == "goal_achieved"
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["extract_person", "extract_star_sign"]

    def test_horoscope_only_goal(self) -> None:
        """Partial goal: only retrieve horoscope, skip news and writeup."""
        actions = star_news_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_horoscope": True}),
            world_state={"has_user_input": True},
        )

        assert result["status"] == "goal_achieved"
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "retrieve_horoscope" in successful
        assert "star_news_writeup" not in successful
        assert "find_news_stories" not in successful


# ===========================================================================
# Example 2: Meal Preparation
# ===========================================================================


class TestMealPreparation:
    """Meal Preparation: parallel preconditions merging at goal."""

    def test_full_meal_preparation(self) -> None:
        """Planner discovers choose_cook + take_order → prepare_meal."""
        actions = meal_prep_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"meal_ready": True}),
            world_state={"has_user_input": True},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["meal_ready"] is True
        assert ws["meal"]["dish"] == "Coq au Vin"
        assert ws["meal"]["prepared_by"] == "Chef Marie"

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "choose_cook" in successful
        assert "take_order" in successful
        assert "prepare_meal" in successful
        # prepare_meal must come after both prerequisites
        assert successful.index("prepare_meal") > successful.index("choose_cook")
        assert successful.index("prepare_meal") > successful.index("take_order")

    def test_partial_prerequisites_met(self) -> None:
        """When cook is already chosen, only take_order and prepare needed."""
        actions = meal_prep_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"meal_ready": True}),
            world_state={
                "has_user_input": True,
                "has_cook": True,
                "cook": {"name": "Chef Pierre", "specialty": "Italian"},
            },
        )

        assert result["status"] == "goal_achieved"
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "choose_cook" not in successful
        assert "take_order" in successful
        assert "prepare_meal" in successful

    def test_goal_already_satisfied(self) -> None:
        """When meal is already ready, no actions needed."""
        actions = meal_prep_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"meal_ready": True}),
            world_state={"meal_ready": True},
        )

        assert result["status"] == "goal_achieved"
        history = result["execution_history"]
        # No actions should have been executed
        successful = [h.action_name for h in history if h.success]
        assert len(successful) == 0


# ===========================================================================
# Example 3: Write and Review Agent
#
# Workflow (stateful with replanning loop):
#   craft_story(user_input) → story
#   assess_story(story) → assessment (accept/reject)
#   If rejected: revise_story(story, feedback) → story (loop back)
#   If accepted: finalize_review(story) → reviewed_story  (goal)
#
# In GOAP: each phase is a separate action with honest declared effects.
# The assess action writes story_approved (distinct key, never undoes
# has_story). On rejection it also sets needs_revision=True, enabling
# revise_story whose extra precondition wins on A* specificity tie-break.
# The observer detects the deviation (story_approved=False vs declared
# True) and triggers replanning.
# ===========================================================================


class TestWriteAndReview:
    """Write and Review: creative content pipeline with revise-on-reject.

    Uses a three-action pipeline — craft → assess → finalize — where the
    assess action writes to story_approved (a distinct key) without undoing
    has_story.  On rejection, needs_revision=True enables a revise_story
    action that the replanner discovers via specificity tie-breaking.
    """

    def test_write_and_review_happy_path(self) -> None:
        """Planner discovers craft → assess → finalize path."""
        actions = write_review_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"review_complete": True}),
            world_state={
                "has_user_input": True,
                "user_input": "a brave astronaut explored Mars",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["review_complete"] is True
        assert "astronaut" in ws["story"]
        assert "NYT Book Review" in ws["reviewed_story"]["reviewer"]

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["craft_story", "assess_story", "finalize_review"]

    def test_revision_via_replanning(self) -> None:
        """Story rejected on first pass, revised via replanning.

        Exercises the assess → revise loop.  assess_story returns
        story_approved=False (deviating from declared True) and sets
        needs_revision=True.  The observer triggers replanning.  In the
        new state, revise_story is preferred over assess_story via A*
        specificity tie-breaking (2 preconditions vs 1).  revise_story
        rewrites the story and produces story_approved=True, after which
        finalize_review completes the goal.
        """

        actions = write_review_actions(
            assess_fn=assess_story_strict,
            include_revise=True,
            revise_fn=revise_story,
        )

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(
                conditions={"review_complete": True},
                replan_strategy=ReplanStrategy.ON_DEVIATION,
            ),
            world_state={"has_user_input": True, "user_input": "dragons and knights"},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        ws = result["world_state"]
        assert "REVISED" in ws["story"]
        # has_story was never undone — it stays True throughout
        assert ws["has_story"] is True

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "revise_story" in successful
        assert "finalize_review" in successful

    def test_never_strategy_accepts_first_draft(self) -> None:
        """With NEVER strategy, story is assessed once and finalized."""
        actions = write_review_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(
                conditions={"review_complete": True},
                replan_strategy=ReplanStrategy.NEVER,
            ),
            world_state={"has_user_input": True, "user_input": "a cat named Whiskers"},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 0
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["craft_story", "assess_story", "finalize_review"]


# ===========================================================================
# Example 4: Fact Checker
#
# Workflow:
#   extract_assertions(content) → factual_assertions
#   rationalize_assertions(factual_assertions) → rationalized_assertions
#   check_facts(rationalized_assertions) → fact_check  (goal)
#
# Simplified version without multi-model ensemble.
# ===========================================================================


class TestFactChecker:
    """Fact Checker: multi-step verification pipeline."""

    def test_full_fact_check_pipeline(self) -> None:
        """Planner discovers extract → rationalize → check sequence."""
        actions = [
            ActionSpec(
                name="extract_assertions",
                preconditions={"has_content": True},
                effects={"has_assertions": True},
                execute=extract_assertions,
            ),
            ActionSpec(
                name="rationalize_assertions",
                preconditions={"has_assertions": True},
                effects={"has_rationalized": True},
                execute=rationalize_assertions,
            ),
            ActionSpec(
                name="check_facts",
                preconditions={"has_rationalized": True},
                effects={"fact_check_complete": True},
                execute=check_facts,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"fact_check_complete": True}),
            world_state={
                "has_content": True,
                "content": "Python was created by Guido van Rossum in 1991. "
                "LangChain was released in 2022 for LLM app development.",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        fc = ws["fact_check"]
        assert fc["total_claims"] == 2
        assert fc["verified"] == 2
        assert all(c["verified"] for c in fc["checks"])

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == [
            "extract_assertions",
            "rationalize_assertions",
            "check_facts",
        ]

    def test_partial_goal_extraction_only(self) -> None:
        """Can stop after extraction without full fact check."""
        actions = [
            ActionSpec(
                name="extract_assertions",
                preconditions={"has_content": True},
                effects={"has_assertions": True},
                execute=extract_assertions,
            ),
            ActionSpec(
                name="rationalize_assertions",
                preconditions={"has_assertions": True},
                effects={"has_rationalized": True},
                execute=rationalize_assertions,
            ),
            ActionSpec(
                name="check_facts",
                preconditions={"has_rationalized": True},
                effects={"fact_check_complete": True},
                execute=check_facts,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_assertions": True}),
            world_state={"has_content": True, "content": "Some factual content."},
        )

        assert result["status"] == "goal_achieved"
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["extract_assertions"]


# ===========================================================================
# Example 5: Cost-Based Agent Selection
#
# Mark expensive actions with a high ``cost`` to push them to last resort.
# This test verifies that LangGoap's A* planner correctly prefers cheap
# actions over expensive ones, only falling back when necessary.
# ===========================================================================


class TestCostBasedPlanning:
    """Cost-based selection: action cost controls planning preferences."""

    def test_cheap_action_preferred(self) -> None:
        """Low-cost cache lookup preferred over expensive API call."""

        def cache_lookup(ws: dict[str, Any]) -> dict[str, Any]:
            return {"has_answer": True, "answer": "cached result", "source": "cache"}

        def api_call(ws: dict[str, Any]) -> dict[str, Any]:
            return {"has_answer": True, "answer": "api result", "source": "api"}

        actions = [
            ActionSpec(
                name="cache_lookup",
                preconditions={"has_query": True},
                effects={"has_answer": True},
                cost=1.0,
                execute=cache_lookup,
            ),
            ActionSpec(
                name="api_call",
                preconditions={"has_query": True},
                effects={"has_answer": True},
                cost=100.0,  # high cost = last resort
                execute=api_call,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_answer": True}),
            world_state={"has_query": True, "query": "What is GOAP?"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["source"] == "cache"

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["cache_lookup"]

    def test_multi_path_cost_optimization(self) -> None:
        """Planner picks cheapest multi-step path to goal."""

        def fast_extract(ws: dict[str, Any]) -> dict[str, Any]:
            return {"has_data": True, "data": "fast"}

        def thorough_extract(ws: dict[str, Any]) -> dict[str, Any]:
            return {"has_data": True, "data": "thorough"}

        def process(ws: dict[str, Any]) -> dict[str, Any]:
            return {"processed": True, "result": f"processed {ws.get('data', '')}"}

        actions = [
            ActionSpec(
                name="fast_extract",
                preconditions={"has_input": True},
                effects={"has_data": True},
                cost=1.0,
                execute=fast_extract,
            ),
            ActionSpec(
                name="thorough_extract",
                preconditions={"has_input": True},
                effects={"has_data": True},
                cost=10.0,
                execute=thorough_extract,
            ),
            ActionSpec(
                name="process",
                preconditions={"has_data": True},
                effects={"processed": True},
                cost=1.0,
                execute=process,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"processed": True}),
            world_state={"has_input": True},
        )

        assert result["status"] == "goal_achieved"
        # Should use fast_extract (cost 1) + process (cost 1) = 2
        # instead of thorough_extract (cost 10) + process (cost 1) = 11
        assert result["world_state"]["result"] == "processed fast"
