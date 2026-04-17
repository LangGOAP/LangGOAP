"""Integration tests for canonical agent workflow patterns in LangGOAP.

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

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Deterministic stubs for GOAP mechanics tests.
# These mirror the original tutorial_examples functions but are self-contained
# so the tutorial module can evolve to require a real LLM without breaking
# planning-focused tests.
# ---------------------------------------------------------------------------

# ---- Star News Finder stubs ----


def _extract_person(ws: dict[str, Any]) -> dict[str, Any]:
    user_input = ws.get("user_input", "")
    return {
        "has_person": True,
        "person": {"name": "Alice", "extracted_from": user_input},
    }


def _extract_star_sign(ws: dict[str, Any]) -> dict[str, Any]:
    person = ws.get("person", {})
    return {"has_star_person": True, "star_person": {**person, "sign": "Aries"}}


def _retrieve_horoscope(ws: dict[str, Any]) -> dict[str, Any]:
    star_person = ws.get("star_person", {})
    sign = star_person.get("sign", "Unknown")
    return {
        "has_horoscope": True,
        "horoscope": f"Today {sign} will experience great fortune in technology.",
    }


def _find_news_stories(ws: dict[str, Any]) -> dict[str, Any]:
    person = ws.get("star_person", {})
    name = person.get("name", "")
    return {
        "has_news": True,
        "news_stories": [
            f"{name} featured in AI conference keynote",
            f"New developments in {person.get('sign', '')} season",
        ],
    }


def _star_news_writeup(ws: dict[str, Any]) -> dict[str, Any]:
    person = ws.get("star_person", {})
    horoscope = ws.get("horoscope", "")
    news = ws.get("news_stories", [])
    return {
        "writeup_complete": True,
        "writeup": (
            f"Star News for {person.get('name', '')}: "
            f"{horoscope} "
            f"In the news: {'; '.join(news)}"
        ),
    }


def _star_news_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="extract_person",
            preconditions={"has_user_input": True},
            effects={"has_person": True},
            execute=_extract_person,
        ),
        ActionSpec(
            name="extract_star_sign",
            preconditions={"has_person": True},
            effects={"has_star_person": True},
            execute=_extract_star_sign,
        ),
        ActionSpec(
            name="retrieve_horoscope",
            preconditions={"has_star_person": True},
            effects={"has_horoscope": True},
            execute=_retrieve_horoscope,
        ),
        ActionSpec(
            name="find_news_stories",
            preconditions={"has_star_person": True},
            effects={"has_news": True},
            cost=2.0,
            execute=_find_news_stories,
        ),
        ActionSpec(
            name="star_news_writeup",
            preconditions={"has_horoscope": True, "has_news": True},
            effects={"writeup_complete": True},
            execute=_star_news_writeup,
        ),
    ]


# ---- Meal Preparation stubs ----


def _choose_cook(ws: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_cook": True,
        "cook": {"name": "Chef Marie", "specialty": "French cuisine"},
    }


def _take_order(ws: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_order": True,
        "order": {"dish": "Coq au Vin", "special_requests": "no mushrooms"},
    }


def _prepare_meal(ws: dict[str, Any]) -> dict[str, Any]:
    cook = ws.get("cook", {})
    order = ws.get("order", {})
    return {
        "meal_ready": True,
        "meal": {
            "dish": order.get("dish", ""),
            "prepared_by": cook.get("name", ""),
            "quality": "excellent",
        },
    }


def _meal_prep_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="choose_cook",
            preconditions={"has_user_input": True},
            effects={"has_cook": True},
            execute=_choose_cook,
        ),
        ActionSpec(
            name="take_order",
            preconditions={"has_user_input": True},
            effects={"has_order": True},
            execute=_take_order,
        ),
        ActionSpec(
            name="prepare_meal",
            preconditions={"has_cook": True, "has_order": True},
            effects={"meal_ready": True},
            execute=_prepare_meal,
        ),
    ]


# ---- Write and Review stubs ----


def _craft_story(ws: dict[str, Any]) -> dict[str, Any]:
    user_input = ws.get("user_input", "a space adventure")
    return {
        "has_story": True,
        "story": f"Once upon a time, in a galaxy far away, {user_input}...",
    }


def _assess_story(ws: dict[str, Any]) -> dict[str, Any]:
    return {"story_approved": True}


def _assess_story_strict(ws: dict[str, Any]) -> dict[str, Any]:
    story = ws.get("story", "")
    if "REVISED" not in story:
        return {"story_approved": False, "needs_revision": True}
    return {"story_approved": True}


def _revise_story(ws: dict[str, Any]) -> dict[str, Any]:
    user_input = ws.get("user_input", "")
    return {
        "has_story": True,
        "story": f"[REVISED] An epic saga: {user_input}",
        "story_approved": True,
    }


def _finalize_review(ws: dict[str, Any]) -> dict[str, Any]:
    story = ws.get("story", "")
    return {
        "review_complete": True,
        "reviewed_story": {
            "story": story,
            "review": "A captivating narrative with excellent pacing.",
            "reviewer": "NYT Book Review",
        },
    }


def _write_review_actions(
    *,
    assess_fn: Any = None,
    include_revise: bool = False,
    revise_fn: Any | None = None,
) -> list[ActionSpec]:
    if assess_fn is None:
        assess_fn = _assess_story
    actions = [
        ActionSpec(
            name="craft_story",
            preconditions={"has_user_input": True},
            effects={"has_story": True},
            execute=_craft_story,
        ),
        ActionSpec(
            name="assess_story",
            preconditions={"has_story": True},
            effects={"story_approved": True},
            execute=assess_fn,
        ),
        ActionSpec(
            name="finalize_review",
            preconditions={"story_approved": True},
            effects={"review_complete": True},
            execute=_finalize_review,
        ),
    ]
    if include_revise and revise_fn is not None:
        actions.append(
            ActionSpec(
                name="revise_story",
                preconditions={"has_story": True, "needs_revision": True},
                effects={"story_approved": True},
                execute=revise_fn,
            )
        )
    return actions


# ---- Fact Checker stubs ----


def _extract_assertions(ws: dict[str, Any]) -> dict[str, Any]:
    content = ws.get("content", "")
    return {
        "has_assertions": True,
        "assertions": [
            {"claim": "Python was created in 1991", "source": content[:50]},
            {"claim": "LangChain was released in 2022", "source": content[:50]},
        ],
    }


def _rationalize_assertions(ws: dict[str, Any]) -> dict[str, Any]:
    assertions = ws.get("assertions", [])
    return {
        "has_rationalized": True,
        "rationalized_assertions": [{**a, "importance": "high"} for a in assertions],
    }


def _check_facts(ws: dict[str, Any]) -> dict[str, Any]:
    assertions = ws.get("rationalized_assertions", [])
    checks = [
        {"claim": a["claim"], "verified": True, "confidence": 0.95} for a in assertions
    ]
    return {
        "fact_check_complete": True,
        "fact_check": {
            "total_claims": len(checks),
            "verified": sum(1 for c in checks if c["verified"]),
            "checks": checks,
        },
    }


class TestStarNewsFinder:
    """Star News Finder: multi-step LLM pipeline with web search."""

    def test_full_pipeline(self) -> None:
        """Planner discovers full extraction → retrieval → writeup path."""
        actions = _star_news_actions()
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
        actions = _star_news_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_star_person": True}),
            world_state={"has_user_input": True, "user_input": "Bob, Sagittarius"},
        )

        assert result["status"] == "goal_achieved"
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["extract_person", "extract_star_sign"]

    def test_horoscope_only_goal(self) -> None:
        """Partial goal: only retrieve horoscope, skip news and writeup."""
        actions = _star_news_actions()
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
        actions = _meal_prep_actions()
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
        actions = _meal_prep_actions()
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
        actions = _meal_prep_actions()
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
        actions = _write_review_actions()
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

        actions = _write_review_actions(
            assess_fn=_assess_story_strict,
            include_revise=True,
            revise_fn=_revise_story,
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
        actions = _write_review_actions()
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
                execute=_extract_assertions,
            ),
            ActionSpec(
                name="rationalize_assertions",
                preconditions={"has_assertions": True},
                effects={"has_rationalized": True},
                execute=_rationalize_assertions,
            ),
            ActionSpec(
                name="check_facts",
                preconditions={"has_rationalized": True},
                effects={"fact_check_complete": True},
                execute=_check_facts,
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
                execute=_extract_assertions,
            ),
            ActionSpec(
                name="rationalize_assertions",
                preconditions={"has_assertions": True},
                effects={"has_rationalized": True},
                execute=_rationalize_assertions,
            ),
            ActionSpec(
                name="check_facts",
                preconditions={"has_rationalized": True},
                effects={"fact_check_complete": True},
                execute=_check_facts,
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
# This test verifies that LangGOAP's A* planner correctly prefers cheap
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


# ---------------------------------------------------------------------------
# Real LLM integration tests — run with ``uv run pytest -m api``
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestAgentPatternsWithLLM:
    """Exercises the LLM-powered tutorial_examples factories end-to-end."""

    @pytest.fixture
    def llm(self) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def test_star_news_with_llm(self, llm: Any) -> None:
        """LLM-powered star news pipeline reaches goal_achieved."""
        from tutorial_examples.agent_pattern_examples import star_news_actions

        actions = star_news_actions(llm)
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
        assert isinstance(ws["writeup"], str)
        assert len(ws["writeup"]) > 20

    def test_write_and_review_with_llm(self, llm: Any) -> None:
        """LLM-powered write → assess → finalize reaches goal_achieved."""
        from tutorial_examples.agent_pattern_examples import write_review_actions

        actions = write_review_actions(llm)
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
        assert isinstance(ws["reviewed_story"]["review"], str)

    def test_fact_checker_with_llm(self, llm: Any) -> None:
        """LLM-powered fact checking pipeline reaches goal_achieved."""
        from tutorial_examples.agent_pattern_examples import fact_checker_actions

        actions = fact_checker_actions(llm)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"fact_check_complete": True}),
            world_state={
                "has_content": True,
                "content": "Python was created by Guido van Rossum in 1991. "
                "The Earth orbits the Sun.",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        fc = ws["fact_check"]
        assert fc["total_claims"] >= 1
        assert isinstance(fc["checks"], list)
