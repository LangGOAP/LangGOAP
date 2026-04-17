"""Shared execute functions for the Agent Pattern Examples tutorial.

Five canonical agent workflow patterns rendered in LangGOAP:

1. Star News Finder — multi-step LLM pipeline with web search.
2. Meal Preparation — parallel preconditions merging at a single goal.
3. Write and Review — iterative refinement via replanning on deviation.
4. Fact Checker — multi-step verification pipeline.
5. Cost-Based Selection — planner prefers cheap actions over expensive
   ones when both satisfy the same goal.

Imported by:
- ``tests/integration/test_agent_pattern_examples.py``
- ``examples/tutorials/agent_pattern_examples_goapified.ipynb``
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec

# ===========================================================================
# Example 1: Star News Finder
# ===========================================================================


def extract_person(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate LLM extracting a person from user input."""
    user_input = ws.get("user_input", "")
    return {"has_person": True, "person": {"name": "Alice", "extracted_from": user_input}}


def extract_star_sign(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate LLM extracting star sign from person context."""
    person = ws.get("person", {})
    return {"has_star_person": True, "star_person": {**person, "sign": "Aries"}}


def retrieve_horoscope(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate calling a horoscope service."""
    star_person = ws.get("star_person", {})
    sign = star_person.get("sign", "Unknown")
    return {"has_horoscope": True, "horoscope": f"Today {sign} will experience great fortune in technology."}


def find_news_stories(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate web search for news related to person and horoscope."""
    person = ws.get("star_person", {})
    name = person.get("name", "")
    return {
        "has_news": True,
        "news_stories": [
            f"{name} featured in AI conference keynote",
            f"New developments in {person.get('sign', '')} season",
        ],
    }


def star_news_writeup(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate generating the final star news writeup."""
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


def star_news_actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="extract_person", preconditions={"has_user_input": True}, effects={"has_person": True}, execute=extract_person),
        ActionSpec(name="extract_star_sign", preconditions={"has_person": True}, effects={"has_star_person": True}, execute=extract_star_sign),
        ActionSpec(name="retrieve_horoscope", preconditions={"has_star_person": True}, effects={"has_horoscope": True}, execute=retrieve_horoscope),
        ActionSpec(name="find_news_stories", preconditions={"has_star_person": True}, effects={"has_news": True}, cost=2.0, execute=find_news_stories),
        ActionSpec(name="star_news_writeup", preconditions={"has_horoscope": True, "has_news": True}, effects={"writeup_complete": True}, execute=star_news_writeup),
    ]


# ===========================================================================
# Example 2: Meal Preparation
# ===========================================================================


def choose_cook(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate selecting a cook based on available staff."""
    return {"has_cook": True, "cook": {"name": "Chef Marie", "specialty": "French cuisine"}}


def take_order(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate taking a customer order."""
    return {"has_order": True, "order": {"dish": "Coq au Vin", "special_requests": "no mushrooms"}}


def prepare_meal(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate meal preparation by the selected cook."""
    cook = ws.get("cook", {})
    order = ws.get("order", {})
    return {
        "meal_ready": True,
        "meal": {"dish": order.get("dish", ""), "prepared_by": cook.get("name", ""), "quality": "excellent"},
    }


def meal_prep_actions() -> list[ActionSpec]:
    return [
        ActionSpec(name="choose_cook", preconditions={"has_user_input": True}, effects={"has_cook": True}, execute=choose_cook),
        ActionSpec(name="take_order", preconditions={"has_user_input": True}, effects={"has_order": True}, execute=take_order),
        ActionSpec(name="prepare_meal", preconditions={"has_cook": True, "has_order": True}, effects={"meal_ready": True}, execute=prepare_meal),
    ]


# ===========================================================================
# Example 3: Write and Review
# ===========================================================================


def craft_story(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate LLM generating a story from user input."""
    user_input = ws.get("user_input", "a space adventure")
    return {"has_story": True, "story": f"Once upon a time, in a galaxy far away, {user_input}..."}


def assess_story(ws: dict[str, Any]) -> dict[str, Any]:
    """Assess a story — always approves (happy path)."""
    return {"story_approved": True}


def assess_story_strict(ws: dict[str, Any]) -> dict[str, Any]:
    """Assess a story — rejects unless the story has been revised.

    Returns ``story_approved=False, needs_revision=True`` on first pass,
    which deviates from the declared effect and triggers replanning.
    """
    story = ws.get("story", "")
    if "REVISED" not in story:
        return {"story_approved": False, "needs_revision": True}
    return {"story_approved": True}


def revise_story(ws: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the story with improvements (prefixes [REVISED])."""
    user_input = ws.get("user_input", "")
    return {"has_story": True, "story": f"[REVISED] An epic saga: {user_input}", "story_approved": True}


def finalize_review(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate producing the final reviewed story."""
    story = ws.get("story", "")
    return {
        "review_complete": True,
        "reviewed_story": {"story": story, "review": "A captivating narrative with excellent pacing.", "reviewer": "NYT Book Review"},
    }


def write_review_actions(
    *,
    assess_fn: Any = assess_story,
    include_revise: bool = False,
    revise_fn: Any | None = None,
) -> list[ActionSpec]:
    """Build Write-and-Review action set.

    Args:
        assess_fn: Execute callable for ``assess_story``.  Swap in
            ``assess_story_strict`` to simulate rejection and trigger revision.
        include_revise: When ``True``, adds a ``revise_story`` action whose
            extra precondition (``needs_revision``) wins via A* specificity.
        revise_fn: Execute callable for ``revise_story`` (required when
            ``include_revise=True``).
    """
    actions = [
        ActionSpec(name="craft_story", preconditions={"has_user_input": True}, effects={"has_story": True}, execute=craft_story),
        ActionSpec(name="assess_story", preconditions={"has_story": True}, effects={"story_approved": True}, execute=assess_fn),
        ActionSpec(name="finalize_review", preconditions={"story_approved": True}, effects={"review_complete": True}, execute=finalize_review),
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


# ===========================================================================
# Example 4: Fact Checker
# ===========================================================================


def extract_assertions(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate extracting factual claims from content."""
    content = ws.get("content", "")
    return {
        "has_assertions": True,
        "assertions": [
            {"claim": "Python was created in 1991", "source": content[:50]},
            {"claim": "LangChain was released in 2022", "source": content[:50]},
        ],
    }


def rationalize_assertions(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate deduplicating and rationalizing assertions."""
    assertions = ws.get("assertions", [])
    return {"has_rationalized": True, "rationalized_assertions": [{**a, "importance": "high"} for a in assertions]}


def check_facts(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate verifying each assertion against known sources."""
    assertions = ws.get("rationalized_assertions", [])
    checks = [{"claim": a["claim"], "verified": True, "confidence": 0.95} for a in assertions]
    return {
        "fact_check_complete": True,
        "fact_check": {"total_claims": len(checks), "verified": sum(1 for c in checks if c["verified"]), "checks": checks},
    }
