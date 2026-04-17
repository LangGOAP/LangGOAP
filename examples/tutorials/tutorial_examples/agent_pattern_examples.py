r"""LLM-powered execute functions for the Agent Pattern Examples tutorial.

Five canonical agent workflow patterns rendered in LangGOAP:

1. Star News Finder — multi-step LLM pipeline with web search.
2. Meal Preparation — parallel preconditions merging at a single goal.
3. Write and Review — iterative refinement via replanning on deviation.
4. Fact Checker — multi-step verification pipeline.
5. Cost-Based Selection — planner prefers cheap actions over expensive
   ones when both satisfy the same goal.

All patterns except Cost-Based Selection use real LLM calls for extraction,
generation, and assessment.  ``extract_star_sign`` and ``retrieve_horoscope``
stay as lookups (the originals also didn't use LLM for these).

Imported by:

- ``examples/tutorials/agent_pattern_examples_goapified.ipynb``
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgoap import ActionSpec

_Execute = Callable[[dict[str, Any]], dict[str, Any]]

# ===========================================================================
# Example 1: Star News Finder
# ===========================================================================


def _make_extract_person(llm: BaseChatModel) -> _Execute:
    """LLM extracts a person's name from user input."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        user_input = ws.get("user_input", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract the person's name from the user input. "
                        "Return ONLY the name, nothing else."
                    )
                ),
                HumanMessage(content=user_input),
            ]
        )
        name = response.content.strip()
        return {
            "has_person": True,
            "person": {"name": name, "extracted_from": user_input},
        }

    return execute


def extract_star_sign(ws: dict[str, Any]) -> dict[str, Any]:
    """Extract star sign from person context (lookup, not LLM)."""
    person = ws.get("person", {})
    text = person.get("extracted_from", "")
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
    ]
    sign = "Unknown"
    for s in signs:
        if s.lower() in text.lower():
            sign = s
            break
    return {"has_star_person": True, "star_person": {**person, "sign": sign}}


def retrieve_horoscope(ws: dict[str, Any]) -> dict[str, Any]:
    """Retrieve horoscope for the star person (service call, not LLM)."""
    star_person = ws.get("star_person", {})
    sign = star_person.get("sign", "Unknown")
    return {
        "has_horoscope": True,
        "horoscope": f"Today {sign} will experience great fortune in technology.",
    }


def _make_find_news_stories(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> _Execute:
    """LLM finds news stories related to the person."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        person = ws.get("star_person", {})
        name = person.get("name", "")
        sign = person.get("sign", "")

        if search_tool is not None:
            raw = search_tool.invoke(f"{name} news")
            items = raw if isinstance(raw, list) else [raw]
            stories = [
                r.get("content", r.get("snippet", str(r))) for r in items
            ]
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a news search agent. Given a person's "
                            "name and star sign, generate 2 plausible recent "
                            "news headlines about them. Return each headline "
                            "on its own line."
                        )
                    ),
                    HumanMessage(
                        content=f"Find news for {name} (star sign: {sign})"
                    ),
                ]
            )
            stories = [
                line.strip().lstrip("- ").lstrip("0123456789.").strip()
                for line in response.content.strip().split("\n")
                if line.strip()
            ]
        return {"has_news": True, "news_stories": stories}

    return execute


def _make_star_news_writeup(llm: BaseChatModel) -> _Execute:
    """LLM composes the final star news writeup."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        person = ws.get("star_person", {})
        horoscope = ws.get("horoscope", "")
        news = ws.get("news_stories", [])
        name = person.get("name", "")
        sign = person.get("sign", "")
        news_text = "\n".join(f"- {s}" for s in news)

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a celebrity news writer. Given a person's "
                        "horoscope and recent news, write a fun, concise "
                        "star news writeup (3-4 sentences). Include the "
                        "person's name and star sign."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Person: {name} ({sign})\n"
                        f"Horoscope: {horoscope}\n"
                        f"News:\n{news_text}\n\n"
                        "Write the star news writeup:"
                    )
                ),
            ]
        )
        return {
            "writeup_complete": True,
            "writeup": response.content.strip(),
        }

    return execute


def star_news_actions(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> list[ActionSpec]:
    """Build the Star News Finder action set.

    Args:
        llm: Language model for extraction and generation.
        search_tool: Optional search tool for news lookup.
    """
    return [
        ActionSpec(
            name="extract_person",
            preconditions={"has_user_input": True},
            effects={"has_person": True},
            execute=_make_extract_person(llm),
        ),
        ActionSpec(
            name="extract_star_sign",
            preconditions={"has_person": True},
            effects={"has_star_person": True},
            execute=extract_star_sign,
        ),
        ActionSpec(
            name="retrieve_horoscope",
            preconditions={"has_star_person": True},
            effects={"has_horoscope": True},
            execute=retrieve_horoscope,
        ),
        ActionSpec(
            name="find_news_stories",
            preconditions={"has_star_person": True},
            effects={"has_news": True},
            cost=2.0,
            execute=_make_find_news_stories(llm, search_tool=search_tool),
        ),
        ActionSpec(
            name="star_news_writeup",
            preconditions={"has_horoscope": True, "has_news": True},
            effects={"writeup_complete": True},
            execute=_make_star_news_writeup(llm),
        ),
    ]


# ===========================================================================
# Example 2: Meal Preparation
# ===========================================================================


def _make_choose_cook(llm: BaseChatModel) -> _Execute:
    """LLM selects a cook based on available context."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a restaurant manager. Choose a cook for "
                        "the kitchen. Return a JSON-like response with the "
                        "cook's name and specialty, e.g.:\n"
                        "Name: Chef Marie\n"
                        "Specialty: French cuisine"
                    )
                ),
                HumanMessage(content="Select a cook for tonight's service."),
            ]
        )
        text = response.content.strip()
        name = "Chef"
        specialty = "General cuisine"
        for line in text.split("\n"):
            if "name" in line.lower() and ":" in line:
                name = line.split(":", 1)[1].strip()
            elif "specialty" in line.lower() and ":" in line:
                specialty = line.split(":", 1)[1].strip()
        return {
            "has_cook": True,
            "cook": {"name": name, "specialty": specialty},
        }

    return execute


def _make_take_order(llm: BaseChatModel) -> _Execute:
    """LLM takes a customer order."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        user_input = ws.get("user_input", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a waiter taking an order. Based on the "
                        "customer's input, determine the dish and any "
                        "special requests. Return:\n"
                        "Dish: <name>\n"
                        "Special requests: <requests or 'none'>"
                    )
                ),
                HumanMessage(
                    content=f"Customer says: {user_input}"
                    if user_input
                    else "The customer would like to order."
                ),
            ]
        )
        text = response.content.strip()
        dish = "Chef's special"
        requests = "none"
        for line in text.split("\n"):
            if "dish" in line.lower() and ":" in line:
                dish = line.split(":", 1)[1].strip()
            elif "special" in line.lower() and ":" in line:
                requests = line.split(":", 1)[1].strip()
        return {
            "has_order": True,
            "order": {"dish": dish, "special_requests": requests},
        }

    return execute


def _make_prepare_meal(llm: BaseChatModel) -> _Execute:
    """LLM describes the meal preparation."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        cook = ws.get("cook", {})
        order = ws.get("order", {})
        cook_name = cook.get("name", "The chef")
        dish = order.get("dish", "the dish")
        requests = order.get("special_requests", "none")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a kitchen narrator. Briefly describe the "
                        "meal preparation in 1-2 sentences. Include the "
                        "cook's name and the dish being prepared."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Cook: {cook_name}\n"
                        f"Dish: {dish}\n"
                        f"Special requests: {requests}\n\n"
                        "Describe the preparation:"
                    )
                ),
            ]
        )
        return {
            "meal_ready": True,
            "meal": {
                "dish": dish,
                "prepared_by": cook_name,
                "quality": "excellent",
                "description": response.content.strip(),
            },
        }

    return execute


def meal_prep_actions(llm: BaseChatModel) -> list[ActionSpec]:
    """Build the Meal Preparation action set.

    Args:
        llm: Language model for cook selection, ordering, and preparation.
    """
    return [
        ActionSpec(
            name="choose_cook",
            preconditions={"has_user_input": True},
            effects={"has_cook": True},
            execute=_make_choose_cook(llm),
        ),
        ActionSpec(
            name="take_order",
            preconditions={"has_user_input": True},
            effects={"has_order": True},
            execute=_make_take_order(llm),
        ),
        ActionSpec(
            name="prepare_meal",
            preconditions={"has_cook": True, "has_order": True},
            effects={"meal_ready": True},
            execute=_make_prepare_meal(llm),
        ),
    ]


# ===========================================================================
# Example 3: Write and Review
# ===========================================================================


def _make_craft_story(llm: BaseChatModel) -> _Execute:
    """LLM generates a short story from user input."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        user_input = ws.get("user_input", "a space adventure")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a creative writer. Given a topic or prompt, "
                        "write a very short story (3-4 sentences). Be "
                        "creative and engaging."
                    )
                ),
                HumanMessage(content=f"Write a short story about: {user_input}"),
            ]
        )
        return {"has_story": True, "story": response.content.strip()}

    return execute


def _make_assess_story(llm: BaseChatModel) -> _Execute:
    """LLM assesses a story — approves if quality is sufficient."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        story = ws.get("story", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a story editor. Assess the following story. "
                        "If it is reasonably well-written and engaging, "
                        "respond with EXACTLY 'APPROVED'. Otherwise respond "
                        "with EXACTLY 'REJECTED' followed by brief feedback."
                    )
                ),
                HumanMessage(content=f"Story to assess:\n{story}"),
            ]
        )
        text = response.content.strip()
        approved = text.upper().startswith("APPROVED")
        return {"story_approved": approved}

    return execute


def _make_assess_story_strict(llm: BaseChatModel) -> _Execute:
    """LLM assesses a story — rejects unless it has been revised.

    Returns ``story_approved=False, needs_revision=True`` on first pass,
    which deviates from the declared effect and triggers replanning.
    """

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        story = ws.get("story", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict story editor. Assess the story. "
                        "If the story contains '[REVISED]' at the start, "
                        "it has been improved — respond with 'APPROVED'. "
                        "Otherwise, respond with 'REJECTED: <brief feedback>'."
                    )
                ),
                HumanMessage(content=f"Story to assess:\n{story}"),
            ]
        )
        text = response.content.strip()
        if "REVISED" in story and text.upper().startswith("APPROVED"):
            return {"story_approved": True}
        return {"story_approved": False, "needs_revision": True}

    return execute


def _make_revise_story(llm: BaseChatModel) -> _Execute:
    """LLM rewrites the story with improvements."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        story = ws.get("story", "")
        user_input = ws.get("user_input", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a story revision agent. Rewrite and improve "
                        "the story below. Make it more engaging and polished. "
                        "Start the revised version with '[REVISED] '."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original topic: {user_input}\n\n"
                        f"Story to revise:\n{story}"
                    )
                ),
            ]
        )
        text = response.content.strip()
        if not text.startswith("[REVISED]"):
            text = f"[REVISED] {text}"
        return {"has_story": True, "story": text, "story_approved": True}

    return execute


def _make_finalize_review(llm: BaseChatModel) -> _Execute:
    """LLM produces the final reviewed story."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        story = ws.get("story", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a literary reviewer. Write a brief review "
                        "(1-2 sentences) of the following story. Be "
                        "constructive and mention what works well."
                    )
                ),
                HumanMessage(content=f"Story to review:\n{story}"),
            ]
        )
        return {
            "review_complete": True,
            "reviewed_story": {
                "story": story,
                "review": response.content.strip(),
                "reviewer": "NYT Book Review",
            },
        }

    return execute


def write_review_actions(
    llm: BaseChatModel,
    *,
    strict: bool = False,
    include_revise: bool = False,
) -> list[ActionSpec]:
    """Build the Write-and-Review action set.

    Args:
        llm: Language model for story generation and assessment.
        strict: When True, uses the strict assessor that rejects
            unrevised stories (triggers replanning).
        include_revise: When True, adds a ``revise_story`` action whose
            extra precondition (``needs_revision``) wins via A* specificity.
    """
    assess_fn = (
        _make_assess_story_strict(llm) if strict else _make_assess_story(llm)
    )
    actions = [
        ActionSpec(
            name="craft_story",
            preconditions={"has_user_input": True},
            effects={"has_story": True},
            execute=_make_craft_story(llm),
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
            execute=_make_finalize_review(llm),
        ),
    ]
    if include_revise:
        actions.append(
            ActionSpec(
                name="revise_story",
                preconditions={"has_story": True, "needs_revision": True},
                effects={"story_approved": True},
                execute=_make_revise_story(llm),
            )
        )
    return actions


# ===========================================================================
# Example 4: Fact Checker
# ===========================================================================


def _make_extract_assertions(llm: BaseChatModel) -> _Execute:
    """LLM extracts factual claims from content."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        content = ws.get("content", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract individual factual claims from the text. "
                        "Return each claim on its own line, prefixed with "
                        "'- '. Be specific and extract concrete assertions."
                    )
                ),
                HumanMessage(content=f"Extract claims from:\n{content}"),
            ]
        )
        claims = [
            line.strip().lstrip("- ").strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != "-"
        ]
        return {
            "has_assertions": True,
            "assertions": [
                {"claim": claim, "source": content[:50]}
                for claim in claims
            ],
        }

    return execute


def _make_rationalize_assertions(llm: BaseChatModel) -> _Execute:
    """LLM deduplicates and rationalizes assertions."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        assertions = ws.get("assertions", [])
        claims_text = "\n".join(
            f"- {a['claim']}" for a in assertions
        )
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a fact-check preparation agent. Given "
                        "extracted claims, deduplicate, rationalize, and "
                        "rank them by importance. Return each claim on its "
                        "own line, prefixed with '- '."
                    )
                ),
                HumanMessage(
                    content=f"Claims to rationalize:\n{claims_text}"
                ),
            ]
        )
        rationalized = [
            line.strip().lstrip("- ").strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != "-"
        ]
        return {
            "has_rationalized": True,
            "rationalized_assertions": [
                {**a, "importance": "high", "rationalized": r}
                for a, r in zip(
                    assertions,
                    rationalized + [""] * max(0, len(assertions) - len(rationalized)),
                )
            ],
        }

    return execute


def _make_check_facts(llm: BaseChatModel) -> _Execute:
    """LLM verifies each assertion."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        assertions = ws.get("rationalized_assertions", [])
        claims_text = "\n".join(
            f"{i+1}. {a['claim']}"
            for i, a in enumerate(assertions)
        )
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a fact checker. For each numbered claim, "
                        "determine if it is TRUE or FALSE. Return one line "
                        "per claim in the format: '<number>. TRUE/FALSE'"
                    )
                ),
                HumanMessage(content=f"Verify these claims:\n{claims_text}"),
            ]
        )
        lines = response.content.strip().split("\n")
        checks = []
        for i, a in enumerate(assertions):
            verified = True
            for line in lines:
                if line.strip().startswith(f"{i+1}."):
                    verified = "TRUE" in line.upper()
                    break
            checks.append({
                "claim": a["claim"],
                "verified": verified,
                "confidence": 0.95 if verified else 0.3,
            })
        return {
            "fact_check_complete": True,
            "fact_check": {
                "total_claims": len(checks),
                "verified": sum(1 for c in checks if c["verified"]),
                "checks": checks,
            },
        }

    return execute


def fact_checker_actions(llm: BaseChatModel) -> list[ActionSpec]:
    """Build the Fact Checker action set.

    Args:
        llm: Language model for claim extraction and verification.
    """
    return [
        ActionSpec(
            name="extract_assertions",
            preconditions={"has_content": True},
            effects={"has_assertions": True},
            execute=_make_extract_assertions(llm),
        ),
        ActionSpec(
            name="rationalize_assertions",
            preconditions={"has_assertions": True},
            effects={"has_rationalized": True},
            execute=_make_rationalize_assertions(llm),
        ),
        ActionSpec(
            name="check_facts",
            preconditions={"has_rationalized": True},
            effects={"fact_check_complete": True},
            execute=_make_check_facts(llm),
        ),
    ]
