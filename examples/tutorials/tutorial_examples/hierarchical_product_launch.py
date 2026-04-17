r"""Hierarchical product launch — Tier 3 ``MultiGoal`` sequential tutorial.

A SaaS product launch decomposes naturally into three stages that must
run in order: you cannot build what you have not specified, and you
cannot announce what you have not built.  LangGOAP expresses this with
a :class:`~langgoap.MultiGoal` in ``"sequential"`` mode — the observer
plans and executes sub-goal 0 to completion, then uses the resulting
world state as the starting state for sub-goal 1, and so on.

The tutorial spotlights four features at once:

1. **``MultiGoal`` sequential decomposition** — three sub-goals in one
   ``graph.invoke`` call; the observer chains the world state between
   stages with no user-side coordination.
2. **Per-stage A* planning** — each sub-goal plans independently, so
   the plan for stage 2 never mentions stage 3's actions (the planner
   literally never sees them during stage 2).
3. **State hand-off** — effects from stage 1 (``prd_approved=True``)
   become preconditions for stage 2's actions; effects from stage 2
   (``qa_passed=True``) satisfy stage 3's launch preconditions.
4. **Per-sub-goal accounting reset** — ``replan_count``,
   ``blacklisted_actions``, and ``action_failure_counts`` all reset at
   every sub-goal advance, so a transient failure while writing the
   PRD does not eat into the build stage's replan budget.

Each content-generating action (``research_market``, ``write_prd``,
``prepare_marketing``, ``announce_launch``) uses the LLM for real text
generation.  Engineering actions (``implement_features``, ``qa_test``)
remain stubs — they represent code-level work that a text LLM cannot
meaningfully perform.

Stage breakdown
---------------

- **Stage 1 — Discovery** (``prd_approved=True``).
  ``research_market`` unlocks ``write_prd``, which locks the product
  requirements doc.
- **Stage 2 — Build** (``qa_passed=True``).
  ``implement_features`` requires ``prd_approved`` (handed off from
  stage 1); ``qa_test`` gates the build on passing tests.
- **Stage 3 — Launch** (``launched=True``).
  ``prepare_marketing`` requires the QA gate; ``announce_launch``
  requires the marketing gate and cements ``launched=True``.

Action catalog
--------------

+---------------------+-----------------------------------+----------------------------------+------+
| Action              | Preconditions                     | Effects                          | Cost |
+=====================+===================================+==================================+======+
| research_market     | (none)                            | market_data=True                 | 2.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| write_prd           | market_data=True                  | prd_approved=True                | 3.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| implement_features  | prd_approved=True                 | code_written=True                | 5.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| qa_test             | code_written=True                 | qa_passed=True                   | 2.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| prepare_marketing   | qa_passed=True                    | marketing_ready=True             | 2.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| announce_launch     | marketing_ready=True,             | launched=True                    | 1.0  |
|                     | qa_passed=True                    |                                  |      |
+---------------------+-----------------------------------+----------------------------------+------+
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgoap import ActionSpec, GoalSpec, MultiGoal

_Execute = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# LLM-powered execute factories
# ---------------------------------------------------------------------------


def _make_research_market(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> _Execute:
    """Research the market landscape for a product."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        product = ws.get("product_name", ws.get("task", "SaaS product"))

        if search_tool is not None:
            raw = search_tool.invoke(f"{product} market analysis")
            items = raw if isinstance(raw, list) else [raw]
            findings = "\n".join(
                r.get("content", r.get("snippet", str(r))) for r in items
            )
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a market research analyst. Given a product "
                            "description, provide a concise market analysis "
                            "covering: target market, competitors, market size, "
                            "and key trends. Be specific and data-oriented."
                        )
                    ),
                    HumanMessage(
                        content=f"Research the market for: {product}"
                    ),
                ]
            )
            findings = response.content.strip()

        return {"market_data": True, "market_research": findings}

    return execute


def _make_write_prd(llm: BaseChatModel) -> _Execute:
    """Write a product requirements document from market research."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        research = ws.get("market_research", "")
        product = ws.get("product_name", ws.get("task", "SaaS product"))

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a product manager. Given market research, "
                        "write a concise PRD (Product Requirements Document) "
                        "that includes: problem statement, target users, "
                        "key features (3-5), success metrics, and timeline. "
                        "Keep it actionable and specific."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Product: {product}\n\n"
                        f"Market research:\n{research}\n\n"
                        "Write the PRD:"
                    )
                ),
            ]
        )
        return {"prd_approved": True, "prd_content": response.content.strip()}

    return execute


def _make_implement_features() -> _Execute:
    """Implement features — stub (code-level work, not LLM-suitable)."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        del ws
        return {"code_written": True}

    return execute


def _make_qa_test() -> _Execute:
    """Run QA tests — stub (testing infrastructure, not LLM-suitable)."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        del ws
        return {"qa_passed": True}

    return execute


def _make_prepare_marketing(llm: BaseChatModel) -> _Execute:
    """Prepare marketing materials for the product launch."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        prd = ws.get("prd_content", "")
        product = ws.get("product_name", ws.get("task", "SaaS product"))

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a marketing strategist. Given a product "
                        "and its PRD, create a concise marketing brief "
                        "that includes: value proposition, key messages "
                        "(3 bullet points), target channels, and a tagline. "
                        "Make it compelling and launch-ready."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Product: {product}\n\n"
                        f"PRD summary:\n{prd}\n\n"
                        "Prepare the marketing brief:"
                    )
                ),
            ]
        )
        return {
            "marketing_ready": True,
            "marketing_content": response.content.strip(),
        }

    return execute


def _make_announce_launch(llm: BaseChatModel) -> _Execute:
    """Write the public launch announcement."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        marketing = ws.get("marketing_content", "")
        product = ws.get("product_name", ws.get("task", "SaaS product"))

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a launch communications specialist. Given "
                        "a product and marketing brief, write a concise "
                        "launch announcement (150-250 words) suitable for "
                        "a blog post or press release. Include the key value "
                        "proposition and a call to action."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Product: {product}\n\n"
                        f"Marketing brief:\n{marketing}\n\n"
                        "Write the launch announcement:"
                    )
                ),
            ]
        )
        return {
            "launched": True,
            "launch_announcement": response.content.strip(),
        }

    return execute


# ---------------------------------------------------------------------------
# Action catalog factory
# ---------------------------------------------------------------------------


def product_launch_actions(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> list[ActionSpec]:
    """Return the six-action catalog that spans all three launch stages.

    Content-generating actions use the LLM for real text generation.
    Engineering actions (implement_features, qa_test) remain stubs.

    Costs reflect rough relative effort so ``Plan.total_cost`` is a
    meaningful readout: discovery 5, build 7, launch 3 (sum 15).

    Args:
        llm: Language model powering content-generating actions.
        search_tool: Optional search tool for ``research_market``
            (e.g. ``TavilySearchResults``).
    """
    return [
        ActionSpec(
            name="research_market",
            preconditions={},
            effects={"market_data": True},
            cost=2.0,
            execute=_make_research_market(llm, search_tool=search_tool),
        ),
        ActionSpec(
            name="write_prd",
            preconditions={"market_data": True},
            effects={"prd_approved": True},
            cost=3.0,
            execute=_make_write_prd(llm),
        ),
        ActionSpec(
            name="implement_features",
            preconditions={"prd_approved": True},
            effects={"code_written": True},
            cost=5.0,
            execute=_make_implement_features(),
        ),
        ActionSpec(
            name="qa_test",
            preconditions={"code_written": True},
            effects={"qa_passed": True},
            cost=2.0,
            execute=_make_qa_test(),
        ),
        ActionSpec(
            name="prepare_marketing",
            preconditions={"qa_passed": True},
            effects={"marketing_ready": True},
            cost=2.0,
            execute=_make_prepare_marketing(llm),
        ),
        ActionSpec(
            name="announce_launch",
            preconditions={"marketing_ready": True, "qa_passed": True},
            effects={"launched": True},
            cost=1.0,
            execute=_make_announce_launch(llm),
        ),
    ]


# ---------------------------------------------------------------------------
# Starting state and goal
# ---------------------------------------------------------------------------


def product_launch_start() -> dict[str, Any]:
    """Clean-slate world state: no milestones reached yet.

    All six boolean flags are seeded ``False`` so the planner has an
    unambiguous starting point and the notebook can flip individual
    flags to ``True`` to demonstrate pre-satisfied sub-goal advance.
    """
    return {
        "market_data": False,
        "prd_approved": False,
        "code_written": False,
        "qa_passed": False,
        "marketing_ready": False,
        "launched": False,
    }


def product_launch_goal() -> MultiGoal:
    """Return the three-stage sequential ``MultiGoal`` for the launch.

    Sub-goal 0 locks the PRD, sub-goal 1 ships a QA-passing build,
    sub-goal 2 announces the launch.  Each sub-goal is a plain
    :class:`~langgoap.GoalSpec` with a single boolean condition — the
    observer chains them in order.
    """
    return MultiGoal(
        goals=(
            GoalSpec(conditions={"prd_approved": True}),  # Stage 1 — Discovery
            GoalSpec(conditions={"qa_passed": True}),  # Stage 2 — Build
            GoalSpec(conditions={"launched": True}),  # Stage 3 — Launch
        ),
        mode="sequential",
    )
