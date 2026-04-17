r"""Content Builder Agent — Tier 3 tutorial for multi-objective CSP and
the fluent :class:`~langgoap.constraints.ConstraintBuilder`.

Translates the deepagents *content builder* pattern — a multi-format
content marketing workflow driven by subagents that produce blog posts,
LinkedIn updates, and Twitter threads — into LangGOAP.  Where the
original pattern uses a supervisor LLM and subagent routing, the GOAP
version replaces the orchestration layer with A\* → CSP planning while
retaining **real LLM intelligence** inside each content-generating action.

What this tutorial spotlights
-----------------------------

1. **Conditional format generation via CSP enumeration** — there are
   two competing blog writers (``write_blog_fast``, ``write_blog_deep``)
   with the same effect ``blog_drafted=True`` but different cost,
   writer_hours, and quality profiles.  A\* by itself always picks the
   cheaper ``write_blog_fast``.  When the goal adds a **hard**
   ``quality_score >= 8`` constraint, the primary plan fails CSP
   validation, the pipeline enumerates alternatives by blacklisting
   each action in the rejected plan, and the deep writer is selected.
   This is the cleanest in-tree demonstration of the
   ``planner/pipeline.py::enumerate_alternatives`` path — it is
   otherwise invisible to notebook readers.

2. **Multi-objective CSP with hard and soft levels** — the premium
   campaign goal carries a hard ``cost_usd`` cap (violation →
   INFEASIBLE), a soft ``writer_hours`` cap (violation → penalty in
   :class:`~langgoap.score.HardSoftScore`), and both a MINIMIZE and a
   MAXIMIZE objective.  The CSP phase populates the HardSoftScore so
   the notebook can show the feasibility flag, hard penalty, and soft
   penalty side by side.

3. **Fluent ConstraintBuilder as a ConstraintProvider analogue** —
   every goal in this module has a hand-rolled variant and a
   ``ConstraintBuilder`` variant, and the integration test pins that
   the two produce structurally equivalent
   :class:`~langgoap.goals.GoalSpec` instances.  This is the
   fluent constraint-provider pattern and the reason
   ``GoalSpec.from_builder`` exists.

GOAP modelling
--------------

**World state flags** (all ``False`` at start):

- ``research_done`` — topic research complete.
- ``outline_ready`` — content outline drafted.
- ``blog_drafted`` / ``blog_cover_ready`` / ``blog_live`` — blog path.
- ``linkedin_drafted`` / ``linkedin_image_ready`` / ``linkedin_live``
  — LinkedIn path.
- ``twitter_drafted`` / ``twitter_image_ready`` / ``twitter_live`` —
  Twitter path.

**Action catalog** (12 actions):

+-----------------------+-------------------+----------------------+-----+-------------------------------+
| Action                | Pre               | Effects              | Cst | Resources                     |
+=======================+===================+======================+=====+===============================+
| research_topic        | (none)            | research_done        | 1.0 | writer_hours=1, cost_usd=5    |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| draft_outline         | research_done     | outline_ready        | 1.0 | writer_hours=2, cost_usd=10   |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| write_blog_fast       | outline_ready     | blog_drafted         | 3.0 | writer_hours=2, cost_usd=30,  |
|                       |                   |                      |     | quality_score=5               |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| write_blog_deep       | outline_ready     | blog_drafted         | 6.0 | writer_hours=6, cost_usd=80,  |
|                       |                   |                      |     | quality_score=10              |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| generate_blog_cover   | blog_drafted      | blog_cover_ready     | 2.0 | gpu_minutes=5, cost_usd=15    |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| publish_blog          | blog_drafted,     | blog_live            | 1.0 | (none)                        |
|                       | blog_cover_ready  |                      |     |                               |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| write_linkedin_post   | outline_ready     | linkedin_drafted     | 2.0 | writer_hours=1, cost_usd=10,  |
|                       |                   |                      |     | quality_score=4               |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| generate_linkedin_img | linkedin_drafted  | linkedin_image_ready | 1.0 | gpu_minutes=2, cost_usd=5     |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| publish_linkedin      | linkedin_drafted, | linkedin_live        | 1.0 | (none)                        |
|                       | linkedin_image... |                      |     |                               |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| write_twitter_thread  | outline_ready     | twitter_drafted      | 1.0 | writer_hours=0.5, cost_usd=4, |
|                       |                   |                      |     | quality_score=2               |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| generate_twitter_img  | twitter_drafted   | twitter_image_ready  | 1.0 | gpu_minutes=1, cost_usd=2     |
+-----------------------+-------------------+----------------------+-----+-------------------------------+
| publish_twitter       | twitter_drafted,  | twitter_live         | 1.0 | (none)                        |
|                       | twitter_image_... |                      |     |                               |
+-----------------------+-------------------+----------------------+-----+-------------------------------+

Resource numbers pin every plan's aggregated totals so the integration
test can assert exact values rather than ``>=`` bounds — structural
tests per the notebook 11/12 audit.

LLM vs stub actions
-------------------

Content-generating actions (research, outline, blog writing, LinkedIn,
Twitter) use the LLM for real text generation.  Image generation and
publish actions remain stubs because a text LLM cannot produce images
and platform publishing requires API credentials.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgoap import ActionSpec, ConstraintSpec, GoalSpec
from langgoap.constraints import ConstraintBuilder
from langgoap.goals import ObjectiveDirection

_Execute = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Stub execute — for actions that cannot use a text LLM (image gen, publish)
# ---------------------------------------------------------------------------


def _make_stub_execute(effects: dict[str, Any]) -> _Execute:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        del ws
        return dict(effects)

    return execute


# ---------------------------------------------------------------------------
# LLM-powered execute factories
# ---------------------------------------------------------------------------


def _make_research_topic(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> _Execute:
    """Research a topic using LLM (or optional search tool)."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        topic = ws.get("topic", ws.get("task", "content marketing"))

        if search_tool is not None:
            raw = search_tool.invoke(topic)
            items = raw if isinstance(raw, list) else [raw]
            findings = "\n".join(
                r.get("content", r.get("snippet", str(r))) for r in items
            )
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a research agent. Given a topic, provide "
                            "3-5 key findings with specific facts and data points "
                            "that would be useful for creating content. Return each "
                            "finding on its own line, prefixed with '- '."
                        )
                    ),
                    HumanMessage(
                        content=f"Research the topic: {topic}"
                    ),
                ]
            )
            findings = response.content.strip()

        return {"research_done": True, "research_findings": findings}

    return execute


def _make_draft_outline(llm: BaseChatModel) -> _Execute:
    """Draft a content outline from research findings."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        findings = ws.get("research_findings", "")
        topic = ws.get("topic", ws.get("task", ""))

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a content strategist. Given research findings "
                        "about a topic, create a structured outline with 5-7 "
                        "sections suitable for a blog post, LinkedIn article, "
                        "and Twitter thread. Return each section on its own "
                        "line as a numbered item."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Research findings:\n{findings}\n\n"
                        "Create a content outline:"
                    )
                ),
            ]
        )
        return {
            "outline_ready": True,
            "content_outline": response.content.strip(),
        }

    return execute


def _make_write_blog_fast(llm: BaseChatModel) -> _Execute:
    """Write a concise, quick-turnaround blog post."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        outline = ws.get("content_outline", "")
        topic = ws.get("topic", ws.get("task", ""))
        findings = ws.get("research_findings", "")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a content writer focused on speed and clarity. "
                        "Given a topic, outline, and research, write a concise "
                        "blog post (300-500 words). Be direct and informative "
                        "without deep analysis. Include a compelling title."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Outline:\n{outline}\n\n"
                        f"Research:\n{findings}\n\n"
                        "Write a concise blog post:"
                    )
                ),
            ]
        )
        return {
            "blog_drafted": True,
            "blog_content": response.content.strip(),
        }

    return execute


def _make_write_blog_deep(llm: BaseChatModel) -> _Execute:
    """Write a thorough, in-depth blog post."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        outline = ws.get("content_outline", "")
        topic = ws.get("topic", ws.get("task", ""))
        findings = ws.get("research_findings", "")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior content writer producing in-depth "
                        "articles. Given a topic, outline, and research, write "
                        "a comprehensive blog post (800-1200 words). Include "
                        "detailed analysis, examples, data points from the "
                        "research, and actionable takeaways. Use section "
                        "headings matching the outline. Include a compelling title."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Outline:\n{outline}\n\n"
                        f"Research:\n{findings}\n\n"
                        "Write an in-depth blog post:"
                    )
                ),
            ]
        )
        return {
            "blog_drafted": True,
            "blog_content": response.content.strip(),
        }

    return execute


def _make_write_linkedin_post(llm: BaseChatModel) -> _Execute:
    """Write a LinkedIn post from the content outline."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        outline = ws.get("content_outline", "")
        topic = ws.get("topic", ws.get("task", ""))
        findings = ws.get("research_findings", "")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a LinkedIn content specialist. Given a topic, "
                        "outline, and research, write a professional LinkedIn "
                        "post (150-300 words). Use a hook opening, bullet points "
                        "for key insights, and a call-to-action. Include "
                        "relevant hashtags."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Outline:\n{outline}\n\n"
                        f"Research:\n{findings}\n\n"
                        "Write a LinkedIn post:"
                    )
                ),
            ]
        )
        return {
            "linkedin_drafted": True,
            "linkedin_content": response.content.strip(),
        }

    return execute


def _make_write_twitter_thread(llm: BaseChatModel) -> _Execute:
    """Write a Twitter thread from the content outline."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        outline = ws.get("content_outline", "")
        topic = ws.get("topic", ws.get("task", ""))
        findings = ws.get("research_findings", "")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a Twitter thread writer. Given a topic, "
                        "outline, and research, write a 5-7 tweet thread. "
                        "Start with a hook tweet, provide key insights in "
                        "subsequent tweets (each under 280 characters), and "
                        "end with a summary/CTA tweet. Number each tweet."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Outline:\n{outline}\n\n"
                        f"Research:\n{findings}\n\n"
                        "Write a Twitter thread:"
                    )
                ),
            ]
        )
        return {
            "twitter_drafted": True,
            "twitter_content": response.content.strip(),
        }

    return execute


# ---------------------------------------------------------------------------
# Action catalog factory
# ---------------------------------------------------------------------------


def content_builder_actions(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> list[ActionSpec]:
    """Return the 12-action content marketing catalog.

    Content-generating actions use the LLM for real text generation.
    Image generation and publish actions remain stubs (a text LLM cannot
    produce images and publishing requires platform API credentials).

    Costs are tuned so A\\* naturally prefers the *fast* blog writer
    and the cheaper publishing paths.  The integration test relies on
    this ordering to drive CSP enumeration on the quality constraint.

    Args:
        llm: Language model powering content-generating actions.
        search_tool: Optional search tool for ``research_topic``
            (e.g. ``TavilySearchResults``).
    """
    return [
        # Shared discovery & outline.
        ActionSpec(
            name="research_topic",
            preconditions={},
            effects={"research_done": True},
            cost=1.0,
            resources={"writer_hours": 1.0, "cost_usd": 5.0},
            execute=_make_research_topic(llm, search_tool=search_tool),
        ),
        ActionSpec(
            name="draft_outline",
            preconditions={"research_done": True},
            effects={"outline_ready": True},
            cost=1.0,
            resources={"writer_hours": 2.0, "cost_usd": 10.0},
            execute=_make_draft_outline(llm),
        ),
        # Two competing blog writers — same effect, different profile.
        ActionSpec(
            name="write_blog_fast",
            preconditions={"outline_ready": True},
            effects={"blog_drafted": True},
            cost=3.0,
            resources={
                "writer_hours": 2.0,
                "cost_usd": 30.0,
                "quality_score": 5.0,
            },
            execute=_make_write_blog_fast(llm),
        ),
        ActionSpec(
            name="write_blog_deep",
            preconditions={"outline_ready": True},
            effects={"blog_drafted": True},
            cost=6.0,
            resources={
                "writer_hours": 6.0,
                "cost_usd": 80.0,
                "quality_score": 10.0,
            },
            execute=_make_write_blog_deep(llm),
        ),
        # Image gen: text LLM cannot produce images — stub.
        ActionSpec(
            name="generate_blog_cover",
            preconditions={"blog_drafted": True},
            effects={"blog_cover_ready": True},
            cost=2.0,
            resources={"gpu_minutes": 5.0, "cost_usd": 15.0},
            execute=_make_stub_execute({"blog_cover_ready": True}),
        ),
        ActionSpec(
            name="publish_blog",
            preconditions={"blog_drafted": True, "blog_cover_ready": True},
            effects={"blog_live": True},
            cost=1.0,
            execute=_make_stub_execute({"blog_live": True}),
        ),
        # LinkedIn path.
        ActionSpec(
            name="write_linkedin_post",
            preconditions={"outline_ready": True},
            effects={"linkedin_drafted": True},
            cost=2.0,
            resources={
                "writer_hours": 1.0,
                "cost_usd": 10.0,
                "quality_score": 4.0,
            },
            execute=_make_write_linkedin_post(llm),
        ),
        ActionSpec(
            name="generate_linkedin_image",
            preconditions={"linkedin_drafted": True},
            effects={"linkedin_image_ready": True},
            cost=1.0,
            resources={"gpu_minutes": 2.0, "cost_usd": 5.0},
            execute=_make_stub_execute({"linkedin_image_ready": True}),
        ),
        ActionSpec(
            name="publish_linkedin",
            preconditions={
                "linkedin_drafted": True,
                "linkedin_image_ready": True,
            },
            effects={"linkedin_live": True},
            cost=1.0,
            execute=_make_stub_execute({"linkedin_live": True}),
        ),
        # Twitter path.
        ActionSpec(
            name="write_twitter_thread",
            preconditions={"outline_ready": True},
            effects={"twitter_drafted": True},
            cost=1.0,
            resources={
                "writer_hours": 0.5,
                "cost_usd": 4.0,
                "quality_score": 2.0,
            },
            execute=_make_write_twitter_thread(llm),
        ),
        ActionSpec(
            name="generate_twitter_image",
            preconditions={"twitter_drafted": True},
            effects={"twitter_image_ready": True},
            cost=1.0,
            resources={"gpu_minutes": 1.0, "cost_usd": 2.0},
            execute=_make_stub_execute({"twitter_image_ready": True}),
        ),
        ActionSpec(
            name="publish_twitter",
            preconditions={
                "twitter_drafted": True,
                "twitter_image_ready": True,
            },
            effects={"twitter_live": True},
            cost=1.0,
            execute=_make_stub_execute({"twitter_live": True}),
        ),
    ]


def content_builder_start() -> dict[str, Any]:
    """Clean-slate world state — no milestones reached."""
    return {
        "research_done": False,
        "outline_ready": False,
        "blog_drafted": False,
        "blog_cover_ready": False,
        "blog_live": False,
        "linkedin_drafted": False,
        "linkedin_image_ready": False,
        "linkedin_live": False,
        "twitter_drafted": False,
        "twitter_image_ready": False,
        "twitter_live": False,
    }


# ---------------------------------------------------------------------------
# Goal factories — hand-rolled and fluent-builder variants
# ---------------------------------------------------------------------------


def blog_only_goal() -> GoalSpec:
    """Simplest goal: ``blog_live=True`` with no constraints.

    A\\* picks the cheapest blog chain (``write_blog_fast``) and the
    pipeline skips CSP entirely because
    :func:`~langgoap.planner.pipeline.needs_csp` returns ``False``.
    """
    return GoalSpec(conditions={"blog_live": True})


def multi_channel_goal(
    *,
    max_cost_usd: float | None = None,
    max_cost_level: Literal["hard", "soft"] = "hard",
    minimize_cost: bool = True,
) -> GoalSpec:
    """All three channels live, with an optional cost cap and objective.

    Args:
        max_cost_usd: Optional aggregated ``cost_usd`` ceiling.  When
            ``level="hard"`` and the primary plan exceeds the cap the
            pipeline returns ``CSPStatus.INFEASIBLE`` because no
            alternative path exists (LinkedIn and Twitter chains are
            single-writer).
        max_cost_level: ``"hard"`` or ``"soft"``.
        minimize_cost: Whether to attach a
            ``cost_usd → MINIMIZE`` objective (routes the goal through
            CSP so a :class:`~langgoap.score.HardSoftScore` is
            populated even without any constraints).
    """
    conditions: dict[str, Any] = {
        "blog_live": True,
        "linkedin_live": True,
        "twitter_live": True,
    }
    constraints: tuple[ConstraintSpec, ...] = ()
    if max_cost_usd is not None:
        constraints = (
            ConstraintSpec(
                key="cost_usd",
                max=float(max_cost_usd),
                level=max_cost_level,
            ),
        )
    objectives: dict[str, ObjectiveDirection] | None = None
    if minimize_cost:
        objectives = {"cost_usd": ObjectiveDirection.MINIMIZE}
    return GoalSpec(
        conditions=conditions,
        constraints=constraints,
        objectives=objectives,
    )


def quality_blog_goal(*, min_quality: float) -> GoalSpec:
    """Blog-only goal with a **hard** ``quality_score >= min_quality``.

    When ``min_quality > 5`` the fast writer's aggregated
    ``quality_score`` is below the floor.  The primary A\\* plan
    therefore fails CSP validation and the pipeline's
    :func:`~langgoap.planner.pipeline.enumerate_alternatives` path runs:
    it blacklists each action in the rejected plan one at a time and
    re-runs A\\*.  The blacklist on ``write_blog_fast`` produces the
    alternative ``write_blog_deep`` plan with ``quality_score=10``,
    which CSP then selects.

    This is the canonical in-tree demo of enumeration triggered by a
    hard-min violation.
    """
    return GoalSpec(
        conditions={"blog_live": True},
        constraints=(
            ConstraintSpec(
                key="quality_score",
                min=float(min_quality),
                level="hard",
            ),
        ),
        objectives={"quality_score": ObjectiveDirection.MAXIMIZE},
    )


def quality_blog_goal_fluent(*, min_quality: float) -> GoalSpec:
    """Same as :func:`quality_blog_goal` but built via the fluent builder.

    Used by the integration test to pin that hand-rolled and
    ``ConstraintBuilder``-produced :class:`~langgoap.goals.GoalSpec`
    instances are structurally identical.
    """
    output = ConstraintBuilder.build(
        ConstraintBuilder.for_plan()
        .sum_resource("quality_score")
        .bounded(min=float(min_quality))
        .penalize(level="hard", weight=1.0)
        .as_constraint("quality_score"),
        ConstraintBuilder.for_plan()
        .sum_resource("quality_score")
        .maximize()
        .as_objective("quality_score"),
    )
    return GoalSpec.from_builder(
        conditions={"blog_live": True},
        builder_output=output,
    )


def premium_campaign_goal(
    *,
    hard_cost_cap: float,
    soft_writer_hours_cap: float,
) -> GoalSpec:
    """Multi-channel goal exercising hard + soft + MIN + MAX at once.

    - Hard: ``cost_usd <= hard_cost_cap`` (violation → INFEASIBLE).
    - Soft: ``writer_hours <= soft_writer_hours_cap`` (violation →
      penalty in ``HardSoftScore.soft``; feasibility preserved).
    - Objective: ``cost_usd → MINIMIZE`` (subtracted from soft).
    - Objective: ``quality_score → MAXIMIZE`` (added to soft).
    """
    return GoalSpec(
        conditions={
            "blog_live": True,
            "linkedin_live": True,
            "twitter_live": True,
        },
        constraints=(
            ConstraintSpec(
                key="cost_usd", max=hard_cost_cap, level="hard"
            ),
            ConstraintSpec(
                key="writer_hours",
                max=soft_writer_hours_cap,
                level="soft",
                weight=1.0,
            ),
        ),
        objectives={
            "cost_usd": ObjectiveDirection.MINIMIZE,
            "quality_score": ObjectiveDirection.MAXIMIZE,
        },
    )


def premium_campaign_goal_fluent(
    *,
    hard_cost_cap: float,
    soft_writer_hours_cap: float,
) -> GoalSpec:
    """Fluent-builder twin of :func:`premium_campaign_goal`.

    Shows every ``ConstraintBuilder`` terminator in one chain set:
    ``as_constraint`` with ``level="hard"``, ``as_constraint`` with
    ``level="soft"``, and two ``as_objective`` chains (one MINIMIZE,
    one MAXIMIZE).
    """
    output = ConstraintBuilder.build(
        ConstraintBuilder.for_plan()
        .sum_resource("cost_usd")
        .bounded(max=hard_cost_cap)
        .penalize(level="hard", weight=1.0)
        .as_constraint("cost_usd"),
        ConstraintBuilder.for_plan()
        .sum_resource("writer_hours")
        .bounded(max=soft_writer_hours_cap)
        .penalize(level="soft", weight=1.0)
        .as_constraint("writer_hours"),
        ConstraintBuilder.for_plan()
        .sum_resource("cost_usd")
        .minimize()
        .as_objective("cost_usd"),
        ConstraintBuilder.for_plan()
        .sum_resource("quality_score")
        .maximize()
        .as_objective("quality_score"),
    )
    return GoalSpec.from_builder(
        conditions={
            "blog_live": True,
            "linkedin_live": True,
            "twitter_live": True,
        },
        builder_output=output,
    )
