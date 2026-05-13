"""Integration tests for ``langgoap.integrations.prebuilt.create_goap_agent``.

Layer A of the three-layer low-code on-ramp (AD-2).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from langgoap.goals import GoalSpec
from langgoap.integrations import create_goap_agent

# Repo-relative path where the README-companion artifacts (PNGs and
# transcripts) live. The api-marked README test re-generates these so
# the README stays in sync with the live behaviour of create_goap_agent.
_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / ".github" / "images"


@tool
def gather_data() -> str:
    """Gather raw data."""
    return "raw data"


@tool
def process_data() -> str:
    """Process gathered data."""
    return "processed"


@tool
def write_report() -> str:
    """Write the final report."""
    return "report.pdf"


class TestCreateGoapAgentBasic:
    def test_returns_compiled_state_graph(self) -> None:
        agent = create_goap_agent(
            tools=[gather_data, process_data, write_report],
            goal=GoalSpec(conditions={"report_written": True}),
            effects={
                "gather_data": {"has_data": True},
                "process_data": {"data_processed": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "process_data": {"has_data": True},
                "write_report": {"data_processed": True},
            },
        )
        assert isinstance(agent, CompiledStateGraph)

    def test_end_to_end_execution(self) -> None:
        agent = create_goap_agent(
            tools=[gather_data, process_data, write_report],
            goal=GoalSpec(conditions={"report_written": True}),
            effects={
                "gather_data": {"has_data": True},
                "process_data": {"data_processed": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "process_data": {"has_data": True},
                "write_report": {"data_processed": True},
            },
        )
        result = agent.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"report_written": True}),
            }
        )
        assert result["status"] == "goal_achieved"
        # plan should chain the three tools
        plan = result["plan"]
        assert plan is not None
        assert plan.action_names == ["gather_data", "process_data", "write_report"]


class TestCreateGoapAgentErrors:
    def test_string_goal_without_llm_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="string goal but no llm"):
            create_goap_agent(
                tools=[gather_data],
                goal="write a report",
            )

    def test_warns_on_tools_without_effects(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="langgoap.integrations"):
            create_goap_agent(
                tools=[gather_data, process_data],
                goal=GoalSpec(conditions={"done": True}),
            )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("gather_data" in w.message for w in warnings)
        assert any("process_data" in w.message for w in warnings)

    def test_no_warning_when_all_tools_have_effects(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="langgoap.integrations"):
            create_goap_agent(
                tools=[gather_data],
                goal=GoalSpec(conditions={"has_data": True}),
                effects={"gather_data": {"has_data": True}},
            )
        warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "has no" in r.message
        ]
        assert warnings == []


class TestCreateGoapAgentNLGoal:
    def test_string_goal_with_llm_invokes_interpreter(self) -> None:
        from langgoap.interpreter import InterpretedGoal
        from tests.conftest import FakeStructuredModel

        fake = FakeStructuredModel(
            response=InterpretedGoal(
                conditions={"report_written": True},
                reasoning="user asked for a written report",
            )
        )
        agent = create_goap_agent(
            tools=[gather_data, write_report],
            goal="please write me a report",
            llm=fake,
            effects={
                "gather_data": {"has_data": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "write_report": {"has_data": True},
            },
        )
        # The resolved goal is exposed as a public attribute so callers
        # can pass it back to ``invoke({"goal": ...})``.
        resolved = agent.goap_goal
        assert dict(resolved.conditions) == {"report_written": True}
        # And the agent still runs end-to-end.
        result = agent.invoke(
            {
                "world_state": {},
                "goal": resolved,
            }
        )
        assert result["status"] == "goal_achieved"


def _quickstart_writers(fail_fast_n_times: int) -> tuple[BaseTool, BaseTool]:
    """Build the two-writer pair used by the README Quickstart demo.

    The fast writer raises ``RuntimeError`` on its first
    ``fail_fast_n_times`` invocations to demonstrate the executor's
    blacklist-on-failure → replan flow.  ``fail_fast_n_times=0`` is the
    happy path (fast always succeeds and A* picks it for its lower cost);
    ``fail_fast_n_times>=1`` blacklists the fast writer and forces a
    replan through the premium writer.
    """
    state = {"fast_calls": 0}

    @tool
    def write_article_fast(brief: str) -> str:
        """Quickly draft an article from a brief. Cheaper, occasionally flaky."""
        state["fast_calls"] += 1
        if state["fast_calls"] <= fail_fast_n_times:
            raise RuntimeError(f"upstream rate limit (attempt {state['fast_calls']})")
        return f"Fast draft: {brief}"

    @tool
    def write_article_premium(brief: str) -> str:
        """Premium-quality article. Higher cost, always reliable."""
        return f"Premium draft: {brief}"

    return write_article_fast, write_article_premium


def _build_quickstart_agent(llm: Any, fail_fast_n_times: int) -> CompiledStateGraph:
    """Build the four-tool README Quickstart agent.

    Two writers (``write_article_fast`` cost=1.0, ``write_article_premium``
    cost=5.0) both produce ``have_draft: True``.  A* picks the cheaper
    one; if it gets blacklisted the planner falls back to the premium
    writer without any routing code in the caller.
    """
    write_fast, write_premium = _quickstart_writers(fail_fast_n_times)

    @tool
    def research_topic(topic: str) -> str:
        """Produce a short research brief for a topic."""
        return f"Brief on {topic}"

    @tool
    def publish_article(draft: str) -> str:
        """Publish an article draft."""
        return f"Published: {draft}"

    return create_goap_agent(
        tools=[research_topic, write_fast, write_premium, publish_article],
        goal="Publish an article about GOAP for LangGraph",
        llm=llm,
        preconditions={
            "write_article_fast": {"have_brief": True},
            "write_article_premium": {"have_brief": True},
            "publish_article": {"have_draft": True},
        },
        effects={
            "research_topic": {"have_brief": True},
            "write_article_fast": {"have_draft": True},
            "write_article_premium": {"have_draft": True},
            "publish_article": {"published": True},
        },
        costs={
            "research_topic": 1.0,
            "write_article_fast": 1.0,
            "write_article_premium": 5.0,
            "publish_article": 1.0,
        },
        result_keys={
            "research_topic": "brief",
            "write_article_fast": "draft",
            "write_article_premium": "draft",
        },
    )


def _summarize_result(result: dict[str, Any]) -> str:
    """Pretty-print the bits of a quickstart run that matter for the README."""
    plan = result.get("plan")
    history = result.get("execution_history", [])
    lines = [
        f"status:              {result.get('status')!r}",
        f"replan_count:        {result.get('replan_count', 0)}",
        f"blacklisted_actions: {list(result.get('blacklisted_actions', []))}",
        f"plan.action_names:   {plan.action_names if plan else None}",
        f"plan.total_cost:     {plan.total_cost if plan else None}",
        "execution_history:",
    ]
    for i, h in enumerate(history, 1):
        outcome = "ok " if h.success else "FAIL"
        err = f"  ({h.error})" if not h.success and h.error else ""
        lines.append(f"  {i:>2}. [{outcome}] {h.action_name}{err}")
    ws = result.get("world_state", {})
    relevant = {k: ws[k] for k in ("topic", "brief", "draft") if k in ws}
    lines.append(f"world_state (relevant keys): {relevant}")
    return "\n".join(lines)


@pytest.mark.api
class TestReadmeQuickstart:
    """Real-API end-to-end test for the README Quickstart example.

    Uses live ``ChatOpenAI`` (gpt-4o-mini, temperature=0) so any drift
    between the README snippet and the actual ``create_goap_agent`` API
    surfaces here first.  Runs only when ``-m api`` is selected and
    ``OPENAI_API_KEY`` is present in the environment (loaded from
    ``.env`` by :mod:`tests.conftest`).

    Side effect: regenerates the README's companion artifacts in
    ``.github/images/`` — one StateGraph PNG plus per-scenario Plan
    PNGs and transcript ``.txt`` files.  Keeps the README byte-aligned
    with live behaviour.
    """

    @pytest.fixture
    def llm(self) -> Any:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key or key.startswith("sk-..."):
            pytest.skip("OPENAI_API_KEY not set in .env")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)

    @pytest.fixture(autouse=True)
    def _ensure_artifacts_dir(self) -> None:
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _save_plan_png(result: dict[str, Any], filename: str) -> None:
        from langgoap.viz import draw_mermaid_png

        png = draw_mermaid_png(result["plan"])
        (_ARTIFACTS_DIR / filename).write_bytes(png)

    @staticmethod
    def _save_stategraph_png(agent: CompiledStateGraph, filename: str) -> None:
        png = agent.get_graph().draw_mermaid_png()
        (_ARTIFACTS_DIR / filename).write_bytes(png)

    @staticmethod
    def _save_transcript(result: dict[str, Any], filename: str) -> None:
        (_ARTIFACTS_DIR / filename).write_text(
            _summarize_result(result) + "\n", encoding="utf-8"
        )

    def test_happy_path(self, llm: Any) -> None:
        """Fast writer always succeeds; A* picks it for its lower cost."""
        agent = _build_quickstart_agent(llm, fail_fast_n_times=0)
        # The natural-language goal should resolve to a published-article flag.
        assert agent.goap_goal.conditions.get("published") is True

        result = agent.invoke(
            {
                "world_state": {"topic": "GOAP for LangGraph"},
                "goal": agent.goap_goal,
            }
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 0
        assert list(result.get("blacklisted_actions", [])) == []
        plan = result["plan"]
        assert plan.action_names == [
            "research_topic",
            "write_article_fast",
            "publish_article",
        ]
        # 1.0 + 1.0 + 1.0
        assert plan.total_cost == pytest.approx(3.0)
        ws = result["world_state"]
        assert ws["brief"] == "Brief on GOAP for LangGraph"
        assert ws["draft"] == "Fast draft: Brief on GOAP for LangGraph"

        # Refresh README companion artifacts (idempotent on disk).
        self._save_stategraph_png(agent, "quickstart-stategraph.png")
        self._save_plan_png(result, "quickstart-plan-happy.png")
        self._save_transcript(result, "quickstart-happy.txt")
        print("\n" + _summarize_result(result))

    def test_replan_to_premium(self, llm: Any) -> None:
        """Fast writer fails once → blacklist → replan picks premium writer."""
        agent = _build_quickstart_agent(llm, fail_fast_n_times=1)
        result = agent.invoke(
            {
                "world_state": {"topic": "GOAP for LangGraph"},
                "goal": agent.goap_goal,
            }
        )

        assert result["status"] == "goal_achieved"
        assert "write_article_fast" in result.get("blacklisted_actions", [])
        # At least one replan should have happened (the fast writer
        # failed, triggering a planner reroute through premium).
        assert result["replan_count"] >= 1
        # The *final* plan that succeeded must use the premium writer.
        names = result["plan"].action_names
        assert "write_article_premium" in names
        assert "write_article_fast" not in names
        # Execution history records both the failed fast attempt and
        # the successful premium completion.
        history = result["execution_history"]
        failed = [h for h in history if not h.success]
        assert any(h.action_name == "write_article_fast" for h in failed)
        succeeded = [h.action_name for h in history if h.success]
        assert "write_article_premium" in succeeded
        assert succeeded[-1] == "publish_article"
        ws = result["world_state"]
        assert ws["draft"] == "Premium draft: Brief on GOAP for LangGraph"

        self._save_plan_png(result, "quickstart-plan-replan.png")
        self._save_transcript(result, "quickstart-replan.txt")
        print("\n" + _summarize_result(result))


class TestCreateGoapAgentResources:
    def test_forwards_resources(self) -> None:
        agent = create_goap_agent(
            tools=[gather_data, write_report],
            goal=GoalSpec(conditions={"report_written": True}),
            effects={
                "gather_data": {"has_data": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "write_report": {"has_data": True},
            },
            resources={
                "gather_data": {"cost_usd": 0.01},
                "write_report": {"cost_usd": 0.05},
            },
        )
        result = agent.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"report_written": True}),
            }
        )
        assert result["status"] == "goal_achieved"
