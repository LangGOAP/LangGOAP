"""Real-LLM integration tests for the NL → GoalSpec interpreter.

These tests make live API calls to OpenAI and Anthropic.  They are gated
behind the ``api`` pytest marker so they are skipped in normal CI:

    uv run pytest -m api          # run only API tests
    uv run pytest -m "not api"   # skip API tests (default CI)

Keys are loaded from ``langgoap/.env`` via python-dotenv.  Copy
``.env.example`` to ``.env`` and fill in your real keys before running.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from langgoap import (
    ActionSpec,
    GoalInterpreter,
    GoalSpec,
    GoapGraph,
)

# ---------------------------------------------------------------------------
# Load keys from .env (no-op if already set in the environment)
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_ENV_FILE, override=False)


# ---------------------------------------------------------------------------
# Shared action setup (same pipeline as the mocked integration tests)
# ---------------------------------------------------------------------------


def _report_pipeline_actions() -> list[ActionSpec]:
    """3-action data pipeline with resource costs."""
    return [
        ActionSpec(
            name="fetch_data",
            preconditions={},
            effects={"data_fetched": True},
            resources={"cost_usd": 0.5, "api_calls": 1},
            metadata={"description": "Fetch raw data from the external API"},
        ),
        ActionSpec(
            name="clean_data",
            preconditions={"data_fetched": True},
            effects={"data_clean": True},
            resources={"cost_usd": 0.1},
            metadata={"description": "Validate and normalise the fetched dataset"},
        ),
        ActionSpec(
            name="generate_report",
            preconditions={"data_clean": True},
            effects={"report_complete": True},
            resources={"cost_usd": 1.0, "tokens": 500},
            metadata={"description": "Produce a summary report from clean data"},
        ),
    ]


# ---------------------------------------------------------------------------
# OpenAI tests
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestGoalInterpreterOpenAI:
    """Live tests against OpenAI's structured-output API."""

    @pytest.fixture
    def llm(self) -> "ChatOpenAI":  # type: ignore[name-defined]
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key or key.startswith("sk-..."):
            pytest.skip("OPENAI_API_KEY not set in .env")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)

    # OpenAI strict JSON-schema mode requires additionalProperties:false on all
    # objects, which is incompatible with the open-ended conditions dict.
    # Use method="function_calling" for all OpenAI interpreter calls.
    _SOK: dict[str, str] = {"method": "function_calling"}

    def test_interpret_simple_report_goal(self, llm: object) -> None:
        """Real LLM correctly maps 'generate a report' to report_complete=True."""
        interpreter = GoalInterpreter(
            llm=llm,  # type: ignore[arg-type]
            actions=_report_pipeline_actions(),
            structured_output_kwargs=self._SOK,
        )
        goal = interpreter.interpret("Generate a report from the raw data")
        assert isinstance(goal, GoalSpec)
        assert "report_complete" in goal.conditions
        assert goal.conditions["report_complete"] is True

    def test_interpret_budget_constraint(self, llm: object) -> None:
        """Real LLM extracts a cost_usd constraint from budget language.

        Uses ``interpret_raw()`` to test constraint extraction in isolation —
        this avoids the ``conditions`` completeness check in ``to_goal_spec()``
        (smaller models in function-calling mode can omit ``conditions`` when
        the constraint phrase dominates) and more directly tests the capability
        we care about: did the LLM understand "$5" as a cost_usd bound?
        """
        from langgoap.interpreter import InterpretedGoal

        interpreter = GoalInterpreter(
            llm=llm,  # type: ignore[arg-type]
            actions=_report_pipeline_actions(),
            structured_output_kwargs=self._SOK,
        )
        raw = interpreter.interpret_raw("Generate a report, but keep costs under $5")
        assert isinstance(raw, InterpretedGoal)
        assert any(c.key == "cost_usd" for c in raw.constraints), (
            f"Expected cost_usd constraint; got: {raw.constraints}"
        )
        cost_c = next(c for c in raw.constraints if c.key == "cost_usd")
        assert cost_c.max is not None and cost_c.max <= 5.0

    def test_interpret_raw_preserves_reasoning(self, llm: object) -> None:
        """interpret_raw() returns an InterpretedGoal with non-empty reasoning."""
        from langgoap.interpreter import InterpretedGoal

        interpreter = GoalInterpreter(
            llm=llm,  # type: ignore[arg-type]
            actions=_report_pipeline_actions(),
            structured_output_kwargs=self._SOK,
        )
        raw = interpreter.interpret_raw("Generate a report")
        assert isinstance(raw, InterpretedGoal)
        assert raw.reasoning, "LLM should populate the reasoning field"

    def test_invoke_nl_end_to_end(self, llm: object) -> None:
        """GoapGraph.invoke_nl() with a real LLM achieves the goal."""
        actions = _report_pipeline_actions()
        graph = GoapGraph(actions=actions)
        result = graph.invoke_nl(
            "Generate a report from the raw data",
            llm=llm,  # type: ignore[arg-type]
            structured_output_kwargs=self._SOK,
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("report_complete") is True

    @pytest.mark.asyncio
    async def test_ainvoke_nl_end_to_end(self, llm: object) -> None:
        """Async GoapGraph.ainvoke_nl() with a real LLM achieves the goal."""
        actions = _report_pipeline_actions()
        graph = GoapGraph(actions=actions)
        result = await graph.ainvoke_nl(
            "Generate a report from the raw data",
            llm=llm,  # type: ignore[arg-type]
            structured_output_kwargs=self._SOK,
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("report_complete") is True


# ---------------------------------------------------------------------------
# Anthropic tests
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestGoalInterpreterAnthropic:
    """Live tests against Anthropic's structured-output API."""

    @pytest.fixture
    def llm(self) -> "ChatAnthropic":  # type: ignore[name-defined]
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or key.startswith("sk-ant-..."):
            pytest.skip("ANTHROPIC_API_KEY not set in .env")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0,
            anthropic_api_key=key,  # type: ignore[call-arg]
        )

    def test_interpret_simple_report_goal(self, llm: object) -> None:
        """Anthropic correctly maps 'generate a report' to report_complete=True."""
        interpreter = GoalInterpreter(llm=llm, actions=_report_pipeline_actions())  # type: ignore[arg-type]
        goal = interpreter.interpret("Generate a report from the raw data")
        assert isinstance(goal, GoalSpec)
        assert "report_complete" in goal.conditions
        assert goal.conditions["report_complete"] is True

    def test_interpret_budget_constraint(self, llm: object) -> None:
        """Anthropic extracts a cost_usd constraint from budget language.

        Uses ``interpret_raw()`` — see OpenAI counterpart for rationale.
        """
        from langgoap.interpreter import InterpretedGoal

        interpreter = GoalInterpreter(llm=llm, actions=_report_pipeline_actions())  # type: ignore[arg-type]
        raw = interpreter.interpret_raw("Generate a report, but keep costs under $5")
        assert isinstance(raw, InterpretedGoal)
        assert any(c.key == "cost_usd" for c in raw.constraints), (
            f"Expected cost_usd constraint; got: {raw.constraints}"
        )
        cost_c = next(c for c in raw.constraints if c.key == "cost_usd")
        assert cost_c.max is not None and cost_c.max <= 5.0

    def test_invoke_nl_end_to_end(self, llm: object) -> None:
        """GoapGraph.invoke_nl() with Anthropic achieves the goal."""
        actions = _report_pipeline_actions()
        graph = GoapGraph(actions=actions)
        result = graph.invoke_nl(
            "Generate a report from the raw data",
            llm=llm,  # type: ignore[arg-type]
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("report_complete") is True
