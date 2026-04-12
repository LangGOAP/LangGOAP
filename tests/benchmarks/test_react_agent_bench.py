"""Head-to-head latency comparison: GoapGraph vs create_react_agent.

Builds the same 5-step sequential task twice:

1. Using ``langgraph.prebuilt.create_react_agent`` with a
   ``FakeListChatModel`` that returns a deterministic sequence of
   tool-call messages.
2. Using ``langgoap.GoapGraph`` with explicit ActionSpecs and no LLM.

Both paths execute the same logical work (5 sequential tool calls).
The benchmark reports wall-clock time for each.

The GOAP planner does strictly more work (explicit A* planning +
execution loop) but should be within reasonable range of the React agent
on small linear tasks.  Only the GOAP baseline is regression-gated via
``make benchmark-compare`` — the React agent timing depends on LangGraph
internals outside our control.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph

# ---------------------------------------------------------------------------
# Shared task: 5-step sequential pipeline
# ---------------------------------------------------------------------------


def _goap_actions() -> list[ActionSpec]:
    """5-step linear chain with trivial execute callables."""
    actions = []
    for i in range(5):
        pre = {f"s{i}": True} if i > 0 else {}
        eff = {f"s{i + 1}": True}
        actions.append(
            ActionSpec(
                name=f"step_{i}",
                preconditions=pre,
                effects=eff,
                cost=1.0,
                execute=lambda s, idx=i: {f"s{idx + 1}": True},
            )
        )
    return actions


def _goap_goal() -> GoalSpec:
    return GoalSpec(conditions={"s5": True})


# ---------------------------------------------------------------------------
# GoapGraph benchmark
# ---------------------------------------------------------------------------


def _run_goap() -> dict[str, Any]:
    graph = GoapGraph(_goap_actions())
    result = graph.invoke(goal=_goap_goal(), world_state={})
    return result


@pytest.mark.bench
class TestGoapVsReactAgent:
    """Latency comparison between GoapGraph and create_react_agent."""

    def test_goap_5_step(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark GoapGraph on a 5-step linear plan."""
        result = benchmark(_run_goap)
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["s5"] is True

    def test_react_agent_5_step(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark create_react_agent with a fake model on 5 tool calls.

        Uses FakeMessagesListChatModel with tool-call AIMessages.
        Skipped when the fake model does not support ``bind_tools``
        (required by ``create_react_agent``).
        """
        try:
            from langchain_core.messages import AIMessage, ToolMessage
            from langchain_core.tools import tool
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            pytest.skip("langgraph.prebuilt not available")

        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel,
        )

        # Build 5 tools with distinct names
        @tool
        def step_0(input: str = "") -> str:  # noqa: ARG001
            """Execute step 0."""
            return "step_0 done"

        @tool
        def step_1(input: str = "") -> str:  # noqa: ARG001
            """Execute step 1."""
            return "step_1 done"

        @tool
        def step_2(input: str = "") -> str:  # noqa: ARG001
            """Execute step 2."""
            return "step_2 done"

        @tool
        def step_3(input: str = "") -> str:  # noqa: ARG001
            """Execute step 3."""
            return "step_3 done"

        @tool
        def step_4(input: str = "") -> str:  # noqa: ARG001
            """Execute step 4."""
            return "step_4 done"

        tools = [step_0, step_1, step_2, step_3, step_4]

        # Build fake model responses: each response calls the next tool
        responses: list[AIMessage] = []
        for i in range(5):
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{i}",
                        "name": f"step_{i}",
                        "args": {"input": "go"},
                    }
                ],
            )
            responses.append(msg)
        # Final response: no tool call, just text
        responses.append(AIMessage(content="All steps complete."))

        fake_model = FakeMessagesListChatModel(responses=responses)

        # create_react_agent calls bind_tools on the model.
        # FakeMessagesListChatModel may not support it.
        try:
            agent = create_react_agent(fake_model, tools)
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip(
                "FakeMessagesListChatModel does not support bind_tools; "
                "react-agent benchmark requires a real or compatible model"
            )

        def run() -> dict[str, Any]:
            return agent.invoke({"messages": [("user", "run all steps")]})

        result = benchmark(run)
        # Verify all 5 tools were called
        msgs = result["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 5
