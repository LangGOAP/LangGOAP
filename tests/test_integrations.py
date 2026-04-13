"""Unit tests for the LangGoap integration adapters.

Covers all three layers of the low-code on-ramp:
- Layer B: ``goapify_tool`` (BaseTool -> ActionSpec)
- Layer A: ``create_goap_agent`` (prebuilt GOAP agent factory)
- Layer C: ``GoapSubgraph`` and ``add_goap_subgraph``
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgoap.goals import GoalSpec
from langgoap.integrations.prebuilt import create_goap_agent
from langgoap.integrations.subgraph import GoapSubgraph, add_goap_subgraph
from langgoap.integrations.tools import goapify_tool
from langgoap.interpreter import InterpretedGoal
from langgoap.testing import FakeStructuredModel

# ---------------------------------------------------------------------------
# Inline mock tools (simple @tool functions for test isolation)
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def noop_tool() -> str:
    """A zero-arg tool that does nothing useful."""
    return "done"


# ---------------------------------------------------------------------------
# Layer B: goapify_tool
# ---------------------------------------------------------------------------


class TestGoapifyToolBasicConversion:
    """Basic conversion from @tool function to ActionSpec."""

    def test_name_matches_tool_name(self) -> None:
        action = goapify_tool(
            greet,
            preconditions={"has_name": True},
            effects={"greeted": True},
        )
        assert action.name == "greet"

    def test_preconditions_and_effects_transferred(self) -> None:
        pre = {"has_name": True}
        eff = {"greeted": True}
        action = goapify_tool(greet, preconditions=pre, effects=eff)
        assert dict(action.preconditions) == pre
        assert dict(action.effects) == eff

    def test_cost_passed_through(self) -> None:
        action = goapify_tool(greet, cost=5.0)
        assert action.cost == 5.0

    def test_resources_passed_through(self) -> None:
        res = {"tokens": 200, "cost_usd": 0.01}
        action = goapify_tool(greet, resources=res)
        assert action.resources is not None
        assert dict(action.resources) == res

    def test_duration_passed_through(self) -> None:
        dur = timedelta(seconds=10)
        action = goapify_tool(greet, duration=dur)
        assert action.duration == dur

    def test_max_retries_passed_through(self) -> None:
        action = goapify_tool(greet, max_retries=3)
        assert action.max_retries == 3

    def test_effect_validator_passed_through(self) -> None:
        def my_validator(pre: dict[str, Any], post: dict[str, Any]) -> bool:
            return post.get("greeted") is True

        action = goapify_tool(greet, effect_validator=my_validator)
        assert action.effect_validator is my_validator


class TestGoapifyToolDefaults:
    """Default values when optional kwargs are omitted."""

    def test_default_cost_is_one(self) -> None:
        action = goapify_tool(greet)
        assert action.cost == 1.0

    def test_default_preconditions_empty(self) -> None:
        action = goapify_tool(greet)
        assert dict(action.preconditions) == {}

    def test_default_effects_empty(self) -> None:
        action = goapify_tool(greet)
        assert dict(action.effects) == {}

    def test_default_resources_none(self) -> None:
        action = goapify_tool(greet)
        assert action.resources is None

    def test_default_duration_none(self) -> None:
        action = goapify_tool(greet)
        assert action.duration is None

    def test_default_max_retries_zero(self) -> None:
        action = goapify_tool(greet)
        assert action.max_retries == 0

    def test_default_effect_validator_none(self) -> None:
        action = goapify_tool(greet)
        assert action.effect_validator is None


class TestGoapifyToolExecuteWrapper:
    """The execute wrapper calls the underlying tool correctly."""

    def test_execute_calls_tool_and_returns_effects(self) -> None:
        eff = {"greeted": True}
        action = goapify_tool(greet, preconditions={"has_name": True}, effects=eff)
        assert action.execute is not None
        result = action.execute({"name": "Alice", "has_name": True})
        assert result == eff

    def test_execute_filters_state_to_tool_input_schema(self) -> None:
        """Only keys matching the tool's args_schema are forwarded."""
        eff = {"sum_ready": True}
        action = goapify_tool(add_numbers, preconditions={}, effects=eff)
        assert action.execute is not None
        # Pass extra keys that are not in add_numbers' schema (a, b).
        result = action.execute({"a": 3, "b": 4, "irrelevant": "noise"})
        assert result == eff

    def test_execute_zero_arg_tool(self) -> None:
        """Zero-arg tools receive no input from state."""
        eff = {"noop_done": True}
        action = goapify_tool(noop_tool, effects=eff)
        assert action.execute is not None
        result = action.execute({"some_state_key": 42})
        assert result == eff

    def test_execute_returns_fresh_copy_of_effects(self) -> None:
        """Each call returns a new dict, not the same mutable reference."""
        eff = {"greeted": True}
        action = goapify_tool(greet, effects=eff)
        assert action.execute is not None
        result1 = action.execute({"name": "A"})
        result2 = action.execute({"name": "B"})
        assert result1 == result2
        assert result1 is not result2


class TestGoapifyToolResultKey:
    """result_key stores the tool's raw return value in world_state."""

    def test_result_key_none_returns_only_effects(self) -> None:
        """Default (no result_key): execute returns exactly the effects dict."""
        eff = {"done": True}
        action = goapify_tool(greet, effects=eff)
        assert action.execute is not None
        result = action.execute({"name": "Alice"})
        assert result == eff

    def test_result_key_captures_tool_return_value(self) -> None:
        """With result_key, the tool's return value appears alongside effects."""
        action = goapify_tool(
            greet,
            effects={"greeted": True},
            result_key="greeting_text",
        )
        assert action.execute is not None
        result = action.execute({"name": "Bob"})
        # Effect flag is preserved
        assert result["greeted"] is True
        # Tool output captured under result_key
        assert result["greeting_text"] == "Hello, Bob!"

    def test_result_key_does_not_appear_in_effects(self) -> None:
        """result_key must NOT pollute ActionSpec.effects — A* stays boolean."""
        action = goapify_tool(
            greet,
            effects={"greeted": True},
            result_key="greeting_text",
        )
        assert "greeting_text" not in dict(action.effects)

    def test_result_key_with_numeric_return_value(self) -> None:
        """Numeric tool outputs are stored verbatim."""
        action = goapify_tool(
            add_numbers,
            effects={"sum_ready": True},
            result_key="sum_value",
        )
        assert action.execute is not None
        result = action.execute({"a": 3, "b": 4})
        assert result["sum_ready"] is True
        assert result["sum_value"] == 7

    def test_result_key_with_zero_arg_tool(self) -> None:
        """Zero-arg tools with result_key also capture output."""
        action = goapify_tool(
            noop_tool,
            effects={"noop_done": True},
            result_key="noop_output",
        )
        assert action.execute is not None
        result = action.execute({})
        assert result["noop_done"] is True
        assert result["noop_output"] == "done"

    def test_result_key_value_reaches_world_state_via_executor(self) -> None:
        """End-to-end: result_key value lands in world_state after execution."""
        from langgoap.graph.nodes import GoapExecutor
        from langgoap.graph.state import GoapState
        from tests.conftest import make_plan

        action = goapify_tool(
            greet,
            preconditions={},
            effects={"greeted": True},
            result_key="greeting_text",
        )
        plan = make_plan(action)
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {"name": "Carol"},
            "plan": plan,
            "current_step": 0,
        }
        update = executor(state)
        ws = update["world_state"]
        assert ws["greeted"] is True
        assert ws["greeting_text"] == "Hello, Carol!"

    def test_result_key_does_not_affect_planning(self) -> None:
        """A* plans correctly when result_key is set; only effects matter."""
        from langgoap.goals import GoalSpec
        from langgoap.graph.nodes import GoapPlanner
        from langgoap.graph.state import GoapState

        action = goapify_tool(
            greet,
            preconditions={"has_name": True},
            effects={"greeted": True},
            result_key="greeting_text",
        )
        planner = GoapPlanner([action])
        goal = GoalSpec(conditions={"greeted": True})
        state: GoapState = {"world_state": {"has_name": True}, "goal": goal}
        result = planner(state)
        plan = result.get("plan")
        assert plan is not None
        assert len(plan) == 1
        assert plan.actions[0].name == "greet"
        # result_key must not appear in the action's effects
        assert "greeting_text" not in dict(plan.actions[0].effects)

    def test_each_execute_call_is_independent(self) -> None:
        """Multiple execute calls each return their own result_key value."""
        action = goapify_tool(
            greet,
            effects={"greeted": True},
            result_key="greeting_text",
        )
        assert action.execute is not None
        r1 = action.execute({"name": "Alice"})
        r2 = action.execute({"name": "Dave"})
        assert r1["greeting_text"] == "Hello, Alice!"
        assert r2["greeting_text"] == "Hello, Dave!"
        assert r1 is not r2


class TestGoapifyToolTypeError:
    """goapify_tool rejects non-BaseTool inputs."""

    def test_raises_type_error_for_plain_function(self) -> None:
        def plain_func(x: str) -> str:
            return x

        with pytest.raises(TypeError, match="BaseTool"):
            goapify_tool(plain_func)  # type: ignore[arg-type]

    def test_raises_type_error_for_string(self) -> None:
        with pytest.raises(TypeError, match="BaseTool"):
            goapify_tool("not_a_tool")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Layer A: create_goap_agent
# ---------------------------------------------------------------------------


class TestCreateGoapAgentStringGoalRequiresLlm:
    """ValueError when string goal is passed without an LLM."""

    def test_string_goal_without_llm_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="string goal but no llm"):
            create_goap_agent(
                tools=[greet],
                goal="Say hello to everyone",
            )


class TestCreateGoapAgentGoalSpecWithoutLlm:
    """GoalSpec goal works without LLM."""

    def test_goalspec_without_llm_returns_compiled_graph(self) -> None:
        goal = GoalSpec(conditions={"greeted": True})
        compiled = create_goap_agent(
            tools=[greet],
            goal=goal,
            preconditions={"greet": {"has_name": True}},
            effects={"greet": {"greeted": True}},
        )
        assert isinstance(compiled, CompiledStateGraph)
        assert hasattr(compiled, "invoke")

    def test_goalspec_goal_attached_as_attribute(self) -> None:
        goal = GoalSpec(conditions={"greeted": True})
        compiled = create_goap_agent(
            tools=[greet],
            goal=goal,
            effects={"greet": {"greeted": True}},
        )
        assert hasattr(compiled, "goap_goal")
        assert getattr(compiled, "goap_goal") is goal


class TestCreateGoapAgentWarningOnNoEffects:
    """Warning emission when tools have no effects declared."""

    def test_warns_when_tools_have_no_effects(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        goal = GoalSpec(conditions={"done": True})
        with caplog.at_level(logging.WARNING, logger="langgoap.integrations.prebuilt"):
            create_goap_agent(
                tools=[greet, add_numbers],
                goal=goal,
                # Deliberately pass no effects for either tool.
            )
        assert "no effects or resources" in caplog.text
        assert "greet" in caplog.text
        assert "add_numbers" in caplog.text

    def test_no_warning_when_all_tools_have_effects(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        goal = GoalSpec(conditions={"greeted": True})
        with caplog.at_level(logging.WARNING, logger="langgoap.integrations.prebuilt"):
            create_goap_agent(
                tools=[greet],
                goal=goal,
                effects={"greet": {"greeted": True}},
            )
        assert "no effects or resources" not in caplog.text

    def test_no_warning_when_tool_has_resources_but_no_effects(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Resources alone satisfy the 'plan-visible' requirement."""
        goal = GoalSpec(conditions={"done": True})
        with caplog.at_level(logging.WARNING, logger="langgoap.integrations.prebuilt"):
            create_goap_agent(
                tools=[greet],
                goal=goal,
                resources={"greet": {"tokens": 100}},
            )
        assert "greet" not in caplog.text


class TestCreateGoapAgentCompiledGraphShape:
    """Returns a compiled graph with expected attributes."""

    def test_compiled_graph_has_invoke_and_ainvoke(self) -> None:
        goal = GoalSpec(conditions={"greeted": True})
        compiled = create_goap_agent(
            tools=[greet],
            goal=goal,
            effects={"greet": {"greeted": True}},
        )
        assert callable(getattr(compiled, "invoke", None))
        assert callable(getattr(compiled, "ainvoke", None))


class TestCreateGoapAgentCostsOverride:
    """Cost overrides are forwarded to goapify_tool."""

    def test_custom_cost_applied(self) -> None:
        goal = GoalSpec(conditions={"greeted": True})
        compiled = create_goap_agent(
            tools=[greet],
            goal=goal,
            effects={"greet": {"greeted": True}},
            costs={"greet": 7.5},
        )
        # The goap_goal is attached; the graph compiled successfully.
        assert isinstance(compiled, CompiledStateGraph)


class TestCreateGoapAgentStringGoalWithLlm:
    """String goal with LLM triggers NL interpretation."""

    def test_string_goal_with_fake_llm_returns_compiled_graph(self) -> None:
        fake_llm = FakeStructuredModel(
            response=InterpretedGoal(
                conditions={"greeted": True},
                constraints=[],
                objectives=[],
                reasoning="User wants greeting.",
            )
        )
        compiled = create_goap_agent(
            tools=[greet],
            goal="Greet everyone",
            llm=fake_llm,
            effects={"greet": {"greeted": True}},
        )
        assert isinstance(compiled, CompiledStateGraph)
        resolved = getattr(compiled, "goap_goal", None)
        assert resolved is not None
        assert isinstance(resolved, GoalSpec)
        assert dict(resolved.conditions) == {"greeted": True}


# ---------------------------------------------------------------------------
# Layer C: GoapSubgraph and add_goap_subgraph
# ---------------------------------------------------------------------------


def _make_simple_actions() -> list[Any]:
    """Create a minimal action set for subgraph tests."""
    from langgoap.actions import ActionSpec

    return [
        ActionSpec(
            name="step_a",
            preconditions={},
            effects={"a_done": True},
            cost=1.0,
            execute=lambda state: {"a_done": True},
        ),
        ActionSpec(
            name="step_b",
            preconditions={"a_done": True},
            effects={"b_done": True},
            cost=1.0,
            execute=lambda state: {"b_done": True},
        ),
    ]


class TestGoapSubgraphConstruction:
    """Construction with actions and goal."""

    def test_stores_actions_and_goal(self) -> None:
        actions = _make_simple_actions()
        goal = GoalSpec(conditions={"b_done": True})
        sub = GoapSubgraph(actions=actions, goal=goal)
        assert sub.actions is actions
        assert sub.goal is goal

    def test_compile_returns_compiled_state_graph(self) -> None:
        actions = _make_simple_actions()
        goal = GoalSpec(conditions={"b_done": True})
        sub = GoapSubgraph(actions=actions, goal=goal)
        compiled = sub.compile()
        assert isinstance(compiled, CompiledStateGraph)

    def test_compiled_graph_has_invoke(self) -> None:
        actions = _make_simple_actions()
        goal = GoalSpec(conditions={"b_done": True})
        sub = GoapSubgraph(actions=actions, goal=goal)
        compiled = sub.compile()
        assert callable(getattr(compiled, "invoke", None))
        assert callable(getattr(compiled, "ainvoke", None))


class TestAddGoapSubgraph:
    """add_goap_subgraph adds a node to a parent StateGraph."""

    def test_adds_named_node_to_parent(self) -> None:
        from typing import TypedDict

        class ParentState(TypedDict, total=False):
            world_state: dict[str, Any]
            plan_result: dict[str, Any]

        actions = _make_simple_actions()
        goal = GoalSpec(conditions={"b_done": True})

        parent = StateGraph(ParentState)
        add_goap_subgraph(
            parent,
            name="goap_planner",
            actions=actions,
            goal=goal,
        )
        # The node should be registered in the parent graph's nodes.
        assert "goap_planner" in parent.nodes

    def test_custom_input_output_keys(self) -> None:
        from typing import TypedDict

        class ParentState(TypedDict, total=False):
            my_world: dict[str, Any]
            my_result: dict[str, Any]

        actions = _make_simple_actions()
        goal = GoalSpec(conditions={"b_done": True})

        parent = StateGraph(ParentState)
        add_goap_subgraph(
            parent,
            name="planner_node",
            actions=actions,
            goal=goal,
            input_key="my_world",
            output_key="my_result",
        )
        assert "planner_node" in parent.nodes

    def test_subgraph_node_has_invokable_runnable(self) -> None:
        """The registered node wraps a Runnable with an invoke method."""
        from typing import TypedDict

        class ParentState(TypedDict, total=False):
            world_state: dict[str, Any]
            plan_result: dict[str, Any]

        actions = _make_simple_actions()
        goal = GoalSpec(conditions={"b_done": True})

        parent = StateGraph(ParentState)
        add_goap_subgraph(
            parent,
            name="goap_node",
            actions=actions,
            goal=goal,
        )
        node_spec = parent.nodes["goap_node"]
        # LangGraph wraps nodes in StateNodeSpec; the runnable is inside.
        assert hasattr(node_spec, "runnable")
        # RunnableCallable exposes .invoke(), not __call__.
        assert hasattr(node_spec.runnable, "invoke")
