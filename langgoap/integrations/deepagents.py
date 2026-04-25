"""DeepAgents integration — expose GOAP planning as a tool or subagent.

Two integration APIs for embedding LangGOAP inside a DeepAgents agent:

* :func:`create_goap_tool` — wraps a ``GoapGraph`` as a LangChain
  :class:`~langchain_core.tools.BaseTool`.  The main agent calls it via
  the normal tool-calling flow.
* :func:`create_goap_subagent` — returns a
  :class:`~deepagents.middleware.subagents.CompiledSubAgent`-compatible
  dict whose ``runnable`` accepts a ``{messages: [HumanMessage(...)]}``
  state and returns an ``{messages: [AIMessage(...)]}`` state after the
  GOAP loop completes.

Both use :class:`~langgoap.interpreter.GoalInterpreter` to convert the
natural-language request to a :class:`~langgoap.goals.GoalSpec` before
running the planner.

No hard dependency on the ``deepagents`` package — everything is built
on ``langchain-core`` and ``langgraph`` primitives.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from langgoap.actions import ActionSpec
from langgoap.graph.builder import GoapGraph

__all__ = [
    "create_goap_tool",
    "create_goap_subagent",
    "format_goap_result",
]


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def format_goap_result(result: dict[str, Any]) -> str:
    """Format a ``GoapState`` result dict as a human-readable report.

    Args:
        result: The dict returned by ``GoapGraph.invoke`` / ``ainvoke``
            (or ``invoke_nl`` / ``ainvoke_nl``).

    Returns:
        A multi-line string summarising the outcome, executed actions,
        and final world state.
    """
    lines: list[str] = []
    status = result.get("status", "unknown")
    lines.append(f"GOAP Result: {status}")

    history = result.get("execution_history")
    if history:
        lines.append("")
        lines.append("Executed actions:")
        for entry in history:
            if isinstance(entry, dict):
                name = entry.get("action", "?")
                ok = entry.get("success", "?")
                lines.append(f"  - {name} (success={ok})")
            else:
                lines.append(f"  - {entry}")

    world_state = result.get("world_state")
    if world_state:
        lines.append("")
        lines.append("Final world state:")
        for k, v in sorted(world_state.items()):
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool input schema
# ---------------------------------------------------------------------------


class _GoapToolInput(BaseModel):
    """Input schema for the GOAP planning tool."""

    request: str = Field(
        description="Natural-language description of the goal to achieve."
    )


# ---------------------------------------------------------------------------
# create_goap_tool
# ---------------------------------------------------------------------------


def create_goap_tool(
    actions: list[ActionSpec],
    *,
    llm: BaseChatModel,
    name: str = "goap_planner",
    description: str = "Run a GOAP (Goal-Oriented Action Planning) loop to achieve a goal described in natural language.",
    world_state: dict[str, Any] | None = None,
    graph_kwargs: dict[str, Any] | None = None,
) -> StructuredTool:
    """Create a LangChain ``BaseTool`` that runs a GOAP planning loop.

    The returned tool accepts a ``request`` string (natural language),
    interprets it as a ``GoalSpec`` via ``GoalInterpreter``, and runs
    the ``GoapGraph`` to completion.

    Args:
        actions: The action library for the planner.
        llm: Chat model for natural-language goal interpretation.
        name: Tool name visible to the calling agent.
        description: Tool description for the agent's tool catalog.
        world_state: Default initial world state.  May be overridden
            per-invocation if the caller passes a ``world_state`` key.
        graph_kwargs: Extra keyword arguments forwarded to
            :class:`~langgoap.graph.builder.GoapGraph`.

    Returns:
        A :class:`~langchain_core.tools.StructuredTool` with both sync
        and async implementations.
    """
    gkw = dict(graph_kwargs or {})
    graph = GoapGraph(actions=actions, **gkw)
    default_ws = dict(world_state or {})

    def _run(request: str) -> str:
        result = graph.invoke_nl(
            request,
            world_state=dict(default_ws),
            llm=llm,
        )
        return format_goap_result(dict(result))

    async def _arun(request: str) -> str:
        result = await graph.ainvoke_nl(
            request,
            world_state=dict(default_ws),
            llm=llm,
        )
        return format_goap_result(dict(result))

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=name,
        description=description,
        args_schema=_GoapToolInput,
    )


# ---------------------------------------------------------------------------
# create_goap_subagent (CompiledSubAgent-compatible dict)
# ---------------------------------------------------------------------------


class _BridgeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_goap_subagent(
    actions: list[ActionSpec],
    *,
    llm: BaseChatModel,
    name: str = "goap_planner",
    description: str = "A GOAP (Goal-Oriented Action Planning) subagent that achieves goals described in natural language.",
    world_state: dict[str, Any] | None = None,
    graph_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a ``CompiledSubAgent``-compatible dict for DeepAgents.

    The returned dict has ``name``, ``description``, and ``runnable``
    keys.  The ``runnable`` is a compiled ``StateGraph`` whose state
    includes a ``messages`` list — the DeepAgents subagent protocol.

    The bridge graph has a single node that:

    1. Extracts the task from the last ``HumanMessage``.
    2. Runs ``GoapGraph.invoke_nl`` / ``ainvoke_nl``.
    3. Returns an ``AIMessage`` with the formatted result.

    Args:
        actions: The action library for the planner.
        llm: Chat model for natural-language goal interpretation.
        name: Subagent name.
        description: Subagent description for routing.
        world_state: Default initial world state.
        graph_kwargs: Extra keyword arguments forwarded to
            :class:`~langgoap.graph.builder.GoapGraph`.

    Returns:
        A dict matching the ``CompiledSubAgent`` protocol:
        ``{"name": str, "description": str, "runnable": Runnable}``.
    """
    gkw = dict(graph_kwargs or {})
    graph = GoapGraph(actions=actions, **gkw)
    default_ws = dict(world_state or {})

    def _extract_task(messages: list[BaseMessage]) -> str:
        """Extract the task description from the last HumanMessage."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return str(messages[-1].content) if messages else ""

    def _goap_node(state: _BridgeState) -> dict[str, Any]:
        task = _extract_task(state["messages"])
        result = graph.invoke_nl(
            task,
            world_state=dict(default_ws),
            llm=llm,
        )
        report = format_goap_result(dict(result))
        return {"messages": [AIMessage(content=report)]}

    async def _agoap_node(state: _BridgeState) -> dict[str, Any]:
        task = _extract_task(state["messages"])
        result = await graph.ainvoke_nl(
            task,
            world_state=dict(default_ws),
            llm=llm,
        )
        report = format_goap_result(dict(result))
        return {"messages": [AIMessage(content=report)]}

    builder = StateGraph(_BridgeState)
    builder.add_node(
        "goap",
        RunnableLambda(func=_goap_node, afunc=_agoap_node),
    )
    builder.set_entry_point("goap")
    builder.set_finish_point("goap")
    compiled = builder.compile()

    return {
        "name": name,
        "description": description,
        "runnable": compiled,
    }
