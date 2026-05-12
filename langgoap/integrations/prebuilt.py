"""Prebuilt GOAP agent one-liner.

Layer A of the three-layer low-code on-ramp (AD-2).  Mirrors
``langgraph.prebuilt.create_react_agent`` in shape: pass tools, a
goal, and optionally an LLM; receive a compiled LangGraph.

**Preconditions and effects are never LLM-inferred.**  The caller
either passes them explicitly, keyed by tool name, or accepts empty
pre/eff actions (which is almost never what you want for a
multi-step planner — the function logs a ``WARNING`` per affected
tool so the misuse is visible).

When ``goal`` is a natural-language string, an ``llm`` must be
provided — :class:`~langgoap.interpreter.GoalInterpreter` converts
the string to a :class:`~langgoap.goals.GoalSpec` once, at
construction time, before the graph is compiled.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.integrations.tools import goapify_tool

logger = logging.getLogger(__name__)


def create_goap_agent(
    tools: list[BaseTool],
    goal: str | GoalSpec,
    *,
    llm: BaseChatModel | None = None,
    preconditions: dict[str, dict[str, Any]] | None = None,
    effects: dict[str, dict[str, Any]] | None = None,
    resources: dict[str, dict[str, float]] | None = None,
    costs: dict[str, float] | None = None,
    result_keys: dict[str, str] | None = None,
    **graph_kwargs: Any,
) -> CompiledStateGraph:
    """Create a compiled GOAP agent from a list of LangChain tools and a goal.

    **NL-at-invocation-time limitation**: the returned ``CompiledStateGraph``
    only exposes LangGraph's ``invoke({"goal": GoalSpec, "world_state":
    dict})`` and ``ainvoke`` interfaces.  It does **not** have
    :meth:`~langgoap.graph.builder.GoapGraph.invoke_nl`, because that
    method lives on the ``GoapGraph`` builder, not on the compiled
    graph.  For single-shot NL execution pass ``goal=request_string``
    directly (NL interpretation happens once at construction).  For
    repeated NL-driven invocations against the same tool set,
    construct a :class:`GoapGraph` directly and use ``invoke_nl()``.

    Args:
        tools: The LangChain tools the agent may call.
        goal: A :class:`GoalSpec`, or a natural-language string.  If a
            string, ``llm`` must be provided.
        llm: Chat model used to interpret a string goal.  Ignored when
            ``goal`` is already a :class:`GoalSpec`.
        preconditions: Optional mapping of tool name → preconditions
            dict.  Missing tools get empty preconditions.
        effects: Optional mapping of tool name → effects dict.
            Missing tools get empty effects.
        resources: Optional mapping of tool name → resources dict.
        costs: Optional mapping of tool name → action cost override.
        result_keys: Optional mapping of tool name → world-state key
            that should receive the tool's raw return value at
            execution time.  Use this to wire one tool's output into
            the next tool's input — e.g. ``{"research_topic":
            "brief"}`` makes the return value of ``research_topic``
            available to a downstream ``write_article(brief)`` tool
            via ``world_state["brief"]``.  Planning is unaffected;
            A* still reasons over the boolean flags in ``effects``.
            A key here must not collide with any key declared for the
            same tool in ``effects`` (see :func:`goapify_tool`).
        **graph_kwargs: Forwarded to
            :meth:`GoapGraph.compile` (e.g. ``checkpointer``, ``store``).

    Returns:
        A compiled :class:`CompiledStateGraph` ready for ``.invoke()``
        or ``.ainvoke()``.

    Raises:
        ValueError: If ``goal`` is a string but ``llm`` is ``None``.
    """
    actions = _wrap_tools_as_actions(
        tools,
        preconditions=preconditions or {},
        effects=effects or {},
        resources=resources or {},
        costs=costs or {},
        result_keys=result_keys or {},
    )
    resolved_goal = _resolve_goal(goal, llm=llm, actions=actions)

    graph = GoapGraph(actions=actions)
    compiled = graph.compile(**graph_kwargs)
    # Attach the resolved goal as a public attribute so callers can
    # pass it back to ``invoke({"goal": agent.goap_goal, ...})`` — the
    # resolved goal is otherwise opaque to the caller when the input
    # was a natural-language string.
    setattr(compiled, "goap_goal", resolved_goal)
    return compiled


def _wrap_tools_as_actions(
    tools: list[BaseTool],
    *,
    preconditions: dict[str, dict[str, Any]],
    effects: dict[str, dict[str, Any]],
    resources: dict[str, dict[str, float]],
    costs: dict[str, float],
    result_keys: dict[str, str],
) -> list[ActionSpec]:
    """Wrap every tool with ``goapify_tool`` and warn on empty-effect tools."""
    actions: list[ActionSpec] = []
    tools_without_eff: list[str] = []
    for tool in tools:
        tool_eff = effects.get(tool.name)
        tool_res = resources.get(tool.name)
        if not tool_eff and not tool_res:
            tools_without_eff.append(tool.name)
        actions.append(
            goapify_tool(
                tool,
                preconditions=preconditions.get(tool.name),
                effects=tool_eff,
                cost=costs.get(tool.name, 1.0),
                resources=tool_res,
                result_key=result_keys.get(tool.name),
            )
        )
    if tools_without_eff:
        logger.warning(
            "create_goap_agent: the following tools have no effects or "
            "resources declared and will be treated as no-op actions by "
            "the planner: %s. Pass effects={<tool>: {...}} to make them "
            "plan-visible.",
            tools_without_eff,
        )
    return actions


def _resolve_goal(
    goal: str | GoalSpec,
    *,
    llm: BaseChatModel | None,
    actions: list[ActionSpec],
) -> GoalSpec:
    """Return a ``GoalSpec``, interpreting a natural-language string once."""
    if not isinstance(goal, str):
        return goal
    if llm is None:
        raise ValueError(
            "create_goap_agent received a string goal but no llm. "
            "Pass llm=ChatOpenAI(...) or convert to a GoalSpec beforehand."
        )
    # Local import to keep interpreter dependencies lazy.
    from langgoap.interpreter import GoalInterpreter

    return GoalInterpreter(llm=llm, actions=actions).interpret(goal)


__all__ = ["create_goap_agent"]
