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
        **graph_kwargs: Forwarded to
            :meth:`GoapGraph.compile` (e.g. ``checkpointer``, ``store``).

    Returns:
        A compiled :class:`CompiledStateGraph` ready for ``.invoke()``
        or ``.ainvoke()``.

    Raises:
        ValueError: If ``goal`` is a string but ``llm`` is ``None``.
    """
    preconditions = preconditions or {}
    effects = effects or {}
    resources = resources or {}
    costs = costs or {}

    # Wrap every tool with goapify_tool — Layer A delegates to Layer B.
    actions: list[ActionSpec] = []
    tools_without_eff: list[str] = []
    for tool in tools:
        tool_pre = preconditions.get(tool.name)
        tool_eff = effects.get(tool.name)
        tool_res = resources.get(tool.name)
        tool_cost = costs.get(tool.name, 1.0)
        if not tool_eff and not tool_res:
            tools_without_eff.append(tool.name)
        actions.append(
            goapify_tool(
                tool,
                preconditions=tool_pre,
                effects=tool_eff,
                cost=tool_cost,
                resources=tool_res,
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

    # Resolve a string goal via the interpreter (once, at construction).
    if isinstance(goal, str):
        if llm is None:
            raise ValueError(
                "create_goap_agent received a string goal but no llm. "
                "Pass llm=ChatOpenAI(...) or convert to a GoalSpec beforehand."
            )
        # Local import to keep interpreter dependencies lazy.
        from langgoap.interpreter import GoalInterpreter

        interpreter = GoalInterpreter(llm=llm, actions=actions)
        resolved_goal: GoalSpec = interpreter.interpret(goal)
    else:
        resolved_goal = goal

    graph = GoapGraph(actions=actions)
    compiled = graph.compile(**graph_kwargs)
    # Attach the resolved goal as a public attribute so callers can
    # pass it back to ``invoke({"goal": agent.goap_goal, ...})`` — the
    # resolved goal is otherwise opaque to the caller when the input
    # was a natural-language string.
    setattr(compiled, "goap_goal", resolved_goal)
    return compiled


__all__ = ["create_goap_agent"]
