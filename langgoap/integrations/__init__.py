"""Low-code LangGraph integration layer.

Three on-ramps from "figure it out for me" to "full control":

* :func:`create_goap_agent` (Layer A) — prebuilt ``CompiledStateGraph``
  from a tool list and a goal, mirroring ``langgraph.prebuilt
  .create_react_agent``.  Fastest way to get a running GOAP agent.
* :func:`goapify_tool` (Layer B) — deterministic adapter from a
  LangChain :class:`~langchain_core.tools.BaseTool` to a LangGOAP
  :class:`~langgoap.actions.ActionSpec`.  Used internally by Layer A
  and directly by users who want to build their own action list.
* :class:`GoapSubgraph` and :func:`add_goap_subgraph` (Layer C) —
  embed a GOAP sub-graph inside an existing LangGraph application
  without leaking GOAP internals into the parent state schema.

All three layers build on each other: Layer A uses Layer B; Layer C
uses :class:`~langgoap.graph.builder.GoapGraph` internally.  Nothing
is duplicated.
"""

from __future__ import annotations

from langgoap.integrations.deepagents import (
    create_goap_subagent,
    create_goap_tool,
    format_goap_result,
)
from langgoap.integrations.langgraph_deploy import scaffold_deployment
from langgoap.integrations.prebuilt import create_goap_agent
from langgoap.integrations.subgraph import GoapSubgraph, add_goap_subgraph
from langgoap.integrations.tools import goapify_tool

__all__ = [
    "create_goap_agent",
    "create_goap_tool",
    "create_goap_subagent",
    "format_goap_result",
    "goapify_tool",
    "GoapSubgraph",
    "add_goap_subgraph",
    "scaffold_deployment",
]
