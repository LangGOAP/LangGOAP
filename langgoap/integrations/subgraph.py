"""Embed a GOAP planner as a sub-graph inside a larger LangGraph application.

Layer C of the three-layer low-code on-ramp (AD-2).  Offers the most
control: construct a :class:`GoapSubgraph` explicitly, compile it,
and drop it into a parent :class:`~langgraph.graph.StateGraph` via
:func:`add_goap_subgraph`.  All GOAP-internal state (``plan``,
``current_step``, ``execution_history``, ``blacklisted_actions``,
``action_failure_counts``) stays sealed inside the sub-graph; the
parent only sees two keys (default ``"world_state"`` and
``"plan_result"``).

Routing note — **LangGraph ``Send`` is not involved.**  ``Send`` is
the fan-out primitive for a single node; subgraph routing uses the
normal ``add_node`` / ``add_edge`` API.  A GOAP executor may still
use ``Send`` internally to dispatch parallel actions to sibling
worker nodes — see the ``flexible_job_shop`` tutorial for an example.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph


class GoapSubgraph:
    """A GOAP planner packaged as a reusable sub-graph.

    Args:
        actions: Actions the planner may use.
        goal: Goal specification.  Pre-resolved at construction time;
            NL interpretation must happen before this class is built.

    Example::

        sub = GoapSubgraph(
            actions=[a1, a2, a3],
            goal=GoalSpec(conditions={"done": True}),
        )
        compiled = sub.compile()
        result = compiled.invoke({"world_state": {}, "goal": goal})
    """

    def __init__(
        self,
        actions: list[ActionSpec],
        goal: GoalSpec,
    ) -> None:
        self.actions = actions
        self.goal = goal

    def compile(self, **kwargs: Any) -> CompiledStateGraph:
        """Compile the sub-graph.

        Args:
            **kwargs: Forwarded to
                :meth:`GoapGraph.compile` (``checkpointer``, ``store``).

        Returns:
            A compiled :class:`CompiledStateGraph` using the ``GoapState``
            schema internally.
        """
        graph = GoapGraph(actions=self.actions)
        return graph.compile(**kwargs)


def add_goap_subgraph(
    parent_builder: StateGraph,
    *,
    name: str,
    actions: list[ActionSpec],
    goal: GoalSpec,
    input_key: str = "world_state",
    output_key: str = "plan_result",
) -> None:
    """Attach a GOAP sub-graph as a node inside a parent ``StateGraph``.

    The sub-graph is compiled and wrapped in a closure that:

    1. Reads ``state[input_key]`` from the parent state.
    2. Invokes the sub-graph with a fresh ``GoapState`` carrying that
       world state and the configured ``goal``.
    3. Writes the final sub-graph state into ``state[output_key]`` as
       a plain dict so the parent schema stays decoupled from
       ``GoapState``.

    The parent graph must declare ``input_key`` and ``output_key`` in
    its state schema (both are plain ``dict[str, Any]`` slots).

    Args:
        parent_builder: The parent :class:`StateGraph` under
            construction.
        name: Node name to register in the parent graph.
        actions: Actions for the sub-graph's planner.
        goal: Goal specification for the sub-graph.
        input_key: Parent-state key providing the initial world state.
            Defaults to ``"world_state"``.
        output_key: Parent-state key where the sub-graph's final
            state will be stored.  Defaults to ``"plan_result"``.
    """
    sub = GoapSubgraph(actions=actions, goal=goal)
    compiled_sub = sub.compile()

    def _node(parent_state: dict[str, Any]) -> dict[str, Any]:
        world_state = dict(parent_state.get(input_key) or {})
        sub_result = compiled_sub.invoke({"world_state": world_state, "goal": goal})
        return {output_key: dict(sub_result)}

    # StateGraph.add_node's signature is parameterised on the parent
    # state's TypedDict, which we do not know statically — cast to a
    # generic Runnable so mypy accepts the insertion.  The runtime
    # contract still holds: the node receives and returns plain dicts.
    parent_builder.add_node(name, cast(Runnable[Any, Any], _node))


__all__ = ["GoapSubgraph", "add_goap_subgraph"]
