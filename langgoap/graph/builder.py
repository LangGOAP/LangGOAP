"""Convenience builder for GOAP execution graphs.

GoapGraph assembles a LangGraph StateGraph with planner, executor,
and observer nodes wired together for the GOAP execution loop.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.nodes import GoapExecutor, GoapObserver, GoapPlanner
from langgoap.graph.state import GoapState
from langgoap.history import StoreExecutionHistory
from langgoap.tracing import PlanningTracer


class GoapGraph:
    """Builder that produces a compiled LangGraph for GOAP execution.

    Supports two usage styles:

    **Explicit compile + invoke** (for power users who need the compiled graph)::

        graph = GoapGraph(actions=[action1, action2, ...])
        compiled = graph.compile(checkpointer=saver)
        result = compiled.invoke({
            "goal": GoalSpec(conditions={"done": True}),
            "world_state": {"a": True},
        })

    **Convenience invoke** (single-shot, no persistence)::

        graph = GoapGraph(actions=[action1, action2, ...])
        result = graph.invoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={"a": True},
        )

    The graph structure is::

        START → planner → executor → observer ──→ END
                  ↑                     │
                  └─────────────────────┘
    """

    def __init__(
        self,
        actions: list[ActionSpec],
        *,
        tracer: PlanningTracer | None = None,
        history: StoreExecutionHistory | None = None,
    ) -> None:
        self.actions = actions
        self._tracer = tracer
        self._history = history

    def compile(
        self,
        checkpointer: Checkpointer | None = None,
        store: Any = None,
    ) -> CompiledStateGraph:
        """Build and compile the GOAP StateGraph.

        Args:
            checkpointer: Optional LangGraph checkpointer for persistence.
            store: Optional LangGraph store for shared state.

        Returns:
            A compiled StateGraph ready for invocation.
        """
        builder = StateGraph(GoapState)

        # All three nodes are wrapped with RunnableLambda so LangGraph
        # dispatches to the async variant under ``ainvoke`` and the sync
        # variant under ``invoke``.  Without this wiring, the planner
        # and observer would only ever see the sync ``__call__`` path
        # and their async tracer hooks would never fire (audit NS2).
        planner = GoapPlanner(self.actions, tracer=self._tracer)
        executor = GoapExecutor(tracer=self._tracer)
        observer = GoapObserver(
            self.actions, tracer=self._tracer, history=self._history
        )
        builder.add_node(
            "planner",
            RunnableLambda(func=planner.__call__, afunc=planner.acall),
        )
        builder.add_node(
            "executor",
            RunnableLambda(func=executor.__call__, afunc=executor.acall),
        )
        builder.add_node(
            "observer",
            RunnableLambda(func=observer.__call__, afunc=observer.acall),
        )

        # Wire edges
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "executor")
        builder.add_edge("executor", "observer")
        # Observer uses Command(goto=...) for routing — no explicit edges needed

        return builder.compile(
            checkpointer=checkpointer,
            store=store,
        )

    def invoke(
        self,
        goal: GoalSpec | MultiGoal,
        world_state: dict[str, Any] | None = None,
        config: RunnableConfig | None = None,
    ) -> GoapState:
        """Convenience method: compile and invoke the graph in one call.

        Suitable for single-shot executions that don't need persistence or
        time-travel.  For repeated invocations with the same compiled graph,
        use :meth:`compile` directly.

        Args:
            goal: The goal to achieve.  May be a single :class:`GoalSpec`
                or a :class:`MultiGoal` wrapping several sub-goals.
            world_state: Initial world state (defaults to an empty dict).
            config: Optional LangGraph run configuration
                (e.g. ``{"configurable": {"thread_id": "..."}}``)

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes.
        """
        compiled = self.compile()
        input_state: GoapState = {
            "goal": goal,
            "world_state": world_state or {},
        }
        # compiled.invoke returns dict[str, Any]; cast to GoapState for callers.
        return compiled.invoke(input_state, config=config)  # type: ignore[return-value]

    async def ainvoke(
        self,
        goal: GoalSpec | MultiGoal,
        world_state: dict[str, Any] | None = None,
        config: RunnableConfig | None = None,
    ) -> GoapState:
        """Async convenience method: compile and invoke the graph.

        Identical to :meth:`invoke` but uses ``ainvoke`` on the compiled
        graph, enabling native async execution of action callables.

        Args:
            goal: The goal to achieve.  May be a single :class:`GoalSpec`
                or a :class:`MultiGoal` wrapping several sub-goals.
            world_state: Initial world state (defaults to an empty dict).
            config: Optional LangGraph run configuration.

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes.
        """
        compiled = self.compile()
        input_state: GoapState = {
            "goal": goal,
            "world_state": world_state or {},
        }
        return await compiled.ainvoke(input_state, config=config)  # type: ignore[return-value]

    def invoke_nl(
        self,
        request: str,
        world_state: dict[str, Any] | None = None,
        *,
        llm: BaseChatModel,
        config: RunnableConfig | None = None,
        structured_output_kwargs: dict[str, Any] | None = None,
    ) -> GoapState:
        """Interpret a natural language request and execute the GOAP loop.

        Convenience method that creates a :class:`~langgoap.interpreter.GoalInterpreter`,
        converts the request to a :class:`~langgoap.goals.GoalSpec`, and invokes
        the graph in one call.

        Args:
            request: Natural language description of the goal.
            world_state: Initial world state (defaults to an empty dict).
            llm: LangChain chat model for goal interpretation.
            config: Optional LangGraph run configuration.
            structured_output_kwargs: Forwarded to
                :class:`~langgoap.interpreter.GoalInterpreter` and then to
                ``llm.with_structured_output()``.  Use
                ``{"method": "function_calling"}`` for OpenAI when the
                conditions dict must stay open-ended.

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes.
        """
        from langgoap.interpreter import GoalInterpreter

        interpreter = GoalInterpreter(
            llm=llm,
            actions=self.actions,
            structured_output_kwargs=structured_output_kwargs,
        )
        goal = interpreter.interpret(request, world_state=world_state)
        return self.invoke(goal=goal, world_state=world_state, config=config)

    async def ainvoke_nl(
        self,
        request: str,
        world_state: dict[str, Any] | None = None,
        *,
        llm: BaseChatModel,
        config: RunnableConfig | None = None,
        structured_output_kwargs: dict[str, Any] | None = None,
    ) -> GoapState:
        """Async variant of :meth:`invoke_nl`.

        Args:
            request: Natural language description of the goal.
            world_state: Initial world state (defaults to an empty dict).
            llm: LangChain chat model for goal interpretation.
            config: Optional LangGraph run configuration.
            structured_output_kwargs: Forwarded to
                :class:`~langgoap.interpreter.GoalInterpreter` and then to
                ``llm.with_structured_output()``.  Use
                ``{"method": "function_calling"}`` for OpenAI when the
                conditions dict must stay open-ended.

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes.
        """
        from langgoap.interpreter import GoalInterpreter

        interpreter = GoalInterpreter(
            llm=llm,
            actions=self.actions,
            structured_output_kwargs=structured_output_kwargs,
        )
        goal = await interpreter.ainterpret(request, world_state=world_state)
        return await self.ainvoke(goal=goal, world_state=world_state, config=config)
