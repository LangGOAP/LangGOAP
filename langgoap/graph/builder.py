"""Convenience builder for GOAP execution graphs.

GoapGraph assembles a LangGraph StateGraph with planner, executor,
and observer nodes wired together for the GOAP execution loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from langgoap.actions import ActionSpec
from langgoap.conditions import AsyncConditionResolver, ConditionResolver
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.nodes import GoapExecutor, GoapObserver, GoapPlanner
from langgoap.graph.state import GoapState
from langgoap.guards import ActionGuard, AsyncActionGuard
from langgoap.history import StoreExecutionHistory
from langgoap.sensors import AsyncSensor, Sensor
from langgoap.serde import install_langgoap_serde
from langgoap.tracing import PlanningTracer

if TYPE_CHECKING:
    from langgoap.planner.strategy import PlanningStrategy


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

    **Custom planning strategy**::

        from langgoap.planner.strategy import LazyDecompositionStrategy
        graph = GoapGraph(
            actions=[action1, action2, ...],
            strategy=LazyDecompositionStrategy(lookahead=2),
        )
        result = graph.invoke(goal=goal, world_state={})

    The graph structure is::

        START → planner → executor → observer ──→ END
                  ↑                     │
                  └─────────────────────┘
    """

    def __init__(
        self,
        actions: list[ActionSpec],
        *,
        strategy: PlanningStrategy | None = None,
        tracer: PlanningTracer | None = None,
        history: StoreExecutionHistory | None = None,
        sensors: list[Sensor | AsyncSensor] | None = None,
        guards: list[ActionGuard | AsyncActionGuard] | None = None,
        resolvers: list[ConditionResolver | AsyncConditionResolver] | None = None,
    ) -> None:
        self.actions = actions
        self._strategy = strategy
        self._tracer = tracer
        self._history = history
        self._sensors = sensors
        self._guards = guards
        self._resolvers = resolvers

    def compile(
        self,
        checkpointer: Checkpointer | None = None,
        store: Any = None,
        *,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ) -> CompiledStateGraph:
        """Build and compile the GOAP StateGraph.

        Args:
            checkpointer: Optional LangGraph checkpointer for persistence.
                When provided, the checkpointer's serde is swapped for a
                LangGoap-aware subclass (see :mod:`langgoap.serde`) so that
                frozen dataclasses using ``MappingProxyType`` fields
                round-trip through msgpack correctly.
            store: Optional LangGraph store for shared state.
            interrupt_before: Optional list of node names to interrupt
                before.  Forwarded to ``StateGraph.compile``.  Use
                ``["executor"]`` to gate each action execution behind a
                human-in-the-loop checkpoint boundary, then resume via
                ``compiled.invoke(None, config=...)``.
            interrupt_after: Optional list of node names to interrupt
                after.  Forwarded to ``StateGraph.compile``.

        Returns:
            A compiled StateGraph ready for invocation.
        """
        # Install LangGoap's MappingProxyType-aware serde on the
        # checkpointer so frozen dataclasses serialize cleanly.  Skipped
        # silently when no checkpointer is provided — no-ops don't need
        # a custom serde because state never hits disk.
        # isinstance narrows Checkpointer (= BaseCheckpointSaver | None |
        # Literal[False]) to BaseCheckpointSaver, satisfying install_langgoap_serde's
        # parameter type without a cast or type: ignore.
        if isinstance(checkpointer, BaseCheckpointSaver):
            install_langgoap_serde(checkpointer)

        builder = StateGraph(GoapState)

        # All three nodes are wrapped with RunnableLambda so LangGraph
        # dispatches to the async variant under ``ainvoke`` and the sync
        # variant under ``invoke``.  Without this wiring, the planner
        # and observer would only ever see the sync ``__call__`` path
        # and their async tracer hooks would never fire (audit NS2).
        planner = GoapPlanner(
            self.actions,
            strategy=self._strategy,
            tracer=self._tracer,
            sensors=self._sensors,
            resolvers=self._resolvers,
        )
        executor = GoapExecutor(tracer=self._tracer, guards=self._guards)
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

        compile_kwargs: dict[str, Any] = {
            "checkpointer": checkpointer,
            "store": store,
        }
        if interrupt_before is not None:
            compile_kwargs["interrupt_before"] = interrupt_before
        if interrupt_after is not None:
            compile_kwargs["interrupt_after"] = interrupt_after
        return builder.compile(**compile_kwargs)

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
        # The compiled graph's invoke() is typed as returning dict[str, Any];
        # cast to GoapState because we own the graph's output schema entirely.
        return cast(GoapState, compiled.invoke(input_state, config=config))

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
        return cast(GoapState, await compiled.ainvoke(input_state, config=config))

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
