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

import random
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgoap.actions import ActionSpec
from langgoap.conditions import AsyncConditionResolver, ConditionResolver
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.guards import ActionGuard, AsyncActionGuard
from langgoap.history import StoreExecutionHistory
from langgoap.integrations._compile_options import _check_compile_kwargs
from langgoap.planner.transitions import TransitionModel
from langgoap.sensors import AsyncSensor, Sensor
from langgoap.tracing import PlanningTracer

if TYPE_CHECKING:
    from langgoap.planner.strategy import PlanningStrategy
    from langgoap.stuck import StuckHandler
    from langgoap.termination import TerminationPolicy


class GoapSubgraph:
    """A GOAP planner packaged as a reusable sub-graph.

    Args:
        actions: Actions the planner may use.
        goal: Goal specification.  Pre-resolved at construction time;
            NL interpretation must happen before this class is built.
        strategy, tracer, history, sensors, guards, resolvers,
        record_expansions, transition_model, rng, stuck_handlers,
        max_stuck_iterations, termination_policies, executor, history_state: Forwarded verbatim to
            :class:`~langgoap.graph.builder.GoapGraph`; see its docstring.

    Example::

        sub = GoapSubgraph(
            actions=[a1, a2, a3],
            goal=GoalSpec(conditions={"done": True}),
        )
        compiled = sub.compile()
        result = compiled.invoke({"world_state": {}, "goal": goal})

    ``result`` is the JSON-native state (``execution_history`` entries
    are dicts); :func:`langgoap.rehydrate_state` converts it back to
    typed objects.  When the sub-graph is compiled with a checkpointer,
    pass ``goal.to_dict()`` as the input goal.
    """

    def __init__(
        self,
        actions: list[ActionSpec],
        goal: GoalSpec,
        *,
        strategy: PlanningStrategy | None = None,
        tracer: PlanningTracer | None = None,
        history: StoreExecutionHistory | None = None,
        sensors: list[Sensor | AsyncSensor] | None = None,
        guards: list[ActionGuard | AsyncActionGuard] | None = None,
        resolvers: list[ConditionResolver | AsyncConditionResolver] | None = None,
        record_expansions: bool = False,
        transition_model: TransitionModel | None = None,
        rng: random.Random | None = None,
        stuck_handlers: list[StuckHandler] | None = None,
        max_stuck_iterations: int = 3,
        termination_policies: list[TerminationPolicy] | None = None,
        executor: Literal["sequential", "parallel"] = "sequential",
        history_state: Literal["full", "delta", "planning"] = "full",
    ) -> None:
        self.actions = actions
        self.goal = goal
        self.graph = GoapGraph(
            actions=actions,
            strategy=strategy,
            tracer=tracer,
            history=history,
            sensors=sensors,
            guards=guards,
            resolvers=resolvers,
            record_expansions=record_expansions,
            transition_model=transition_model,
            rng=rng,
            stuck_handlers=stuck_handlers,
            max_stuck_iterations=max_stuck_iterations,
            termination_policies=termination_policies,
            executor=executor,
            history_state=history_state,
        )

    def compile(self, **compile_kwargs: Any) -> CompiledStateGraph:
        """Compile the sub-graph.

        Args:
            **compile_kwargs: Forwarded to :meth:`GoapGraph.compile`
                (``checkpointer``, ``store``, ``interrupt_before``,
                ``interrupt_after``).  Any other name raises ``TypeError``
                naming the compile options and the ``GoapGraph`` options.

        Returns:
            A compiled :class:`CompiledStateGraph` using the
            :class:`~langgoap.graph.state.GoapStateChannels` schema
            internally.
        """
        _check_compile_kwargs("GoapSubgraph.compile", compile_kwargs)
        return self.graph.compile(**compile_kwargs)


def add_goap_subgraph(
    parent_builder: StateGraph,
    *,
    name: str,
    actions: list[ActionSpec],
    goal: GoalSpec,
    input_key: str = "world_state",
    output_key: str = "plan_result",
    compile_kwargs: dict[str, Any] | None = None,
    strategy: PlanningStrategy | None = None,
    tracer: PlanningTracer | None = None,
    history: StoreExecutionHistory | None = None,
    sensors: list[Sensor | AsyncSensor] | None = None,
    guards: list[ActionGuard | AsyncActionGuard] | None = None,
    resolvers: list[ConditionResolver | AsyncConditionResolver] | None = None,
    record_expansions: bool = False,
    transition_model: TransitionModel | None = None,
    rng: random.Random | None = None,
    stuck_handlers: list[StuckHandler] | None = None,
    max_stuck_iterations: int = 3,
    termination_policies: list[TerminationPolicy] | None = None,
    executor: Literal["sequential", "parallel"] = "sequential",
    history_state: Literal["full", "delta", "planning"] = "full",
) -> None:
    """Attach a GOAP sub-graph as a node inside a parent ``StateGraph``.

    The sub-graph is compiled and registered as a node with sync and
    async variants.  Each variant:

    1. Reads ``state[input_key]`` from the parent state.
    2. Invokes the sub-graph (``invoke`` under ``parent.invoke``,
       ``ainvoke`` under ``parent.ainvoke``) with a fresh JSON-native
       input carrying that world state and the configured ``goal``, and
       with the parent's ``RunnableConfig``, so the parent's
       checkpointer, callbacks, tags and thread reach the child.  An
       ``interrupt()`` raised inside the child (human approval) pauses
       the parent run and resumes with ``Command(resume=...)`` on the
       parent.
    3. Writes the final sub-graph state into ``state[output_key]`` as
       the plain JSON dict LangGraph produced, so the parent schema stays
       decoupled from the GOAP state and any parent checkpointer can
       persist it.  Use :func:`langgoap.rehydrate_state` to get typed
       objects back.

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
        compile_kwargs: Optional :meth:`GoapGraph.compile` options for the
            child (``interrupt_before`` / ``interrupt_after``; a
            checkpointer is normally inherited from the parent).
        strategy, tracer, history, sensors, guards, resolvers,
        record_expansions, transition_model, rng, stuck_handlers,
        max_stuck_iterations, termination_policies, executor, history_state: Forwarded verbatim to
            :class:`~langgoap.graph.builder.GoapGraph`; see its docstring.
    """
    sub = GoapSubgraph(
        actions=actions,
        goal=goal,
        strategy=strategy,
        tracer=tracer,
        history=history,
        sensors=sensors,
        guards=guards,
        resolvers=resolvers,
        record_expansions=record_expansions,
        transition_model=transition_model,
        rng=rng,
        stuck_handlers=stuck_handlers,
        max_stuck_iterations=max_stuck_iterations,
        termination_policies=termination_policies,
        executor=executor,
        history_state=history_state,
    )
    _check_compile_kwargs("add_goap_subgraph", dict(compile_kwargs or {}))
    compiled_sub = sub.compile(**(compile_kwargs or {}))

    def _sub_input(parent_state: dict[str, Any]) -> dict[str, Any]:
        world_state = dict(parent_state.get(input_key) or {})
        return sub.graph.initial_state(goal, world_state)

    def _node(parent_state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        sub_result = compiled_sub.invoke(_sub_input(parent_state), config=config)
        return {output_key: dict(sub_result)}

    async def _anode(
        parent_state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        sub_result = await compiled_sub.ainvoke(_sub_input(parent_state), config=config)
        return {output_key: dict(sub_result)}

    # StateGraph.add_node's signature is parameterised on the parent
    # state's TypedDict, which we do not know statically — cast to a
    # generic Runnable so mypy accepts the insertion.  The runtime
    # contract still holds: the node receives and returns plain dicts.
    parent_builder.add_node(
        name, cast(Runnable[Any, Any], RunnableLambda(func=_node, afunc=_anode))
    )


__all__ = ["GoapSubgraph", "add_goap_subgraph"]
