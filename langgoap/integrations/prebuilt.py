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
import random
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from langgoap.actions import ActionSpec
from langgoap.conditions import AsyncConditionResolver, ConditionResolver
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.guards import ActionGuard, AsyncActionGuard
from langgoap.history import StoreExecutionHistory
from langgoap.integrations._compile_options import (
    COMPILE_OPTIONS,
    _check_compile_kwargs,
)
from langgoap.integrations.tools import EffectValidator, goapify_tool
from langgoap.planner.transitions import TransitionModel
from langgoap.qos import ActionQos
from langgoap.sensors import AsyncSensor, Sensor
from langgoap.tracing import PlanningTracer
from langgoap.types import CostFunction

if TYPE_CHECKING:
    from langgoap.planner.strategy import PlanningStrategy
    from langgoap.stuck import StuckHandler
    from langgoap.termination import TerminationPolicy

logger = logging.getLogger(__name__)

# ``COMPILE_OPTIONS`` / ``_check_compile_kwargs`` live in ``_compile_options`` and are
# re-exported here for callers that imported them from this module.


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
    max_retries: dict[str, int] | None = None,
    durations: dict[str, timedelta] | None = None,
    effect_validators: dict[str, EffectValidator] | None = None,
    qos: dict[str, ActionQos] | None = None,
    require_human_approval: dict[str, bool | type] | None = None,
    human_input_key: dict[str, str] | None = None,
    can_rerun: dict[str, bool] | None = None,
    read_only: dict[str, bool] | None = None,
    utility: dict[str, float | CostFunction] | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
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
    **compile_kwargs: Any,
) -> CompiledStateGraph:
    """Create a compiled GOAP agent from a list of LangChain tools and a goal.

    The returned graph exposes LangGraph's ``invoke`` / ``ainvoke`` and
    returns the JSON-native :class:`~langgoap.graph.state.GoapStateChannels`
    form (``plan`` and ``execution_history`` entries are dicts); use
    :func:`langgoap.rehydrate_state` to get typed objects.  Pass the goal
    as ``agent.goap_goal`` (or ``goal.to_dict()`` when a checkpointer is
    attached, because LangGraph persists the raw input).

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
        max_retries: Optional mapping of tool name → planner-level
            retry budget.  ``0`` (default) blacklists the action on
            its first failure; ``N`` allows ``N`` extra planner-level
            replans before blacklisting (i.e. ``N + 1`` total
            failures).  This is the knob that lets a transient
            failure trigger replanning without giving up entirely.
        durations: Optional mapping of tool name → :class:`datetime.timedelta`.
            Used by the CSP scheduler when computing parallel
            execution windows.  Tools with no entry default to
            instantaneous actions.
        effect_validators: Optional mapping of tool name → callable
            ``(pre_state, post_state) -> bool`` invoked after
            execution.  Returning ``False`` signals the action did
            not produce its declared effects (soft failure → replan).
        qos: Optional mapping of tool name →
            :class:`~langgoap.qos.ActionQos` retry / idempotency policy.
        require_human_approval: Optional mapping of tool name → ``True``
            (approve/deny gate) or a Pydantic ``BaseModel`` subclass
            (typed form) that pauses the run with ``interrupt()`` before
            the tool executes.
        human_input_key: Optional mapping of tool name → world-state key
            receiving the approved form.
        can_rerun: Optional mapping of tool name → ``False`` for tools
            that must not run twice in one run.
        read_only: Optional mapping of tool name → ``True`` for tools
            that do not mutate the world.
        utility: Optional mapping of tool name → utility score for
            :class:`~langgoap.planner.utility.UtilityStrategy`.
        metadata: Optional mapping of tool name → free-form metadata dict.
        strategy, tracer, history, sensors, guards, resolvers,
        record_expansions, transition_model, rng, stuck_handlers,
        max_stuck_iterations, termination_policies, executor, history_state: Forwarded verbatim to
            :class:`~langgoap.graph.builder.GoapGraph`; see its docstring.
        **compile_kwargs: Forwarded to :meth:`GoapGraph.compile`
            (``checkpointer``, ``store``, ``interrupt_before``,
            ``interrupt_after``).  Any other name raises ``TypeError``.

    Returns:
        A compiled :class:`CompiledStateGraph` ready for ``.invoke()``
        or ``.ainvoke()``.  It carries two attributes for convenience:
        ``goap_goal`` (the resolved :class:`GoalSpec`) and ``goap_actions``
        (the wrapped :class:`ActionSpec` list, e.g. for
        :func:`langgoap.rehydrate_state`).

    Raises:
        ValueError: If ``goal`` is a string but ``llm`` is ``None``.
        TypeError: If ``**compile_kwargs`` contains a name that
            :meth:`GoapGraph.compile` does not accept.
    """
    _check_compile_kwargs("create_goap_agent", compile_kwargs)
    actions = _wrap_tools_as_actions(
        tools,
        preconditions=preconditions or {},
        effects=effects or {},
        resources=resources or {},
        costs=costs or {},
        result_keys=result_keys or {},
        max_retries=max_retries or {},
        durations=durations or {},
        effect_validators=effect_validators or {},
        qos=qos or {},
        require_human_approval=require_human_approval or {},
        human_input_key=human_input_key or {},
        can_rerun=can_rerun or {},
        read_only=read_only or {},
        utility=utility or {},
        metadata=metadata or {},
    )
    resolved_goal = _resolve_goal(goal, llm=llm, actions=actions)

    graph = GoapGraph(
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
    compiled = graph.compile(**compile_kwargs)
    # Attach the resolved goal and the wrapped actions as public attributes
    # so callers can pass the goal back to ``invoke({"goal":
    # agent.goap_goal, ...})`` (the resolved goal is otherwise opaque when
    # the input was a natural-language string) and rehydrate the JSON
    # result with ``rehydrate_state(result, agent.goap_actions)``.
    setattr(compiled, "goap_goal", resolved_goal)
    setattr(compiled, "goap_actions", actions)
    return compiled


def _wrap_tools_as_actions(
    tools: list[BaseTool],
    *,
    preconditions: dict[str, dict[str, Any]],
    effects: dict[str, dict[str, Any]],
    resources: dict[str, dict[str, float]],
    costs: dict[str, float],
    result_keys: dict[str, str],
    max_retries: dict[str, int],
    durations: dict[str, timedelta],
    effect_validators: dict[str, EffectValidator],
    qos: dict[str, ActionQos],
    require_human_approval: dict[str, bool | type],
    human_input_key: dict[str, str],
    can_rerun: dict[str, bool],
    read_only: dict[str, bool],
    utility: dict[str, float | CostFunction],
    metadata: dict[str, dict[str, Any]],
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
                max_retries=max_retries.get(tool.name, 0),
                duration=durations.get(tool.name),
                effect_validator=effect_validators.get(tool.name),
                qos=qos.get(tool.name),
                require_human_approval=require_human_approval.get(tool.name, False),
                human_input_key=human_input_key.get(tool.name),
                can_rerun=can_rerun.get(tool.name, True),
                read_only=read_only.get(tool.name, False),
                utility=utility.get(tool.name),
                metadata=metadata.get(tool.name),
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
