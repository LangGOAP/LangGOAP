"""Adapter from LangChain ``BaseTool`` to LangGOAP ``ActionSpec``.

Layer B of the three-layer low-code on-ramp (AD-2).  Fully
deterministic — preconditions, effects, cost, resources, duration,
and retry policy all come from the caller, never from the LLM.

The produced :class:`~langgoap.actions.ActionSpec` carries an
``execute`` wrapper that calls the underlying tool and returns the
declared ``effects`` dict so the GOAP executor can apply them to the
world state.

**Tool return values and ``result_key``**

By default the tool's own return value is discarded — actions produce
*state-transition flags* declared up front via ``effects``, not raw
tool output.  This keeps the A* search space boolean and deterministic.

When you need the tool's runtime output available to downstream actions
(e.g. a ``search_web`` tool that returns results the next action must
read), pass ``result_key="<key>"``.  The tool's return value is then
stored in ``world_state[result_key]`` at execution time while the
``effects`` dict continues to drive planning unchanged:

.. code-block:: python

    search_action = goapify_tool(
        search_tool,
        preconditions={"query_ready": True},
        effects={"search_done": True},      # A* sees this boolean flag
        result_key="search_results",        # executor stores raw output here
    )
    # After execution: world_state["search_results"] == [list of results]
    # A* still plans using world_state["search_done"] == True
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable

from langchain_core.tools import BaseTool

from langgoap.actions import ActionSpec
from langgoap.qos import ActionQos
from langgoap.types import CostFunction

EffectValidator = Callable[[dict[str, Any], dict[str, Any]], bool]


def _tool_input(tool: BaseTool, state: dict[str, Any]) -> dict[str, Any]:
    """Filter *state* to the arguments *tool* declares.

    Tools with no ``args_schema`` (zero-arg tools) receive an empty dict,
    which ``StructuredTool.invoke`` / ``StructuredTool.ainvoke`` accept.

    * A Pydantic ``args_schema`` is read field by field: a value present
      under the field name *or* its alias is forwarded under the alias,
      which is the key the model validates by (issue #38).
    * A JSON-schema ``args_schema`` (a ``dict``, common in langchain-core
      1.x) and any other shape go through ``tool.args``, whose keys are the
      declared argument names.
    """
    schema = tool.args_schema
    if schema is None:
        return {}
    fields = getattr(schema, "model_fields", None) if isinstance(schema, type) else None
    if fields:
        payload: dict[str, Any] = {}
        for name, info in fields.items():
            alias = getattr(info, "alias", None) or name
            if alias in state:
                payload[alias] = state[alias]
            elif name in state:
                payload[alias] = state[name]
        return payload
    try:
        declared = list(tool.args.keys())
    except Exception:  # pragma: no cover - exotic schema objects
        declared = list(getattr(schema, "__fields__", {}).keys())
    return {k: state[k] for k in declared if k in state}


def _strict_runner(tool: BaseTool) -> BaseTool:
    """Return *tool*, or a copy of it that raises instead of returning errors.

    ``handle_tool_error`` / ``handle_validation_error`` turn a failing tool
    call into an error *string*; wrapped as an action that would be applied
    as a success and the observer would never replan (issue #37).  The copy
    shares the tool's callable, callbacks and schema; the caller's object is
    left untouched.
    """
    if getattr(tool, "handle_tool_error", False) or getattr(
        tool, "handle_validation_error", False
    ):
        return tool.model_copy(
            update={"handle_tool_error": False, "handle_validation_error": False}
        )
    return tool


def goapify_tool(
    tool: BaseTool,
    *,
    preconditions: dict[str, Any] | None = None,
    effects: dict[str, Any] | None = None,
    cost: float = 1.0,
    resources: dict[str, float] | None = None,
    duration: timedelta | None = None,
    max_retries: int = 0,
    effect_validator: EffectValidator | None = None,
    result_key: str | None = None,
    qos: ActionQos | None = None,
    require_human_approval: bool | type = False,
    human_input_key: str | None = None,
    can_rerun: bool = True,
    read_only: bool = False,
    utility: float | CostFunction | None = None,
    metadata: dict[str, Any] | None = None,
) -> ActionSpec:
    """Wrap a LangChain :class:`BaseTool` into a LangGOAP :class:`ActionSpec`.

    The resulting action's ``execute`` callable invokes ``tool`` with the
    state dict filtered to match the tool's declared input schema (if any)
    and returns the declared ``effects`` dict.  A tool failure is an action
    failure: tools created with ``handle_tool_error`` / ``handle_validation_error``
    are invoked through a copy with those options off, so a ``ToolException``
    raises (and the observer blacklists and replans) instead of the error
    text being applied as a successful result.  Passing no
    ``preconditions``/``effects`` yields an action with empty pre/eff —
    legal but rarely useful, which is why :func:`create_goap_agent` emits a
    warning in that case.

    Args:
        tool: Any LangChain ``BaseTool`` instance (e.g. produced by
            the ``@tool`` decorator).
        preconditions: World-state conditions required before the tool
            runs.  Keys must be hashable scalars.
        effects: World-state changes produced by executing the tool.
            These boolean/scalar flags are what A* reasons over.
        cost: Static action cost for A* search.  Defaults to 1.0.
        resources: Per-run resource usage for CSP optimization
            (e.g. ``{"cost_usd": 0.02, "tokens": 500}``).
        duration: Estimated wall-clock duration for temporal scheduling.
        max_retries: Number of retries before the executor blacklists
            the action.
        effect_validator: Optional postcondition checker.
        result_key: When set, the tool's raw return value is stored in
            ``world_state[result_key]`` after execution so that downstream
            actions can read it.  Planning is unaffected — A* continues to
            reason over the boolean flags in ``effects``.  Typical usage::

                search = goapify_tool(
                    search_tool,
                    effects={"search_done": True},
                    result_key="search_results",
                )
                # After execution: world_state["search_results"] == <raw output>

            ``result_key`` must not overlap with any key in ``effects`` —
            a collision would silently overwrite the planning flag with raw
            tool output, corrupting the world state A* reasons over.
        qos: Optional :class:`~langgoap.qos.ActionQos` retry / idempotency
            policy applied by the executor.
        require_human_approval: ``True`` to gate the tool behind a LangGraph
            ``interrupt()`` approve/deny prompt, or a Pydantic ``BaseModel``
            subclass to collect a typed form on resume.
        human_input_key: World-state key that receives the approved form
            (defaults to the action name).
        can_rerun: ``False`` for actions that must not run twice within one
            run (the planner will not re-plan through a spent instance).
        read_only: ``True`` for actions that do not mutate the world; a
            planning hint consumed by strategies and validators.
        utility: Optional utility score (static or callable) consumed by
            :class:`~langgoap.planner.utility.UtilityStrategy`.
        metadata: Free-form, JSON-serialisable metadata attached to the
            action.

        Every remaining :class:`ActionSpec` field has the same meaning and
        default as on :class:`ActionSpec` itself, so a tool wrapped here is
        a first-class action.

    Returns:
        A frozen :class:`ActionSpec` ready to feed into
        :class:`~langgoap.graph.builder.GoapGraph`.

    Raises:
        TypeError:  If ``tool`` is not a :class:`BaseTool` instance.
        ValueError: If ``result_key`` matches a key already declared in
            ``effects``.  This would silently overwrite the planning flag
            with raw tool output at execution time.
    """
    if not isinstance(tool, BaseTool):
        raise TypeError(
            f"goapify_tool expected a LangChain BaseTool, got "
            f"{type(tool).__name__}. Wrap plain functions with "
            f"@langchain_core.tools.tool first."
        )

    declared_effects = dict(effects or {})
    declared_preconditions = dict(preconditions or {})

    # Guard: a result_key that matches an effects key would silently overwrite
    # the planning flag with raw tool output at execution time.
    if result_key is not None and result_key in declared_effects:
        raise ValueError(
            f"goapify_tool: result_key={result_key!r} collides with a key "
            f"already declared in effects. Use a distinct key so that the "
            f"planning flag and the tool's runtime output do not overwrite "
            f"each other in world_state."
        )

    _result_key = result_key  # capture in closure

    def _merge(raw_output: Any) -> dict[str, Any]:
        """Merge declared effects with the optional result_key output."""
        update = dict(declared_effects)
        if _result_key is not None:
            update[_result_key] = raw_output
        return update

    runner = _strict_runner(tool)

    def _execute(state: dict[str, Any]) -> dict[str, Any]:
        raw_output = runner.invoke(_tool_input(tool, state))
        return _merge(raw_output)

    async def _aexecute(state: dict[str, Any]) -> dict[str, Any]:
        # BaseTool.ainvoke is part of the Runnable contract and is always
        # available; for sync-only tools it delegates to a thread executor
        # internally.
        raw_output = await runner.ainvoke(_tool_input(tool, state))
        return _merge(raw_output)

    return ActionSpec(
        name=tool.name,
        preconditions=declared_preconditions,
        effects=declared_effects,
        cost=cost,
        execute=_execute,
        aexecute=_aexecute,
        effect_validator=effect_validator,
        max_retries=max_retries,
        resources=resources,
        duration=duration,
        qos=qos,
        require_human_approval=require_human_approval,
        human_input_key=human_input_key,
        can_rerun=can_rerun,
        read_only=read_only,
        utility=utility,
        metadata=metadata,
    )


__all__ = ["goapify_tool", "EffectValidator"]
