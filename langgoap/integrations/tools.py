"""Adapter from LangChain ``BaseTool`` to LangGoap ``ActionSpec``.

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

EffectValidator = Callable[[dict[str, Any], dict[str, Any]], bool]


def _tool_input(tool: BaseTool, state: dict[str, Any]) -> dict[str, Any]:
    """Filter *state* to the keys declared in *tool*'s args schema.

    Tools with no ``args_schema`` (zero-arg tools) receive an empty dict,
    which ``StructuredTool.invoke`` / ``StructuredTool.ainvoke`` accept.
    Supports both Pydantic v2 (``.model_fields``) and v1 (``.__fields__``).
    """
    schema = tool.args_schema
    if schema is None:
        return {}
    fields = getattr(schema, "model_fields", None)
    if fields is None:
        fields = getattr(schema, "__fields__", {})
    return {k: state[k] for k in fields if k in state}


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
) -> ActionSpec:
    """Wrap a LangChain :class:`BaseTool` into a LangGoap :class:`ActionSpec`.

    The resulting action's ``execute`` callable invokes ``tool`` with the
    state dict filtered to match the tool's declared input schema (if any)
    and returns the declared ``effects`` dict.  Passing no
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

    def _execute(state: dict[str, Any]) -> dict[str, Any]:
        raw_output = tool.invoke(_tool_input(tool, state))
        return _merge(raw_output)

    async def _aexecute(state: dict[str, Any]) -> dict[str, Any]:
        # BaseTool.ainvoke is part of the Runnable contract and is always
        # available; for sync-only tools it delegates to a thread executor
        # internally.
        raw_output = await tool.ainvoke(_tool_input(tool, state))
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
    )


__all__ = ["goapify_tool", "EffectValidator"]
