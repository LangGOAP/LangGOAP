"""Adapter from LangChain ``BaseTool`` to LangGoap ``ActionSpec``.

Layer B of the three-layer low-code on-ramp (AD-2).  Fully
deterministic — preconditions, effects, cost, resources, duration,
and retry policy all come from the caller, never from the LLM.

The produced :class:`~langgoap.actions.ActionSpec` carries an
``execute`` wrapper that calls the underlying tool and returns the
declared ``effects`` dict so the GOAP executor can apply them to the
world state.  The tool's own return value is **not** used to drive
state transitions; actions produce side effects that are declared up
front via ``effects``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable

from langchain_core.tools import BaseTool

from langgoap.actions import ActionSpec

EffectValidator = Callable[[dict[str, Any], dict[str, Any]], bool]


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
) -> ActionSpec:
    """Wrap a LangChain :class:`BaseTool` into a LangGoap :class:`ActionSpec`.

    The resulting action's ``execute`` callable invokes ``tool`` with the
    state dict filtered to match the tool's declared input schema (if
    any) and returns the declared ``effects`` dict.  Passing no
    ``preconditions``/``effects`` yields an action with empty
    pre/eff — legal but rarely useful, which is why
    :func:`create_goap_agent` emits a warning in that case.

    Args:
        tool: Any LangChain ``BaseTool`` instance (e.g. produced by
            the ``@tool`` decorator).
        preconditions: World-state conditions required before the tool
            runs.  Keys must be hashable scalars.
        effects: World-state changes produced by executing the tool.
        cost: Static action cost for A* search.  Defaults to 1.0.
        resources: Per-run resource usage for CSP optimization
            (e.g. ``{"cost_usd": 0.02, "tokens": 500}``).
        duration: Estimated wall-clock duration for temporal scheduling.
        max_retries: Number of retries before the executor blacklists
            the action.
        effect_validator: Optional postcondition checker.

    Returns:
        A frozen :class:`ActionSpec` ready to feed into
        :class:`~langgoap.graph.builder.GoapGraph`.

    Raises:
        TypeError: If ``tool`` is not a :class:`BaseTool` instance.
    """
    if not isinstance(tool, BaseTool):
        raise TypeError(
            f"goapify_tool expected a LangChain BaseTool, got "
            f"{type(tool).__name__}. Wrap plain functions with "
            f"@langchain_core.tools.tool first."
        )

    declared_effects = dict(effects or {})
    declared_preconditions = dict(preconditions or {})

    def _execute(state: dict[str, Any]) -> dict[str, Any]:
        # Filter the state dict down to the keys the tool declares in
        # its args_schema.  Tools with no args_schema (zero-arg tools)
        # receive an empty dict, which StructuredTool accepts.
        tool_input: dict[str, Any] = {}
        schema = tool.args_schema
        if schema is not None:
            # Pydantic v2: .model_fields; v1: .__fields__ fallback.
            fields = getattr(schema, "model_fields", None)
            if fields is None:
                fields = getattr(schema, "__fields__", {})
            tool_input = {k: state[k] for k in fields if k in state}
        # Invoke the tool — its return value is discarded (effects
        # drive state transitions, not return values).
        tool.invoke(tool_input)
        return dict(declared_effects)

    return ActionSpec(
        name=tool.name,
        preconditions=declared_preconditions,
        effects=declared_effects,
        cost=cost,
        execute=_execute,
        effect_validator=effect_validator,
        max_retries=max_retries,
        resources=resources,
        duration=duration,
    )


__all__ = ["goapify_tool", "EffectValidator"]
