"""Natural language to GoalSpec interpreter using LLM structured output.

Converts free-text requests like "Generate a report under $5" into formal
GoalSpec objects via LangChain's ``with_structured_output()`` mechanism.
Provider-agnostic: works with any ``BaseChatModel`` that supports structured
output (OpenAI, Anthropic, Google, local models, etc.).

Architecture::

    "Generate a report under $5"
            ↓
    [GoalInterpreter.interpret()]
      ├─ Build prompt (system message with action catalog + user request)
      ├─ LLM call via with_structured_output(InterpretedGoal)
      └─ Convert InterpretedGoal → GoalSpec
            ↓
    GoalSpec(conditions={"report_complete": True}, ...)
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.types import ObjectiveDirection

# ---------------------------------------------------------------------------
# Pydantic schemas (LLM-facing)
# ---------------------------------------------------------------------------


class InterpretedConstraint(BaseModel):
    """A resource constraint extracted from natural language."""

    key: str = Field(description="Resource key (e.g. 'cost_usd', 'total_tokens')")
    max: float | None = Field(default=None, description="Upper bound, or null")
    min: float | None = Field(default=None, description="Lower bound, or null")
    weight: float = Field(default=1.0, description="Relative importance (default 1.0)")


class InterpretedObjective(BaseModel):
    """An optimization objective extracted from natural language."""

    metric: str = Field(description="Metric name to optimize")
    direction: Literal["minimize", "maximize"] = Field(
        description="Optimization direction — must be 'minimize' or 'maximize'"
    )


class InterpretedGoal(BaseModel):
    """Structured goal extracted from a natural language request.

    This is a transient LLM output — deliberately a mutable Pydantic model,
    not a frozen dataclass.  Converted to ``GoalSpec`` before use in planning.

    Note:
        ``conditions`` defaults to an empty dict so that LLM structured output
        always parses successfully even when a model omits the field (common
        with smaller models in function-calling mode when constraints dominate
        the request).  Emptiness is validated by :func:`to_goal_spec`, which
        raises ``ValueError`` if no conditions were extracted.
    """

    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Required world-state conditions (key → hashable scalar value)",
    )
    constraints: list[InterpretedConstraint] = Field(
        default_factory=list,
        description="Hard resource/budget constraints",
    )
    objectives: list[InterpretedObjective] = Field(
        default_factory=list,
        description="Optimization targets (minimize/maximize)",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of how the goal was derived",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_action_catalog(actions: list[ActionSpec]) -> str:
    """Format a list of actions into a human-readable catalog for the prompt.

    Includes name, preconditions, effects, description (from metadata),
    and resources for each action.
    """
    if not actions:
        return "(no actions available)"

    lines: list[str] = []
    for action in actions:
        parts = [f"- **{action.name}**"]
        if action.preconditions:
            parts.append(f"  Preconditions: {dict(action.preconditions)}")
        if action.effects:
            parts.append(f"  Effects: {dict(action.effects)}")
        if action.metadata and action.metadata.get("description"):
            parts.append(f"  Description: {action.metadata['description']}")
        if action.resources:
            parts.append(f"  Resources: {dict(action.resources)}")
        lines.append("\n".join(parts))

    return "\n".join(lines)


def to_goal_spec(interpreted: InterpretedGoal) -> GoalSpec:
    """Convert an InterpretedGoal (LLM output) to a GoalSpec (planning type).

    Maps InterpretedConstraint → ConstraintSpec,
    InterpretedObjective → ObjectiveDirection enum.

    Raises:
        ValueError: If ``conditions`` is empty; if a condition value is not a
            hashable scalar (``bool``, ``str``, ``int``, or ``float``); if a
            constraint has ``min > max`` (validated by
            :class:`~langgoap.goals.ConstraintSpec`).  All three errors can
            occur when an LLM hallucinates invalid values — callers should
            catch ``ValueError`` and prompt the user to rephrase.

    Note:
        Objective ``direction`` is constrained to ``Literal["minimize",
        "maximize"]`` on :class:`InterpretedObjective` itself — Pydantic
        rejects any other value before this function is called.
    """
    if not interpreted.conditions:
        raise ValueError(
            "LLM returned empty conditions — cannot create a meaningful GoalSpec. "
            "Ask the user to rephrase the request with a concrete end-state."
        )

    _SCALAR_TYPES = (bool, int, float, str)
    bad = {
        k: type(v).__name__
        for k, v in interpreted.conditions.items()
        if not isinstance(v, _SCALAR_TYPES)
    }
    if bad:
        raise ValueError(
            f"Condition values must be bool, int, float, or str. "
            f"Non-scalar keys returned by LLM: {bad}. "
            f"Ask the user to rephrase so the goal uses only scalar values."
        )

    constraints = tuple(
        ConstraintSpec(
            key=c.key,
            max=c.max,
            min=c.min,
            weight=c.weight,
        )
        for c in interpreted.constraints
    )

    # InterpretedObjective.direction is Literal["minimize", "maximize"]; Pydantic
    # guarantees no other value reaches here.
    _direction_map: dict[str, ObjectiveDirection] = {
        "minimize": ObjectiveDirection.MINIMIZE,
        "maximize": ObjectiveDirection.MAXIMIZE,
    }
    objectives: MappingProxyType[str, ObjectiveDirection] | None = None
    if interpreted.objectives:
        objectives = MappingProxyType(
            {obj.metric: _direction_map[obj.direction] for obj in interpreted.objectives}
        )

    # MappingProxyType wrapping satisfies GoalSpec's field type annotation.
    # GoalSpec.__post_init__ detects it is already a MappingProxyType and skips re-wrapping.
    return GoalSpec(
        conditions=MappingProxyType(interpreted.conditions),
        constraints=constraints,
        objectives=objectives,
    )


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a goal interpreter for a Goal-Oriented Action Planning (GOAP) system.
Given a natural language request, extract a structured goal specification.

## Available Actions

{action_catalog}

## Current World State

{world_state}

## Instructions

1. **conditions**: Identify the desired end-state as key-value pairs. Use \
condition keys that appear in the action effects above. Values must be \
hashable scalars: bool, str, int, or float.

2. **constraints**: Extract resource constraints from budget/limit language \
(e.g. "under $5" → key="cost_usd", max=5.0). Only include constraints \
explicitly stated in the request.

3. **objectives**: Extract optimization targets from language like \
"minimize", "cheapest", "fastest", "maximize quality". Map to the \
appropriate metric and direction ("minimize" or "maximize").

4. **reasoning**: Briefly explain how you derived the goal from the request.

Return ONLY the structured goal. Do not include actions or plans.\
"""


# ---------------------------------------------------------------------------
# GoalInterpreter
# ---------------------------------------------------------------------------


class GoalInterpreter:
    """Converts natural language requests into GoalSpec objects via LLM.

    Uses ``BaseChatModel.with_structured_output()`` to extract structured
    goals from free-text requests.  Provider-agnostic: any chat model
    that supports structured output works.

    Args:
        llm: A LangChain chat model (e.g. ``ChatOpenAI``, ``ChatAnthropic``).
        actions: Available GOAP actions (used to build the prompt catalog).
        system_prompt: Custom system prompt template.  Must contain
            ``{action_catalog}`` and ``{world_state}`` placeholders.
            Defaults to :data:`DEFAULT_SYSTEM_PROMPT`.
        structured_output_kwargs: Optional keyword arguments forwarded to
            ``llm.with_structured_output()``.  Use this for provider-specific
            options, e.g. ``{"method": "function_calling"}`` for OpenAI when
            ``conditions`` must remain an open dict (OpenAI strict JSON-schema
            mode requires ``additionalProperties: false`` on all objects, which
            is incompatible with a free-form conditions mapping).
    """

    def __init__(
        self,
        llm: BaseChatModel,
        actions: list[ActionSpec],
        system_prompt: str | None = None,
        structured_output_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._llm = llm
        self._actions = actions
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._structured_llm = llm.with_structured_output(
            InterpretedGoal, **(structured_output_kwargs or {})
        )

    def build_messages(
        self, request: str, world_state: dict[str, Any] | None
    ) -> list[SystemMessage | HumanMessage]:
        catalog = build_action_catalog(self._actions)
        ws_str = str(world_state) if world_state else "{}"
        system_text = self._system_prompt.format(
            action_catalog=catalog,
            world_state=ws_str,
        )
        return [SystemMessage(content=system_text), HumanMessage(content=request)]

    def interpret(
        self, request: str, world_state: dict[str, Any] | None = None
    ) -> GoalSpec:
        """Interpret a natural language request and return a GoalSpec."""
        raw = self.interpret_raw(request, world_state=world_state)
        return to_goal_spec(raw)

    async def ainterpret(
        self, request: str, world_state: dict[str, Any] | None = None
    ) -> GoalSpec:
        """Async variant of :meth:`interpret`."""
        raw = await self.ainterpret_raw(request, world_state=world_state)
        return to_goal_spec(raw)

    def interpret_raw(
        self, request: str, world_state: dict[str, Any] | None = None
    ) -> InterpretedGoal:
        """Interpret a request and return the raw InterpretedGoal.

        Preserves the ``reasoning`` field for debugging/transparency.
        """
        messages = self.build_messages(request, world_state)
        return cast(InterpretedGoal, self._structured_llm.invoke(messages))

    async def ainterpret_raw(
        self, request: str, world_state: dict[str, Any] | None = None
    ) -> InterpretedGoal:
        """Async variant of :meth:`interpret_raw`."""
        messages = self.build_messages(request, world_state)
        return cast(InterpretedGoal, await self._structured_llm.ainvoke(messages))
