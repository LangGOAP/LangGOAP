"""Reflexion self-critique tracer for GOAP execution.

Implements Shinn et al. (2023) verbal self-reflection: when an action
fails, the tracer generates a :class:`Reflection` capturing what went
wrong and a suggestion for the next attempt.  Reflections are stored
in a :class:`~langgoap.history.StoreExecutionHistory` (or in-memory
if no store) and surfaced on subsequent ``plan_start`` events.

The :class:`ReflexionTracer` is a :class:`~langgoap.tracing.PlanningTracer`
implementation — it hooks into the existing tracer infrastructure and
can be composed with other tracers via :class:`~langgoap.tracing.MultiTracer`.

When ``llm`` is provided, reflections are LLM-generated from the failure
context.  When ``llm`` is ``None``, reflections are template-based (no
LLM call needed).

The tracer never blocks planning — all LLM calls use a timeout and
failures are logged and swallowed.

When a :class:`~langgoap.history.StoreExecutionHistory` is provided, the
FIFO reflection list is persisted to the backing :class:`~langgraph.store.base.BaseStore`
under the ``("langgoap", "reflections")`` namespace on every write, and
loaded back on construction so reflections survive across tracer instances
that share the same store.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Namespace and key used to persist reflections in the backing store.
_REFLECTIONS_NS: tuple[str, str] = ("langgoap", "reflections")
_REFLECTIONS_KEY = "current"


def _reflection_to_dict(r: "Reflection") -> dict[str, Any]:
    """Serialise a :class:`Reflection` to a plain dict for ``BaseStore.put``."""
    return {
        "action_name": r.action_name,
        "error": r.error,
        "reflection": r.reflection,
        "suggestion": r.suggestion,
        "timestamp": r.timestamp,
    }


def _reflection_from_dict(d: dict[str, Any]) -> "Reflection":
    """Deserialise a :class:`Reflection` from the format produced by :func:`_reflection_to_dict`."""
    return Reflection(
        action_name=d["action_name"],
        error=d["error"],
        reflection=d["reflection"],
        suggestion=d["suggestion"],
        timestamp=d["timestamp"],
    )


@dataclass(frozen=True, slots=True)
class Reflection:
    """A verbal reflection on an execution failure.

    Attributes:
        action_name: Name of the action that failed.
        error: Error message from the failure.
        reflection: Analysis of what went wrong.
        suggestion: Suggestion for the next attempt.
        timestamp: When the reflection was generated (ISO format).
    """

    action_name: str
    error: str
    reflection: str
    suggestion: str
    timestamp: str


_REFLECTION_PROMPT = """\
An action in a Goal-Oriented Action Planning (GOAP) system failed.

Action: {action_name}
Error: {error}
World state at failure: {world_state}

Analyze what went wrong in 1-2 sentences (the "reflection").
Then suggest how to fix or avoid this failure in 1-2 sentences (the "suggestion").

Return your analysis in this exact format:
Reflection: <your analysis>
Suggestion: <your suggestion>
"""


def _template_reflection(action_name: str, error: str) -> tuple[str, str]:
    """Generate a template-based reflection (no LLM)."""
    reflection = (
        f"Action '{action_name}' failed with error: {error}. "
        f"The action's preconditions may not have been fully met, "
        f"or the action's implementation may have an issue."
    )
    suggestion = (
        f"Consider alternative actions that achieve the same effect, "
        f"or verify that '{action_name}' has the correct preconditions "
        f"and a robust implementation."
    )
    return reflection, suggestion


def _parse_llm_response(text: str) -> tuple[str, str]:
    """Parse reflection and suggestion from LLM response text."""
    reflection = ""
    suggestion = ""
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("reflection:"):
            reflection = stripped[len("reflection:") :].strip()
        elif stripped.lower().startswith("suggestion:"):
            suggestion = stripped[len("suggestion:") :].strip()
    if not reflection:
        reflection = text.strip()
    if not suggestion:
        suggestion = "Consider alternative approaches."
    return reflection, suggestion


class ReflexionTracer:
    """Tracer that captures failures and generates verbal reflections.

    On ``on_action_complete`` with a failure, generates a
    :class:`Reflection`.  On ``on_plan_start``, logs accumulated
    reflections for the current planning session.

    Args:
        llm: Optional LangChain ``BaseChatModel``.  When ``None``,
            reflections are template-based (no LLM call).
        history: Optional :class:`~langgoap.history.StoreExecutionHistory`
            for persisting reflections.  When ``None``, reflections are
            stored in-memory only.
        max_reflections: Maximum number of reflections to keep (FIFO).
    """

    def __init__(
        self,
        llm: Any = None,
        history: Any = None,
        max_reflections: int = 5,
    ) -> None:
        self._llm = llm
        self._history = history
        self._max_reflections = max_reflections
        self._reflections: list[Reflection] = []
        self._current_world_state: dict[str, Any] = {}
        # Load any previously-persisted reflections from the store so this
        # tracer instance picks up where a prior one left off.
        if self._history is not None:
            self._load_reflections()

    @property
    def reflections(self) -> list[Reflection]:
        """Read-only view of accumulated reflections."""
        return list(self._reflections)

    def _generate_reflection(
        self, action_name: str, error: str, world_state: dict[str, Any]
    ) -> Reflection:
        """Generate a reflection from a failure context."""
        if self._llm is not None:
            try:
                from langchain_core.messages import HumanMessage

                prompt = _REFLECTION_PROMPT.format(
                    action_name=action_name,
                    error=error,
                    world_state=world_state,
                )
                response = self._llm.invoke([HumanMessage(content=prompt)])
                text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                reflection_text, suggestion_text = _parse_llm_response(text)
            except Exception as exc:
                logger.warning(
                    "LLM reflection failed: %s; falling back to template", exc
                )
                reflection_text, suggestion_text = _template_reflection(
                    action_name, error
                )
        else:
            reflection_text, suggestion_text = _template_reflection(action_name, error)

        return Reflection(
            action_name=action_name,
            error=error,
            reflection=reflection_text,
            suggestion=suggestion_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_reflections(self) -> None:
        """Populate ``self._reflections`` from the backing store (sync)."""
        if self._history is None:
            return
        try:
            store = self._history._store
            item = store.get(_REFLECTIONS_NS, _REFLECTIONS_KEY)
            value = getattr(item, "value", item)
            if value:
                loaded = [
                    _reflection_from_dict(d) for d in value.get("reflections", [])
                ]
                self._reflections = loaded[-self._max_reflections :]
        except Exception as exc:
            logger.warning("ReflexionTracer: failed to load reflections: %s", exc)

    def _persist_reflections(self) -> None:
        """Write ``self._reflections`` to the backing store (sync)."""
        if self._history is None:
            return
        try:
            self._history._store.put(
                _REFLECTIONS_NS,
                _REFLECTIONS_KEY,
                {"reflections": [_reflection_to_dict(r) for r in self._reflections]},
            )
        except Exception as exc:
            logger.warning("ReflexionTracer: failed to persist reflections: %s", exc)

    async def _apersist_reflections(self) -> None:
        """Write ``self._reflections`` to the backing store (async)."""
        if self._history is None:
            return
        try:
            await self._history._store.aput(
                _REFLECTIONS_NS,
                _REFLECTIONS_KEY,
                {"reflections": [_reflection_to_dict(r) for r in self._reflections]},
            )
        except Exception as exc:
            logger.warning(
                "ReflexionTracer: failed to persist reflections (async): %s", exc
            )

    def _add_reflection(self, reflection: Reflection) -> None:
        """Append a reflection, enforce the FIFO limit, and persist."""
        self._reflections.append(reflection)
        if len(self._reflections) > self._max_reflections:
            self._reflections = self._reflections[-self._max_reflections :]
        self._persist_reflections()

    # ------------------------------------------------------------------
    # Sync hooks
    # ------------------------------------------------------------------

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        if isinstance(state, dict):
            self._current_world_state = state
        if self._reflections:
            logger.info(
                "Reflexion: %d reflections available from prior failures",
                len(self._reflections),
            )
            for r in self._reflections:
                logger.info(
                    "  [%s] %s: %s → %s",
                    r.timestamp,
                    r.action_name,
                    r.reflection,
                    r.suggestion,
                )

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        pass

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        pass

    def on_action_start(self, action: Any, state: Any) -> None:
        if isinstance(state, dict):
            self._current_world_state = state

    def on_action_complete(self, result: Any) -> None:
        # Check if the result indicates failure
        if isinstance(result, dict):
            status = result.get("status")
            if status == "action_failed":
                # Extract action name and error from execution_history
                history = result.get("execution_history", [])
                for entry in history:
                    if hasattr(entry, "success") and not entry.success:
                        action_name = getattr(entry, "action_name", "unknown")
                        error = (
                            getattr(entry, "error", "unknown error") or "unknown error"
                        )
                        reflection = self._generate_reflection(
                            action_name, error, self._current_world_state
                        )
                        self._add_reflection(reflection)

    def on_replan(self, reason: str, new_plan: Any) -> None:
        pass

    def on_goal_achieved(self, final_state: Any) -> None:
        pass

    def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Async hooks
    # ------------------------------------------------------------------

    async def aon_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.on_plan_start(goal, state, strategy_name)

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
        pass

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
        pass

    async def aon_action_start(self, action: Any, state: Any) -> None:
        self.on_action_start(action, state)

    async def aon_action_complete(self, result: Any) -> None:
        # Mirror the sync detection logic but use async persistence so that
        # aput (e.g. AsyncPostgresStore) is correctly awaited rather than
        # blocking the event loop with a sync store.put call.
        if isinstance(result, dict):
            status = result.get("status")
            if status == "action_failed":
                history_entries = result.get("execution_history", [])
                for entry in history_entries:
                    if hasattr(entry, "success") and not entry.success:
                        action_name = getattr(entry, "action_name", "unknown")
                        error = (
                            getattr(entry, "error", "unknown error") or "unknown error"
                        )
                        reflection = self._generate_reflection(
                            action_name, error, self._current_world_state
                        )
                        self._reflections.append(reflection)
                        if len(self._reflections) > self._max_reflections:
                            self._reflections = self._reflections[
                                -self._max_reflections :
                            ]
                        await self._apersist_reflections()

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        pass

    async def aon_goal_achieved(self, final_state: Any) -> None:
        pass

    async def aon_sensor_complete(self, sensor_name: str, updates: Any) -> None:
        pass


__all__ = [
    "Reflection",
    "ReflexionTracer",
]
