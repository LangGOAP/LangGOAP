"""Tests for the Reflexion self-critique tracer (langgoap.reflexion)."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.graph.state import ActionResult

# _parse_llm_response is private but tested directly here because it is a pure
# parsing function with no side effects.  Testing it directly is more explicit
# than routing through a mock LLM; this is an accepted pattern for pure helpers
# inside the same package.
from langgoap.reflexion import Reflection, ReflexionTracer, _parse_llm_response
from langgoap.tracing import MultiTracer, NullTracer

# ---------------------------------------------------------------------------
# Reflection dataclass
# ---------------------------------------------------------------------------


def test_reflection_creation() -> None:
    r = Reflection(
        action_name="act",
        error="boom",
        reflection="something went wrong",
        suggestion="try again",
        timestamp="2024-01-01T00:00:00+00:00",
    )
    assert r.action_name == "act"
    assert r.error == "boom"
    assert r.reflection == "something went wrong"
    assert r.suggestion == "try again"


# ---------------------------------------------------------------------------
# Template-based reflection (no LLM)
# ---------------------------------------------------------------------------


def test_template_reflection_on_failure() -> None:
    tracer = ReflexionTracer()
    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(
                action_name="flaky_action",
                success=False,
                error="connection timeout",
            )
        ],
    }
    tracer.on_action_complete(result)

    assert len(tracer.reflections) == 1
    r = tracer.reflections[0]
    assert r.action_name == "flaky_action"
    assert r.error == "connection timeout"
    assert "flaky_action" in r.reflection
    assert len(r.suggestion) > 0
    assert r.timestamp  # non-empty


def test_template_reflection_no_failure_no_reflection() -> None:
    tracer = ReflexionTracer()
    result: dict[str, Any] = {
        "world_state": {"a": True},
        "current_step": 1,
        "execution_history": [ActionResult(action_name="good_action", success=True)],
    }
    tracer.on_action_complete(result)
    assert len(tracer.reflections) == 0


# ---------------------------------------------------------------------------
# LLM-based reflection with FakeStructuredModel
# ---------------------------------------------------------------------------


def test_llm_based_reflection() -> None:
    """ReflexionTracer generates LLM-based reflections when llm is set."""
    from langchain_core.messages import AIMessage

    class FakeLLM:
        def invoke(self, messages: Any) -> AIMessage:
            return AIMessage(
                content="Reflection: The API endpoint was unreachable.\n"
                "Suggestion: Add a retry mechanism with backoff."
            )

    tracer = ReflexionTracer(llm=FakeLLM())
    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(
                action_name="call_api",
                success=False,
                error="HTTP 503",
            )
        ],
    }
    tracer.on_action_complete(result)

    assert len(tracer.reflections) == 1
    r = tracer.reflections[0]
    assert r.action_name == "call_api"
    assert "unreachable" in r.reflection
    assert "retry" in r.suggestion


def test_llm_failure_falls_back_to_template() -> None:
    class FailingLLM:
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("LLM unavailable")

    tracer = ReflexionTracer(llm=FailingLLM())
    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(action_name="act", success=False, error="error msg")
        ],
    }
    tracer.on_action_complete(result)

    # Should still produce a template-based reflection
    assert len(tracer.reflections) == 1
    assert "act" in tracer.reflections[0].reflection


# ---------------------------------------------------------------------------
# max_reflections limits stored history
# ---------------------------------------------------------------------------


def test_max_reflections_limit() -> None:
    tracer = ReflexionTracer(max_reflections=3)
    for i in range(5):
        result: dict[str, Any] = {
            "status": "action_failed",
            "execution_history": [
                ActionResult(
                    action_name=f"action_{i}",
                    success=False,
                    error=f"error_{i}",
                )
            ],
        }
        tracer.on_action_complete(result)

    assert len(tracer.reflections) == 3
    # Should keep the 3 most recent
    assert tracer.reflections[0].action_name == "action_2"
    assert tracer.reflections[2].action_name == "action_4"


# ---------------------------------------------------------------------------
# Multiple failures → multiple reflections in order
# ---------------------------------------------------------------------------


def test_multiple_failures_in_order() -> None:
    tracer = ReflexionTracer()
    for name in ["a", "b", "c"]:
        result: dict[str, Any] = {
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name=name, success=False, error=f"err_{name}")
            ],
        }
        tracer.on_action_complete(result)

    assert len(tracer.reflections) == 3
    assert [r.action_name for r in tracer.reflections] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Empty failure history → no reflections
# ---------------------------------------------------------------------------


def test_empty_failure_history() -> None:
    tracer = ReflexionTracer()
    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [],
    }
    tracer.on_action_complete(result)
    assert len(tracer.reflections) == 0


# ---------------------------------------------------------------------------
# on_plan_start logs reflections
# ---------------------------------------------------------------------------


def test_on_plan_start_with_existing_reflections(caplog: Any) -> None:
    import logging

    tracer = ReflexionTracer()
    # Add a reflection manually
    tracer._reflections.append(
        Reflection(
            action_name="prev_action",
            error="prev_error",
            reflection="analysis",
            suggestion="fix it",
            timestamp="2024-01-01T00:00:00+00:00",
        )
    )

    with caplog.at_level(logging.INFO, logger="langgoap.reflexion"):
        tracer.on_plan_start(None, {}, "AStar")

    assert "1 reflections available" in caplog.text


def test_on_plan_start_no_reflections_no_log(caplog: Any) -> None:
    import logging

    tracer = ReflexionTracer()
    with caplog.at_level(logging.INFO, logger="langgoap.reflexion"):
        tracer.on_plan_start(None, {}, "AStar")

    assert "reflections available" not in caplog.text


# ---------------------------------------------------------------------------
# Async hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_hooks_work() -> None:
    tracer = ReflexionTracer()
    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(action_name="async_act", success=False, error="async err")
        ],
    }
    await tracer.aon_action_complete(result)
    assert len(tracer.reflections) == 1
    assert tracer.reflections[0].action_name == "async_act"

    await tracer.aon_plan_start(None, {}, "AStar")  # no crash


# ---------------------------------------------------------------------------
# Composable via MultiTracer
# ---------------------------------------------------------------------------


def test_composable_via_multi_tracer() -> None:
    reflexion = ReflexionTracer()
    null = NullTracer()
    multi = MultiTracer([reflexion, null])

    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(action_name="multi_act", success=False, error="multi err")
        ],
    }
    multi.on_action_complete(result)

    assert len(reflexion.reflections) == 1
    assert reflexion.reflections[0].action_name == "multi_act"


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------


def test_parse_llm_response_standard_format() -> None:
    text = "Reflection: something broke\nSuggestion: fix it"
    r, s = _parse_llm_response(text)
    assert r == "something broke"
    assert s == "fix it"


def test_parse_llm_response_no_format() -> None:
    text = "The action failed because the API was down"
    r, s = _parse_llm_response(text)
    assert r == text
    assert s == "Consider alternative approaches."


# ---------------------------------------------------------------------------
# Tracer exception in reflection generation doesn't crash
# ---------------------------------------------------------------------------


def test_tracer_exception_in_generation_doesnt_crash() -> None:
    """Even if the LLM raises during invoke, the tracer still works."""

    class BrokenLLM:
        def invoke(self, messages: Any) -> Any:
            raise Exception("total catastrophe")

    tracer = ReflexionTracer(llm=BrokenLLM())
    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(action_name="broken", success=False, error="err")
        ],
    }
    # Should not raise
    tracer.on_action_complete(result)
    # Falls back to template
    assert len(tracer.reflections) == 1
    assert "broken" in tracer.reflections[0].reflection


# ---------------------------------------------------------------------------
# Store-backed persistence (F9 fix)
# ---------------------------------------------------------------------------


def test_reflections_persisted_to_store() -> None:
    """on_action_complete writes reflections into the backing store."""
    from langgraph.store.memory import InMemoryStore

    from langgoap.history import StoreExecutionHistory
    from langgoap.reflexion import _REFLECTIONS_KEY, _REFLECTIONS_NS

    store = InMemoryStore()
    history = StoreExecutionHistory(store)
    tracer = ReflexionTracer(history=history)

    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(action_name="persist_act", success=False, error="persist err")
        ],
    }
    tracer.on_action_complete(result)

    # The in-memory store should now hold the reflection.
    item = store.get(_REFLECTIONS_NS, _REFLECTIONS_KEY)
    value = getattr(item, "value", item)
    assert value is not None
    stored = value.get("reflections", [])
    assert len(stored) == 1
    assert stored[0]["action_name"] == "persist_act"
    assert stored[0]["error"] == "persist err"


def test_reflections_loaded_from_store_on_init() -> None:
    """A new ReflexionTracer with the same store picks up prior reflections."""
    from langgraph.store.memory import InMemoryStore

    from langgoap.history import StoreExecutionHistory

    store = InMemoryStore()
    history = StoreExecutionHistory(store)

    # First tracer captures a failure and persists it.
    tracer1 = ReflexionTracer(history=history)
    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(action_name="old_action", success=False, error="old error")
        ],
    }
    tracer1.on_action_complete(result)
    assert len(tracer1.reflections) == 1

    # Second tracer shares the same store — must load tracer1's reflection.
    tracer2 = ReflexionTracer(history=history)
    assert len(tracer2.reflections) == 1
    assert tracer2.reflections[0].action_name == "old_action"
    assert tracer2.reflections[0].error == "old error"


@pytest.mark.asyncio
async def test_async_reflections_persisted_to_store() -> None:
    """aon_action_complete persists reflections via async store.aput."""
    from langgraph.store.memory import InMemoryStore

    from langgoap.history import StoreExecutionHistory
    from langgoap.reflexion import _REFLECTIONS_KEY, _REFLECTIONS_NS

    store = InMemoryStore()
    history = StoreExecutionHistory(store)
    tracer = ReflexionTracer(history=history)

    result: dict[str, Any] = {
        "status": "action_failed",
        "execution_history": [
            ActionResult(action_name="async_act", success=False, error="async err")
        ],
    }
    await tracer.aon_action_complete(result)

    item = store.get(_REFLECTIONS_NS, _REFLECTIONS_KEY)
    value = getattr(item, "value", item)
    assert value is not None
    stored = value.get("reflections", [])
    assert len(stored) == 1
    assert stored[0]["action_name"] == "async_act"


# ---------------------------------------------------------------------------
# F10 — reflection_context surfaced into GoapState by GoapPlanner
# ---------------------------------------------------------------------------


def test_reflection_context_injected_into_goapstate_after_failure() -> None:
    """After a failure, GoapPlanner injects reflection_context from the tracer into state.

    The planner reads the `.reflections` property of the tracer and serialises
    each reflection as a human-readable string into ``GoapState["reflection_context"]``.
    Downstream components (e.g. PromptConditions) can read this field without
    coupling to the tracer directly.
    """
    from langgoap.actions import ActionSpec
    from langgoap.goals import GoalSpec
    from langgoap.graph.nodes import GoapPlanner

    tracer = ReflexionTracer()

    # Manually inject a reflection (simulates a prior failure that triggered
    # the reflexion hook) so the planner has something to surface.
    tracer._add_reflection(
        Reflection(
            action_name="broken_action",
            error="connection refused",
            reflection="The API was down during the last attempt.",
            suggestion="Wait and retry with exponential back-off.",
            timestamp=0.0,
        )
    )

    action = ActionSpec(name="a", effects={"done": True})
    goal = GoalSpec(conditions={"done": True})
    planner = GoapPlanner([action], tracer=tracer)  # type: ignore[arg-type]

    result = planner({"goal": goal, "world_state": {}})

    assert result.get("status") == "executing"
    ctx = result.get("reflection_context")
    assert ctx is not None and len(ctx) == 1
    # Each entry is "[action_name] reflection → suggestion"
    assert "[broken_action]" in ctx[0]
    assert "connection refused" not in ctx[0]  # error not repeated; reflection text is
    assert "API was down" in ctx[0]
    assert "exponential back-off" in ctx[0]


def test_reflection_context_surfaced_through_multitracer() -> None:
    """MultiTracer.reflections aggregates from inner ReflexionTracers."""
    inner = ReflexionTracer()
    inner._add_reflection(
        Reflection(
            action_name="act",
            error="err",
            reflection="Something failed.",
            suggestion="Try again.",
            timestamp=0.0,
        )
    )
    multi = MultiTracer([NullTracer(), inner])  # type: ignore[list-item]

    # MultiTracer.reflections delegates to inner tracers' .reflections properties.
    assert len(multi.reflections) == 1
    assert multi.reflections[0].action_name == "act"
