"""Integration tests for form-binding HITL.

Mirrors Embabel's typed-form binding contract on
``research/repos/embabel-agent/embabel-agent-api/src/test/kotlin/
com/embabel/agent/core/hitl/FormBindingRequestTest.kt``: when an
action requires human input shaped by a typed schema, the executor
emits an interrupt carrying the schema; on resume with a matching
payload, the parsed model is merged into world state and execution
continues.

LangGOAP's form-binding extends ``ActionSpec.require_human_approval``
from ``bool`` to ``bool | type[BaseModel]``:

* ``False`` (default) — no HITL gate.
* ``True`` — boolean approve/deny gate (existing behaviour).
* ``BaseModel`` subclass — typed-form gate.  The interrupt payload
  carries the model's JSON schema; the resume payload is validated
  via ``Model.model_validate`` and merged into ``world_state`` under
  ``ActionSpec.human_input_key`` (or the action name if unset).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph


class DeliveryConfirmation(BaseModel):
    """Typed form a delivery operator must complete to confirm dispatch."""

    delivery_date: date
    notes: str
    signature_waiver: bool = False


def _confirm_delivery_actions() -> list[ActionSpec]:
    """Two-action plan: prepare → confirm_delivery (gated by typed form)."""

    def prepare(world_state: dict[str, Any]) -> dict[str, Any]:
        return {"prepared": True}

    def dispatch(world_state: dict[str, Any]) -> dict[str, Any]:
        return {"dispatched": True}

    return [
        ActionSpec(
            name="prepare",
            preconditions={},
            effects={"prepared": True},
            execute=prepare,
        ),
        ActionSpec(
            name="confirm_delivery",
            preconditions={"prepared": True},
            effects={"dispatched": True},
            execute=dispatch,
            require_human_approval=DeliveryConfirmation,
            human_input_key="confirmation",
        ),
    ]


def _interrupted_payload(result: Any) -> dict[str, Any]:
    """Extract the first interrupt payload from a graph invoke result."""
    interrupts = result.get("__interrupt__", [])
    assert interrupts, f"expected __interrupt__ in result, got {list(result.keys())!r}"
    interrupt_obj = interrupts[0]
    return interrupt_obj.value


# ---------------------------------------------------------------------------
# Sync — typed-form happy path
# ---------------------------------------------------------------------------


class TestFormBindingHitlSync:
    def test_pydantic_form_interrupt_carries_schema(self) -> None:
        """The interrupt payload must include the Pydantic JSON schema so
        downstream UI / clients can render the form."""
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-1"}}

        result = graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}),
                "world_state": {},
            },
            config=config,
        )

        payload = _interrupted_payload(result)
        assert payload["type"] == "goap_action_form"
        assert payload["action"] == "confirm_delivery"
        assert "form_schema" in payload
        # The Pydantic schema must include the field names.
        schema = payload["form_schema"]
        assert "delivery_date" in schema["properties"]
        assert "notes" in schema["properties"]
        assert "signature_waiver" in schema["properties"]

    def test_resume_with_valid_form_merges_parsed_model_into_state(self) -> None:
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-2"}}

        # First call: hits the form interrupt before confirm_delivery runs.
        graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}),
                "world_state": {},
            },
            config=config,
        )

        # Resume with a valid form payload.
        resume_payload = {
            "delivery_date": "2026-05-15",
            "notes": "Leave at side door",
            "signature_waiver": True,
        }
        result = graph.invoke(
            Command(resume=resume_payload),
            config=config,
        )

        assert result["status"] == "goal_achieved"
        # Parsed form merged into world_state under the configured key.
        confirmation = result["world_state"]["confirmation"]
        # Pydantic v2 round-trip: instance carrying the typed fields.
        assert isinstance(confirmation, DeliveryConfirmation) or isinstance(
            confirmation, dict
        )
        if isinstance(confirmation, DeliveryConfirmation):
            assert confirmation.notes == "Leave at side door"
            assert confirmation.signature_waiver is True
        else:
            # Some serializers round-trip through dict; check field presence.
            assert confirmation["notes"] == "Leave at side door"
            assert confirmation["signature_waiver"] is True
        assert result["world_state"]["dispatched"] is True

    def test_resume_with_invalid_form_treated_as_denial(self) -> None:
        """A resume payload that fails Pydantic validation must abort the
        action with a deny-style failure (matches Embabel's
        ``IllegalStateException`` on invalid submission)."""
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-3"}}

        graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}),
                "world_state": {},
            },
            config=config,
        )

        # Missing required ``notes`` and bad ``delivery_date``.
        bad_payload = {"delivery_date": "not-a-date"}
        result = graph.invoke(
            Command(resume=bad_payload),
            config=config,
        )

        # Confirm: action did not run, no goal achievement.
        assert "dispatched" not in result["world_state"]
        # Failure surfaces as a structured error in execution_history.
        history = result["execution_history"]
        confirm_history = [h for h in history if h.action_name == "confirm_delivery"]
        assert confirm_history, "confirm_delivery should appear in history"
        last = confirm_history[-1]
        assert last.success is False
        assert "form" in (last.error or "").lower()


# ---------------------------------------------------------------------------
# Backwards compatibility — bool gate still works
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_bool_approval_unchanged(self) -> None:
        """``require_human_approval=True`` keeps its existing
        approve/deny semantics — no Pydantic involvement."""

        def dispatch(world_state: dict[str, Any]) -> dict[str, Any]:
            return {"dispatched": True}

        actions = [
            ActionSpec(
                name="confirm",
                preconditions={},
                effects={"dispatched": True},
                execute=dispatch,
                require_human_approval=True,
            )
        ]
        graph = GoapGraph(actions).compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-bool"}}

        result = graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}),
                "world_state": {},
            },
            config=config,
        )

        payload = _interrupted_payload(result)
        assert payload["type"] == "goap_action_approval"

        result = graph.invoke(Command(resume=True), config=config)
        assert result["status"] == "goal_achieved"

    def test_human_input_key_defaults_to_action_name(self) -> None:
        graph = GoapGraph(
            [
                ActionSpec(
                    name="prepare", preconditions={}, effects={"prepared": True}
                ),
                ActionSpec(
                    name="confirm_delivery",
                    preconditions={"prepared": True},
                    effects={"dispatched": True},
                    execute=lambda ws: {"dispatched": True},
                    require_human_approval=DeliveryConfirmation,
                    # human_input_key omitted — defaults to action name
                ),
            ]
        ).compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-default-key"}}

        graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}),
                "world_state": {},
            },
            config=config,
        )
        result = graph.invoke(
            Command(
                resume={
                    "delivery_date": "2026-05-15",
                    "notes": "n",
                }
            ),
            config=config,
        )

        # Default key is the action name.
        assert "confirm_delivery" in result["world_state"]
