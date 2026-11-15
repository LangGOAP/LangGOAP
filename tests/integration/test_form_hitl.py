"""Integration tests for form-binding HITL.

Mirrors Embabel's typed-form binding contract on
``https://github.com/embabel/embabel-agent/blob/HEAD/embabel-agent-api/src/test/kotlin/
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
  ``ActionSpec.human_input_key`` (or the action name if unset).  A
  payload that fails validation is asked again with the errors in the
  next interrupt; ``False`` / ``{"approved": False}`` declines the form
  and is recorded like a denied boolean gate.
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
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
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
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
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

    def test_invalid_payload_is_asked_again_with_the_errors(self) -> None:
        """A resume payload that fails Pydantic validation is an input error,
        not a decision: the operator is asked again and the new interrupt
        carries the validation errors.  Nothing is recorded as a failure and
        the action is not blacklisted."""
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-3"}}

        first = graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
                "world_state": {},
            },
            config=config,
        )
        assert _interrupted_payload(first)["attempt"] == 1
        assert "validation_errors" not in _interrupted_payload(first)

        # Missing required ``notes`` and bad ``delivery_date``.
        bad_payload = {"delivery_date": "not-a-date"}
        again = graph.invoke(Command(resume=bad_payload), config=config)

        payload = _interrupted_payload(again)
        assert payload["type"] == "goap_action_form"
        assert payload["action"] == "confirm_delivery"
        assert payload["attempt"] == 2
        assert payload["rejected_payload"] == bad_payload
        locations = {tuple(e["loc"]) for e in payload["validation_errors"]}
        assert locations == {("delivery_date",), ("notes",)}
        assert all(
            {"loc", "msg", "type"} <= set(e) for e in payload["validation_errors"]
        )
        assert "dispatched" not in again["world_state"]
        assert all(h["success"] for h in again["execution_history"])
        assert again.get("blacklisted_actions", []) == []

        # A corrected submission goes through.
        result = graph.invoke(Command(resume=_VALID_CONFIRMATION), config=config)
        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["confirmation"]["notes"] == "Leave at side door"
        assert [h["action_name"] for h in result["execution_history"]] == [
            "prepare",
            "confirm_delivery",
        ]

    def test_wrong_shape_payload_is_asked_again(self) -> None:
        """``True`` or ``None`` is not a form submission; ask again and say why."""
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-3b"}}
        graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
                "world_state": {},
            },
            config=config,
        )
        again = graph.invoke(Command(resume=True), config=config)
        payload = _interrupted_payload(again)
        assert payload["attempt"] == 2
        assert payload["validation_errors"] == [
            {
                "loc": [],
                "msg": "form payload must be an object matching the schema",
                "type": "wrong_shape",
            }
        ]
        result = graph.invoke(Command(resume=_VALID_CONFIRMATION), config=config)
        assert result["status"] == "goal_achieved"

    def test_form_can_be_denied_like_a_boolean_gate(self) -> None:
        """``False`` (or ``{"approved": False, "reason": ...}``) declines the
        form.  That is a decision, so it is recorded and blacklisted exactly
        like a denied boolean gate."""
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-3c"}}
        graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
                "world_state": {},
            },
            config=config,
        )
        result = graph.invoke(
            Command(resume={"approved": False, "reason": "wrong address"}),
            config=config,
        )
        assert "__interrupt__" not in result
        assert result["status"] == "no_plan"
        assert result["blacklisted_actions"] == ["confirm_delivery"]
        last = result["execution_history"][-1]
        assert last["action_name"] == "confirm_delivery"
        assert last["success"] is False
        assert "human_approval_denied: wrong address" in last["error"]
        assert "dispatched" not in result["world_state"]

    def test_false_denies_the_form(self) -> None:
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-3d"}}
        graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
                "world_state": {},
            },
            config=config,
        )
        result = graph.invoke(Command(resume=False), config=config)
        assert result["blacklisted_actions"] == ["confirm_delivery"]
        assert "human_approval_denied" in result["execution_history"][-1]["error"]

    @pytest.mark.asyncio
    async def test_async_invalid_payload_is_asked_again(self) -> None:
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "test-form-3e"}}
        await graph.ainvoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
                "world_state": {},
            },
            config=config,
        )
        again = await graph.ainvoke(Command(resume={"notes": "no date"}), config=config)
        payload = _interrupted_payload(again)
        assert payload["attempt"] == 2
        assert [tuple(e["loc"]) for e in payload["validation_errors"]] == [
            ("delivery_date",)
        ]
        result = await graph.ainvoke(Command(resume=_VALID_CONFIRMATION), config=config)
        assert result["status"] == "goal_achieved"


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
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
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
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
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


# ---------------------------------------------------------------------------
# #75 — the operator's answer survives an action failure and is not asked twice
# ---------------------------------------------------------------------------

_VALID_CONFIRMATION = {
    "delivery_date": "2026-05-15",
    "notes": "Leave at side door",
    "signature_waiver": True,
}


def _flaky_confirm_actions(fail_times: int = 1) -> list[ActionSpec]:
    """``confirm_delivery`` raises ``fail_times`` times, then succeeds.

    ``max_retries`` allows one re-plan, so the second attempt runs the same
    action again; it must reuse the operator's stored answer instead of
    interrupting a second time.
    """
    calls = {"n": 0}

    def dispatch(world_state: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError("carrier API timeout")
        return {"dispatched": True}

    def prepare(world_state: dict[str, Any]) -> dict[str, Any]:
        return {"prepared": True}

    return [
        ActionSpec(name="prepare", effects={"prepared": True}, execute=prepare),
        ActionSpec(
            name="confirm_delivery",
            preconditions={"prepared": True},
            effects={"dispatched": True},
            execute=dispatch,
            require_human_approval=DeliveryConfirmation,
            human_input_key="confirmation",
            max_retries=1,
        ),
    ]


def _goal_input() -> dict[str, Any]:
    return {
        "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
        "world_state": {},
    }


class TestFormAnswerSurvivesFailure:
    @pytest.mark.parametrize("executor", ["sequential", "parallel"])
    def test_retry_reuses_the_answer_without_a_second_prompt(
        self, executor: str
    ) -> None:
        graph = GoapGraph(_flaky_confirm_actions(), executor=executor).compile(  # type: ignore[arg-type]
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": f"form-survives-{executor}"}}
        graph.invoke(_goal_input(), config=config)

        result = graph.invoke(Command(resume=_VALID_CONFIRMATION), config=config)

        assert "__interrupt__" not in result, "the operator was asked a second time"
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["confirmation"]["notes"] == "Leave at side door"
        attempts = [
            h
            for h in result["execution_history"]
            if h["action_name"] == "confirm_delivery"
        ]
        assert [h["success"] for h in attempts] == [False, True]

    def test_answer_is_checkpointed_after_the_failure(self) -> None:
        """Even before the retry runs, the failed step's checkpoint holds the answer."""
        graph = GoapGraph(_flaky_confirm_actions(fail_times=5)).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "form-checkpointed"}}
        graph.invoke(_goal_input(), config=config)

        result = graph.invoke(Command(resume=_VALID_CONFIRMATION), config=config)

        # max_retries=1 is exhausted by the persistent failure: the run ends
        # without the goal, but the answer the operator gave is still there.
        assert result["status"] != "goal_achieved"
        assert result["world_state"]["confirmation"]["notes"] == "Leave at side door"
        history = [snap.values for snap in graph.get_state_history(config)]
        failed_steps = [
            s
            for s in history
            if any(not h["success"] for h in s.get("execution_history", []))
        ]
        assert failed_steps and all(
            "confirmation" in s["world_state"] for s in failed_steps
        )

    async def test_async_path_reuses_the_answer(self) -> None:
        graph = GoapGraph(_flaky_confirm_actions()).compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "form-survives-async"}}
        await graph.ainvoke(_goal_input(), config=config)

        result = await graph.ainvoke(Command(resume=_VALID_CONFIRMATION), config=config)

        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"

    def test_prefilled_valid_answer_skips_the_prompt(self) -> None:
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "form-prefilled"}}
        result = graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
                "world_state": {"confirmation": dict(_VALID_CONFIRMATION)},
            },
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"

    def test_stored_answer_that_fails_validation_is_prompted_again(self) -> None:
        graph = GoapGraph(_confirm_delivery_actions()).compile(
            checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "form-stale"}}
        result = graph.invoke(
            {
                "goal": GoalSpec(conditions={"dispatched": True}).to_dict(),
                "world_state": {"confirmation": {"bogus": 1}},
            },
            config=config,
        )

        assert _interrupted_payload(result)["type"] == "goap_action_form"
