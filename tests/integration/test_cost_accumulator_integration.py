"""End-to-end: ``CostAccumulator`` wired into a real LangChain LLM call,
populating the keys that ``MaxCostPolicy`` reads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from langgoap.actions import ActionSpec
from langgoap.callbacks import CostAccumulator
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.termination import MaxCostPolicy


class _FakeLLM:
    """Minimal LLM stand-in that fires ``on_llm_end`` on the supplied callbacks.

    Avoids a real network call.  The callback contract under test is the
    one between ``CostAccumulator`` and the LLM client, not the LLM
    client and the network.
    """

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    def invoke(self, prompt: str, callbacks: list[Any]) -> str:
        response = SimpleNamespace(
            generations=[],
            llm_output={
                "token_usage": {
                    "prompt_tokens": self._prompt_tokens,
                    "completion_tokens": self._completion_tokens,
                    "total_tokens": (self._prompt_tokens + self._completion_tokens),
                },
                "model_name": "gpt-4o-mini",
            },
        )
        for cb in callbacks:
            cb.on_llm_end(response, run_id=uuid4(), parent_run_id=None)
        return f"answer to: {prompt}"


def _llm_step(name: str, eff_key: str, prompt_tokens: int, completion_tokens: int):
    """Build an action whose execute() invokes a fake LLM with a callback."""

    def execute(world_state: dict[str, Any]) -> dict[str, Any]:
        llm = _FakeLLM(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        accumulator = CostAccumulator(world_state)
        llm.invoke(f"do {name}", callbacks=[accumulator])
        return {eff_key: True}

    return ActionSpec(
        name=name,
        preconditions={},
        effects={eff_key: True, "total_cost_usd": 0.0},
        execute=execute,
    )


class TestCostAccumulatorWithMaxCostPolicy:
    """The whole point: drop a CostAccumulator on your LLM call, set a
    MaxCostPolicy, and the agent halts when the budget runs out.  No
    manual ``world_state['total_cost_usd'] += X`` plumbing in actions.
    """

    def test_budget_terminates_after_real_callback_accounting(self) -> None:
        # Each step: 1000 prompt + 500 completion tokens at gpt-4o-mini
        # pricing = 1000*0.00015/1k + 500*0.0006/1k = 0.00015 + 0.0003 = $0.00045
        # per step.  Cap at $0.001 → step 3 trips it.
        actions = [
            _llm_step("step_a", "a_done", 1000, 500),
            _llm_step("step_b", "b_done", 1000, 500),
            _llm_step("step_c", "c_done", 1000, 500),
            _llm_step("step_d", "d_done", 1000, 500),
        ]
        # The callback also bumps total_tokens / llm_call_count, but
        # the goal is satisfied by chained effects on a/b/c/d_done.
        actions[1] = ActionSpec(
            name="step_b",
            preconditions={"a_done": True},
            effects={"b_done": True, "total_cost_usd": 0.0},
            execute=actions[1].execute,
        )
        actions[2] = ActionSpec(
            name="step_c",
            preconditions={"b_done": True},
            effects={"c_done": True, "total_cost_usd": 0.0},
            execute=actions[2].execute,
        )
        actions[3] = ActionSpec(
            name="step_d",
            preconditions={"c_done": True},
            effects={"d_done": True, "total_cost_usd": 0.0},
            execute=actions[3].execute,
        )
        actions[0] = ActionSpec(
            name="step_a",
            preconditions={},
            effects={"a_done": True, "total_cost_usd": 0.0},
            execute=actions[0].execute,
        )

        graph = GoapGraph(
            actions,
            termination_policies=[MaxCostPolicy(usd=0.001)],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"d_done": True}),
            world_state={"total_cost_usd": 0.0},
        )

        # Steps a, b fit cleanly (totalling $0.00090).  Step c brings
        # the total to $0.00135 — over budget — so the observer
        # terminates BEFORE step d.  step c's effects are already in
        # world_state because the policy halts the next routing
        # decision, not in the middle of an executed action.
        assert result["status"] == "terminated"
        executed = [h.action_name for h in result["execution_history"] if h.success]
        assert executed == ["step_a", "step_b", "step_c"]
        assert "d_done" not in result["world_state"]
        # Costs accumulated through the CALLBACK, not via manual
        # wiring inside execute().
        assert result["world_state"]["total_cost_usd"] > 0.001
        assert result["world_state"]["llm_call_count"] == 3
        assert result["world_state"]["total_tokens"] == 4500

    def test_under_budget_completes_normally(self) -> None:
        actions = [
            _llm_step("only_step", "done", 100, 50),
        ]
        graph = GoapGraph(
            actions,
            termination_policies=[MaxCostPolicy(usd=10.00)],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={"total_cost_usd": 0.0},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["llm_call_count"] == 1
        assert result["world_state"]["total_tokens"] == 150
