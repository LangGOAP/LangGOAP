"""Integration tests for ``TransitionModel`` threaded through the
LangGraph action-executor.

``GoapGraph(transition_model=model)`` causes the executor to call
``model.sample(world_state, action, rng)`` instead of falling back
to ``action.get_effects(world_state)`` whenever the action did not
itself return a dict.

Actions that return a dict at runtime (real LLM calls, real tool
outputs) are unaffected \u2014 their return value remains
authoritative, since the whole point of those actions is that the
real world *is* the runtime oracle.
"""

from __future__ import annotations

from collections.abc import Mapping
from random import Random
from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.planner.transitions import DivergencePolicy


class _NoisyModel:
    """``TransitionModel`` whose sample() adds a flag ``action.get_effects``
    does **not** set.

    The extra key (``"noisy"``) is the test's detection signal: if the
    executor wires the transition model through correctly, it will appear
    in the final world_state.  If the executor still falls back to
    ``action.get_effects``, it will not.
    """

    def __init__(self) -> None:
        self.divergence_policy: DivergencePolicy | None = DivergencePolicy(
            reason="noisy test model", kind="other"
        )
        self.sample_calls: list[tuple[str, dict[str, Any]]] = []

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(
        self,
        state: Mapping[str, Any],
        action: ActionSpec,
        rng: Random,
    ) -> Mapping[str, Any]:
        self.sample_calls.append((action.name, dict(state)))
        declared = dict(action.get_effects(dict(state)))
        declared["noisy"] = True  # marker only sample() ever writes
        return declared


def _two_step_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="step_one",
            preconditions={},
            effects={"stage_a": True},
            cost=1.0,
        ),
        ActionSpec(
            name="step_two",
            preconditions={"stage_a": True},
            effects={"stage_b": True},
            cost=1.0,
        ),
    ]


class TestTransitionModelInExecutor:
    def test_sync_executor_uses_transition_model_sample(self) -> None:
        """Sync path: ``sample()`` is called for every declared-effects action."""
        model = _NoisyModel()
        graph = GoapGraph(actions=_two_step_actions(), transition_model=model)
        compiled = graph.compile()
        result = compiled.invoke(
            {
                "goal": GoalSpec(conditions={"stage_b": True}),
                "world_state": {},
            }
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["stage_a"] is True
        assert result["world_state"]["stage_b"] is True
        # Marker that only sample() sets \u2014 proves the executor consulted the model.
        assert result["world_state"]["noisy"] is True
        # Both actions went through the model.
        assert [name for name, _ in model.sample_calls] == [
            "step_one",
            "step_two",
        ]

    @pytest.mark.asyncio
    async def test_async_executor_uses_transition_model_sample(self) -> None:
        """Async path: ``sample()`` fires from ``ainvoke`` too (parity)."""
        model = _NoisyModel()
        graph = GoapGraph(actions=_two_step_actions(), transition_model=model)
        compiled = graph.compile()
        result = await compiled.ainvoke(
            {
                "goal": GoalSpec(conditions={"stage_b": True}),
                "world_state": {},
            }
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["noisy"] is True
        assert [name for name, _ in model.sample_calls] == [
            "step_one",
            "step_two",
        ]

    def test_action_returning_dict_overrides_transition_model(self) -> None:
        """Real runtime returns (e.g. LLM/tool results) remain authoritative.

        ``TransitionModel`` models the world's response to *symbolic*
        actions.  If an ``ActionSpec.execute`` callable returns a dict,
        that dict *is* the runtime truth and the transition model is
        bypassed, matching the contract documented on ``_apply_result``.
        """

        def _truthy_execute(world: Mapping[str, Any]) -> dict[str, Any]:
            return {"stage_a": True, "authoritative": True}

        actions = [
            ActionSpec(
                name="step_one",
                preconditions={},
                effects={"stage_a": True},
                execute=_truthy_execute,
                cost=1.0,
            ),
        ]
        model = _NoisyModel()
        graph = GoapGraph(actions=actions, transition_model=model)
        compiled = graph.compile()
        result = compiled.invoke(
            {"goal": GoalSpec(conditions={"stage_a": True}), "world_state": {}}
        )
        assert result["world_state"]["authoritative"] is True
        assert "noisy" not in result["world_state"]
        assert model.sample_calls == []
