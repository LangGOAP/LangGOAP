"""Unit tests for :mod:`langgoap.planner.agents`.

``AgentModel`` is the library-level protocol for predicting the
actions of *external* agents — opponents in an adversarial game,
other agents in a multi-agent plan, or any entity whose behaviour
the :class:`TransitionModel` needs to sample to simulate outcomes.
It mirrors the ``expected`` / ``sample`` split already established
by :class:`TransitionModel` so the two compose cleanly.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Sequence


class TestAgentModelProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        from langgoap.planner.agents import AgentModel

        class _MinimalModel:
            def expected(self, state: Mapping[str, Any]) -> str:
                return "Stop"

            def sample(self, state: Mapping[str, Any], rng: random.Random) -> str:
                return "Stop"

        assert isinstance(_MinimalModel(), AgentModel)

    def test_non_conforming_class_fails_isinstance(self) -> None:
        from langgoap.planner.agents import AgentModel

        class _MissingSample:
            def expected(self, state: Mapping[str, Any]) -> str:
                return "Stop"

        assert not isinstance(_MissingSample(), AgentModel)


class TestUniformRandomAgentModel:
    def test_expected_returns_default_when_no_legal_actions(self) -> None:
        from langgoap.planner.agents import UniformRandomAgentModel

        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=lambda state: [],
            default="Stop",
        )
        assert model.expected({}) == "Stop"

    def test_expected_returns_default_when_action_list_nonempty(self) -> None:
        from langgoap.planner.agents import UniformRandomAgentModel

        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=lambda state: ["A", "B", "C"],
            default="Stop",
        )
        assert model.expected({}) == "Stop"

    def test_sample_returns_default_when_no_legal_actions(self) -> None:
        from langgoap.planner.agents import UniformRandomAgentModel

        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=lambda state: [],
            default="Stop",
        )
        rng = random.Random(0)
        assert model.sample({}, rng) == "Stop"

    def test_sample_draws_from_legal_actions(self) -> None:
        from langgoap.planner.agents import UniformRandomAgentModel

        options = ["North", "South", "East", "West"]
        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=lambda state: options,
            default="Stop",
        )
        rng = random.Random(123)
        draws = [model.sample({}, rng) for _ in range(2000)]
        assert set(draws) == set(options)

    def test_sample_distribution_is_uniform(self) -> None:
        from langgoap.planner.agents import UniformRandomAgentModel

        options = ["A", "B", "C", "D"]
        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=lambda state: options,
            default="Stop",
        )
        rng = random.Random(7)
        n = 10_000
        counts = Counter(model.sample({}, rng) for _ in range(n))
        for opt in options:
            ratio = counts[opt] / n
            assert 0.2 <= ratio <= 0.3, f"{opt} ratio {ratio:.3f} outside tolerance"

    def test_enumerate_receives_state_mapping(self) -> None:
        from langgoap.planner.agents import UniformRandomAgentModel

        seen: list[Mapping[str, Any]] = []

        def enumerate_fn(state: Mapping[str, Any]) -> Sequence[str]:
            seen.append(state)
            return ["Stop"]

        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=enumerate_fn, default="Stop"
        )
        payload = {"pos": (1, 2), "tick": 3}
        model.sample(payload, random.Random(0))
        assert seen == [payload]

    def test_sample_is_deterministic_under_seeded_rng(self) -> None:
        from langgoap.planner.agents import UniformRandomAgentModel

        options = ["N", "S", "E", "W"]
        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=lambda state: options,
            default="Stop",
        )
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        seq1 = [model.sample({}, rng1) for _ in range(50)]
        seq2 = [model.sample({}, rng2) for _ in range(50)]
        assert seq1 == seq2

    def test_satisfies_agent_model_protocol(self) -> None:
        from langgoap.planner.agents import AgentModel, UniformRandomAgentModel

        model: UniformRandomAgentModel[str] = UniformRandomAgentModel(
            enumerate_actions=lambda state: ["Stop"],
            default="Stop",
        )
        assert isinstance(model, AgentModel)
