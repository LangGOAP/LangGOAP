"""The class-based and decorator APIs can express every ActionSpec field
(issue #52): resources, duration, metadata, and callable effects with
``effect_keys``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from langgoap import GoapAction, goap_action


class TestGoapActionForwardsEveryField:
    def test_resources_duration_metadata_reach_the_spec(self) -> None:
        class Ship(GoapAction):
            preconditions = {"packed": True}
            effects = {"shipped": True}
            resources = {"cost_usd": 12.5}
            duration = timedelta(hours=2)
            metadata = {"team": "ops"}

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"shipped": True}

        spec = Ship().to_spec()
        assert dict(spec.resources or {}) == {"cost_usd": 12.5}
        assert spec.duration == timedelta(hours=2)
        assert dict(spec.metadata or {}) == {"team": "ops"}

    def test_defaults_stay_none(self) -> None:
        class Plain(GoapAction):
            effects = {"x": True}

        spec = Plain().to_spec()
        assert (
            spec.resources is None and spec.duration is None and spec.metadata is None
        )

    def test_callable_effects_with_effect_keys(self) -> None:
        class Count(GoapAction):
            effect_keys = frozenset({"count"})

            @staticmethod
            def effects(state: dict[str, Any]) -> dict[str, Any]:
                return {"count": state.get("count", 0) + 1}

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"count": state.get("count", 0) + 1}

        spec = Count().to_spec()
        assert spec.has_dynamic_effects
        assert spec.effect_key_set() == frozenset({"count"})
        assert dict(spec.get_effects({"count": 2})) == {"count": 3}


class TestDecoratorAcceptsCallableEffects:
    def test_effect_function_and_keys(self) -> None:
        @goap_action(
            effects=lambda ws: {"count": ws.get("count", 0) + 1},
            effect_keys={"count"},
            resources={"tokens": 10},
        )
        def bump(state: dict[str, Any]) -> dict[str, Any]:
            return {"count": state.get("count", 0) + 1}

        assert bump.has_dynamic_effects
        assert bump.effect_key_set() == frozenset({"count"})
        assert dict(bump.get_effects({"count": 1})) == {"count": 2}
        assert dict(bump.resources or {}) == {"tokens": 10}

    def test_static_effects_still_work(self) -> None:
        @goap_action(effects={"done": True})
        def finish(state: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        assert not finish.has_dynamic_effects
        assert dict(finish.effects) == {"done": True}
