"""Tests for ActionSpec, @goap_action decorator, and GoapAction base class."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec, GoapAction, goap_action


class TestActionSpecCreation:
    def test_create_with_all_fields(self) -> None:
        spec = ActionSpec(
            name="test",
            preconditions={"a": True},
            effects={"b": True},
            cost=2.0,
        )
        assert spec.name == "test"
        assert spec.preconditions == {"a": True}
        assert spec.effects == {"b": True}
        assert spec.cost == 2.0

    def test_frozen(self) -> None:
        spec = ActionSpec(name="test", effects={"a": True})
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        spec = ActionSpec(name="test")
        assert dict(spec.preconditions) == {}
        assert dict(spec.effects) == {}
        assert spec.cost == 1.0
        assert spec.execute is None
        assert spec.resources is None
        assert spec.duration is None
        assert spec.metadata is None
        assert spec.effect_validator is None

    def test_preconditions_are_truly_immutable(self) -> None:
        """Mutation of preconditions/effects must be rejected even after construction."""
        spec = ActionSpec(name="test", preconditions={"a": True}, effects={"b": True})
        with pytest.raises(TypeError):
            spec.preconditions["hacked"] = True  # type: ignore[index]
        with pytest.raises(TypeError):
            spec.effects["hacked"] = True  # type: ignore[index]

    def test_two_subclasses_do_not_share_dicts(self) -> None:
        """GoapAction subclasses must each have their own preconditions/effects dict."""
        from langgoap.actions import GoapAction

        class AlphaAction(GoapAction):
            pass

        class BetaAction(GoapAction):
            pass

        alpha = AlphaAction()
        beta = BetaAction()
        assert alpha.preconditions is not beta.preconditions
        assert alpha.effects is not beta.effects
        # Mutating one must not affect the other
        alpha.preconditions["x"] = True
        assert "x" not in beta.preconditions


class TestActionSpecCost:
    def test_static_cost(self) -> None:
        spec = ActionSpec(name="test", cost=3.5)
        assert spec.get_cost() == 3.5

    def test_dynamic_cost_function(self) -> None:
        def cost_fn(ws: dict[str, Any]) -> float:
            return ws.get("difficulty", 1.0) * 2.0

        spec = ActionSpec(name="test", cost=cost_fn)
        assert spec.get_cost({"difficulty": 3.0}) == 6.0

    def test_dynamic_cost_with_no_state(self) -> None:
        def cost_fn(ws: dict[str, Any]) -> float:
            return ws.get("x", 5.0)

        spec = ActionSpec(name="test", cost=cost_fn)
        assert spec.get_cost() == 5.0


class TestActionSpecValidation:
    def test_has_effects_with_effects(self) -> None:
        spec = ActionSpec(name="test", effects={"a": True})
        assert spec.has_effects() is True

    def test_has_effects_empty(self) -> None:
        spec = ActionSpec(name="test")
        assert spec.has_effects() is False

    def test_validate_effects_default_checks_declared_effects(self) -> None:
        """Default validator checks declared effects exist in post_state."""
        spec = ActionSpec(name="test", effects={"a": True})
        assert spec.validate_effects(pre_state={}, post_state={"a": True}) is True
        assert spec.validate_effects(pre_state={}, post_state={"a": False}) is False
        assert spec.validate_effects(pre_state={}, post_state={}) is False

    def test_validate_effects_custom_validator(self) -> None:
        """Custom effect_validator is called instead of the default."""

        def always_fail(pre: dict, post: dict) -> bool:
            return False

        spec = ActionSpec(
            name="test", effects={"a": True}, effect_validator=always_fail
        )
        assert spec.validate_effects(pre_state={}, post_state={"a": True}) is False

    def test_validate_effects_empty_action_always_passes(self) -> None:
        """An action with no declared effects trivially passes validation."""
        spec = ActionSpec(name="noop")
        assert spec.validate_effects(pre_state={}, post_state={}) is True


class TestGoapActionDecorator:
    def test_decorator_creates_spec(self) -> None:
        @goap_action(
            preconditions={"has_data": True},
            effects={"report_ready": True},
            cost=2.0,
        )
        def generate_report(state: dict[str, Any]) -> dict[str, Any]:
            return {"report_ready": True}

        assert isinstance(generate_report, ActionSpec)
        assert generate_report.name == "generate_report"
        assert generate_report.preconditions == {"has_data": True}
        assert generate_report.effects == {"report_ready": True}
        assert generate_report.cost == 2.0

    def test_decorator_preserves_function(self) -> None:
        @goap_action(effects={"done": True})
        def my_action(state: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        assert my_action.execute is not None
        result = my_action.execute({"x": 1})
        assert result == {"done": True}

    def test_decorator_custom_name(self) -> None:
        @goap_action(name="custom_name", effects={"x": True})
        def something(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        assert something.name == "custom_name"

    def test_decorator_default_name_from_function(self) -> None:
        @goap_action(effects={"x": True})
        def my_func(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        assert my_func.name == "my_func"

    def test_decorator_defaults(self) -> None:
        @goap_action(effects={"x": True})
        def minimal(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        assert minimal.preconditions == {}
        assert minimal.cost == 1.0


class TestGoapActionClass:
    def test_subclass_to_spec(self) -> None:
        class GatherData(GoapAction):
            preconditions = {"has_source": True}
            effects = {"has_data": True}

            def cost(self, world_state: dict[str, Any]) -> float:
                return 1.5

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"has_data": True}

        action = GatherData()
        spec = action.to_spec()

        assert isinstance(spec, ActionSpec)
        assert spec.name == "GatherData"
        assert spec.preconditions == {"has_source": True}
        assert spec.effects == {"has_data": True}
        assert callable(spec.cost)
        assert spec.get_cost({}) == 1.5

    def test_default_cost(self) -> None:
        action = GoapAction()
        assert action.cost({}) == 1.0

    def test_default_execute(self) -> None:
        action = GoapAction()
        assert action.execute({}) == {}

    def test_dynamic_cost_via_class(self) -> None:
        class ExpensiveAction(GoapAction):
            effects = {"done": True}

            def cost(self, world_state: dict[str, Any]) -> float:
                return world_state.get("complexity", 1.0) * 10.0

        spec = ExpensiveAction().to_spec()
        assert spec.get_cost({"complexity": 3.0}) == 30.0
