"""Unit tests for the NL → GoalSpec interpreter."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from langgoap import (
    ActionSpec,
    ConstraintSpec,
    GoalInterpreter,
    GoalSpec,
    InterpretedConstraint,
    InterpretedGoal,
    InterpretedObjective,
    ObjectiveDirection,
)
from langgoap.interpreter import build_action_catalog, to_goal_spec
from tests.conftest import FakeStructuredModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="fetch_data",
            preconditions={"has_source": True},
            effects={"data_fetched": True},
            resources={"api_calls": 1, "cost_usd": 0.01},
            metadata={"description": "Fetch raw data from the source"},
        ),
        ActionSpec(
            name="generate_report",
            preconditions={"data_fetched": True},
            effects={"report_complete": True},
            resources={"tokens": 500, "cost_usd": 0.05},
        ),
    ]


# ---------------------------------------------------------------------------
# TestBuildActionCatalog
# ---------------------------------------------------------------------------


class TestBuildActionCatalog:
    def test_includes_action_names(self) -> None:
        catalog = build_action_catalog(_sample_actions())
        assert "fetch_data" in catalog
        assert "generate_report" in catalog

    def test_includes_effects_and_preconditions(self) -> None:
        catalog = build_action_catalog(_sample_actions())
        assert "has_source" in catalog
        assert "data_fetched" in catalog
        assert "report_complete" in catalog

    def test_includes_metadata_description(self) -> None:
        catalog = build_action_catalog(_sample_actions())
        assert "Fetch raw data from the source" in catalog

    def test_includes_resources(self) -> None:
        catalog = build_action_catalog(_sample_actions())
        assert "api_calls" in catalog
        assert "cost_usd" in catalog

    def test_empty_actions_list(self) -> None:
        catalog = build_action_catalog([])
        assert catalog == "(no actions available)"

    def test_action_with_only_name(self) -> None:
        """Action with no pre/eff/metadata/resources still renders a name line."""
        minimal = ActionSpec(name="do_thing", preconditions={}, effects={})
        catalog = build_action_catalog([minimal])
        assert "do_thing" in catalog
        # No extra sections rendered for empty fields
        assert "Preconditions" not in catalog
        assert "Effects" not in catalog
        assert "Description" not in catalog
        assert "Resources" not in catalog

    def test_action_with_preconditions_but_no_effects(self) -> None:
        """Catalog renders preconditions even when effects are absent."""
        partial = ActionSpec(
            name="check",
            preconditions={"ready": True},
            effects={},
        )
        catalog = build_action_catalog([partial])
        assert "ready" in catalog
        assert "Effects" not in catalog

    def test_action_with_effects_but_no_preconditions(self) -> None:
        """Catalog renders effects even when preconditions are absent."""
        partial = ActionSpec(
            name="init",
            preconditions={},
            effects={"initialized": True},
        )
        catalog = build_action_catalog([partial])
        assert "initialized" in catalog
        assert "Preconditions" not in catalog


# ---------------------------------------------------------------------------
# TestToGoalSpec
# ---------------------------------------------------------------------------


class TestToGoalSpec:
    def test_simple_conditions(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert dict(goal.conditions) == {"report_complete": True}

    def test_conditions_are_mapping_proxy(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert isinstance(goal.conditions, MappingProxyType)

    def test_with_constraints(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"done": True},
            constraints=[
                InterpretedConstraint(key="cost_usd", max=5.0, weight=2.0),
            ],
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert len(goal.constraints) == 1
        c = goal.constraints[0]
        assert isinstance(c, ConstraintSpec)
        assert c.key == "cost_usd"
        assert c.max == 5.0
        assert c.min is None
        assert c.weight == 2.0

    def test_minimize_objective(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"done": True},
            objectives=[InterpretedObjective(metric="cost_usd", direction="minimize")],
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert goal.objectives is not None
        assert goal.objectives["cost_usd"] == ObjectiveDirection.MINIMIZE

    def test_maximize_objective(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"done": True},
            objectives=[InterpretedObjective(metric="quality", direction="maximize")],
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert goal.objectives is not None
        assert goal.objectives["quality"] == ObjectiveDirection.MAXIMIZE

    def test_invalid_direction_raises_at_construction(self) -> None:
        """Invalid direction is rejected by Pydantic at InterpretedObjective construction (H1).

        Since direction is Literal["minimize", "maximize"], any other value raises
        ValidationError before to_goal_spec() is ever reached.  This is the intended
        upstream enforcement — bad values never enter the planning pipeline.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="minimize.*maximize|literal_error"):
            InterpretedObjective(metric="x", direction="sideways")  # type: ignore[arg-type]

    def test_empty_constraints_yields_empty_tuple(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"done": True},
            constraints=[],
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert goal.constraints == ()

    def test_empty_objectives_yields_none(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"done": True},
            objectives=[],
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert goal.objectives is None

    def test_invalid_constraint_min_greater_than_max_raises(self) -> None:
        """LLM hallucinating min > max must raise ValueError via ConstraintSpec validation.

        Regression guard: callers of to_goal_spec() should catch ValueError
        and re-prompt the user when the LLM produces impossible constraints.
        """
        interpreted = InterpretedGoal(
            conditions={"done": True},
            constraints=[
                InterpretedConstraint(key="cost_usd", min=100.0, max=5.0),
            ],
            reasoning="test",
        )
        with pytest.raises(ValueError, match="min.*max|max.*min"):
            to_goal_spec(interpreted)

    def test_empty_conditions_raises(self) -> None:
        """LLM returning empty conditions must raise ValueError (H2).

        An empty GoalSpec would be satisfied immediately by any world state,
        producing a silent logical failure without this guard.
        """
        interpreted = InterpretedGoal(conditions={}, reasoning="test")
        with pytest.raises(ValueError, match="empty conditions"):
            to_goal_spec(interpreted)

    def test_unhashable_condition_value_raises(self) -> None:
        """LLM returning a list/dict condition value must raise ValueError (H3).

        Without this guard, the unhashable value is silently dropped by
        PlanningState.from_dict(), making the goal condition disappear.
        """
        interpreted = InterpretedGoal(
            conditions={"items": [1, 2, 3]},  # type: ignore[dict-item]
            reasoning="test",
        )
        with pytest.raises(ValueError, match="Non-scalar keys"):
            to_goal_spec(interpreted)

    def test_multiple_constraints_all_converted(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"done": True},
            constraints=[
                InterpretedConstraint(key="cost_usd", max=5.0),
                InterpretedConstraint(key="tokens", max=1000.0, weight=0.5),
            ],
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert len(goal.constraints) == 2
        keys = {c.key for c in goal.constraints}
        assert keys == {"cost_usd", "tokens"}

    def test_multiple_objectives_all_converted(self) -> None:
        interpreted = InterpretedGoal(
            conditions={"done": True},
            objectives=[
                InterpretedObjective(metric="cost_usd", direction="minimize"),
                InterpretedObjective(metric="quality", direction="maximize"),
            ],
            reasoning="test",
        )
        goal = to_goal_spec(interpreted)
        assert goal.objectives is not None
        assert goal.objectives["cost_usd"] == ObjectiveDirection.MINIMIZE
        assert goal.objectives["quality"] == ObjectiveDirection.MAXIMIZE


# ---------------------------------------------------------------------------
# TestGoalInterpreter
# ---------------------------------------------------------------------------


class TestGoalInterpreter:
    def _make_interpreter(
        self,
        response: InterpretedGoal,
        system_prompt: str | None = None,
    ) -> GoalInterpreter:
        llm = FakeStructuredModel(response=response)
        return GoalInterpreter(
            llm=llm,
            actions=_sample_actions(),
            system_prompt=system_prompt,
        )

    def test_interpret_returns_goal_spec(self) -> None:
        response = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="User wants a report",
        )
        interp = self._make_interpreter(response)
        goal = interp.interpret("Generate a report")
        assert isinstance(goal, GoalSpec)
        assert dict(goal.conditions) == {"report_complete": True}

    def test_interpret_raw_returns_interpreted_goal(self) -> None:
        response = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="User wants a report",
        )
        interp = self._make_interpreter(response)
        raw = interp.interpret_raw("Generate a report")
        assert isinstance(raw, InterpretedGoal)
        assert raw.reasoning == "User wants a report"

    def test_interpret_returns_goal_spec_not_interpreted_goal(self) -> None:
        """interpret() must return GoalSpec, not the raw InterpretedGoal (M1).

        The previous test checked ``not hasattr(goal, 'reasoning')`` which is
        trivially true for any GoalSpec regardless of implementation.  This
        version verifies the type identity explicitly and cross-checks that the
        raw form *does* carry reasoning while the converted form does not.
        """
        response = InterpretedGoal(
            conditions={"done": True},
            reasoning="Some debug info",
        )
        interp = self._make_interpreter(response)
        raw = interp.interpret_raw("Do it")
        goal = interp.interpret("Do it")
        assert isinstance(raw, InterpretedGoal)
        assert raw.reasoning == "Some debug info"
        assert isinstance(goal, GoalSpec)
        assert not isinstance(goal, InterpretedGoal)

    def test_interpret_with_constraints(self) -> None:
        response = InterpretedGoal(
            conditions={"report_complete": True},
            constraints=[InterpretedConstraint(key="cost_usd", max=5.0)],
            reasoning="Budget limit",
        )
        interp = self._make_interpreter(response)
        goal = interp.interpret("Generate a report under $5")
        assert len(goal.constraints) == 1
        assert goal.constraints[0].key == "cost_usd"
        assert goal.constraints[0].max == 5.0

    @pytest.mark.asyncio
    async def test_ainterpret_returns_goal_spec(self) -> None:
        response = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="async test",
        )
        interp = self._make_interpreter(response)
        goal = await interp.ainterpret("Generate a report")
        assert isinstance(goal, GoalSpec)
        assert dict(goal.conditions) == {"report_complete": True}

    @pytest.mark.asyncio
    async def test_ainterpret_raw_returns_interpreted_goal(self) -> None:
        response = InterpretedGoal(
            conditions={"done": True},
            reasoning="async debug",
        )
        interp = self._make_interpreter(response)
        raw = await interp.ainterpret_raw("Do it")
        assert isinstance(raw, InterpretedGoal)
        assert raw.reasoning == "async debug"

    def test_custom_system_prompt_is_used(self) -> None:
        custom = "Custom: {action_catalog}\nState: {world_state}"
        response = InterpretedGoal(conditions={"x": True}, reasoning="")
        interp = self._make_interpreter(response, system_prompt=custom)
        messages = interp.build_messages("test", None)
        assert messages[0].content.startswith("Custom:")

    def test_world_state_appears_in_prompt(self) -> None:
        response = InterpretedGoal(conditions={"x": True}, reasoning="")
        interp = self._make_interpreter(response)
        messages = interp.build_messages("test", {"has_data": True})
        system_content = messages[0].content
        assert "has_data" in system_content

    def test_action_catalog_appears_in_default_prompt(self) -> None:
        """Action names and effects from _sample_actions() must appear in the system prompt (M5).

        Guards against build_messages() being accidentally decoupled from
        build_action_catalog().
        """
        response = InterpretedGoal(conditions={"x": True}, reasoning="")
        interp = self._make_interpreter(response)
        messages = interp.build_messages("test", None)
        system_content = messages[0].content
        # Names
        assert "fetch_data" in system_content
        assert "generate_report" in system_content
        # Effects / preconditions
        assert "data_fetched" in system_content
        assert "report_complete" in system_content
        # Resources
        assert "cost_usd" in system_content

    def test_interpret_propagates_value_error_from_to_goal_spec(self) -> None:
        """interpret() must propagate ValueError raised by to_goal_spec() (M2).

        If the LLM returns an empty conditions dict, callers of interpret()
        must see the ValueError, not a silent always-satisfied GoalSpec.
        """
        response = InterpretedGoal(conditions={}, reasoning="LLM forgot conditions")
        interp = self._make_interpreter(response)
        with pytest.raises(ValueError, match="empty conditions"):
            interp.interpret("Do something")

    @pytest.mark.asyncio
    async def test_ainterpret_propagates_value_error_from_to_goal_spec(self) -> None:
        """Async interpret must also propagate ValueError (M2 async path)."""
        response = InterpretedGoal(conditions={}, reasoning="LLM forgot conditions")
        interp = self._make_interpreter(response)
        with pytest.raises(ValueError, match="empty conditions"):
            await interp.ainterpret("Do something")

    def test_auto_detect_non_openai_uses_empty_kwargs(self) -> None:
        """Non-OpenAI models get empty structured_output_kwargs by default."""
        from langgoap.interpreter import _default_structured_output_kwargs

        llm = FakeStructuredModel(
            response=InterpretedGoal(conditions={"x": True}, reasoning="")
        )
        kwargs = _default_structured_output_kwargs(llm)
        assert kwargs == {}

    def test_auto_detect_applied_when_none(self) -> None:
        """GoalInterpreter applies auto-detection when structured_output_kwargs is None."""
        response = InterpretedGoal(conditions={"x": True}, reasoning="")
        llm = FakeStructuredModel(response=response)
        # Should not raise — auto-detection returns {} for non-OpenAI
        interp = GoalInterpreter(llm=llm, actions=_sample_actions())
        goal = interp.interpret("Do something")
        assert isinstance(goal, GoalSpec)

    def test_explicit_kwargs_override_auto_detect(self) -> None:
        """Explicit structured_output_kwargs bypasses auto-detection."""
        response = InterpretedGoal(conditions={"x": True}, reasoning="")
        llm = FakeStructuredModel(response=response)
        custom_kwargs = {"method": "json_mode"}
        # Construction with explicit kwargs should not trigger auto-detect
        interp = GoalInterpreter(
            llm=llm,
            actions=_sample_actions(),
            structured_output_kwargs=custom_kwargs,
        )
        goal = interp.interpret("Do something")
        assert isinstance(goal, GoalSpec)

    def test_with_structured_output_called_with_correct_schema(self) -> None:
        """GoalInterpreter must request InterpretedGoal schema (L3).

        FakeStructuredModel.expected_schema raises AssertionError if the wrong
        schema is passed to with_structured_output().
        """
        response = InterpretedGoal(conditions={"x": True}, reasoning="")
        llm = FakeStructuredModel(response=response, expected_schema=InterpretedGoal)
        # Construction calls with_structured_output — error surfaces here if wrong.
        interp = GoalInterpreter(llm=llm, actions=_sample_actions())
        goal = interp.interpret("Do something")
        assert isinstance(goal, GoalSpec)
