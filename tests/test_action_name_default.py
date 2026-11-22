"""``ActionSpec.name`` defaults to the execute function's name (issue #98)."""

from __future__ import annotations

import functools
from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph


def negotiate_terms(world_state: dict[str, Any]) -> None:
    return None


async def fetch_quote(world_state: dict[str, Any]) -> None:
    return None


class TestNameDefault:
    def test_name_comes_from_execute(self) -> None:
        spec = ActionSpec(effects={"terms_agreed": True}, execute=negotiate_terms)
        assert spec.name == "negotiate_terms"

    def test_name_comes_from_aexecute(self) -> None:
        spec = ActionSpec(effects={"quoted": True}, aexecute=fetch_quote)
        assert spec.name == "fetch_quote"

    def test_execute_wins_when_both_are_given(self) -> None:
        spec = ActionSpec(
            effects={"x": True}, execute=negotiate_terms, aexecute=fetch_quote
        )
        assert spec.name == "negotiate_terms"

    def test_an_explicit_name_wins(self) -> None:
        spec = ActionSpec("agree", {}, {"terms_agreed": True}, execute=negotiate_terms)
        assert spec.name == "agree"
        assert ActionSpec(name="agree2", execute=negotiate_terms).name == "agree2"

    def test_a_bound_method_gives_its_name(self) -> None:
        class Desk:
            def stamp(self, world_state: dict[str, Any]) -> None:
                return None

        assert ActionSpec(effects={"x": True}, execute=Desk().stamp).name == "stamp"

    def test_the_planner_runs_with_defaulted_names(self) -> None:
        graph = GoapGraph(
            [ActionSpec(effects={"terms_agreed": True}, execute=negotiate_terms)]
        )
        result = graph.invoke(GoalSpec(conditions={"terms_agreed": True}))
        assert [r.action_name for r in result["execution_history"]] == [
            "negotiate_terms"
        ]

    def test_round_trip_keeps_the_name(self) -> None:
        spec = ActionSpec(effects={"x": True}, execute=negotiate_terms)
        assert ActionSpec.from_dict(spec.to_dict()).name == "negotiate_terms"


class TestNoNameToDefaultTo:
    def test_no_name_and_no_callable_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ActionSpec(effects={"x": True})

    def test_a_lambda_has_no_usable_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ActionSpec(effects={"x": True}, execute=lambda ws: None)

    def test_a_partial_has_no_usable_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ActionSpec(effects={"x": True}, execute=functools.partial(negotiate_terms))

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ActionSpec("", {}, {"x": True})

    def test_a_wait_still_needs_its_name(self) -> None:
        with pytest.raises(TypeError):
            ActionSpec.wait(effects={"x": True})  # type: ignore[call-arg]
