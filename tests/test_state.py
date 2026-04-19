"""Tests for PlanningState and state-related utilities."""

from __future__ import annotations

import logging

import pytest

from langgoap.state import PlanningState


class TestPlanningStateCreation:
    def test_from_dict_creates_state(self) -> None:
        state = PlanningState.from_dict({"a": 1, "b": True})
        assert state.to_dict() == {"a": 1, "b": True}

    def test_from_empty_dict(self) -> None:
        state = PlanningState.from_dict({})
        assert state.to_dict() == {}
        assert len(state) == 0

    def test_to_dict_returns_mutable_copy(self) -> None:
        state = PlanningState.from_dict({"x": 1})
        d = state.to_dict()
        d["x"] = 999
        assert state.to_dict() == {"x": 1}


class TestPlanningStateImmutability:
    def test_frozen_raises_on_attribute_set(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        with pytest.raises(AttributeError):
            state.conditions = frozenset()  # type: ignore[misc]

    def test_hashable(self) -> None:
        s1 = PlanningState.from_dict({"a": 1, "b": 2})
        s2 = PlanningState.from_dict({"a": 1, "b": 2})
        assert hash(s1) == hash(s2)
        assert {s1, s2} == {s1}

    def test_usable_as_dict_key(self) -> None:
        state = PlanningState.from_dict({"x": 10})
        d = {state: "value"}
        assert d[state] == "value"


class TestPlanningStateEquality:
    def test_equal_states(self) -> None:
        s1 = PlanningState.from_dict({"a": 1, "b": 2})
        s2 = PlanningState.from_dict({"b": 2, "a": 1})
        assert s1 == s2

    def test_unequal_states(self) -> None:
        s1 = PlanningState.from_dict({"a": 1})
        s2 = PlanningState.from_dict({"a": 2})
        assert s1 != s2

    def test_different_keys(self) -> None:
        s1 = PlanningState.from_dict({"a": 1})
        s2 = PlanningState.from_dict({"b": 1})
        assert s1 != s2


class TestPlanningStateSatisfies:
    def test_satisfies_all_conditions(self) -> None:
        state = PlanningState.from_dict({"a": 1, "b": True, "c": "yes"})
        assert state.satisfies({"a": 1, "b": True})

    def test_fails_on_wrong_value(self) -> None:
        state = PlanningState.from_dict({"a": 1, "b": False})
        assert not state.satisfies({"b": True})

    def test_fails_on_missing_key(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        assert not state.satisfies({"missing": True})

    def test_empty_conditions_always_satisfied(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        assert state.satisfies({})

    def test_empty_state_fails_nonempty_conditions(self) -> None:
        state = PlanningState.from_dict({})
        assert not state.satisfies({"a": 1})


class TestPlanningStateApply:
    def test_apply_adds_new_keys(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        new = state.apply({"b": 2})
        assert new.to_dict() == {"a": 1, "b": 2}

    def test_apply_overwrites_existing_keys(self) -> None:
        state = PlanningState.from_dict({"a": 1, "b": 2})
        new = state.apply({"b": 99})
        assert new.to_dict() == {"a": 1, "b": 99}

    def test_apply_does_not_mutate_original(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        state.apply({"a": 999, "b": 2})
        assert state.to_dict() == {"a": 1}

    def test_apply_empty_effects_returns_equal_state(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        new = state.apply({})
        assert new == state


class TestPlanningStateKeyFiltering:
    def test_from_dict_filters_by_keys(self) -> None:
        state = PlanningState.from_dict(
            {"a": True, "b": False, "c": 42},
            keys={"a", "c"},
        )
        d = state.to_dict()
        assert d == {"a": True, "c": 42}
        assert "b" not in d

    def test_from_dict_filters_out_unhashable_values(self) -> None:
        """Lists and dicts are silently skipped."""
        state = PlanningState.from_dict(
            {"flag": True, "documents": [1, 2, 3], "meta": {"nested": True}},
        )
        d = state.to_dict()
        assert d == {"flag": True}

    def test_from_dict_key_filter_plus_unhashable(self) -> None:
        """Key filtering and unhashable skipping compose correctly."""
        state = PlanningState.from_dict(
            {"flag": True, "data": [1, 2], "score": 0.9, "extra": "ignored"},
            keys={"flag", "data", "score"},
        )
        d = state.to_dict()
        # data is filtered by keys but still unhashable → skipped
        assert d == {"flag": True, "score": 0.9}

    def test_from_dict_no_keys_preserves_all_hashable(self) -> None:
        """Without keys filter, all hashable values are included."""
        state = PlanningState.from_dict({"a": 1, "b": "two", "c": True})
        assert state.to_dict() == {"a": 1, "b": "two", "c": True}

    def test_satisfies_works_after_filtering(self) -> None:
        state = PlanningState.from_dict(
            {"has_data": True, "question": "what?", "docs": [1, 2]},
            keys={"has_data"},
        )
        assert state.satisfies({"has_data": True})
        assert not state.satisfies({"has_data": False})
        assert not state.satisfies({"question": "what?"})  # filtered out


class TestPlanningStateAccessors:
    def test_get_existing_key(self) -> None:
        state = PlanningState.from_dict({"x": 42})
        assert state.get("x") == 42

    def test_get_missing_key_returns_default(self) -> None:
        state = PlanningState.from_dict({})
        assert state.get("x") is None
        assert state.get("x", "fallback") == "fallback"

    def test_contains(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        assert "a" in state
        assert "b" not in state

    def test_len(self) -> None:
        state = PlanningState.from_dict({"a": 1, "b": 2, "c": 3})
        assert len(state) == 3

    def test_repr(self) -> None:
        state = PlanningState.from_dict({"a": 1})
        r = repr(state)
        assert "PlanningState" in r
        assert "'a'" in r


class TestInferStartState:
    def test_infer_from_actions(self) -> None:
        from langgoap.actions import ActionSpec
        from langgoap.state import infer_start_state

        actions = [
            ActionSpec(name="gather", effects={"has_data": True}),
            ActionSpec(
                name="analyze",
                preconditions={"has_data": True},
                effects={"has_result": True},
            ),
            ActionSpec(
                name="report",
                preconditions={"has_result": True},
                effects={"done": True},
            ),
        ]
        start = infer_start_state(actions)
        assert start == {
            "done": False,
            "has_data": False,
            "has_result": False,
        }

    def test_skips_non_boolean_effects(self) -> None:
        from langgoap.actions import ActionSpec
        from langgoap.state import infer_start_state

        actions = [
            ActionSpec(
                name="score",
                effects={"quality": 0.9, "done": True},
            ),
        ]
        start = infer_start_state(actions)
        # "quality" has a float effect, not bool → excluded
        assert start == {"done": False}

    def test_empty_actions(self) -> None:
        from langgoap.state import infer_start_state

        assert infer_start_state([]) == {}


class TestSetLogLevel:
    def test_set_log_level_changes_logger(self) -> None:
        import langgoap

        langgoap.set_log_level("ERROR")
        assert logging.getLogger("langgoap").level == logging.ERROR
        langgoap.set_log_level("WARNING")
        assert logging.getLogger("langgoap").level == logging.WARNING

    def test_set_log_level_accepts_int(self) -> None:
        import langgoap

        langgoap.set_log_level(logging.DEBUG)
        assert logging.getLogger("langgoap").level == logging.DEBUG
