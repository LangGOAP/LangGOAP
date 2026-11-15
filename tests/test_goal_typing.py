"""``GoalSpec(conditions={...})`` is what every caller writes; it must type-check.

The fields are annotated as ``Mapping`` (what a caller passes) and converted
to an immutable ``MappingProxyType`` at construction (what the planner
relies on).  Annotating them as ``MappingProxyType`` made every dict literal
a mypy error in user code.
"""

from __future__ import annotations

from types import MappingProxyType

from langgoap import GoalSpec, SoftGoal
from langgoap.types import ObjectiveDirection


def _annotation(cls: type, field: str) -> str:
    # ``from __future__ import annotations`` keeps the source text; resolving
    # it with ``get_type_hints`` would need the TYPE_CHECKING-only names.
    return str(cls.__dataclass_fields__[field].type)


class TestAnnotationsAcceptPlainDicts:
    def test_goal_spec_conditions(self) -> None:
        assert _annotation(GoalSpec, "conditions") == "Mapping[str, Any]"

    def test_goal_spec_objectives(self) -> None:
        assert (
            _annotation(GoalSpec, "objectives")
            == "Mapping[str, ObjectiveDirection] | None"
        )

    def test_soft_goal_conditions(self) -> None:
        assert _annotation(SoftGoal, "conditions") == "Mapping[str, Any]"


class TestValuesStayImmutable:
    def test_goal_spec_wraps_dicts(self) -> None:
        goal = GoalSpec(
            conditions={"done": True},
            objectives={"cost_usd": ObjectiveDirection.MINIMIZE},
        )
        assert isinstance(goal.conditions, MappingProxyType)
        assert isinstance(goal.objectives, MappingProxyType)
        assert dict(goal.conditions) == {"done": True}

    def test_soft_goal_wraps_dicts(self) -> None:
        assert isinstance(SoftGoal(conditions={"x": 1}).conditions, MappingProxyType)
