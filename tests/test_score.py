"""Tests for the Score hierarchy.

Covers the three concrete subclasses (SimpleScore, HardSoftScore,
BendableScore), cross-subclass comparison errors, and the sign
convention documented on each class.
"""

from __future__ import annotations

import pytest

from langgoap.score import BendableScore, HardSoftScore, Score, SimpleScore


class TestSimpleScore:
    def test_value_returns_scalar(self) -> None:
        s = SimpleScore(scalar=3.5)
        assert s.value == 3.5

    def test_default_zero(self) -> None:
        s = SimpleScore()
        assert s.scalar == 0.0
        assert s.value == 0.0

    def test_is_feasible_is_always_true(self) -> None:
        assert SimpleScore(scalar=0.0).is_feasible() is True
        assert SimpleScore(scalar=100.0).is_feasible() is True
        assert SimpleScore(scalar=-1.0).is_feasible() is True

    def test_lexicographic_comparison_same_subclass(self) -> None:
        a = SimpleScore(scalar=1.0)
        b = SimpleScore(scalar=2.0)
        assert a < b
        assert a <= b
        assert b > a
        assert b >= a
        assert SimpleScore(scalar=1.0) <= SimpleScore(scalar=1.0)
        assert SimpleScore(scalar=1.0) >= SimpleScore(scalar=1.0)

    def test_repr(self) -> None:
        assert repr(SimpleScore(scalar=2.5)) == "SimpleScore(2.5)"

    def test_cross_subclass_comparison_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot compare SimpleScore"):
            _ = SimpleScore(scalar=1.0) < HardSoftScore(hard=0.0, soft=1.0)
        with pytest.raises(TypeError):
            _ = SimpleScore(scalar=1.0) <= BendableScore()


class TestHardSoftScore:
    def test_defaults(self) -> None:
        s = HardSoftScore()
        assert s.hard == 0.0
        assert s.soft == 0.0
        assert s.is_feasible() is True

    def test_value_is_sum(self) -> None:
        s = HardSoftScore(hard=-3.0, soft=-5.0)
        assert s.value == -8.0

    def test_feasibility_requires_zero_hard(self) -> None:
        assert HardSoftScore(hard=0.0, soft=-100.0).is_feasible() is True
        assert HardSoftScore(hard=-0.01, soft=0.0).is_feasible() is False
        assert HardSoftScore(hard=-1.0, soft=100.0).is_feasible() is False

    def test_lexicographic_ordering_hard_dominates(self) -> None:
        # Feasible beats infeasible regardless of soft
        feasible = HardSoftScore(hard=0.0, soft=-1000.0)
        infeasible = HardSoftScore(hard=-0.1, soft=1000.0)
        assert feasible > infeasible
        assert infeasible < feasible

    def test_ties_broken_by_soft(self) -> None:
        a = HardSoftScore(hard=0.0, soft=-5.0)
        b = HardSoftScore(hard=0.0, soft=-3.0)
        assert a < b
        assert b > a

    def test_soft_has_no_sign_restriction(self) -> None:
        # Maximize objectives can push soft positive
        positive = HardSoftScore(hard=0.0, soft=100.0)
        negative = HardSoftScore(hard=0.0, soft=-100.0)
        assert positive > negative
        assert positive.is_feasible()

    def test_cross_subclass_comparison_raises(self) -> None:
        with pytest.raises(TypeError):
            _ = HardSoftScore(hard=0.0) < SimpleScore(scalar=0.0)

    def test_repr(self) -> None:
        assert "hard=" in repr(HardSoftScore(hard=-1.0, soft=2.0))


class TestBendableScore:
    def test_defaults(self) -> None:
        s = BendableScore()
        assert s.hard_levels == ()
        assert s.soft_levels == ()
        assert s.is_feasible() is True

    def test_value_sums_all_levels(self) -> None:
        s = BendableScore(hard_levels=(-1.0, -2.0), soft_levels=(-3.0, 4.0))
        assert s.value == -2.0

    def test_is_feasible_checks_all_hard_levels(self) -> None:
        assert BendableScore(hard_levels=(0.0, 0.0)).is_feasible()
        assert not BendableScore(hard_levels=(0.0, -0.1)).is_feasible()

    def test_lexicographic_ordering(self) -> None:
        a = BendableScore(hard_levels=(-1.0, 0.0), soft_levels=(0.0,))
        b = BendableScore(hard_levels=(0.0, -1.0), soft_levels=(0.0,))
        # b has a greater first hard level → b > a
        assert b > a
        assert a < b

    def test_shape_mismatch_raises(self) -> None:
        a = BendableScore(hard_levels=(0.0,), soft_levels=(0.0,))
        b = BendableScore(hard_levels=(0.0, 0.0), soft_levels=(0.0,))
        with pytest.raises(TypeError, match="different"):
            _ = a < b

    def test_coerces_lists_to_tuples(self) -> None:
        s = BendableScore(hard_levels=[0.0, -1.0], soft_levels=[2.0])  # type: ignore[arg-type]
        assert isinstance(s.hard_levels, tuple)
        assert isinstance(s.soft_levels, tuple)

    def test_cross_subclass_comparison_raises(self) -> None:
        with pytest.raises(TypeError):
            _ = BendableScore() < SimpleScore(scalar=0.0)


class TestScoreBaseClass:
    def test_abstract_value_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = Score().value  # type: ignore[call-arg]

    def test_abstract_is_feasible_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            Score().is_feasible()  # type: ignore[call-arg]
