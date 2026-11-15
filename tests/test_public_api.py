"""Tests for the public API surface of LangGOAP."""

from __future__ import annotations

import importlib.metadata

import langgoap


class TestPublicAPI:
    def test_all_symbols_importable(self) -> None:
        """Every symbol in __all__ is importable from the top-level package."""
        for name in langgoap.__all__:
            obj = getattr(langgoap, name)
            assert obj is not None, f"{name} is None"

    def test_version_available(self) -> None:
        """__version__ is a non-empty string."""
        assert isinstance(langgoap.__version__, str)
        assert len(langgoap.__version__) > 0
        # Single source of truth is ``langgoap/_version.py`` (hatch reads it);
        # the installed distribution metadata must agree with it.
        assert langgoap.__version__ == importlib.metadata.version("langgoap")

    def test_all_has_no_duplicates(self) -> None:
        assert len(langgoap.__all__) == len(set(langgoap.__all__))

    def test_types_behind_public_fields_are_exported(self) -> None:
        """Types a public field or option refers to are importable from the package (#65)."""
        expected = {
            "PlanQualityMetric",
            "MinimizeFinalStateExpression",
            "MinimizeTrajectorySum",
            "MinimizeTrajectoryMax",
            "RepairResult",
            "repair_plan",
            "AgentModel",
            "UniformRandomAgentModel",
            "ScalarHeuristic",
            "RolloutPolicy",
            "RandomRollout",
            "HeuristicRollout",
            "StochasticRollout",
            "assert_expected_matches_declared",
            "SafeTracerProxy",
            "async_execute_action",
            "EffectValidator",
        }
        missing = expected - set(langgoap.__all__)
        assert not missing, sorted(missing)

    def test_planner_subpackage_reexports_its_public_types(self) -> None:
        import langgoap.planner as planner

        expected = {
            "ScalarHeuristic",
            "PlanQualityMetric",
            "MinimizeFinalStateExpression",
            "MinimizeTrajectorySum",
            "MinimizeTrajectoryMax",
            "RepairResult",
            "repair_plan",
            "AgentModel",
            "UniformRandomAgentModel",
            "RolloutPolicy",
            "RandomRollout",
            "HeuristicRollout",
            "StochasticRollout",
            "assert_expected_matches_declared",
        }
        assert expected <= set(planner.__all__)
        for name in expected:
            assert getattr(planner, name) is getattr(langgoap, name)

    def test_integrations_export_effect_validator(self) -> None:
        import langgoap.integrations as integrations

        assert "EffectValidator" in integrations.__all__
