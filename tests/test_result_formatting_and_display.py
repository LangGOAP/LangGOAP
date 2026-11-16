"""Compact tool/subagent reports (#57) and network-free notebook display (#58)."""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.graph.state import ActionResult
from langgoap.integrations.deepagents import format_goap_result
from langgoap.planner.types import Plan
from langgoap.viz import jupyter as viz_jupyter


class TestFormatGoapResult:
    def _result(self) -> dict[str, Any]:
        big = list(range(50))
        return {
            "status": "goal_achieved",
            "world_state": {"done": True, "items": big, "note": "x" * 500},
            "execution_history": [
                ActionResult(
                    action_name="gather",
                    success=True,
                    state_before={"items": big},
                    state_after={"items": big, "done": True},
                ),
                ActionResult(
                    action_name="verify",
                    success=False,
                    error="boom",
                    state_before={"items": big},
                ),
            ],
        }

    def test_real_action_results_render_as_one_line_each(self) -> None:
        text = format_goap_result(self._result())
        assert "  - gather (success=True)" in text
        assert "  - verify (success=False)" in text
        assert "state_before" not in text and "state_after" not in text

    def test_world_state_values_are_clipped(self) -> None:
        text = format_goap_result(self._result())
        assert len(text) < 900, len(text)
        assert "…" in text or "..." in text
        assert "done: True" in text

    def test_failed_step_shows_its_error(self) -> None:
        assert "boom" in format_goap_result(self._result())


class TestReprMimebundle:
    def _plan(self) -> Plan:
        return GoapGraph([ActionSpec(name="a", effects={"done": True})]).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )["plan"]

    def test_default_bundle_is_offline_markdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGGOAP_DISPLAY_PNG", raising=False)
        called = []
        monkeypatch.setattr(
            "langgoap.viz.mermaid.draw_mermaid_png",
            lambda plan, **kw: called.append(1) or b"png",
        )
        bundle = viz_jupyter.repr_mimebundle(self._plan())
        assert "image/png" not in bundle
        assert "text/markdown" in bundle and "```mermaid" in bundle["text/markdown"]
        assert "text/plain" in bundle
        assert called == [], "no network call on display"

    def test_png_is_opt_in_via_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGGOAP_DISPLAY_PNG", "1")
        monkeypatch.setattr(
            "langgoap.viz.mermaid.draw_mermaid_png", lambda plan, **kw: b"png-bytes"
        )
        bundle = viz_jupyter.repr_mimebundle(self._plan())
        assert bundle["image/png"] == b"png-bytes"

    def test_png_failure_falls_back_to_markdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGGOAP_DISPLAY_PNG", "1")

        def boom(plan: Any, **kw: Any) -> bytes:
            raise RuntimeError("offline")

        monkeypatch.setattr("langgoap.viz.mermaid.draw_mermaid_png", boom)
        bundle = viz_jupyter.repr_mimebundle(self._plan())
        assert "image/png" not in bundle and "text/markdown" in bundle

    def test_plan_mimebundle_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGGOAP_DISPLAY_PNG", raising=False)
        bundle = self._plan()._repr_mimebundle_()
        assert "image/png" not in bundle and "text/markdown" in bundle
