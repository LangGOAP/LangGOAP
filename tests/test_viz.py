"""Unit tests for the plan visualization subpackage."""

from __future__ import annotations

from datetime import timedelta

import pytest

from langgoap import (
    ActionSpec,
    CSPMetadata,
    CSPStatus,
    Plan,
    ResourceUsage,
    ScheduleEntry,
)
from langgoap.planner.types import PlanMetadata
from langgoap.viz import (
    render_ascii,
    render_ascii_gantt,
    render_dot,
    render_mermaid,
    render_mermaid_gantt,
    visualize,
)


def _make_linear_plan() -> Plan:
    a = ActionSpec(name="gather", effects={"has_data": True})
    b = ActionSpec(
        name="analyze",
        preconditions={"has_data": True},
        effects={"has_result": True},
    )
    c = ActionSpec(
        name="report",
        preconditions={"has_result": True},
        effects={"done": True},
    )
    return Plan(actions=(a, b, c), total_cost=3.0)


def _make_parallel_scheduled_plan() -> Plan:
    a = ActionSpec(
        name="fetch_a",
        effects={"got_a": True},
        duration=timedelta(seconds=2),
        resources={"tokens": 100.0, "cost_usd": 0.1},
    )
    b = ActionSpec(
        name="fetch_b",
        effects={"got_b": True},
        duration=timedelta(seconds=3),
        resources={"tokens": 150.0, "cost_usd": 0.15},
    )
    c = ActionSpec(
        name="merge",
        preconditions={"got_a": True, "got_b": True},
        effects={"merged": True},
        duration=timedelta(seconds=1),
        resources={"tokens": 50.0, "cost_usd": 0.05},
    )
    schedule = (
        ScheduleEntry(
            action_name="fetch_a",
            start=timedelta(seconds=0),
            duration=timedelta(seconds=2),
            end=timedelta(seconds=2),
        ),
        ScheduleEntry(
            action_name="fetch_b",
            start=timedelta(seconds=0),
            duration=timedelta(seconds=3),
            end=timedelta(seconds=3),
        ),
        ScheduleEntry(
            action_name="merge",
            start=timedelta(seconds=3),
            duration=timedelta(seconds=1),
            end=timedelta(seconds=4),
        ),
    )
    resource_usage = (
        ResourceUsage(
            key="tokens",
            total=300.0,
            constraint_max=500.0,
            satisfied=True,
        ),
        ResourceUsage(
            key="cost_usd",
            total=0.3,
            constraint_max=1.0,
            satisfied=True,
        ),
    )
    csp = CSPMetadata(
        status=CSPStatus.OPTIMAL,
        schedule=schedule,
        resource_usage=resource_usage,
        makespan=timedelta(seconds=4),
    )
    meta = PlanMetadata(csp=csp)
    return Plan(actions=(a, b, c), total_cost=3.0, metadata=meta)


class TestRenderMermaid:
    def test_empty_plan(self) -> None:
        result = render_mermaid(Plan.empty())
        assert "flowchart TD" in result
        assert "empty" in result

    def test_linear_plan_has_all_node_ids(self) -> None:
        plan = _make_linear_plan()
        src = render_mermaid(plan)
        assert src.startswith("flowchart TD")
        for action in plan.actions:
            assert action.name in src
        # sequential dependency edges from gather -> analyze -> report
        assert "-->" in src

    def test_linear_plan_edges_come_from_precondition_match(self) -> None:
        plan = _make_linear_plan()
        src = render_mermaid(plan)
        # Two edges expected: gather->analyze, analyze->report.
        assert src.count("-->") == 2

    def test_scheduled_plan_groups_parallel_actions(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_mermaid(plan)
        # Parallel slot should produce a subgraph block.
        assert "subgraph" in src
        assert "t=0s" in src
        # Merge is later, so no subgraph for it (only one action at t=3).
        assert "merge" in src

    def test_scheduled_plan_renders_resource_summary(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_mermaid(plan)
        assert "Resource usage" in src
        assert "tokens" in src
        assert "cost_usd" in src
        assert "[OK]" in src

    def test_resource_summary_disabled(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_mermaid(plan, show_resources=False)
        assert "Resource usage" not in src

    def test_violation_is_reported(self) -> None:
        a = ActionSpec(name="burn", resources={"tokens": 1000.0})
        csp = CSPMetadata(
            status=CSPStatus.INFEASIBLE,
            resource_usage=(
                ResourceUsage(
                    key="tokens",
                    total=1000.0,
                    constraint_max=500.0,
                    satisfied=False,
                ),
            ),
        )
        plan = Plan(actions=(a,), total_cost=1.0, metadata=PlanMetadata(csp=csp))
        src = render_mermaid(plan)
        assert "[VIOLATED]" in src

    def test_node_ids_are_safe_for_special_characters(self) -> None:
        a = ActionSpec(name="run.sql query-1", effects={"ok": True})
        plan = Plan(actions=(a,), total_cost=1.0)
        src = render_mermaid(plan)
        # Must not contain dots or spaces inside the generated identifier.
        # The label can still contain them — we only care about the id.
        assert "a0_run_sql_query_1" in src


class TestRenderMermaidGantt:
    def test_requires_schedule(self) -> None:
        plan = _make_linear_plan()
        with pytest.raises(ValueError, match="schedule"):
            render_mermaid_gantt(plan)

    def test_scheduled_plan_produces_gantt(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_mermaid_gantt(plan)
        assert src.startswith("gantt")
        for entry in plan.metadata.csp.schedule:  # type: ignore[union-attr]
            assert entry.action_name in src


class TestRenderDot:
    def test_empty_plan(self) -> None:
        src = render_dot(Plan.empty())
        assert "digraph" in src
        assert "empty" in src

    def test_linear_plan_is_valid_dot(self) -> None:
        plan = _make_linear_plan()
        src = render_dot(plan)
        assert src.startswith("digraph Plan {")
        assert src.rstrip().endswith("}")
        assert "->" in src
        for action in plan.actions:
            assert action.name in src

    def test_scheduled_plan_creates_cluster(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_dot(plan)
        assert "subgraph cluster_" in src
        assert "t=0s" in src
        assert "legend" in src
        assert "tokens" in src

    def test_resource_legend_disabled(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_dot(plan, show_resources=False)
        assert "legend" not in src


class TestRenderAscii:
    def test_empty_plan(self) -> None:
        src = render_ascii(Plan.empty())
        assert "empty" in src.lower()

    def test_linear_plan(self) -> None:
        plan = _make_linear_plan()
        src = render_ascii(plan)
        assert "Plan (3 steps" in src
        assert "gather" in src
        assert "analyze" in src
        assert "report" in src
        assert "(after: [0])" in src  # analyze depends on gather
        assert "(after: [1])" in src  # report depends on analyze

    def test_scheduled_plan_renders_parallel_block(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_ascii(plan)
        assert "t=0s (parallel)" in src
        assert "fetch_a" in src
        assert "fetch_b" in src
        assert "makespan" in src
        assert "tokens" in src

    def test_resource_disabled(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_ascii(plan, show_resources=False)
        assert "Resources:" not in src


class TestRenderAsciiGantt:
    def test_requires_schedule(self) -> None:
        with pytest.raises(ValueError, match="schedule"):
            render_ascii_gantt(_make_linear_plan())

    def test_scheduled_plan_produces_bars(self) -> None:
        plan = _make_parallel_scheduled_plan()
        src = render_ascii_gantt(plan)
        assert "fetch_a" in src
        assert "fetch_b" in src
        assert "merge" in src
        # Must contain bar characters.
        assert "#" in src

    def test_width_validated(self) -> None:
        with pytest.raises(ValueError, match="width"):
            render_ascii_gantt(_make_parallel_scheduled_plan(), width=5)


class TestVisualizeDispatch:
    def test_ascii_format_returns_str(self) -> None:
        result = visualize(_make_linear_plan(), format="ascii")
        assert isinstance(result, str)
        assert "gather" in result

    def test_ascii_gantt_requires_schedule(self) -> None:
        with pytest.raises(ValueError, match="schedule"):
            visualize(_make_linear_plan(), format="ascii_gantt")

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            visualize(_make_linear_plan(), format="bogus")  # type: ignore[arg-type]

    def test_mermaid_format_produces_string_when_ipython_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force "IPython absent" to get a deterministic string return type.
        monkeypatch.setattr("langgoap.viz.jupyter._ipython_available", lambda: False)
        result = visualize(_make_linear_plan(), format="mermaid")
        assert isinstance(result, str)
        assert "flowchart TD" in result

    def test_auto_without_ipython_falls_back_to_ascii(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("langgoap.viz.jupyter._ipython_available", lambda: False)
        result = visualize(_make_linear_plan(), format="auto")
        assert isinstance(result, str)
        assert "Plan" in result


class TestPlanViz:
    def test_plan_methods_delegate(self) -> None:
        plan = _make_linear_plan()
        assert "flowchart TD" in plan.to_mermaid()
        assert "digraph" in plan.to_dot()
        assert "Plan (3 steps" in plan.to_ascii()

    def test_plan_save_ascii(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        plan = _make_linear_plan()
        out = plan.save(tmp_path / "plan.txt")
        assert out.exists()
        text = out.read_text()
        assert "gather" in text

    def test_plan_save_mermaid_extension(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        plan = _make_linear_plan()
        out = plan.save(tmp_path / "plan.mmd")
        text = out.read_text()
        assert "flowchart TD" in text

    def test_plan_save_dot_extension(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        plan = _make_linear_plan()
        out = plan.save(tmp_path / "plan.dot")
        text = out.read_text()
        assert "digraph" in text

    def test_plan_save_explicit_format(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        plan = _make_linear_plan()
        out = plan.save(tmp_path / "plan.unknown", format="mermaid")
        assert "flowchart TD" in out.read_text()

    def test_plan_visualize_returns_string_when_ipython_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("langgoap.viz.jupyter._ipython_available", lambda: False)
        result = _make_linear_plan().visualize(format="mermaid")
        assert isinstance(result, str)
