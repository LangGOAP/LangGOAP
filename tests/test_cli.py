"""Tests for the LangGOAP CLI.

Uses Click's ``CliRunner`` for in-process testing — no process spawning.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec

# ---------------------------------------------------------------------------
# Helpers — write fixture modules to tmp_path
# ---------------------------------------------------------------------------


def _write_fixture_module(tmp_path: Path) -> Path:
    """Write a Python module with ACTIONS and GOAL to tmp_path."""
    mod = tmp_path / "cli_fixture.py"
    mod.write_text(textwrap.dedent("""\
        from langgoap.actions import ActionSpec
        from langgoap.goals import GoalSpec

        ACTIONS = [
            ActionSpec(name="gather", preconditions={}, effects={"data": True}, cost=1.0),
            ActionSpec(
                name="process",
                preconditions={"data": True},
                effects={"done": True},
                cost=1.0,
            ),
        ]

        GOAL = GoalSpec(conditions={"done": True})

        IMPOSSIBLE_GOAL = GoalSpec(conditions={"impossible": True})

        WORLD_STATE = {"existing": True}
        """))
    return mod


def _write_plan_module(tmp_path: Path) -> Path:
    """Write a Python module with a Plan object to tmp_path."""
    mod = tmp_path / "cli_plan_fixture.py"
    mod.write_text(textwrap.dedent("""\
        from langgoap.actions import ActionSpec
        from langgoap.planner.types import Plan

        MY_PLAN = Plan(
            actions=(
                ActionSpec(name="gather", preconditions={}, effects={"data": True}, cost=1.0),
                ActionSpec(name="process", preconditions={"data": True}, effects={"done": True}, cost=1.0),
            ),
            total_cost=2.0,
        )
        """))
    return mod


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a clean environment for CLI tests.

    Adds ``tmp_path`` to ``sys.path`` and removes any stale module
    cache entries for our fixture modules.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    # Remove stale module cache entries.
    for mod_name in list(sys.modules):
        if mod_name.startswith("cli_fixture") or mod_name.startswith(
            "cli_plan_fixture"
        ):
            del sys.modules[mod_name]
    return tmp_path


# ---------------------------------------------------------------------------
# TestLoader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_actions(self, cli_env: Path) -> None:
        from langgoap.cli._loader import load_actions

        _write_fixture_module(cli_env)
        actions = load_actions("cli_fixture:ACTIONS")
        assert len(actions) == 2
        assert all(isinstance(a, ActionSpec) for a in actions)

    def test_load_goal(self, cli_env: Path) -> None:
        from langgoap.cli._loader import load_goal

        _write_fixture_module(cli_env)
        goal = load_goal("cli_fixture:GOAL")
        assert isinstance(goal, GoalSpec)
        assert dict(goal.conditions) == {"done": True}

    def test_load_world_state_none(self) -> None:
        from langgoap.cli._loader import load_world_state

        assert load_world_state(None) == {}

    def test_load_world_state_json(self, tmp_path: Path) -> None:
        from langgoap.cli._loader import load_world_state

        jf = tmp_path / "state.json"
        jf.write_text('{"ready": true}')
        ws = load_world_state(str(jf))
        assert ws == {"ready": True}

    def test_load_world_state_module_ref(self, cli_env: Path) -> None:
        from langgoap.cli._loader import load_world_state

        _write_fixture_module(cli_env)
        ws = load_world_state("cli_fixture:WORLD_STATE")
        assert ws == {"existing": True}

    def test_invalid_ref_raises_value_error(self) -> None:
        from langgoap.cli._loader import load_python_variable

        with pytest.raises(ValueError, match="Invalid reference"):
            load_python_variable("no_colon_here")

    def test_bad_module_raises_import_error(self) -> None:
        from langgoap.cli._loader import load_python_variable

        with pytest.raises(ImportError):
            load_python_variable("nonexistent_module_xyz:VAR")

    def test_bad_variable_raises_attribute_error(self, cli_env: Path) -> None:
        from langgoap.cli._loader import load_python_variable

        _write_fixture_module(cli_env)
        with pytest.raises(AttributeError):
            load_python_variable("cli_fixture:DOES_NOT_EXIST")

    def test_load_actions_wrong_type_raises_type_error(self, cli_env: Path) -> None:
        from langgoap.cli._loader import load_actions

        _write_fixture_module(cli_env)
        with pytest.raises(TypeError, match="list"):
            load_actions("cli_fixture:GOAL")  # GOAL is a GoalSpec, not list


# ---------------------------------------------------------------------------
# TestPlanCmd
# ---------------------------------------------------------------------------


class TestPlanCmd:
    def test_help_output(self) -> None:
        from langgoap.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "--help"])
        assert result.exit_code == 0
        assert "ACTIONS_REF" in result.output
        assert "GOAL_REF" in result.output

    def test_successful_plan(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["plan", "cli_fixture:ACTIONS", "cli_fixture:GOAL"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "gather" in result.output
        assert "process" in result.output

    def test_json_format(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["plan", "cli_fixture:ACTIONS", "cli_fixture:GOAL", "-f", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        # ``plan`` only plans (issue #40); ``--run`` reports the graph status.
        assert data["status"] == "planned"
        assert data["strategy"] == "astar"
        assert "gather" in data["actions"]
        assert "process" in data["actions"]

    def test_mermaid_format(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["plan", "cli_fixture:ACTIONS", "cli_fixture:GOAL", "-f", "mermaid"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "flowchart" in result.output.lower() or "graph" in result.output.lower()

    def test_output_to_file(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        out_file = cli_env / "plan.txt"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plan",
                "cli_fixture:ACTIONS",
                "cli_fixture:GOAL",
                "-o",
                str(out_file),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "gather" in content

    def test_unreachable_goal_exits_nonzero(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["plan", "cli_fixture:ACTIONS", "cli_fixture:IMPOSSIBLE_GOAL"],
        )
        assert result.exit_code != 0

    def test_bad_ref_reports_error(self) -> None:
        from langgoap.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "no_colon", "also_no_colon"])
        assert result.exit_code != 0
        assert "Error" in result.output


# ---------------------------------------------------------------------------
# TestVisualizeCmd
# ---------------------------------------------------------------------------


class TestVisualizeCmd:
    def test_help_output(self) -> None:
        from langgoap.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "PLAN_REF" in result.output

    def test_ascii_visualization(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_plan_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["visualize", "cli_plan_fixture:MY_PLAN"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "gather" in result.output

    def test_mermaid_visualization(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_plan_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["visualize", "cli_plan_fixture:MY_PLAN", "-f", "mermaid"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# TestExplainCmd
# ---------------------------------------------------------------------------


class TestExplainCmd:
    def test_help_output(self) -> None:
        from langgoap.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["explain", "--help"])
        assert result.exit_code == 0
        assert "ACTIONS_REF" in result.output

    def test_explain_unreachable(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["explain", "cli_fixture:ACTIONS", "cli_fixture:IMPOSSIBLE_GOAL"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "impossible" in result.output.lower()

    def test_explain_achievable(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["explain", "cli_fixture:ACTIONS", "cli_fixture:GOAL"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "producible" in result.output.lower()


# ---------------------------------------------------------------------------
# TestActionsCmd
# ---------------------------------------------------------------------------


class TestActionsCmd:
    def test_help_output(self) -> None:
        from langgoap.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["actions", "--help"])
        assert result.exit_code == 0
        assert "ACTIONS_REF" in result.output

    def test_table_format(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["actions", "cli_fixture:ACTIONS"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "gather" in result.output
        assert "process" in result.output
        assert "2 action(s)" in result.output

    def test_json_format(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_fixture_module(cli_env)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["actions", "cli_fixture:ACTIONS", "-f", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["name"] == "gather"
        assert data[1]["name"] == "process"


# ---------------------------------------------------------------------------
# TestVersionFlag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    def test_version_output(self) -> None:
        from langgoap.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "langgoap" in result.output


# ---------------------------------------------------------------------------
# ``plan`` plans; ``--run`` executes; renderers fail cleanly (issue #40)
# ---------------------------------------------------------------------------


def _write_side_effect_module(tmp_path: Path) -> Path:
    """Actions whose ``execute`` leaves a file behind, to detect execution."""
    mod = tmp_path / "cli_fixture.py"
    marker = (tmp_path / "executed.marker").as_posix()
    mod.write_text(textwrap.dedent(f"""\
        from pathlib import Path

        from langgoap.actions import ActionSpec
        from langgoap.goals import ConstraintSpec, GoalSpec

        MARKER = Path({marker!r})

        def _touch(ws):
            MARKER.write_text("ran")
            return {{"done": True}}

        ACTIONS = [
            ActionSpec(name="gather", effects={{"data": True}}, cost=1.0,
                       resources={{"cost_usd": 1.0}}),
            ActionSpec(name="process", preconditions={{"data": True}},
                       effects={{"done": True}}, cost=1.0, execute=_touch,
                       resources={{"cost_usd": 1.0}}),
            ActionSpec(name="dynamic", effects={{"other": True}},
                       cost=lambda ws: 2.0),
        ]

        GOAL = GoalSpec(conditions={{"done": True}})
        CAPPED_GOAL = GoalSpec(
            conditions={{"done": True}},
            constraints=(ConstraintSpec(key="cost_usd", max=10.0),),
        )
        """))
    return mod


class TestPlanCmdPlansOnly:
    def test_plan_does_not_execute_actions(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_side_effect_module(cli_env)
        result = CliRunner().invoke(
            cli,
            ["plan", "cli_fixture:ACTIONS", "cli_fixture:GOAL"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "process" in result.output
        assert not (cli_env / "executed.marker").exists(), "plan must not run executors"

    def test_run_flag_executes_actions(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_side_effect_module(cli_env)
        result = CliRunner().invoke(
            cli,
            ["plan", "cli_fixture:ACTIONS", "cli_fixture:GOAL", "--run", "-f", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert (cli_env / "executed.marker").exists()
        assert json.loads(result.output)["status"] == "goal_achieved"

    def test_json_output_names_the_strategy_that_ran(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_side_effect_module(cli_env)
        runner = CliRunner()
        explicit = runner.invoke(
            cli,
            [
                "plan",
                "cli_fixture:ACTIONS",
                "cli_fixture:CAPPED_GOAL",
                "-s",
                "astar",
                "-f",
                "json",
            ],
            catch_exceptions=False,
        )
        auto = runner.invoke(
            cli,
            [
                "plan",
                "cli_fixture:ACTIONS",
                "cli_fixture:CAPPED_GOAL",
                "-s",
                "auto",
                "-f",
                "json",
            ],
            catch_exceptions=False,
        )
        assert explicit.exit_code == 0 and auto.exit_code == 0
        assert json.loads(explicit.output)["strategy"] == "astar"
        assert json.loads(auto.output)["strategy"] == "pipeline"

    def test_default_strategy_is_auto(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        result = CliRunner().invoke(cli, ["plan", "--help"])
        assert "[default: auto]" in result.output

    def test_gantt_without_schedule_is_a_clean_error(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_side_effect_module(cli_env)
        result = CliRunner().invoke(
            cli, ["plan", "cli_fixture:ACTIONS", "cli_fixture:GOAL", "-f", "gantt"]
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "schedule" in result.output.lower()


class TestActionsCmdDynamicCost:
    def test_callable_cost_renders_as_dyn(self, cli_env: Path) -> None:
        from langgoap.cli.main import cli

        _write_side_effect_module(cli_env)
        table = CliRunner().invoke(
            cli, ["actions", "cli_fixture:ACTIONS"], catch_exceptions=False
        )
        assert table.exit_code == 0, table.output
        assert "dyn" in table.output
        as_json = CliRunner().invoke(
            cli,
            ["actions", "cli_fixture:ACTIONS", "-f", "json"],
            catch_exceptions=False,
        )
        assert as_json.exit_code == 0, as_json.output
        entries = {e["name"]: e for e in json.loads(as_json.output)}
        assert entries["dynamic"]["cost"] == "dyn"
        assert entries["gather"]["cost"] == 1.0
