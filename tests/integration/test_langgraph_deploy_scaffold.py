"""Integration tests for the LangGraph-deployment scaffolder.

The scaffolder generates the four files needed to deploy a LangGOAP-built
graph via ``langgraph dev`` / ``langgraph deploy``.  Once deployed, the
graph is automatically callable as an MCP tool — ``/mcp`` routes are
enabled by default in every LangGraph deployment (see
``research/repos/langgraph/libs/cli/schemas/schema.json`` — the
``disable_mcp`` field defaults to ``False``).

These tests verify the scaffolder produces the right shape; the actual
``langgraph dev`` server bring-up and Claude-Desktop-as-client wiring are
manual steps documented in the tutorial notebook.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from langgoap.cli.main import cli
from langgoap.integrations.langgraph_deploy import scaffold_deployment


def _scaffold_minimal(out_dir: Path) -> Path:
    """Run the scaffolder with a tiny GoapGraph factory snippet."""
    return scaffold_deployment(
        out_dir,
        graph_factory_module="my_agent.graph",
        graph_factory_attr="build_graph",
        agent_name="my_agent",
        agent_description="A small demo agent.",
        python_version="3.12",
        extra_dependencies=("langchain-openai>=0.3",),
    )


# ---------------------------------------------------------------------------
# Scaffold output shape
# ---------------------------------------------------------------------------


class TestScaffoldOutput:
    def test_writes_all_four_files(self, tmp_path: Path) -> None:
        out = _scaffold_minimal(tmp_path / "deploy")
        assert (out / "langgraph.json").is_file()
        assert (out / "graph.py").is_file()
        assert (out / ".env.example").is_file()
        assert (out / "README.md").is_file()

    def test_langgraph_json_validates_against_schema(self, tmp_path: Path) -> None:
        """Generated config must match the langgraph-cli schema shape:
        ``graphs`` mapping name → ``module:attr`` reference, ``env``
        path, and a python_version.  We don't run the CLI's own
        validator (extra dep) but check the documented contract."""
        out = _scaffold_minimal(tmp_path / "deploy")
        config = json.loads((out / "langgraph.json").read_text())

        # Required: graphs mapping, each value a 'path:attr' reference.
        assert "graphs" in config
        assert "my_agent" in config["graphs"]
        ref = config["graphs"]["my_agent"]
        assert ref.endswith(":graph"), f"unexpected ref shape: {ref!r}"

        # Optional but recommended: python_version, env, dependencies.
        assert config["python_version"] == "3.12"
        assert config["env"] == ".env"
        assert "langgoap" in config.get("dependencies", [])
        assert "langchain-openai>=0.3" in config["dependencies"]

    def test_entrypoint_is_importable_python(self, tmp_path: Path) -> None:
        """The generated graph.py must be syntactically valid Python and
        define the ``graph`` symbol the langgraph.json reference points
        to.  We import via ``importlib`` rather than executing it as a
        script (the import would fail because the user-supplied factory
        module 'my_agent.graph' doesn't exist in the test sandbox), so we
        compile-check it instead."""
        out = _scaffold_minimal(tmp_path / "deploy")
        source = (out / "graph.py").read_text()

        # Compile-check: must parse cleanly.
        compile(source, str(out / "graph.py"), "exec")

        # Surface check: must reference the supplied factory and bind a
        # module-level ``graph`` name (the symbol langgraph.json points to).
        assert "from my_agent.graph import build_graph" in source
        assert "build_graph" in source
        assert "graph =" in source

    def test_env_example_lists_recommended_vars(self, tmp_path: Path) -> None:
        out = _scaffold_minimal(tmp_path / "deploy")
        env_example = (out / ".env.example").read_text()
        # OpenAI is the canonical LLM provider used in LangGOAP tutorials.
        assert "OPENAI_API_KEY" in env_example
        assert "LANGCHAIN_API_KEY" in env_example  # for LangSmith tracing

    def test_readme_documents_the_three_step_workflow(self, tmp_path: Path) -> None:
        out = _scaffold_minimal(tmp_path / "deploy")
        readme = (out / "README.md").read_text().lower()
        for keyword in ("langgraph dev", "mcp", "claude desktop"):
            assert keyword.lower() in readme, f"missing keyword {keyword!r}"

    def test_returns_output_directory_path(self, tmp_path: Path) -> None:
        out = scaffold_deployment(
            tmp_path / "agent",
            graph_factory_module="my_agent.graph",
            graph_factory_attr="build_graph",
            agent_name="my_agent",
            agent_description="x",
        )
        assert out == tmp_path / "agent"
        assert out.is_dir()

    def test_does_not_overwrite_existing_files_without_force(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "deploy"
        _scaffold_minimal(out)
        with pytest.raises(FileExistsError):
            _scaffold_minimal(out)

    def test_force_overwrites_existing_files(self, tmp_path: Path) -> None:
        out = tmp_path / "deploy"
        _scaffold_minimal(out)
        # Mutate one file so we can detect the rewrite.
        (out / "README.md").write_text("custom content")
        scaffold_deployment(
            out,
            graph_factory_module="my_agent.graph",
            graph_factory_attr="build_graph",
            agent_name="my_agent",
            agent_description="x",
            force=True,
        )
        readme = (out / "README.md").read_text()
        assert "custom content" not in readme


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestDeployInitCli:
    def test_help_shows_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "deploy-init" in result.output

    def test_subcommand_scaffolds_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "agent"
        result = runner.invoke(
            cli,
            [
                "deploy-init",
                str(out),
                "--graph-factory",
                "my_agent.graph:build_graph",
                "--name",
                "my_agent",
                "--description",
                "demo",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "langgraph.json").is_file()
        assert (out / "graph.py").is_file()

    def test_subcommand_rejects_existing_dir_without_force(
        self, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "agent"
        out.mkdir()
        (out / "langgraph.json").write_text("{}")
        result = runner.invoke(
            cli,
            [
                "deploy-init",
                str(out),
                "--graph-factory",
                "my_agent.graph:build_graph",
                "--name",
                "my_agent",
                "--description",
                "demo",
            ],
        )
        assert result.exit_code != 0
        assert "exists" in result.output.lower() or "already" in result.output.lower()

    def test_subcommand_force_flag_overwrites(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "agent"
        out.mkdir()
        (out / "langgraph.json").write_text("{}")
        result = runner.invoke(
            cli,
            [
                "deploy-init",
                str(out),
                "--graph-factory",
                "my_agent.graph:build_graph",
                "--name",
                "my_agent",
                "--description",
                "demo",
                "--force",
            ],
        )
        assert result.exit_code == 0, result.output
        config = json.loads((out / "langgraph.json").read_text())
        assert "graphs" in config


# ---------------------------------------------------------------------------
# Round-trip with a real GoapGraph factory
# ---------------------------------------------------------------------------


class TestRoundTripWithRealFactory:
    """Scaffold + load + invoke pattern.

    Generates a deployment dir whose ``graph.py`` references a tiny
    factory we install on ``sys.path`` for the duration of the test.
    Imports the generated module and asserts the bound ``graph`` is a
    compiled LangGraph that we can invoke.
    """

    def test_generated_graph_module_loads_and_invokes(self, tmp_path: Path) -> None:
        # 1. Set up a fake user module on disk.
        pkg_dir = tmp_path / "fake_agent"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "graph.py").write_text(
            "from langgoap import ActionSpec, GoalSpec, GoapGraph\n"
            "\n"
            "def build_graph():\n"
            "    return GoapGraph([ActionSpec(name='noop', effects={'done': True})])\n"
        )
        sys.path.insert(0, str(tmp_path))
        try:
            # 2. Scaffold a deployment that points at it.
            out = scaffold_deployment(
                tmp_path / "deploy",
                graph_factory_module="fake_agent.graph",
                graph_factory_attr="build_graph",
                agent_name="noop_agent",
                agent_description="A noop agent.",
            )
            # 3. Import the generated graph.py and verify the symbol works.
            spec = importlib.util.spec_from_file_location(
                "deploy_graph", out / "graph.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            graph = module.graph
            # Compiled LangGraph exposes invoke().
            from langgraph.graph.state import CompiledStateGraph

            assert isinstance(graph, CompiledStateGraph)
            from langgoap import GoalSpec

            result = graph.invoke(
                {
                    "goal": GoalSpec(conditions={"done": True}),
                    "world_state": {},
                }
            )
            assert result["status"] == "goal_achieved"
        finally:
            sys.path.remove(str(tmp_path))
