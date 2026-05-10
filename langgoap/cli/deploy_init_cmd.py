"""``langgoap deploy-init`` — scaffold a LangGraph deployment for a LangGOAP agent.

Generates the four files needed to deploy a user-supplied ``GoapGraph``
via ``langgraph dev`` / ``langgraph deploy``.  Once deployed, the
graph is automatically callable as an MCP tool — ``/mcp`` routes are
enabled by default in every LangGraph deployment.

Usage::

    langgoap deploy-init ./my_agent_deploy \\
        --graph-factory my_agent.graph:build_graph \\
        --name my_agent \\
        --description "An agent that does X."
"""

from __future__ import annotations

from pathlib import Path

import click

from langgoap.integrations.langgraph_deploy import scaffold_deployment


@click.command("deploy-init")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--graph-factory",
    "graph_factory",
    required=True,
    help=(
        "Importable reference to a no-argument callable returning a "
        "configured GoapGraph, in the form 'module.path:attr_name'."
    ),
)
@click.option(
    "--name",
    "agent_name",
    required=True,
    help="Short identifier for the deployed graph (becomes the MCP tool name).",
)
@click.option(
    "--description",
    "agent_description",
    required=True,
    help="Human-readable description, surfaced in the generated README.",
)
@click.option(
    "--python-version",
    default="3.12",
    show_default=True,
    help="Python version pinned in langgraph.json (>= 3.11).",
)
@click.option(
    "--dep",
    "extra_dependencies",
    multiple=True,
    help=(
        "Extra pip dependency to append to the default ``langgoap``. "
        "Repeat to add multiple."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files in OUT_DIR.",
)
def deploy_init(
    out_dir: Path,
    graph_factory: str,
    agent_name: str,
    agent_description: str,
    python_version: str,
    extra_dependencies: tuple[str, ...],
    force: bool,
) -> None:
    """Scaffold a deployable LangGraph directory for a LangGOAP agent.

    Once the directory exists, run ``langgraph dev`` inside it to bring
    up a local server with ``/mcp`` routes enabled — any MCP client
    can then discover and call the agent.
    """
    if ":" not in graph_factory:
        raise click.UsageError(
            "--graph-factory must be in 'module.path:attr_name' form, "
            f"got {graph_factory!r}"
        )
    module_path, attr_name = graph_factory.rsplit(":", 1)

    try:
        result = scaffold_deployment(
            out_dir,
            graph_factory_module=module_path,
            graph_factory_attr=attr_name,
            agent_name=agent_name,
            agent_description=agent_description,
            python_version=python_version,
            extra_dependencies=tuple(extra_dependencies),
            force=force,
        )
    except FileExistsError as exc:
        raise click.UsageError(
            f"{exc}.  Pass --force to overwrite, or choose a fresh OUT_DIR."
        ) from exc

    click.echo(f"Scaffolded LangGraph deployment at {result}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. cd {result}")
    click.echo("  2. cp .env.example .env  # fill in OPENAI_API_KEY etc.")
    click.echo("  3. langgraph dev         # /mcp endpoint live by default")
    click.echo("")
    click.echo("See the generated README.md for Claude Desktop wiring.")


__all__ = ["deploy_init"]
