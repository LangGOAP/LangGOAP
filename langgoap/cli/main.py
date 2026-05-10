"""Main CLI entry point.

Registers all subcommands under the ``langgoap`` group.
"""

from __future__ import annotations

import click

from langgoap._version import __version__
from langgoap.cli.actions_cmd import actions
from langgoap.cli.deploy_init_cmd import deploy_init
from langgoap.cli.explain_cmd import explain
from langgoap.cli.plan_cmd import plan
from langgoap.cli.visualize_cmd import visualize


@click.group()
@click.version_option(version=__version__, prog_name="langgoap")
def cli() -> None:
    """LangGOAP — Goal-Oriented Action Planning for LangGraph."""


cli.add_command(plan)
cli.add_command(visualize)
cli.add_command(explain)
cli.add_command(actions)
cli.add_command(deploy_init)
