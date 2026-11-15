"""``langgoap plan`` command."""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgoap.planner.types import Plan

import click

from langgoap.cli._loader import load_actions, load_goal, load_world_state

logger = logging.getLogger(__name__)


@click.command("plan")
@click.argument("actions_ref")
@click.argument("goal_ref")
@click.option(
    "-w",
    "--world-state",
    "world_state_ref",
    default=None,
    help="Module:variable or JSON file for initial world state.",
)
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(["ascii", "mermaid", "dot", "json", "gantt"]),
    default="ascii",
    show_default=True,
    help="Output format for the plan.",
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["auto", "astar", "pipeline", "mcts"]),
    default="auto",
    show_default=True,
    help=(
        "Planning strategy. 'auto' routes to the CSP pipeline when the goal "
        "carries constraints, objectives or soft goals and to A* otherwise; "
        "the other names force that strategy."
    ),
)
@click.option(
    "--run/--no-run",
    default=False,
    show_default=True,
    help=(
        "Execute the plan through the GOAP graph (runs every action's "
        "execute callable). By default the command only plans."
    ),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write output to a file instead of stdout.",
)
def plan(
    actions_ref: str,
    goal_ref: str,
    world_state_ref: str | None,
    fmt: str,
    strategy: str,
    run: bool,
    output_path: str | None,
) -> None:
    """Plan a sequence of actions to achieve a goal.

    ACTIONS_REF and GOAL_REF are Python module:variable references
    (e.g. my_module:ACTIONS, my_module:GOAL).

    Planning is side-effect free: no action's ``execute`` callable runs
    unless ``--run`` is given.
    """
    try:
        actions_list = load_actions(actions_ref)
        goal = load_goal(goal_ref)
        ws = load_world_state(world_state_ref)
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        raise click.ClickException(str(exc)) from exc

    strategy_name = _effective_strategy_name(strategy, goal)
    strategy_obj = _resolve_strategy(strategy)

    result: dict[str, Any]
    if run:
        from langgoap.graph.builder import GoapGraph

        graph = GoapGraph(actions=actions_list, strategy=strategy_obj)
        result = dict(graph.invoke(goal=goal, world_state=ws))
    else:
        planned = _plan_only(strategy_obj, actions_list, goal, ws)
        result = {
            "plan": planned,
            "status": "planned" if planned is not None else "no_plan",
        }
    result["strategy"] = strategy_name

    status = result.get("status", "unknown")
    result_plan: Plan | None = result.get("plan")

    if status == "no_plan" or result_plan is None:
        click.echo(f"No plan found (status={status}).", err=True)
        _auto_explain(actions_list, goal, ws)
        sys.exit(1)

    try:
        output = _render_plan(result_plan, fmt, result)
    except ValueError as exc:
        raise click.ClickException(f"cannot render the plan as {fmt!r}: {exc}") from exc
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        click.echo(f"Plan written to {output_path}")
    else:
        click.echo(output)


def _effective_strategy_name(name: str, goal: Any) -> str:
    """Name the strategy that will actually run for *name* and *goal*."""
    if name != "auto":
        return name
    from langgoap.planner.pipeline import needs_csp

    return "pipeline" if needs_csp(goal) else "astar"


def _plan_only(
    strategy_obj: Any, actions_list: list[Any], goal: Any, ws: dict[str, Any]
) -> Any:
    """Plan without executing anything (the default ``langgoap plan`` mode)."""
    from langgoap.graph.nodes._helpers import _planning_keys
    from langgoap.state import PlanningState

    start = PlanningState.from_dict(ws, keys=_planning_keys(actions_list, goal))
    if strategy_obj is not None:
        return strategy_obj.plan(start, goal, actions_list)
    from langgoap.planner.pipeline import needs_csp

    if needs_csp(goal):
        from langgoap.planner.pipeline import plan as pipeline_plan

        return pipeline_plan(start, goal, actions_list)
    from langgoap.planner.astar import plan as astar_plan

    return astar_plan(start, goal, actions_list)


def _resolve_strategy(name: str) -> Any:
    """Resolve a strategy name to a strategy instance (``None`` = auto routing)."""
    if name == "auto":
        return None
    if name == "astar":
        from langgoap.planner.strategy import AStarStrategy

        return AStarStrategy()
    if name == "pipeline":
        from langgoap.planner.strategy import TwoPhasePipelineStrategy

        return TwoPhasePipelineStrategy()
    if name == "mcts":
        from langgoap.planner.mcts import MCTSStrategy

        return MCTSStrategy()
    return None


def _render_plan(plan_obj: "Plan", fmt: str, result: dict[str, Any]) -> str:
    """Render a plan in the requested format."""
    if fmt == "ascii":
        return plan_obj.to_ascii()
    elif fmt == "mermaid":
        return plan_obj.to_mermaid()
    elif fmt == "dot":
        return plan_obj.to_dot()
    elif fmt == "gantt":
        return plan_obj.to_gantt()
    elif fmt == "json":
        return json.dumps(
            {
                "status": result.get("status"),
                "strategy": result.get("strategy"),
                "actions": plan_obj.action_names,
                "total_cost": plan_obj.total_cost,
                "steps": len(plan_obj),
            },
            indent=2,
        )
    return str(plan_obj)


def _auto_explain(
    actions_list: list[Any],
    goal: Any,
    ws: dict[str, Any],
) -> None:
    """Print an explanation when no plan is found."""
    try:
        from langgoap.planner.explain import explain_no_plan
        from langgoap.state import PlanningState

        start = PlanningState.from_dict(ws)
        explanation = explain_no_plan(start, goal, actions_list)
        click.echo("\nExplanation:", err=True)
        if explanation.unreachable_conditions:
            click.echo(
                f"  Unreachable conditions: {list(explanation.unreachable_conditions)}",
                err=True,
            )
        if explanation.missing_preconditions:
            click.echo(
                f"  Missing preconditions: {list(explanation.missing_preconditions)}",
                err=True,
            )
        if explanation.suggestion:
            click.echo(f"  Suggestion: {explanation.suggestion}", err=True)
    except Exception as exc:
        # Best-effort explainer — the CLI must keep going if the helper
        # hits an edge case.  Surface the cause under DEBUG logging.
        logger.debug("auto-explain failed: %s", exc, exc_info=True)
