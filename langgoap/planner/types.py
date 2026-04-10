"""Types for GOAP planning results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langgoap.actions import ActionSpec
from langgoap.score import Score, SimpleScore
from langgoap.state import PlanningState

if TYPE_CHECKING:
    from langgoap.planner.csp import CSPMetadata


@dataclass(frozen=True)
class PlanMetadata:
    """Metadata about how a plan was generated.

    Attributes:
        nodes_explored: Number of A* nodes expanded during search.
        planning_time_ms: Wall-clock time spent planning in milliseconds.
        actions_pruned: Number of actions removed by optimization passes.
        csp: Results from CSP constraint validation/optimization, or ``None``
            when no constraints or objectives were specified.
    """

    nodes_explored: int = 0
    planning_time_ms: float = 0.0
    actions_pruned: int = 0
    csp: CSPMetadata | None = None


@dataclass(frozen=True)
class Plan:
    """A sequence of actions that achieves a goal from a given start state.

    Attributes:
        actions: Ordered list of ActionSpecs to execute.
        expected_states: The world state expected after each action.
        total_cost: Sum of action costs along the plan.
        metadata: Planning algorithm statistics.
        score: OptaPlanner-style :class:`~langgoap.score.Score`.  A*-only
            plans carry a :class:`~langgoap.score.SimpleScore` equal to
            ``total_cost``; after CSP evaluation, the pipeline replaces
            this with a :class:`~langgoap.score.HardSoftScore` whose
            ``hard`` level is 0.0 for feasible plans and whose ``soft``
            level aggregates weighted objectives and soft penalties.
    """

    actions: tuple[ActionSpec, ...]
    expected_states: tuple[PlanningState, ...] = ()
    total_cost: float = 0.0
    metadata: PlanMetadata = field(default_factory=PlanMetadata)
    score: Score = field(default_factory=lambda: SimpleScore(scalar=0.0))

    @property
    def action_names(self) -> list[str]:
        """Return the names of all actions in the plan."""
        return [a.name for a in self.actions]

    def __len__(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        return (
            f"Plan(actions={self.action_names!r}, "
            f"total_cost={self.total_cost:.4g}, "
            f"steps={len(self)})"
        )

    @classmethod
    def empty(cls) -> Plan:
        """Create an empty plan (goal already satisfied)."""
        return cls(actions=(), expected_states=(), total_cost=0.0)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_mermaid(
        self,
        *,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str:
        """Render this plan as a Mermaid ``flowchart TD`` diagram.

        Returns:
            Mermaid source code as a plain string.
        """
        from langgoap.viz.mermaid import render_mermaid

        return render_mermaid(
            self,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def to_dot(
        self,
        *,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str:
        """Render this plan as Graphviz DOT source.

        Returns:
            DOT source code as a plain string.
        """
        from langgoap.viz.dot import render_dot

        return render_dot(
            self,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def to_ascii(
        self,
        *,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str:
        """Render this plan as an ASCII tree.

        Returns:
            Plain-text ASCII representation of the plan.
        """
        from langgoap.viz.ascii import render_ascii

        return render_ascii(
            self,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def visualize(
        self,
        *,
        format: Literal[
            "auto", "mermaid", "dot", "ascii", "gantt", "ascii_gantt"
        ] = "auto",
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str | Any:
        """Display-aware visualization helper.

        When IPython is available, ``format="auto"`` returns an
        ``IPython.display.Markdown`` wrapping the Mermaid source so
        Jupyter renders it as a diagram.  When IPython is absent, returns
        the ASCII rendering as a plain string.

        See :func:`langgoap.viz.jupyter.visualize` for the full format
        table.
        """
        from langgoap.viz.jupyter import visualize as _visualize

        return _visualize(
            self,
            format=format,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["mermaid", "dot", "ascii"] | None = None,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> Path:
        """Write a rendered representation of this plan to ``path``.

        The format is inferred from the path suffix when ``format`` is
        ``None``: ``.mmd``/``.mermaid`` → Mermaid, ``.dot``/``.gv`` →
        DOT, ``.txt``/``.ascii`` or anything else → ASCII.

        Args:
            path: Destination file path.
            format: Explicit format override.
            show_resources: Forwarded to the renderer.
            show_schedule: Forwarded to the renderer.

        Returns:
            The ``Path`` that was written.
        """
        dest = Path(path)
        if format is None:
            suffix = dest.suffix.lower().lstrip(".")
            if suffix in {"mmd", "mermaid"}:
                format = "mermaid"
            elif suffix in {"dot", "gv"}:
                format = "dot"
            else:
                format = "ascii"

        if format == "mermaid":
            source = self.to_mermaid(
                show_resources=show_resources, show_schedule=show_schedule
            )
        elif format == "dot":
            source = self.to_dot(
                show_resources=show_resources, show_schedule=show_schedule
            )
        else:
            source = self.to_ascii(
                show_resources=show_resources, show_schedule=show_schedule
            )

        dest.write_text(source, encoding="utf-8")
        return dest
