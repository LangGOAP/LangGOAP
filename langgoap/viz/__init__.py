"""Plan visualization utilities.

Pure-Python renderers for :class:`~langgoap.planner.types.Plan` objects.
Produces Mermaid ``flowchart TD`` diagrams, Graphviz DOT graphs, ASCII
tree layouts, and schedule Gantt charts.

Design notes:

* Every renderer returns a plain ``str``.  No IPython coupling happens in
  the renderer modules themselves — the display-aware wrapper lives in
  :mod:`langgoap.viz.jupyter`.
* Dependency edges come from :func:`langgoap.planner.csp.build_dependency_graph`
  so what the visualizer draws is exactly what the scheduler sees.
* Schedule overlays and resource bars are rendered only when the plan
  carries :class:`~langgoap.planner.csp.CSPMetadata`.
"""

from __future__ import annotations

from langgoap.viz.ascii import render_ascii, render_ascii_gantt
from langgoap.viz.dot import render_dot
from langgoap.viz.jupyter import visualize
from langgoap.viz.mermaid import render_mermaid, render_mermaid_gantt

__all__ = [
    "render_mermaid",
    "render_mermaid_gantt",
    "render_dot",
    "render_ascii",
    "render_ascii_gantt",
    "visualize",
]
