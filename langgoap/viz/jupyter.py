"""Display-aware visualization helper for Jupyter environments.

The helper detects IPython at call time and returns an
``IPython.display.Markdown`` or ``IPython.display.Image`` instance when
available so notebooks can render rich output; otherwise it returns the
raw string.  Keeping the detection at call time (rather than at import
time) lets the core package stay a zero-weight dependency on IPython.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Literal

from langgoap.viz.ascii import render_ascii, render_ascii_gantt
from langgoap.viz.dot import render_dot
from langgoap.viz.mermaid import render_mermaid, render_mermaid_gantt

if TYPE_CHECKING:
    from langgoap.planner.types import Plan

logger = logging.getLogger(__name__)

VizFormat = Literal["auto", "mermaid", "dot", "ascii", "gantt", "ascii_gantt"]


def _ipython_available() -> bool:
    """Return True when IPython is importable in the current environment."""
    return importlib.util.find_spec("IPython") is not None


def visualize(
    plan: Plan,
    *,
    format: VizFormat = "auto",
    show_resources: bool = True,
    show_schedule: bool = True,
) -> str | Any:
    """Render a Plan for display.

    Return types:

    * ``format="ascii"`` or ``format="ascii_gantt"`` → always a ``str``.
    * ``format="mermaid"`` → ``IPython.display.Markdown`` if IPython is
      importable, otherwise a plain ``str``.
    * ``format="gantt"`` → same detection rules as ``"mermaid"``.
    * ``format="dot"`` → ``IPython.display.Image`` when IPython *and* the
      ``graphviz`` Python package with the ``dot`` binary are available.
      Otherwise returns the DOT source as ``str``.
    * ``format="auto"`` → ``"mermaid"`` when IPython is available,
      ``"ascii"`` otherwise.

    Args:
        plan: Plan to render.
        format: Output format selector.
        show_resources: Forwarded to the underlying renderer.
        show_schedule: Forwarded to the underlying renderer.

    Returns:
        Rendered output; exact type depends on ``format`` and the
        presence of IPython in the current environment.
    """
    if format == "auto":
        format = "mermaid" if _ipython_available() else "ascii"

    if format == "ascii":
        return render_ascii(
            plan, show_resources=show_resources, show_schedule=show_schedule
        )

    if format == "ascii_gantt":
        return render_ascii_gantt(plan)

    if format == "mermaid":
        source = render_mermaid(
            plan, show_resources=show_resources, show_schedule=show_schedule
        )
        if _ipython_available():
            from IPython.display import Markdown

            return Markdown(f"```mermaid\n{source}\n```")
        return source

    if format == "gantt":
        source = render_mermaid_gantt(plan)
        if _ipython_available():
            from IPython.display import Markdown

            return Markdown(f"```mermaid\n{source}\n```")
        return source

    if format == "dot":
        source = render_dot(
            plan, show_resources=show_resources, show_schedule=show_schedule
        )
        if not _ipython_available():
            return source
        try:
            import graphviz
        except ImportError:
            logger.warning(
                "graphviz Python package not available; returning DOT source."
            )
            return source
        try:
            src = graphviz.Source(source)
            png_bytes = src.pipe(format="png")
        except Exception as exc:  # pragma: no cover — environment-dependent
            logger.warning("graphviz render failed (%s); returning DOT source.", exc)
            return source
        from IPython.display import Image

        return Image(png_bytes)

    raise ValueError(f"Unknown visualization format: {format!r}")
