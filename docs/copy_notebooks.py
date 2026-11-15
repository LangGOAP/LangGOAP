#!/usr/bin/env python3
"""Copy example notebooks into docs/examples/ for Sphinx rendering.

This script is idempotent — safe to run multiple times. It copies
the notebooks listed in :data:`CATALOG` from the project's
``examples/`` directory into ``docs/examples/`` subdirectories so
``myst_nb`` can render them during the Sphinx build.

Only notebooks explicitly listed in :data:`CATALOG` are copied.
Notebooks that live in ``examples/tutorials/`` but are not part of
the shipped catalog (LangGraph reference originals, exploratory
drafts, etc.) are intentionally excluded so the Sphinx toctree stays
aligned with ``docs/examples/index.md``.

Usage:
    python docs/copy_notebooks.py
"""

import json
import re
from pathlib import Path

# Resolve paths relative to this script's location
DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = PROJECT_ROOT / "examples"
EXAMPLES_DST = DOCS_DIR / "examples"

#: Where relative links that leave the rendered notebook tree point instead.
GITHUB_BLOB = "https://github.com/LangGOAP/LangGOAP/blob/main/"

_LINK = re.compile(r"\]\(([^)\s#]+)(#[^)\s]*)?\)")

# Explicit catalog: only these notebooks are copied into the docs
# build.  Anything in examples/ that is not listed here is excluded.
CATALOG: dict[str, list[str]] = {
    "basics": [
        "plan_visualization.ipynb",
        "nl_goal_interpreter.ipynb",
        "create_goap_agent_quickstart.ipynb",
        "termination_policies.ipynb",
        "stuck_handlers.ipynb",
        "typed_form_hitl.ipynb",
        "action_qos.ipynb",
        "cli.ipynb",
        "goap_subgraph.ipynb",
        "tracing.ipynb",
        "stochastic_gridworld.ipynb",
        "mcts_vs_astar_stochastic.ipynb",
    ],
    "tutorials": [
        # Tier 1 — Primers
        "directory_handler.ipynb",
        "robot_navigation.ipynb",
        "hungry_agent.ipynb",
        # Tier 2 — Workflows and constraint optimization
        "cloud_balancing.ipynb",
        "vehicle_routing.ipynb",
        "nurse_rostering.ipynb",
        "project_job_scheduling.ipynb",
        "task_assigning.ipynb",
        "sql_query_agent.ipynb",
        "vulnerability_scanner.ipynb",
        "cost_bounded_research_agent.ipynb",
        "personal_shopper_agent.ipynb",
        "scheduled_delivery_confirmer.ipynb",
        "deepagents_integration.ipynb",
        # Tier 3 — Full-stack showcases
        "deep_research_agent.ipynb",
        "hierarchical_product_launch.ipynb",
        "content_builder_agent.ipynb",
        "temporal_match_cellar.ipynb",
        "flexible_job_shop.ipynb",
        "supply_chain_disruption_mediator.ipynb",
        "code_review_agent_mcp_deployment.ipynb",
        "from_routing_graphs_to_goap.ipynb",
    ],
    "screencast/research_agent": [
        "screencast.ipynb",
    ],
    "screencast/llm_compiler": [
        "llm_compiler_screencast.ipynb",
    ],
    "screencast/email_assistant": [
        "email_assistant_screencast.ipynb",
    ],
    "screencast/weather_alert": [
        "weather_alert_screencast.ipynb",
    ],
}


def _is_rendered_notebook(path: Path) -> bool:
    """True when ``path`` is a catalog notebook, i.e. it exists in the docs tree too."""
    try:
        rel = path.relative_to(EXAMPLES_SRC)
    except ValueError:
        return False
    return rel.name in CATALOG.get(rel.parent.as_posix(), [])


def rewrite_relative_links(markdown: str, notebook_dir: Path) -> str:
    """Point relative links that leave the rendered notebook tree at GitHub.

    A notebook under ``examples/`` may link to its backing test
    (``../../tests/integration/test_x.py``), to a sibling file
    (``script.md``) or to a directory (``../screencast/incident/``).
    Those paths are valid on GitHub but not inside ``docs/examples/``, so
    the copy rendered by Sphinx gets absolute ``GITHUB_BLOB`` URLs
    (issue #66).  Links to other catalog notebooks stay relative, because
    the docs tree mirrors ``examples/`` for them; absolute URLs, anchors
    and targets that do not exist are left alone.
    """

    def _replace(match: re.Match[str]) -> str:
        target, anchor = match.group(1), match.group(2) or ""
        if target.startswith(("http://", "https://", "mailto:")):
            return match.group(0)
        resolved = (notebook_dir / target).resolve()
        if not resolved.exists() or _is_rendered_notebook(resolved):
            return match.group(0)
        try:
            rel = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return match.group(0)
        if target.endswith("/"):
            rel += "/"
        return f"]({GITHUB_BLOB}{rel}{anchor})"

    return _LINK.sub(_replace, markdown)


def _copy_notebook(src_file: Path, dst_file: Path) -> None:
    """Copy one notebook, rewriting the links in its Markdown cells."""
    notebook = json.loads(src_file.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell["source"])
        cell["source"] = rewrite_relative_links(text, src_file.parent).splitlines(
            keepends=True
        )
    dst_file.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def copy_notebooks() -> None:
    """Copy every notebook in :data:`CATALOG` into ``docs/examples/``."""
    copied = 0
    missing: list[str] = []

    for src_subdir, names in CATALOG.items():
        src_path = EXAMPLES_SRC / src_subdir
        dst_path = EXAMPLES_DST / src_subdir
        dst_path.mkdir(parents=True, exist_ok=True)

        # Remove stale notebooks from previous runs so excluded
        # notebooks never linger in the docs build tree.
        for stale in dst_path.glob("*.ipynb"):
            if stale.name not in names:
                stale.unlink()

        for name in names:
            src_file = src_path / name
            if not src_file.exists():
                missing.append(str(src_file.relative_to(PROJECT_ROOT)))
                continue
            dst_file = dst_path / name
            _copy_notebook(src_file, dst_file)
            copied += 1
            print(
                f"  {src_file.relative_to(PROJECT_ROOT)} -> {dst_file.relative_to(DOCS_DIR)}"
            )

    print(f"\nCopied {copied} notebooks.")
    if missing:
        raise SystemExit(
            "Catalog notebooks missing on disk:\n  " + "\n  ".join(missing)
        )


if __name__ == "__main__":
    print("Copying example notebooks into docs/examples/...\n")
    copy_notebooks()
