#!/usr/bin/env python3
"""Copy example notebooks into docs/examples/ for Sphinx rendering.

This script is idempotent — safe to run multiple times. It copies
the notebooks listed in :data:`CATALOG` from the project's
``examples/`` directory into ``docs/examples/`` subdirectories so
``myst_nb`` can render them during the Sphinx build.

Only notebooks explicitly listed in :data:`CATALOG` are copied.
Notebooks that live in ``examples/tutorials/`` but are not part of
the shipped v0.1.0 catalog (LangGraph reference originals, exploratory
drafts, etc.) are intentionally excluded so the Sphinx toctree stays
aligned with ``docs/examples/index.md``.

Usage:
    python docs/copy_notebooks.py
"""
import shutil
from pathlib import Path

# Resolve paths relative to this script's location
DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = PROJECT_ROOT / "examples"
EXAMPLES_DST = DOCS_DIR / "examples"

# Explicit catalog: only these notebooks are copied into the docs
# build.  Anything in examples/ that is not listed here is excluded.
CATALOG: dict[str, list[str]] = {
    "basics": [
        "plan_visualization.ipynb",
        "nl_goal_interpreter.ipynb",
    ],
    "tutorials": [
        # Tier 1 — Primers
        "directory_handler.ipynb",
        "robot_navigation.ipynb",
        "hungry_agent.ipynb",
        # Tier 2 — Constraint optimization + workflow agents
        "cloud_balancing.ipynb",
        "vehicle_routing.ipynb",
        "nurse_rostering.ipynb",
        "project_job_scheduling.ipynb",
        "task_assigning.ipynb",
        "sql_query_agent.ipynb",
        "vulnerability_scanner.ipynb",
        # Tier 3 — Full-feature showcases
        "deep_research_agent.ipynb",
        "hierarchical_product_launch.ipynb",
        "content_builder_agent.ipynb",
        "temporal_match_cellar.ipynb",
        "flexible_job_shop.ipynb",
    ],
}


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
            shutil.copy2(src_file, dst_file)
            copied += 1
            print(f"  {src_file.relative_to(PROJECT_ROOT)} -> {dst_file.relative_to(DOCS_DIR)}")

    print(f"\nCopied {copied} notebooks.")
    if missing:
        raise SystemExit(
            "Catalog notebooks missing on disk:\n  " + "\n  ".join(missing)
        )


if __name__ == "__main__":
    print("Copying example notebooks into docs/examples/...\n")
    copy_notebooks()
