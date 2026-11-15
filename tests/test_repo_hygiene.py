"""Catalog and stale-text guards from the v0.2.0 audit (issue #67).

Each test pins one fact that had drifted: the notebook catalogs agree with
the files on disk, the CHANGELOG dates released versions, removed
mechanisms are no longer documented, shipped text does not cite the
maintainer's private ``research/`` tree, and the copyright year matches
the repository's first commit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from docs.copy_notebooks import CATALOG

ROOT = Path(__file__).resolve().parents[1]
#: Sub-paths of the maintainer's private, gitignored notes tree.
_PRIVATE_TREE = re.compile(r"\bresearch/(?:repos|plans|experiments|papers)/")
EXAMPLES = ROOT / "examples"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _text_files(*roots: str) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        for path in (ROOT / root).rglob("*"):
            if (
                path.suffix in {".py", ".md", ".ipynb", ".toml", ".yml", ".yaml"}
                and path.is_file()
            ):
                if ".ipynb_checkpoints" in path.parts or "_build" in path.parts:
                    continue
                out.append(path)
    return out


class TestNotebookCatalog:
    def test_every_basics_notebook_is_rendered(self) -> None:
        on_disk = sorted(p.name for p in (EXAMPLES / "basics").glob("*.ipynb"))
        assert sorted(CATALOG["basics"]) == on_disk

    def test_examples_page_lists_and_toctrees_every_catalog_notebook(self) -> None:
        page = _read("docs/examples/index.md")
        for subdir, names in CATALOG.items():
            for name in names:
                assert f"`{name}`" in page or f"`{subdir}/{name}`" in page, name
                assert f"{subdir}/{name[:-len('.ipynb')]}" in page, name

    def test_docs_index_counts_match_the_catalog(self) -> None:
        match = re.search(
            r"(\d+) tutorial notebooks and\s+(\d+) basics notebooks",
            _read("docs/index.md"),
        )
        assert match is not None
        assert (int(match.group(1)), int(match.group(2))) == (
            len(CATALOG["tutorials"]),
            len(CATALOG["basics"]),
        )

    def test_examples_readme_lists_every_basics_notebook(self) -> None:
        readme = _read("examples/README.md")
        for path in (EXAMPLES / "basics").glob("*.ipynb"):
            assert f"basics/{path.name}" in readme, path.name

    def test_incident_screencast_notebook_is_referenced(self) -> None:
        assert "incident_response_screencast.ipynb" in _read(
            "examples/screencast/incident/README.md"
        )

    def test_notebooks_use_the_standard_kernel(self) -> None:
        for path in EXAMPLES.rglob("*.ipynb"):
            if ".ipynb_checkpoints" in path.parts:
                continue
            kernel = json.loads(path.read_text())["metadata"].get("kernelspec", {})
            assert kernel.get("name") == "python3", path

    def test_catalog_docstring_is_version_neutral(self) -> None:
        assert "v0.1.0" not in _read("docs/copy_notebooks.py")


class TestChangelog:
    def test_released_versions_are_dated(self) -> None:
        changelog = _read("CHANGELOG.md")
        assert "## [0.1.2] — 2026-05-15" in changelog
        assert "## [0.1.1] — 2026-05-15" in changelog
        assert changelog.count("Unreleased") == 1


class TestStaleText:
    def test_no_private_research_paths_in_shipped_text(self) -> None:
        offenders = [
            str(path.relative_to(ROOT))
            for path in _text_files("langgoap", "tests", "examples")
            if _PRIVATE_TREE.search(path.read_text(errors="ignore"))
        ]
        assert offenders == []

    def test_live_demo_env_gate_is_gone(self) -> None:
        offenders = [
            str(path.relative_to(ROOT))
            for path in _text_files("tests", "examples")
            if path != Path(__file__)
            and "LANGGOAP_RUN_LIVE_DEMO" in path.read_text(errors="ignore")
        ]
        assert offenders == []

    def test_issue_template_links_only_existing_files(self) -> None:
        config = _read(".github/ISSUE_TEMPLATE/config.yml")
        for name in re.findall(r"\b([A-Z_]+\.md)\b", config):
            assert (ROOT / name).exists(), name

    @pytest.mark.parametrize("rel", ["LICENSE", "docs/conf.py"])
    def test_copyright_year_is_the_first_commit_year(self, rel: str) -> None:
        assert "2026" in _read(rel)
        assert "2024" not in _read(rel)

    def test_contributing_clone_step_uses_the_repository_name(self) -> None:
        assert "cd LangGOAP\n" in _read("CONTRIBUTING.md")
        assert "cd langgoap\n" not in _read("CONTRIBUTING.md")

    def test_env_example_spells_the_project_name(self) -> None:
        assert "LangGoap" not in _read(".env.example")

    def test_jupyter_image_does_not_install_unused_langchain(self) -> None:
        assert '"langchain>=' not in _read("examples/Dockerfile.jupyter")

    def test_bench_marker_is_gone(self) -> None:
        assert "bench:" not in _read("pyproject.toml")
        assert not any(
            "pytest.mark.bench" in p.read_text()
            for p in (ROOT / "tests/benchmarks").glob("*.py")
        )
