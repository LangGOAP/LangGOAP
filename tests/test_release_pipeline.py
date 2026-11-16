"""Release mechanics must agree with the package (audit of the v0.2.0 audit).

The version moved to ``langgoap/_version.py`` (``dynamic = ["version"]``),
but the release workflows still read ``[project].version`` from
``pyproject.toml`` and would have failed on the first job of the first
release.  These tests pin the pieces a tag depends on: where the workflows
read the version, that the version file and the CHANGELOG agree, that the
release notes call out breaking changes, and that the release drafter
follows a pre-1.0 versioning policy.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]

import langgoap
from langgoap import __version__

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _top_changelog_heading() -> re.Match[str]:
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\] — (.+)$", _read("CHANGELOG.md"), re.M)
    assert match is not None, "CHANGELOG has no versioned heading"
    return match


class TestReleaseWorkflowsReadTheVersionFile:
    def test_no_workflow_reads_a_static_pyproject_version(self) -> None:
        for name in ("release.yml", "test-release.yml"):
            text = (WORKFLOWS / name).read_text(encoding="utf-8")
            assert "['project']['version']" not in text, name
            assert "langgoap/_version.py" in text, name

    def test_pyproject_declares_the_version_dynamic(self) -> None:
        assert 'dynamic = ["version"]' in _read("pyproject.toml")


class TestVersionAndChangelogAgree:
    def test_version_file_matches_the_top_changelog_entry(self) -> None:
        heading = _top_changelog_heading()
        assert heading.group(1) == __version__

    def test_top_entry_is_dated_or_unreleased(self) -> None:
        marker = _top_changelog_heading().group(2)
        assert marker == "Unreleased" or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", marker
        ), marker

    def test_a_dated_release_with_breaking_changes_says_so(self) -> None:
        changelog = _read("CHANGELOG.md")
        start = changelog.index("## [0.2.0]")
        end = changelog.index("\n## [", start + 1)
        section = changelog[start:end]
        assert "### Breaking changes" in section
        assert "### Removed" in section
        assert (
            "LangGOAPRedisSerializer"
            in section.split("### Removed", 1)[1].split("###", 1)[0]
        )

    def test_jupyter_image_installs_the_current_minor(self) -> None:
        pin = re.search(r"langgoap>=(\d+\.\d+)", _read("examples/Dockerfile.jupyter"))
        assert pin is not None
        assert __version__.startswith(pin.group(1) + ".")

    def test_no_stale_release_names_in_package_text(self) -> None:
        offenders = [
            str(p.relative_to(ROOT))
            for p in Path(langgoap.__file__).resolve().parent.rglob("*.py")
            if "v0.1.0" in p.read_text(encoding="utf-8")
        ]
        assert offenders == []


class TestReleaseDrafterPolicy:
    def test_breaking_changes_bump_the_minor_before_1_0(self) -> None:
        config = yaml.safe_load(_read(".github/release-drafter-config.yml"))
        resolver = config["version-resolver"]
        assert "breakingchange" in resolver["minor"]["labels"]
        assert "breakingchange" not in resolver.get("major", {}).get("labels", [])

    def test_documentation_label_has_a_category(self) -> None:
        config = yaml.safe_load(_read(".github/release-drafter-config.yml"))
        labelled = {
            label
            for cat in config["categories"]
            for label in (cat.get("labels") or [cat.get("label")])
        }
        assert "documentation" in labelled

    def test_autolabeler_runs_on_pull_requests(self) -> None:
        workflow = yaml.safe_load(_read(".github/workflows/release-drafter.yml"))
        triggers = workflow.get(True) or workflow.get(
            "on"
        )  # PyYAML parses bare `on` as True
        assert "pull_request_target" in triggers or "pull_request" in triggers


def _autolabels(config: dict, title: str) -> set[str]:
    """Apply the drafter's title rules the way release-drafter does."""
    labels: set[str] = set()
    for rule in config["autolabeler"]:
        for pattern in rule.get("title", []):
            body, _, flags = pattern.rpartition("/")
            body = body[1:]  # strip the leading slash of ``/.../i``
            if re.search(body, title, re.IGNORECASE if "i" in flags else 0):
                labels.add(rule["label"])
    return labels


class TestAutolabelerLabelsEachPullRequestOnce:
    """A pull request shows up under exactly one heading of the release notes.

    release-drafter lists a PR under every category whose label it carries,
    so a title rule that fires alongside another, or a ``files`` rule that
    marks every docs PR as maintenance, duplicates the entry.
    """

    def test_no_rule_is_keyed_on_files(self) -> None:
        config = yaml.safe_load(_read(".github/release-drafter-config.yml"))
        assert [rule for rule in config["autolabeler"] if "files" in rule] == []

    def test_conventional_titles_get_exactly_one_label(self) -> None:
        config = yaml.safe_load(_read(".github/release-drafter-config.yml"))
        expected = {
            "docs(examples): LangGraph tutorial refactors": {"documentation"},
            "fix!: resolve the audit P0 and P1 defects": {"breakingchange"},
            "feat(graph)!: drop the legacy serializer": {"breakingchange"},
            "feat(graph): compact execution-history snapshots": {"enhancement"},
            "fix(hitl,state,types): least-surprise fixes": {"bug"},
            "release: prepare v0.2.0": {"maintenance"},
            "build(deps): bump the actions group": {"maintenance"},
            "chore(ci): refresh the lock": {"maintenance"},
            "test(mcts): disable the wall-clock cap": {"maintenance"},
        }
        got = {title: _autolabels(config, title) for title in expected}
        assert got == expected

    def test_every_autolabel_has_a_category(self) -> None:
        config = yaml.safe_load(_read(".github/release-drafter-config.yml"))
        categorised = {
            label
            for cat in config["categories"]
            for label in (cat.get("labels") or [cat.get("label")])
        }
        assert {rule["label"] for rule in config["autolabeler"]} <= categorised
