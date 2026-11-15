"""The README's MFCQI badge must show a number CI measured.

Until v0.2.0 the badge read a hand-maintained ``.github/badges/mfcqi.json``
that said ``0.90`` while CI measured ``0.81``.  The badge now comes from the
same gist as the coverage badge, written on every push to ``main`` by the
badge workflow from ``mfcqi badge langgoap`` output.  These tests pin that
wiring so the number cannot silently go stale again.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
WORKFLOW = ROOT / ".github" / "workflows" / "coverage-gist.yml"
STATIC_BADGE = ROOT / ".github" / "badges" / "mfcqi.json"
BADGE_ACTION = "schneegans/dynamic-badges-action"
GIST_RAW = re.compile(
    r"https://gist\.githubusercontent\.com/(?P<user>[^/]+)/(?P<gist>[0-9a-f]+)/raw/(?P<file>[\w.-]+)"
)


def _badge_steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    (job,) = workflow["jobs"].values()
    return job["steps"]


def _gist_badges() -> dict[str, tuple[str, str]]:
    """Map README badge ``alt`` text to the (gist id, file) it reads."""
    found: dict[str, tuple[str, str]] = {}
    for match in re.finditer(r'<img src="([^"]+)" alt="([^"]+)"', README.read_text()):
        src, alt = match.groups()
        gist = GIST_RAW.search(src)
        if gist:
            found[alt] = (gist["gist"], gist["file"])
    return found


class TestNoHandMaintainedBadge:
    def test_static_badge_file_is_gone(self) -> None:
        assert not STATIC_BADGE.exists(), "the MFCQI badge must not be a committed file"

    def test_readme_reads_no_badge_from_the_repository_tree(self) -> None:
        assert ".github/badges/" not in README.read_text()


class TestReadmeBadge:
    def test_mfcqi_badge_reads_the_coverage_gist(self) -> None:
        badges = _gist_badges()
        assert badges["MFCQI Score"][1] == "langgoap-mfcqi.json"
        assert (
            badges["MFCQI Score"][0] == badges["Coverage"][0]
        ), "both badges share one gist"


class TestBadgeWorkflow:
    def test_mfcqi_badge_is_published_from_mfcqi_output(self) -> None:
        steps = _badge_steps()
        (compute,) = [s for s in steps if s.get("id") == "mfcqi"]
        assert "python -m mfcqi badge langgoap -f json" in compute["run"]
        (publish,) = [
            s
            for s in steps
            if s.get("uses", "").startswith(BADGE_ACTION)
            and s["with"]["filename"] == "langgoap-mfcqi.json"
        ]
        assert publish["with"]["label"] == "MFCQI"
        assert publish["with"]["message"] == "${{ steps.mfcqi.outputs.message }}"
        assert publish["with"]["color"] == "${{ steps.mfcqi.outputs.color }}"

    def test_mfcqi_badge_is_published_before_the_test_suite_runs(self) -> None:
        """A failing test must not leave a stale quality number on the README."""
        names = [s.get("name", "") for s in _badge_steps()]
        assert names.index("Create MFCQI Badge") < names.index(
            "Run tests with coverage"
        )

    def test_coverage_badge_still_published(self) -> None:
        steps = _badge_steps()
        filenames = {
            s["with"]["filename"]
            for s in steps
            if s.get("uses", "").startswith(BADGE_ACTION)
        }
        assert filenames == {"langgoap-coverage.json", "langgoap-mfcqi.json"}


class TestMfcqiBadgeContract:
    def test_badge_json_has_the_fields_the_workflow_reads(self, tmp_path: Path) -> None:
        """``mfcqi badge -f json`` must keep emitting ``message`` and ``color``."""
        pytest.importorskip("mfcqi")
        pkg = tmp_path / "tiny"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            '"""Tiny package."""\n\n\ndef add(a: int, b: int) -> int:\n    """Add."""\n    return a + b\n'
        )
        out = tmp_path / "badge.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mfcqi",
                "badge",
                str(tmp_path),
                "-f",
                "json",
                "-o",
                str(out),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        badge = json.loads(out.read_text())
        assert badge["schemaVersion"] == 1
        assert re.fullmatch(r"\d\.\d\d( \(\w+\))?", badge["message"]), badge["message"]
        assert isinstance(badge["color"], str) and badge["color"]
