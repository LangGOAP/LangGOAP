"""The API reference is generated from ``langgoap.__all__`` (issue #66).

``docs/api/index.md`` used to be a hand-written list that named 63 of the
146 public symbols and called itself the "v0.1.0 surface".  It is now the
output of ``docs/generate_api_index.py``, which reads the grouping comments
inside ``langgoap/__init__.py``'s ``__all__``; these tests fail when the
committed page and the generator disagree, or when a public name is not
documented.
"""

from __future__ import annotations

import re
from pathlib import Path

import langgoap
from docs.generate_api_index import INIT_FILE, TARGET, parse_groups, render

ROOT = Path(__file__).resolve().parents[1]


def _documented(markdown: str) -> set[str]:
    """Names on autosummary lines plus constants listed as bullets."""
    summary = re.findall(r"^\s+langgoap\.([A-Za-z_][A-Za-z0-9_]*)$", markdown, re.M)
    constants = re.findall(r"^- `langgoap\.([A-Z][A-Z0-9_]*)`", markdown, re.M)
    return set(summary) | set(constants)


class TestGeneratedApiIndex:
    def test_committed_page_matches_the_generator(self) -> None:
        expected = render(parse_groups(INIT_FILE.read_text()))
        assert TARGET.read_text() == expected, "run `python docs/generate_api_index.py`"

    def test_every_public_name_is_documented_once(self) -> None:
        markdown = TARGET.read_text()
        public = set(langgoap.__all__) - {"__version__"}
        assert _documented(markdown) == public
        entries = re.findall(r"^\s+langgoap\.\w+$", markdown, re.M)
        assert len(entries) == len(set(entries))

    def test_constants_are_listed_not_autosummarised(self) -> None:
        """Instances such as ``FIRE_ONCE`` would duplicate their class's attribute
        descriptions on a stub page, so they are listed with their type instead."""
        markdown = TARGET.read_text()
        assert "- `langgoap.FIRE_ONCE` (`ActionQos`)" in markdown
        assert "- `langgoap.NIRVANA_NAME` (`str`)" in markdown
        assert not re.search(r"^\s+langgoap\.[A-Z][A-Z0-9_]*$", markdown, re.M)

    def test_groups_follow_the_comment_headers(self) -> None:
        groups = parse_groups(INIT_FILE.read_text())
        titles = [title for title, _ in groups]
        assert titles[0] == "Package"
        assert "Planning strategies" in titles
        assert all(names for _, names in groups)

    def test_no_stale_version_claim(self) -> None:
        assert "v0.1.0" not in TARGET.read_text()
