"""``docs/copy_notebooks.py`` rewrites relative links for the docs build (#66).

Notebooks link to their backing tests and to sibling files with paths that
are valid on GitHub (``../../tests/integration/test_x.py``) but not inside
``docs/examples/``.  The copier turns those into absolute GitHub URLs and
leaves links between rendered notebooks relative, so Sphinx resolves them.
"""

from __future__ import annotations

from pathlib import Path

from docs.copy_notebooks import EXAMPLES_SRC, GITHUB_BLOB, rewrite_relative_links

TUTORIALS = EXAMPLES_SRC / "tutorials"


def test_backing_test_links_become_github_urls() -> None:
    text = "See [the test](../../tests/integration/test_form_hitl.py) for details."
    out = rewrite_relative_links(text, TUTORIALS)
    assert (
        out
        == f"See [the test]({GITHUB_BLOB}tests/integration/test_form_hitl.py) for details."
    )


def test_links_between_rendered_notebooks_stay_relative() -> None:
    text = "Primer: [stuck handlers](../basics/stuck_handlers.ipynb)."
    assert rewrite_relative_links(text, TUTORIALS) == text


def test_sibling_markdown_files_become_github_urls() -> None:
    src = EXAMPLES_SRC / "screencast" / "research_agent"
    text = "Read the [script](script.md) and [narrative](narrative.md)."
    out = rewrite_relative_links(text, src)
    assert f"{GITHUB_BLOB}examples/screencast/research_agent/script.md" in out
    assert f"{GITHUB_BLOB}examples/screencast/research_agent/narrative.md" in out


def test_directories_become_github_tree_urls() -> None:
    text = "Scripts in [the incident folder](../screencast/incident/)."
    out = rewrite_relative_links(text, TUTORIALS)
    assert (
        out
        == f"Scripts in [the incident folder]({GITHUB_BLOB}examples/screencast/incident/)."
    )


def test_absolute_anchor_and_missing_targets_are_untouched() -> None:
    text = (
        "[a](https://example.com/x) [b](#section) [c](mailto:x@y.z) "
        "[d](../../does/not/exist.py) [e](../basics/stuck_handlers.ipynb#cell)"
    )
    assert rewrite_relative_links(text, TUTORIALS) == text


def test_anchor_survives_rewriting(tmp_path: Path) -> None:
    text = "[t](../../tests/test_cli.py#L10)"
    assert (
        rewrite_relative_links(text, TUTORIALS)
        == f"[t]({GITHUB_BLOB}tests/test_cli.py#L10)"
    )
