"""Tests for the public API surface of LangGOAP."""

from __future__ import annotations

import langgoap


class TestPublicAPI:
    def test_all_symbols_importable(self) -> None:
        """Every symbol in __all__ is importable from the top-level package."""
        for name in langgoap.__all__:
            obj = getattr(langgoap, name)
            assert obj is not None, f"{name} is None"

    def test_version_available(self) -> None:
        """__version__ is a non-empty string."""
        assert isinstance(langgoap.__version__, str)
        assert len(langgoap.__version__) > 0
        assert langgoap.__version__ == "0.1.1"
