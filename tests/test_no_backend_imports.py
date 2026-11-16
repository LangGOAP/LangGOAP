"""The library must not import any checkpoint backend (issue #17).

Backends are installed through pass-through extras (``langgoap[checkpoint-
redis]`` and friends) and talk to LangGOAP only through LangGraph's
``BaseCheckpointSaver`` / ``SerializerProtocol`` interfaces.  Any import of
a concrete backend, or of a serializer implementation detail such as
``ormsgpack``, inside ``langgoap/`` is a regression.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import langgoap

PACKAGE = Path(langgoap.__file__).resolve().parent  # the checkout or the wheel

FORBIDDEN = [
    r"^\s*(from|import)\s+langgraph\.checkpoint\.(redis|postgres|sqlite|mongodb)",
    r"^\s*(from|import)\s+ormsgpack\b",
    r"^\s*(from|import)\s+redis\b",
    r"^\s*(from|import)\s+psycopg\b",
    r"^\s*(from|import)\s+asyncpg\b",
]


def _source_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


@pytest.mark.parametrize("pattern", FORBIDDEN)
def test_no_backend_specific_imports(pattern: str) -> None:
    offenders: list[str] = []
    regex = re.compile(pattern, re.MULTILINE)
    for path in _source_files():
        for match in regex.finditer(path.read_text(encoding="utf-8")):
            line_no = path.read_text(encoding="utf-8").count("\n", 0, match.start()) + 1
            offenders.append(
                f"{path.relative_to(PACKAGE.parent)}:{line_no}: {match.group(0).strip()}"
            )
    assert not offenders, "backend-specific imports found:\n" + "\n".join(offenders)


def test_package_has_source_files() -> None:
    assert len(_source_files()) > 10
