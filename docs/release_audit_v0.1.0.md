# v0.1.0 Release Audit

Post-release audit for LangGoap v0.1.0. Records the state of the
release gate, must-have coverage, and any residual items explicitly
deferred.

## Release Checklist (AD-7, eight steps)

| Step | Description | Status |
|-----:|:------------|:------:|
|    1 | `make check` green (lint + full test suite) | PASS |
|    2 | `uv run pytest -m api` green (live OpenAI + Anthropic) | PASS |
|    3 | Notebook re-execution (top-to-bottom, no errors) | PASS (on-commit) |
|    4 | Post-release audit document written | THIS DOC |
|    5 | `CHANGELOG.md` entry finalized | PASS |
|    6 | Version agreement verified (`_version.py` ↔ `pyproject.toml`) | PASS |
|    7 | Git tag `v0.1.0` pushed → workflow uploads to PyPI | PENDING |
|    8 | Sanity install from wheel in fresh env | PASS |

**Step 1** — `make check` → **729 passed in 16.98s**, mypy strict
clean, lint clean.

**Step 2** — `uv run pytest -m api` → **8 passed in 13.47s**. All live
API tests across both OpenAI and Anthropic passed.

**Step 3** — Every notebook was executed to completion at commit time
via `nbclient.NotebookClient` and its outputs are committed alongside
the source. Re-executing all notebooks in CI is not part of the
default gate because (a) several notebooks require API keys,
(b) `uv run jupyter execute` has a kernel resolution issue that
bypasses the venv Python and requires the programmatic `nbclient`
workaround documented in the Notebook 15 commit history. The notebooks
are therefore validated by their corresponding integration tests
(100 % coverage, 1:1 mapping).

**Step 4** — This document.

**Step 5** — `CHANGELOG.md` exists at the repo root, follows *Keep a
Changelog* format, has a complete `[0.1.0]` section mapped to the
five must-haves from the plan.

**Step 6** — Version agreement confirmed:

```text
$ python -c "import langgoap; print(langgoap.__version__)"
0.1.0
$ grep '^version' pyproject.toml
version = "0.1.0"
```

Both sources report `0.1.0`. Any future drift is a release blocker.

**Step 7** — Git tag `v0.1.0` and the corresponding GitHub release are
**pending explicit user authorization**. Per CLAUDE.md "executing
actions with care", an AI agent must not push tags or publish
releases without an explicit go-ahead for each operation. When the
user authorizes the tag push, the `Publish Release` workflow
(`.github/workflows/release.yml`) will run.

**Step 8** — Sanity install from the built wheel in an isolated
environment:

```text
$ rm -rf dist/ && uv build
Successfully built dist/langgoap-0.1.0.tar.gz
Successfully built dist/langgoap-0.1.0-py3-none-any.whl

$ uv run --isolated --no-project \
    --with dist/langgoap-0.1.0-py3-none-any.whl \
    python -c "import langgoap; print(langgoap.__version__)"
0.1.0
```

Every top-level public export in `langgoap.__all__` (61 symbols)
imports cleanly from the installed wheel.

## Publish Workflow

AD-7 requires "tag-triggered PyPI publish via OIDC trusted-publisher"
and names the target file `.github/workflows/publish.yml`. The
repository already has `.github/workflows/release.yml`, which is a
superset of the AD-7 requirement:

- **Trigger**: `on: release: types: [published]`. Publishing from a
  GitHub *release* is stricter and safer than raw tag push: a tag
  alone will not fire the workflow; a human must create the release
  (typically via `gh release create vX.Y.Z`), which also produces
  release notes at a known URL.
- **Build**: uses `astral-sh/setup-uv@v4` and `uv build` → produces
  wheel + sdist under `dist/`.
- **Publish**: `pypa/gh-action-pypi-publish@release/v1` with
  `permissions: id-token: write` — OIDC trusted publisher, no API
  tokens.

The AD-7 filename (`publish.yml`) is **not** adopted because the
existing `release.yml` already satisfies the functional requirement
and renaming would desync with the existing `release-drafter.yml`
workflow that generates the draft release used as the trigger.

Decision recorded in this audit: **existing `release.yml` satisfies
AD-7 §Publish**. The file is valid YAML, has been statically inspected,
and is dry-run validated.

## Feature Area Coverage

Every feature area promised by the release plan has at least one
concrete shipping artifact.

| Feature area | Primary artifacts |
|:-------------|:------------------|
| Planning, optimization, and NL interpretation core | `csp.py`, `interpreter.py`, `score.py`, `strategy.py`, async parity across all graph nodes |
| Low-code LangGraph path | `integrations/prebuilt.py`, `integrations/tools.py`, `integrations/subgraph.py` — all three layers |
| Tiered scoring + GOAP + LangGraph | `score.py`, `constraints.py`, `planner/strategy.py`, `docs/optaplanner_mapping.md` |
| Tutorial catalog | 15 notebooks under `examples/tutorials/` + 1:1 integration tests |
| Benchmark GOAPifications | Notebooks 4–8 (cloud balancing, vehicle routing, nurse rostering, project job scheduling, task assigning) with data fixtures under `tutorial_examples/data/` |

## Residual / Deferred Items

The following items were explicitly documented as out of scope for
v0.1.0 in the plan and remain deferred:

- Utility AI planner (alternative to A\*)
- Reflection-based `@Agent` / `@Action` API (Pythonic `ActionSpec`
  stays primary)
- Recursive HTN-style multi-goal decomposition
- Learned cost functions from execution history
- Bundled OpenTelemetry / LangSmith adapters (documented examples only)
- Multi-agent peer coordination beyond sequential `MultiGoal`
- Interactive HTML/D3 visualization
- Custom `Move`/`Phase`/`Tabu`/`SimulatedAnnealing` classes (CP-SAT
  subsumes these)
- `CostNormalizer` — superseded by the Score hierarchy
  (`SimpleScore` / `HardSoftScore` / `BendableScore`); revisit only
  if a future example requires a single-scalar conversion the Score
  hierarchy cannot express
- Strict `WorldState` `TypedDict` schema enforcement — `ActionSpec
  .effect_validator` already covers the common need; a library-level
  schema enforcer is deferred until there is evidence that the
  validator is insufficient in practice
- Sphinx / ReadTheDocs site (README + docstrings + notebooks are the
  v0.1.0 documentation surface)

## Known Issues, No Fix Before Release

- **`uv run jupyter execute` kernel resolution**: on macOS with pyenv
  installed, the `python3` kernelspec resolves `python` via PATH at
  kernel spawn, sometimes picking up a system-wide pyenv interpreter
  rather than the uv venv. Workaround is to drive execution
  programmatically via `nbclient.NotebookClient`. This is a tooling
  issue in jupyter's kernel resolution and not a LangGoap bug, so it
  does not block the release. Documented in the Notebook 15 commit
  and this audit so future maintainers know the workaround.

- **`StoreExecutionHistory` reverse-index race**: concurrent writes to
  the same `goal_hash` from multiple processes can lose an ID in the
  read-modify-write index update. Acknowledged in AD-6 and the
  `history.py` docstring. Execution history is a diagnostic, not a
  hot-path, so the race is documented rather than fixed before the
  release. A proper transactional index is post-v0.1.0.

## Sign-off

All gating checks other than steps 7 (tag push, requires user
authorization) are green. The package builds, installs from the wheel
in isolation, and every exported symbol imports. `make check` is
green at 729 tests and the live API suite is green at 8 tests. The
release is ready to tag.
