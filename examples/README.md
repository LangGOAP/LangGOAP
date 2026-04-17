# LangGOAP Notebooks

This directory contains Jupyter notebooks that serve as **runnable
documentation** for LangGOAP. They come in two flavours:

- **Basics** (`basics/`) — short primers that each demonstrate a single
  mechanic such as plan visualisation or natural-language goal intake.
- **Tutorials** (`tutorials/`) — end-to-end walkthroughs organised into
  three tiers of increasing complexity. See
  [`tutorials/README.md`](tutorials/README.md) for the full v0.1.0
  featured catalog (15 notebooks) and the list of pre-catalog extras.

Every featured notebook has a corresponding integration test under
`tests/integration/` that runs on every `make check`; the notebook is
the runnable documentation while the integration test is the source of
truth.

## Basics

- [`basics/plan_visualization.ipynb`](basics/plan_visualization.ipynb) —
  Mermaid / DOT / ASCII rendering of plans and CSP schedules.
- [`basics/nl_goal_interpreter.ipynb`](basics/nl_goal_interpreter.ipynb) —
  turning plain-English requests into `GoalSpec`s via `GoalInterpreter`.

## Tutorials

See [`tutorials/README.md`](tutorials/README.md) for the tiered catalog.

## Running Notebooks with Docker

```bash
cd examples
docker compose up
```

Open `http://127.0.0.1:8888/tree` in your browser. When running with
Docker Compose, the local library from `../` is mounted and installed
automatically.

To stop:

```bash
docker compose down
```

## Running Notebooks Locally

```bash
pip install langgoap jupyter
jupyter notebook
```
