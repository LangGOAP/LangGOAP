"""Benchmarks for PlanningState hot paths.

PlanningState is the innermost data structure touched by every A* iteration:
  - from_dict  — called once per graph expansion to create the start state
  - apply      — called once per candidate action (state transition)
  - satisfies  — called once per node expansion to check goal / preconditions
  - get        — called in the observer and executor to read individual keys

Varying the state size (SMALL / MEDIUM / LARGE) reveals the frozenset
scaling characteristics and helps detect regressions in the hot path.

Key: all benchmarks are pure CPU — no I/O, no LLM calls.
"""

from __future__ import annotations

import pytest

from langgoap.state import PlanningState

# ---------------------------------------------------------------------------
# Parametrised state sizes
# ---------------------------------------------------------------------------

SIZES = [5, 50, 500]


def _make_dict(n: int) -> dict[str, bool]:
    return {f"flag_{i}": (i % 2 == 0) for i in range(n)}


def _make_state(n: int) -> PlanningState:
    return PlanningState.from_dict(_make_dict(n))


# ---------------------------------------------------------------------------
# from_dict — creation cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_bench_from_dict(benchmark: object, n: int) -> None:
    """PlanningState.from_dict with n scalar key-value pairs."""
    d = _make_dict(n)
    benchmark(PlanningState.from_dict, d)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# apply — state transition (innermost A* loop call)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_bench_apply_single_effect(benchmark: object, n: int) -> None:
    """apply() with a single new effect on a state of size n.

    Models the common case: one action flips one boolean flag.
    """
    state = _make_state(n)
    effects = {"new_flag": True}
    benchmark(state.apply, effects)  # type: ignore[call-arg]


@pytest.mark.parametrize("n", [5, 50])
def test_bench_apply_bulk_effects(benchmark: object, n: int) -> None:
    """apply() with n/2 effects simultaneously (bulk transition)."""
    state = _make_state(n)
    effects = {f"bulk_{i}": True for i in range(n // 2)}
    benchmark(state.apply, effects)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# satisfies — precondition / goal check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_state,n_req", [(50, 2), (50, 10), (500, 2), (500, 50)])
def test_bench_satisfies_hit(benchmark: object, n_state: int, n_req: int) -> None:
    """satisfies() when all n_req conditions are present (full hit).

    Conditions are always satisfied — this measures the fast path where
    every frozenset membership test succeeds.
    """
    state = _make_state(n_state)
    # All keys present and matching
    requirements = {f"flag_{i}": (i % 2 == 0) for i in range(n_req)}
    benchmark(state.satisfies, requirements)  # type: ignore[call-arg]


@pytest.mark.parametrize("n_state", [50, 500])
def test_bench_satisfies_miss(benchmark: object, n_state: int) -> None:
    """satisfies() when the first condition is absent (early exit).

    The frozenset scan terminates at the first miss — this measures
    the fast-fail path relevant to A* precondition checks.
    """
    state = _make_state(n_state)
    requirements = {"absent_key": True}  # guaranteed miss
    benchmark(state.satisfies, requirements)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# get — single key lookup (linear scan over frozenset)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_bench_get_hit(benchmark: object, n: int) -> None:
    """get() when the key is present (linear scan, worst-case = last item)."""
    state = _make_state(n)
    last_key = f"flag_{n - 1}"
    benchmark(state.get, last_key)  # type: ignore[call-arg]


@pytest.mark.parametrize("n", SIZES)
def test_bench_get_miss(benchmark: object, n: int) -> None:
    """get() when the key is absent (full scan, returns default)."""
    state = _make_state(n)
    benchmark(state.get, "absent_key")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# to_dict — snapshot for world_state writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_bench_to_dict(benchmark: object, n: int) -> None:
    """to_dict() materialises a frozenset into a dict (called by executor)."""
    state = _make_state(n)
    benchmark(state.to_dict)  # type: ignore[call-arg]
