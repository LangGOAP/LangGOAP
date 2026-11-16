"""LLMCompiler screencast: a tool DAG planned by A* and run in parallel waves.

The original tutorial hand-rolls a planner that emits ``$N`` dependency
references, a task-fetching unit that schedules them, and a joiner whose
``Replan`` branch four GitHub issues report as never having worked. In the
GOAP version the dependency structure is the precondition/effect graph, the
parallel executor runs independent lookups in one wave, and a failed lookup
is re-planned around with the sibling's result kept.
"""

from __future__ import annotations

from typing import Any

from examples.screencast.llm_compiler.after import (
    CITY_POPULATIONS,
    QUESTION_CITIES,
    build_actions,
    build_goal,
    initial_world_state,
)
from langgoap import GoapGraph

EXPECTED_TOTAL = sum(CITY_POPULATIONS[c] for c in QUESTION_CITIES)


def _executor_steps(graph: Any, state: dict[str, Any]) -> int:
    """Count executor node invocations by streaming node updates."""
    return sum(
        1
        for update in graph.stream(state, stream_mode="updates")
        if "executor" in update
    )


class TestHappyPath:
    def test_answer_combines_both_lookups(self) -> None:
        result = GoapGraph(build_actions(), executor="parallel").invoke(
            goal=build_goal(), world_state=initial_world_state()
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["total_population"] == EXPECTED_TOTAL
        assert f"{EXPECTED_TOTAL:,}" in result["world_state"]["answer"]

    def test_independent_lookups_run_in_one_wave(self) -> None:
        graph = GoapGraph(build_actions(), executor="parallel")
        compiled = graph.compile()
        state = graph.initial_state(build_goal(), initial_world_state())
        # two lookups (one wave) + sum + compose = 3 executor invocations
        assert _executor_steps(compiled, state) == 3

        sequential = GoapGraph(build_actions())
        state = sequential.initial_state(build_goal(), initial_world_state())
        assert _executor_steps(sequential.compile(), state) == 4

    def test_sum_waits_for_both_lookups(self) -> None:
        result = GoapGraph(build_actions(), executor="parallel").invoke(
            goal=build_goal(), world_state=initial_world_state()
        )
        names = [h.action_name for h in result["execution_history"]]
        assert names.index("sum_populations") > max(
            names.index("lookup_springfield"), names.index("lookup_shelbyville")
        )
        assert names[-1] == "compose_answer"
        assert result["replan_count"] == 0


class TestReplanAroundAFailedLookup:
    """The joiner's ``Replan`` branch, done by the observer."""

    def _disrupted(self) -> dict[str, Any]:
        ws = initial_world_state()
        ws["primary_index_down"] = True  # the fast lookup raises for every city
        return ws

    def test_fallback_lookup_completes_the_plan(self) -> None:
        result = GoapGraph(build_actions(), executor="parallel").invoke(
            goal=build_goal(), world_state=self._disrupted()
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["total_population"] == EXPECTED_TOTAL
        names = [h.action_name for h in result["execution_history"] if h.success]
        assert "lookup_springfield_archive" in names
        assert "lookup_shelbyville_archive" in names

    def test_failures_are_recorded_and_blacklisted(self) -> None:
        result = GoapGraph(build_actions(), executor="parallel").invoke(
            goal=build_goal(), world_state=self._disrupted()
        )
        failed = [h.action_name for h in result["execution_history"] if not h.success]
        assert set(failed) == {"lookup_springfield", "lookup_shelbyville"}
        assert set(result["blacklisted_actions"]) == {
            "lookup_springfield",
            "lookup_shelbyville",
        }
        assert result["replan_count"] >= 1

    def test_one_failed_city_keeps_the_siblings_result(self) -> None:
        ws = initial_world_state()
        ws["index_down_for"] = ["Springfield"]
        result = GoapGraph(build_actions(), executor="parallel").invoke(
            goal=build_goal(), world_state=ws
        )
        assert result["status"] == "goal_achieved"
        succeeded = [h.action_name for h in result["execution_history"] if h.success]
        # Shelbyville's fast lookup ran once and was not repeated after the replan.
        assert succeeded.count("lookup_shelbyville") == 1
        assert "lookup_springfield_archive" in succeeded
        assert result["world_state"]["total_population"] == EXPECTED_TOTAL

    async def test_async_path_matches(self) -> None:
        result = await GoapGraph(build_actions(), executor="parallel").ainvoke(
            goal=build_goal(), world_state=self._disrupted()
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["total_population"] == EXPECTED_TOTAL
