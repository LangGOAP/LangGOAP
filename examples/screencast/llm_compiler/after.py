"""LLMCompiler, GOAP version: the tool DAG is the precondition graph.

The original streams a plan of ``$N``-referenced tool tasks, schedules them
by hand as dependencies resolve, and asks a "joiner" LLM whether to answer
or re-plan. Here each tool task is an ``ActionSpec`` whose dependencies are
preconditions: A* orders them, ``executor="parallel"`` runs the independent
lookups in one wave, and the observer re-plans around a failed lookup while
keeping the sibling's result.

Run: uv run python examples/screencast/llm_compiler/after.py
"""

from __future__ import annotations

from typing import Any, Callable

from langgoap import ActionSpec, GoalSpec, GoapGraph

# Hermetic "population index": the numbers a search tool would return.
CITY_POPULATIONS: dict[str, int] = {"Springfield": 30_720, "Shelbyville": 24_118}
QUESTION_CITIES: tuple[str, ...] = ("Springfield", "Shelbyville")
QUESTION = "What is the combined population of Springfield and Shelbyville?"


def _slug(city: str) -> str:
    return city.lower()


def _primary_lookup(city: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The fast index. It raises when the index is down for ``city``."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        if ws.get("primary_index_down") or city in ws.get("index_down_for", []):
            raise RuntimeError(f"population index unavailable for {city}")
        print(f"  [lookup_{_slug(city)}] index -> {CITY_POPULATIONS[city]:,}")
        return {
            f"population_{_slug(city)}": CITY_POPULATIONS[city],
            f"population_{_slug(city)}_known": True,
        }

    return execute


def _archive_lookup(city: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The slow archive: same answer, three times the cost."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        print(f"  [lookup_{_slug(city)}_archive] archive -> {CITY_POPULATIONS[city]:,}")
        return {
            f"population_{_slug(city)}": CITY_POPULATIONS[city],
            f"population_{_slug(city)}_known": True,
        }

    return execute


def sum_populations(ws: dict[str, Any]) -> dict[str, Any]:
    total = sum(ws[f"population_{_slug(c)}"] for c in QUESTION_CITIES)
    print(
        f"  [sum_populations] {' + '.join(str(ws[f'population_{_slug(c)}']) for c in QUESTION_CITIES)} = {total:,}"
    )
    return {"total_population": total, "total_known": True}


def compose_answer(ws: dict[str, Any]) -> dict[str, Any]:
    answer = f"The combined population of {' and '.join(QUESTION_CITIES)} is {ws['total_population']:,}."
    print(f"  [compose_answer] {answer}")
    return {"answer": answer, "answer_ready": True}


def build_actions() -> list[ActionSpec]:
    """Tool tasks as actions. Dependencies are preconditions, not ``$N`` strings."""
    actions: list[ActionSpec] = []
    for city in QUESTION_CITIES:
        known = f"population_{_slug(city)}_known"
        actions.append(
            ActionSpec(
                name=f"lookup_{_slug(city)}",
                preconditions={"question_parsed": True},
                effects={known: True},
                cost=1.0,
                execute=_primary_lookup(city),
            )
        )
        actions.append(
            ActionSpec(
                name=f"lookup_{_slug(city)}_archive",
                preconditions={"question_parsed": True},
                effects={known: True},
                cost=3.0,
                execute=_archive_lookup(city),
            )
        )
    actions.append(
        ActionSpec(
            name="sum_populations",
            preconditions={
                f"population_{_slug(c)}_known": True for c in QUESTION_CITIES
            },
            effects={"total_known": True},
            cost=1.0,
            execute=sum_populations,
        )
    )
    actions.append(
        ActionSpec(
            name="compose_answer",
            preconditions={"total_known": True},
            effects={"answer_ready": True},
            cost=1.0,
            execute=compose_answer,
        )
    )
    return actions


def build_goal() -> GoalSpec:
    return GoalSpec(conditions={"answer_ready": True})


def initial_world_state() -> dict[str, Any]:
    return {"question": QUESTION, "question_parsed": True}


def run() -> None:
    print(f"\n{'=' * 60}")
    print("  AFTER (GOAP) — parallel lookups, A* ordering")
    print(f"{'=' * 60}")
    result = GoapGraph(build_actions(), executor="parallel").invoke(
        goal=build_goal(), world_state=initial_world_state()
    )
    print(f"\n  Status : {result['status']}")
    print(f"  Answer : {result['world_state'].get('answer')}")
    print(
        f"  Plan   : {' → '.join(h.action_name for h in result['execution_history'] if h.success)}"
    )
    print("\n  ROUTING FUNCTIONS: 0   DEPENDENCY PARSER: 0   JOINER: the observer")


if __name__ == "__main__":
    run()
