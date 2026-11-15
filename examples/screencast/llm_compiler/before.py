"""LLMCompiler, hand-wired: planner + task-fetching unit + joiner + replan.

A condensed, hermetic version of the official LLMCompiler tutorial's control
flow (two nodes, one conditional edge). The planner emits tasks with ``$N``
dependency references, a scheduler executes them as dependencies resolve, and
a joiner decides between a final answer and ``Replan``. Routing is encoded in
message *types*: the graph ends when the last message is an ``AIMessage``.

The replan branch is where the tutorial's reported defects live (issues
#274, #1459, #1513, #1565): the feedback is a string the planner must parse,
and the only safety net is the recursion limit.

Run: uv run python examples/screencast/llm_compiler/before.py
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

CITY_POPULATIONS: dict[str, int] = {"Springfield": 30_720, "Shelbyville": 24_118}


class CompilerState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    observations: dict[int, Any]
    index_down_for: list[str]
    attempt: int


# --- tools -----------------------------------------------------------------


def lookup(city: str, *, index_down_for: list[str]) -> Any:
    if city in index_down_for:
        return f"ERROR: population index unavailable for {city}"
    return CITY_POPULATIONS[city]


def archive_lookup(city: str) -> int:
    return CITY_POPULATIONS[city]


def add(values: list[Any]) -> Any:
    if any(isinstance(v, str) for v in values):
        return "ERROR: cannot add a failed lookup"
    return sum(values)


# --- planner: a fixed DAG with $N references (an LLM wrote these in the original)


def plan(attempt: int, feedback: str | None) -> list[dict[str, Any]]:
    # On a replan the planner must *parse* the feedback to pick a different tool;
    # the tutorial fed the feedback back as a SystemMessage and hoped.
    use_archive = bool(feedback and "unavailable" in feedback)
    tool = "archive_lookup" if use_archive else "lookup"
    return [
        {"idx": 1, "tool": tool, "args": {"city": "Springfield"}},
        {"idx": 2, "tool": tool, "args": {"city": "Shelbyville"}},
        {"idx": 3, "tool": "add", "args": {"values": ["$1", "$2"]}},
    ]


# --- task-fetching unit: resolve $N references and run ready tasks ---------

_REF = re.compile(r"^\$(\d+)$")


def _resolve(value: Any, observations: dict[int, Any]) -> Any:
    if isinstance(value, list):  # issue #274: more than one dependency in one argument
        return [_resolve(v, observations) for v in value]
    if isinstance(value, str) and (m := _REF.match(value)):
        return observations[int(m.group(1))]
    return value


def _deps(task: dict[str, Any]) -> set[int]:
    found: set[int] = set()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for item in v.values():
                walk(item)
        elif isinstance(v, list):
            for item in v:
                walk(item)
        elif isinstance(v, str) and (m := _REF.match(v)):
            found.add(int(m.group(1)))

    walk(task["args"])
    return found


def schedule(tasks: list[dict[str, Any]], index_down_for: list[str]) -> dict[int, Any]:
    observations: dict[int, Any] = {}
    pending = {t["idx"]: t for t in tasks}
    with ThreadPoolExecutor() as pool:
        while pending:
            ready = [t for t in pending.values() if _deps(t).issubset(observations)]
            if not ready:
                raise RuntimeError("dependency cycle or unresolved reference")

            def run_task(task: dict[str, Any]) -> tuple[int, Any]:
                args = {k: _resolve(v, observations) for k, v in task["args"].items()}
                if task["tool"] == "lookup":
                    return task["idx"], lookup(**args, index_down_for=index_down_for)
                if task["tool"] == "archive_lookup":
                    return task["idx"], archive_lookup(**args)
                return task["idx"], add(**args)

            for idx, obs in pool.map(run_task, ready):
                observations[idx] = obs
                del pending[idx]
    return observations


# --- graph nodes -----------------------------------------------------------


def plan_and_schedule(state: CompilerState) -> dict[str, Any]:
    attempt = state.get("attempt", 0) + 1
    feedback = next(
        (
            m.content
            for m in reversed(state["messages"])
            if isinstance(m, SystemMessage)
        ),
        None,
    )
    tasks = plan(attempt, str(feedback) if feedback else None)
    print(f"  [plan_and_schedule] attempt {attempt}: {[t['tool'] for t in tasks]}")
    observations = schedule(tasks, state.get("index_down_for", []))
    return {"observations": observations, "attempt": attempt}


def join(state: CompilerState) -> dict[str, Any]:
    total = state["observations"][3]
    if isinstance(total, str):  # Replan: feed the error back as context
        failed = [o for o in state["observations"].values() if isinstance(o, str)]
        print(f"  [join] Replan — {failed[0]}")
        return {
            "messages": [
                SystemMessage(content=f"Context from last attempt: {failed[0]}")
            ]
        }
    print(f"  [join] FinalResponse — {total:,}")
    return {"messages": [AIMessage(content=f"The combined population is {total:,}.")]}


def should_continue(state: CompilerState) -> Literal["plan_and_schedule", "__end__"]:
    # Routing by message type: an AIMessage means "done", anything else means "again".
    if isinstance(state["messages"][-1], AIMessage):
        return "__end__"  # END
    return "plan_and_schedule"


def build_graph() -> StateGraph:
    builder = StateGraph(CompilerState)
    builder.add_node("plan_and_schedule", plan_and_schedule)
    builder.add_node("join", join)
    builder.add_edge(START, "plan_and_schedule")
    builder.add_edge("plan_and_schedule", "join")
    builder.add_conditional_edges("join", should_continue)
    return builder


def run(index_down_for: list[str] | None = None) -> None:
    label = "HAPPY PATH" if not index_down_for else f"INDEX DOWN FOR {index_down_for}"
    print(f"\n{'=' * 60}")
    print(f"  BEFORE (hand-wired LangGraph) — {label}")
    print(f"{'=' * 60}")
    graph = build_graph().compile()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Combined population of Springfield and Shelbyville?"
                )
            ],
            "index_down_for": index_down_for or [],
        },
        config={"recursion_limit": 8},
    )
    print(
        f"\n  Result: {result['messages'][-1].content}  (attempts: {result['attempt']})"
    )


if __name__ == "__main__":
    run()
    run(["Springfield"])
    print(f"\n{'=' * 60}")
    print("  NODES: 2   CONDITIONAL EDGES: 1 (routing by message type)")
    print("  DEPENDENCY PARSER: ~40 lines   REPLAN: a string the planner must parse")
    print("  SAFETY NET: recursion_limit")
    print(f"{'=' * 60}")
