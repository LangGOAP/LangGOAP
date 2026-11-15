"""Generate ``llm_compiler_screencast.ipynb`` (run from the repo root).

    uv run python examples/screencast/llm_compiler/_build_notebook.py
    uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace \
        examples/screencast/llm_compiler/llm_compiler_screencast.ipynb

Markdown cells carry the narration; code cells import the three scripts so
the notebook and the scripts cannot drift apart. Everything is hermetic.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "llm_compiler_screencast.ipynb"
S = nbf.v4.new_markdown_cell
C = nbf.v4.new_code_cell

cells: list = []

cells.append(S("""# LLMCompiler — before / after / after-disrupted

Notebook version of the three-script screencast in
[`examples/screencast/llm_compiler/`](https://github.com/LangGOAP/LangGOAP/tree/main/examples/screencast/llm_compiler).

**The task.** Answer "what is the combined population of Springfield and Shelbyville?":
two independent lookups, one sum, one sentence. Run the lookups in parallel; recover when one fails.

**The original.** LangGraph's LLMCompiler tutorial (462 code lines, 2 nodes, 1 conditional edge)
streams a DAG of tool tasks with `$N` dependency references, schedules them by hand as
dependencies resolve, and asks a *joiner* LLM to choose between a final answer and `Replan`.
Routing is encoded in message types: the graph ends when the last message is an `AIMessage`.
Four issues report the replan branch as broken or untested
([#274](https://github.com/langchain-ai/langgraph/issues/274),
[#1459](https://github.com/langchain-ai/langgraph/issues/1459),
[#1513](https://github.com/langchain-ai/langgraph/issues/1513),
[#1565](https://github.com/langchain-ai/langgraph/issues/1565)).

| | Hand-wired (`before.py`) | GOAP (`after.py`) |
|---|---|---|
| Dependencies | `$N` strings, a resolver, a thread pool | preconditions and effects |
| Ordering | planner prompt + scheduler | A* |
| Parallelism | `ThreadPoolExecutor` over "ready" tasks | `executor="parallel"` waves |
| Recovery | joiner emits `Replan` as a `SystemMessage` the planner must parse | observer blacklists the failed action and re-plans |
| Termination | `recursion_limit` | goal reached, or `no_plan` with an explanation |
"""))

cells.append(
    S(
        """## 1. The hand-wired version

`before.py` is a condensed, hermetic version of the tutorial's control flow. The planner is a fixed
task list (an LLM writes it in the original), the scheduler resolves `$N` references and runs ready
tasks in a thread pool, and `should_continue` routes on `isinstance(messages[-1], AIMessage)`."""
    )
)

cells.append(C("""from examples.screencast.llm_compiler import before

print(before.build_graph().compile().get_graph().draw_mermaid())
before.run()
before.run(index_down_for=["Springfield"])"""))

cells.append(
    S(
        """Two things to notice in the disrupted run: the replan re-ran **both** lookups through the archive
although only Springfield's index was down, and the planner had to recognise the word
"unavailable" in a feedback string to switch tools. Neither is a bug in this condensed version;
both are what the architecture asks you to hand-write."""
    )
)

cells.append(
    S(
        """## 2. The GOAP version: dependencies are preconditions

Each tool task is an `ActionSpec`. The sum's preconditions are the two `*_known` facts the lookups
produce, so A* orders the plan and the parallel executor runs the two lookups in one wave."""
    )
)

cells.append(
    C(
        """from examples.screencast.llm_compiler.after import build_actions, build_goal, initial_world_state
from langgoap import GoapGraph

for a in build_actions():
    print(f"{a.name:28s} cost={a.cost:<4} pre={dict(a.preconditions)}  eff={dict(a.effects)}")

graph = GoapGraph(build_actions(), executor="parallel")
result = graph.invoke(goal=build_goal(), world_state=initial_world_state())
print()
print("status :", result["status"])
print("answer :", result["world_state"]["answer"])
print("history:", [h.action_name for h in result["execution_history"]])"""
    )
)

cells.append(
    S(
        """How many times did the executor node run? Streaming the graph's node updates shows the two
lookups as a single executor invocation (one wave), then the sum, then the answer."""
    )
)

cells.append(
    C(
        """def executor_steps(g: GoapGraph) -> int:
    compiled = g.compile()
    state = g.initial_state(build_goal(), initial_world_state())
    return sum(1 for update in compiled.stream(state, stream_mode="updates") if "executor" in update)

print("parallel executor  :", executor_steps(GoapGraph(build_actions(), executor="parallel")), "executor invocations")
print("sequential executor:", executor_steps(GoapGraph(build_actions())), "executor invocations")"""
    )
)

cells.append(S("""## 3. Disruption: the index is down

The fast lookup raises. `max_retries` is 0, so the observer blacklists it and the planner swaps in
the archive lookup (cost 3). A sibling result already obtained is kept, so only the failed city is
looked up again."""))

cells.append(C("""import logging, sys

# The observer and planner log through ``logging``; send those lines to stdout so
# they appear in order with the prints below instead of in a separate stderr block.
logging.basicConfig(stream=sys.stdout, format="  log  %(message)s", level=logging.WARNING, force=True)


def show(label: str, **flags):
    ws = {**initial_world_state(), **flags}
    actions = build_actions()
    if flags.get("archive_down"):
        actions = [a for a in actions if not a.name.endswith("_archive")]
    result = GoapGraph(actions, executor="parallel").invoke(goal=build_goal(), world_state=ws)
    print(f"--- {label} ---")
    print("status     :", result["status"], "| replans:", result["replan_count"], "| blacklisted:", result["blacklisted_actions"])
    for h in result["execution_history"]:
        print(f"  {'ok  ' if h.success else 'FAIL'} {h.action_name}" + ("" if h.success else f" — {h.error}"))
    if result["status"] == "no_plan":
        print("  explanation:", result["no_plan_explanation"]["suggestion"])
    print()

show("index down for Springfield only", index_down_for=["Springfield"])
show("index down for every city", primary_index_down=True)
show("index and archive both down", primary_index_down=True, archive_down=True)"""))

cells.append(S("""## 4. What replaced what

| Hand-wired piece | Lines in `before.py` | GOAP replacement |
|---|---|---|
| `$N` reference parser and dependency walker | ~35 | preconditions on each `ActionSpec` |
| Ready-task scheduler with a thread pool | ~25 | `GoapGraph(..., executor="parallel")` |
| Joiner deciding `FinalResponse` vs `Replan` | ~10 | the observer: a failed action is blacklisted and the planner runs again |
| Feedback as a `SystemMessage` the planner must parse | ~5 | nothing: the archive action is simply the next cheapest way to the same effect |
| `should_continue` routing on message type | ~5 | nothing: the goal is a world-state fact |
| `recursion_limit` as the safety net | 1 | `no_plan` with an explanation; `TerminationPolicy` if you want a budget |

The integration test
[`tests/integration/test_screencast_llm_compiler.py`](https://github.com/LangGOAP/LangGOAP/blob/main/tests/integration/test_screencast_llm_compiler.py)
pins the wave count, the answer, the replan behaviour and the async path."""))


def main() -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
