"""LLMCompiler, GOAP version under disruption: the index is down.

Same actions as ``after.py``. The fast lookup raises for one city or for
every city; the observer blacklists it and the planner swaps in the archive
lookup while keeping any sibling result already obtained. When both the
index and the archive are down the run ends with ``no_plan`` and an
explanation instead of spinning until a recursion limit.

Run: uv run python examples/screencast/llm_compiler/after_disrupted.py
"""

from __future__ import annotations

from typing import Any

from examples.screencast.llm_compiler.after import (
    build_actions,
    build_goal,
    initial_world_state,
)
from langgoap import GoapGraph


def run(label: str, **flags: Any) -> None:
    print(f"\n{'=' * 60}")
    print(f"  AFTER (GOAP) — {label}")
    print(f"{'=' * 60}")
    ws = {**initial_world_state(), **flags}
    actions = build_actions()
    if flags.get("archive_down"):
        actions = [a for a in actions if not a.name.endswith("_archive")]
    result = GoapGraph(actions, executor="parallel").invoke(
        goal=build_goal(), world_state=ws
    )
    print(f"\n  Status      : {result['status']}")
    print(f"  Answer      : {result['world_state'].get('answer')}")
    print(f"  Replans     : {result['replan_count']}")
    print(f"  Blacklisted : {result['blacklisted_actions']}")
    for h in result["execution_history"]:
        mark = "ok  " if h.success else "FAIL"
        print(f"    {mark} {h.action_name}{'' if h.success else ' — ' + str(h.error)}")
    if result["status"] == "no_plan":
        explanation = result.get("no_plan_explanation") or {}
        print(f"  Explanation : {explanation.get('suggestion')}")


if __name__ == "__main__":
    run("index down for Springfield only", index_down_for=["Springfield"])
    run("index down for every city", primary_index_down=True)
    run("index and archive both down", primary_index_down=True, archive_down=True)
