"""A* ignores actions that cannot contribute to the goal.

An action whose effects touch no goal-relevant key (the goal's keys and,
transitively, the precondition keys of actions that write relevant keys)
cannot change whether any precondition or goal condition holds. Searching
through it only multiplies states, so the planner drops it before the search.
"""

from __future__ import annotations

from langgoap import ActionSpec, GoalSpec, plan
from langgoap.planner.astar import relevant_actions


def layered(
    n_layers: int, width: int, extra: int
) -> tuple[list[ActionSpec], GoalSpec, dict]:
    actions = []
    for layer in range(n_layers):
        for w in range(width):
            actions.append(
                ActionSpec(
                    f"l{layer}_w{w}",
                    {f"s{layer}": True},
                    {f"s{layer + 1}": True},
                    cost=1.0 + w * 0.5,
                )
            )
        for e in range(extra):  # reachable, cheap, and useless
            actions.append(
                ActionSpec(
                    f"l{layer}_x{e}",
                    {f"s{layer}": True},
                    {f"junk{layer}_{e}": True},
                    cost=0.5,
                )
            )
    goal = GoalSpec(conditions={f"s{n_layers}": True})
    start = {"s0": True, **{f"s{i}": False for i in range(1, n_layers + 1)}}
    return actions, goal, start


def test_relevance_is_a_backward_closure_over_keys() -> None:
    goal = {"published": True}
    actions = [
        ActionSpec("write", {"brief": True}, {"draft": True}),
        ActionSpec("research", {}, {"brief": True}),
        ActionSpec("publish", {"draft": True, "reviewed": True}, {"published": True}),
        ActionSpec("review", {"draft": True}, {"reviewed": True}),
        ActionSpec(
            "tweet", {"published": True}, {"tweeted": True}
        ),  # downstream of the goal
        ActionSpec("water_plants", {}, {"plants_watered": True}),  # unrelated
        ActionSpec("log", {}, {}),  # no effects at all
    ]
    kept = [a.name for a in relevant_actions(goal, actions)]
    assert kept == ["write", "research", "publish", "review"]


def test_a_relevant_dynamic_effect_keeps_every_action() -> None:
    """A callable effect may read any key (``eat`` reads ``location``), so nothing is dropped."""
    goal = {"food": frozenset()}
    actions = [
        ActionSpec("goto_a", {}, {"location": "a"}),
        ActionSpec("whistle", {}, {"whistled": True}),
        ActionSpec(
            "eat",
            {},
            lambda ws: {"food": ws["food"] - frozenset({ws["location"]})},
            effect_keys=frozenset({"food"}),
        ),
    ]
    assert [a.name for a in relevant_actions(goal, actions)] == [
        "goto_a",
        "whistle",
        "eat",
    ]
    result = plan(
        {"location": "start", "food": frozenset({"a"})},
        GoalSpec(conditions=goal),
        actions,
    )
    assert result is not None and [a.name for a in result.actions] == ["goto_a", "eat"]


def test_an_irrelevant_dynamic_effect_is_still_dropped() -> None:
    goal = {"done": True}
    actions = [
        ActionSpec("finish", {"ready": True}, {"done": True}),
        ActionSpec("prepare", {}, {"ready": True}),
        ActionSpec(
            "noise", {}, lambda ws: {"other": 1}, effect_keys=frozenset({"other"})
        ),
    ]
    assert [a.name for a in relevant_actions(goal, actions)] == ["finish", "prepare"]


def test_distractors_do_not_cost_nodes() -> None:
    clean, goal, start = layered(6, 3, 0)
    noisy, _, _ = layered(6, 3, 2)
    quiet = plan(start, goal, clean)
    loud = plan(start, goal, noisy)
    assert quiet is not None and loud is not None
    assert [a.name for a in loud.actions] == [a.name for a in quiet.actions]
    assert loud.total_cost == quiet.total_cost
    assert loud.metadata.nodes_explored == quiet.metadata.nodes_explored


def test_a_plan_never_contains_an_irrelevant_action() -> None:
    actions, goal, start = layered(8, 3, 3)
    result = plan(start, goal, actions)
    assert result is not None
    assert all("_x" not in a.name for a in result.actions)
    assert len(result.actions) == 8


def test_pruning_keeps_the_cheapest_route() -> None:
    """The optimum goes through a key that is not a goal key; pruning must keep it."""
    goal = GoalSpec(conditions={"at_goal": True})
    actions = [
        ActionSpec("walk", {"at_start": True}, {"at_goal": True}, cost=10.0),
        ActionSpec("buy_ticket", {"at_start": True}, {"ticket": True}, cost=1.0),
        ActionSpec("ride", {"ticket": True}, {"at_goal": True}, cost=2.0),
        ActionSpec("whistle", {"at_start": True}, {"whistled": True}, cost=0.1),
    ]
    result = plan({"at_start": True, "at_goal": False}, goal, actions)
    assert result is not None and [a.name for a in result.actions] == [
        "buy_ticket",
        "ride",
    ]
    assert result.total_cost == 3.0


def test_unreachable_goals_still_fail_fast() -> None:
    actions, goal, start = layered(4, 2, 1)
    assert plan(start, GoalSpec(conditions={"nowhere": True}), actions) is None


def test_mcts_prunes_the_same_way() -> None:
    """Whatever the tree returns, a distractor is never in it: it was never offered."""
    from langgoap.planner.mcts import MCTSExploration, MCTSReuseConfig, MCTSStrategy
    from langgoap.state import PlanningState

    actions, goal, start = layered(5, 2, 3)
    strategy = MCTSStrategy(
        exploration=MCTSExploration(iterations=600, rollout_depth=10, seed=3),
        reuse=MCTSReuseConfig(anytime_fallback=True),
    )
    result = strategy.plan(PlanningState.from_dict(start), goal, actions)
    assert result is not None and len(result.actions) >= 1
    assert all("_x" not in a.name for a in result.actions)


def test_mcts_keeps_every_action_when_none_can_reach_the_goal() -> None:
    """The anytime fallback still owes a step of progress on a goal the tree cannot see."""
    from langgoap.planner.mcts import MCTSExploration, MCTSReuseConfig, MCTSStrategy
    from langgoap.state import PlanningState

    actions = [ActionSpec("a", {}, {"x": 1}, cost=1.0)]
    strategy = MCTSStrategy(
        exploration=MCTSExploration(
            iterations=16, rollout_depth=2, seed=1, wall_clock_ms=0
        ),
        reuse=MCTSReuseConfig(anytime_fallback=True),
    )
    result = strategy.plan(
        PlanningState.from_dict({}), GoalSpec(conditions={"done": True}), actions
    )
    assert result is not None and result.actions[0].name == "a"
