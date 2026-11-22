"""Observed effects: after an action, the sensors say what happened (issue #96).

With ``effects="observed"`` the executor assumes the declared effects, asks the
sensors, and lets what they report for the action's keys win.  An action no
longer has to re-read its own facts and return them.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph


class Platform:
    """A stand-in for the real system the actions act on."""

    def __init__(self) -> None:
        self.card = "declined"
        self.backup_tried = False
        self.sensed = 0

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        self.sensed += 1
        return {
            "paid": self.card == "ok",
            "backup_card": not self.backup_tried,
            "unrelated": self.sensed,
        }


def _graph(platform: Platform, actions: list[ActionSpec], **options: Any) -> GoapGraph:
    return GoapGraph(actions, sensors=[platform.sense], effects="observed", **options)


class TestObservedEffects:
    def test_the_sensor_overrules_the_declared_effect(self) -> None:
        """The retry is declared to pay; the platform says it did not."""
        platform = Platform()

        def retry(world_state: dict[str, Any]) -> None:
            platform.backup_tried = True  # the charge fails again: card stays declined

        def pay_cash(world_state: dict[str, Any]) -> None:
            platform.card = "ok"

        actions = [
            ActionSpec(
                "retry_card",
                {"paid": False, "backup_card": True},
                {"paid": True},
                cost=1.0,
                execute=retry,
                observe=("backup_card",),
            ),
            ActionSpec(
                "pay_cash", {"paid": False}, {"paid": True}, cost=5.0, execute=pay_cash
            ),
        ]
        result = _graph(platform, actions).invoke(GoalSpec(conditions={"paid": True}))
        assert result["status"] == "goal_achieved"
        assert [r.action_name for r in result["execution_history"]] == [
            "retry_card",
            "pay_cash",
        ]
        first = result["execution_history"][0]
        assert first.success is True  # it ran; the world just did not follow
        assert first.state_after["paid"] is False
        assert first.state_after["backup_card"] is False  # observe= adds the key

    def test_only_the_actions_keys_are_taken_from_the_sensor(self) -> None:
        platform = Platform()

        def pay(world_state: dict[str, Any]) -> None:
            platform.card = "ok"

        actions = [ActionSpec("pay", {"paid": False}, {"paid": True}, execute=pay)]
        result = _graph(platform, actions).invoke(GoalSpec(conditions={"paid": True}))
        record = result["execution_history"][0]
        # "unrelated" changes on every read; the action did not ask for it.
        assert record.state_after["unrelated"] == record.state_before["unrelated"]

    def test_always_observe_adds_keys_to_every_action(self) -> None:
        platform = Platform()

        def pay(world_state: dict[str, Any]) -> None:
            platform.card = "ok"

        actions = [ActionSpec("pay", {"paid": False}, {"paid": True}, execute=pay)]
        graph = _graph(platform, actions, always_observe=("unrelated",))
        record = graph.invoke(GoalSpec(conditions={"paid": True}))["execution_history"][
            0
        ]
        assert record.state_after["unrelated"] > record.state_before["unrelated"]

    def test_an_action_needs_no_execute(self) -> None:
        """A decision-only action: its declared flag is assumed, then the world is read."""
        seen: list[dict[str, Any]] = []

        def sense(world_state: dict[str, Any]) -> dict[str, Any]:
            seen.append(dict(world_state))
            return {
                "kitchen": "cooking" if world_state.get("accepted_delay") else "late"
            }

        actions = [
            ActionSpec(
                "accept_delay",
                {"kitchen": "late"},
                {"kitchen": "cooking", "accepted_delay": True},
            )
        ]
        graph = GoapGraph(actions, sensors=[sense], effects="observed")
        result = graph.invoke(GoalSpec(conditions={"kitchen": "cooking"}))
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["accepted_delay"] is True
        # the sensor ran after the action with the declared flag already in place
        assert seen[-1]["accepted_delay"] is True

    def test_a_returned_dict_is_assumed_before_sensing(self) -> None:
        seen: list[dict[str, Any]] = []

        def sense(world_state: dict[str, Any]) -> dict[str, Any]:
            seen.append(dict(world_state))
            return {"done": bool(world_state.get("note"))}

        actions = [
            ActionSpec(
                "work",
                {"done": False},
                {"done": True},
                execute=lambda ws: {"note": "by hand"},
            )
        ]
        graph = GoapGraph(actions, sensors=[sense], effects="observed")
        result = graph.invoke(GoalSpec(conditions={"done": True}), {"note": ""})
        assert result["world_state"]["note"] == "by hand"
        assert seen[-1]["note"] == "by hand"
        assert result["status"] == "goal_achieved"

    async def test_async_path_observes_too(self) -> None:
        platform = Platform()

        async def pay(world_state: dict[str, Any]) -> None:
            platform.card = "ok"

        actions = [ActionSpec("pay", {"paid": False}, {"paid": True}, aexecute=pay)]
        result = await _graph(platform, actions).ainvoke(
            GoalSpec(conditions={"paid": True})
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["paid"] is True

    def test_parallel_executor_observes_too(self) -> None:
        platform = Platform()

        def pay(world_state: dict[str, Any]) -> None:
            platform.card = "ok"

        actions = [ActionSpec("pay", {"paid": False}, {"paid": True}, execute=pay)]
        graph = _graph(platform, actions, executor="parallel")
        result = graph.invoke(GoalSpec(conditions={"paid": True}))
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["paid"] is True


class TestDeclaredIsStillTheDefault:
    def test_declared_effects_are_believed_without_the_option(self) -> None:
        platform = Platform()
        actions = [
            ActionSpec("pay", {"paid": False}, {"paid": True}, execute=lambda ws: None)
        ]
        graph = GoapGraph(actions, sensors=[platform.sense])
        record = graph.invoke(GoalSpec(conditions={"paid": True}))["execution_history"][
            0
        ]
        assert record.state_after["paid"] is True  # nobody looked

    def test_observed_without_sensors_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sensors"):
            GoapGraph([ActionSpec("a", {}, {"x": True})], effects="observed")

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="effects"):
            GoapGraph([ActionSpec("a", {}, {"x": True})], effects="guessed")  # type: ignore[arg-type]


class TestObserveField:
    def test_observe_is_stored_as_a_tuple_and_round_trips(self) -> None:
        spec = ActionSpec("a", {}, {"x": True}, observe=["y", "z"])
        assert spec.observe == ("y", "z")
        assert ActionSpec.from_dict(spec.to_dict()).observe == ("y", "z")

    def test_default_is_empty(self) -> None:
        assert ActionSpec("a", {}, {"x": True}).observe == ()


class TestPlainFunctionSensors:
    def test_a_function_is_a_sensor_named_after_itself(self) -> None:
        names: list[str] = []

        class Recorder:
            def on_sensor_complete(self, sensor_name: str, updates: Any) -> None:
                names.append(sensor_name)

        def platform_facts(world_state: dict[str, Any]) -> dict[str, Any]:
            return {"x": True}

        graph = GoapGraph(
            [ActionSpec("noop", {"x": True}, {"y": True})],
            sensors=[platform_facts],
            tracer=Recorder(),  # type: ignore[arg-type]
        )
        result = graph.invoke(GoalSpec(conditions={"y": True}))
        assert result["status"] == "goal_achieved"
        assert names and set(names) == {"platform_facts"}
