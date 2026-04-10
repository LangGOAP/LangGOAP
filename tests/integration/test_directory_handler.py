"""Integration test for the Directory Handler tutorial (Tier 1 primer).

Exercises the basic GOAP loop end-to-end with real filesystem
side-effects.  The planner must discover that it has to create the
workspace directory before it can create the marker token, and the
executor must perform the actual ``mkdir`` and ``touch`` calls.

The shared execute functions live in
``examples/tutorials/tutorial_examples/directory_handler.py`` so the
notebook and the test exercise the exact same code.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tutorial_examples.directory_handler import (
    directory_handler_actions,
    initial_world_state,
)

from langgoap import GoalSpec, GoapGraph


class TestDirectoryHandler:
    """Directory Handler: minimal GOAP loop with real filesystem actions."""

    def test_full_pipeline_creates_workspace_and_token(self) -> None:
        """Planner discovers create_workspace → create_token path."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "goap_workspace"
            actions = directory_handler_actions(workspace)

            result = GoapGraph(actions=actions).invoke(
                goal=GoalSpec(
                    conditions={"workspace_exists": True, "token_exists": True}
                ),
                world_state=initial_world_state(workspace),
            )

            assert result["status"] == "goal_achieved"

            # Both filesystem effects were actually performed.
            assert workspace.is_dir()
            assert (workspace / ".token").is_file()

            # World state reflects the effects.
            ws = result["world_state"]
            assert ws["workspace_exists"] is True
            assert ws["token_exists"] is True

            # The plan used both actions in the expected order.
            successful = [
                h.action_name for h in result["execution_history"] if h.success
            ]
            assert successful == ["create_workspace", "create_token"]

    def test_workspace_already_exists_skips_create_workspace(self) -> None:
        """When the workspace exists, only create_token is needed."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "goap_workspace"
            workspace.mkdir()  # pre-create the workspace
            actions = directory_handler_actions(workspace)

            result = GoapGraph(actions=actions).invoke(
                goal=GoalSpec(
                    conditions={"workspace_exists": True, "token_exists": True}
                ),
                world_state=initial_world_state(workspace),
            )

            assert result["status"] == "goal_achieved"
            assert (workspace / ".token").is_file()

            successful = [
                h.action_name for h in result["execution_history"] if h.success
            ]
            assert successful == ["create_token"]

    def test_goal_already_satisfied_executes_nothing(self) -> None:
        """Fully satisfied goal returns immediately with an empty plan."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "goap_workspace"
            workspace.mkdir()
            (workspace / ".token").touch()
            actions = directory_handler_actions(workspace)

            result = GoapGraph(actions=actions).invoke(
                goal=GoalSpec(
                    conditions={"workspace_exists": True, "token_exists": True}
                ),
                world_state=initial_world_state(workspace),
            )

            assert result["status"] == "goal_achieved"
            successful = [
                h.action_name for h in result["execution_history"] if h.success
            ]
            assert successful == []

    def test_partial_goal_workspace_only(self) -> None:
        """Partial goal: plan stops after the workspace is created."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "goap_workspace"
            actions = directory_handler_actions(workspace)

            result = GoapGraph(actions=actions).invoke(
                goal=GoalSpec(conditions={"workspace_exists": True}),
                world_state=initial_world_state(workspace),
            )

            assert result["status"] == "goal_achieved"
            assert workspace.is_dir()
            # create_token must NOT run for this goal
            assert not (workspace / ".token").is_file()

            successful = [
                h.action_name for h in result["execution_history"] if h.success
            ]
            assert successful == ["create_workspace"]
