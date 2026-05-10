"""Shared execution helpers for the executor / parallel-executor nodes.

Pure functions that build executor result dicts (success, failure,
guard-block, human-approval-denied) and handle pre-execution
validation.  Kept module-private so the ``executor.py`` and
``parallel_executor.py`` modules do not have to import each other.
"""

from __future__ import annotations

import random
from typing import Any

from langgraph.types import interrupt

from langgoap.actions import ActionSpec
from langgoap.graph.nodes._helpers import logger
from langgoap.graph.state import ActionResult, GoapState
from langgoap.guards import GuardResult
from langgoap.planner.transitions import TransitionModel
from langgoap.planner.types import Plan


def _apply_result(
    raw_result: Any,
    action: ActionSpec,
    world_state: dict[str, Any],
    *,
    transition_model: TransitionModel | None = None,
    rng: random.Random | None = None,
) -> None:
    """Merge an action's return value into world_state in place.

    If the callable returned a dict, its contents overwrite matching keys \u2014
    real runtime returns (LLM/tool outputs) stay authoritative.  Otherwise
    the executor falls back to either ``transition_model.sample`` (when
    provided) or ``action.get_effects`` (the pre-``TransitionModel``
    default).  Dynamic effects are resolved against ``world_state`` before
    merging in both paths.
    """
    if isinstance(raw_result, dict):
        world_state.update(raw_result)
        return
    if transition_model is not None:
        sample_rng = rng if rng is not None else random.Random()
        world_state.update(transition_model.sample(world_state, action, sample_rng))
        return
    world_state.update(action.get_effects(world_state))


def _build_success(
    action: ActionSpec,
    current_step: int,
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the success return dict for the executor.

    When ``action.effect_validator`` rejects the post-state, the executor
    rolls the world state back to the snapshot taken before the action
    ran.  This is critical: a validator's purpose is to detect actions
    that did not actually accomplish what they claimed, and a rejection
    means the effects are *not trustworthy*.  Leaking the mutated
    world_state through would let the rejected action satisfy the
    planner's goal predicate (because the effect keys are already set),
    short-circuiting the blacklist + replan dance the validator was
    designed to trigger.  The mutated post-state is still exposed via
    ``ActionResult.state_after`` for diagnostics.
    """
    state_after = dict(world_state)
    if action.effect_validator is not None and not action.validate_effects(
        state_before, state_after
    ):
        logger.warning("Action %r effect validation failed", action.name)
        return {
            "status": "action_failed",
            "world_state": state_before,
            "execution_history": [
                ActionResult(
                    action_name=action.name,
                    success=False,
                    state_before=state_before,
                    state_after=state_after,
                    error="Effect validation failed",
                )
            ],
        }
    logger.debug("Action %r succeeded, world_state=%s", action.name, world_state)
    return {
        "world_state": world_state,
        "current_step": current_step + 1,
        "execution_history": [
            ActionResult(
                action_name=action.name,
                success=True,
                state_before=state_before,
                state_after=state_after,
            )
        ],
    }


def _build_failure(
    action: ActionSpec,
    exc: Exception,
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the failure return dict for the executor."""
    logger.warning("Action %r failed: %s", action.name, exc)
    return {
        "status": "action_failed",
        "execution_history": [
            ActionResult(
                action_name=action.name,
                success=False,
                state_before=state_before,
                state_after=dict(world_state),
                error=str(exc),
            )
        ],
    }


def _build_guard_block(
    action: ActionSpec,
    guard_results: list[GuardResult],
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure result dict for a BLOCK-severity guard failure.

    Immediately blacklists the action (failure_count = max_retries + 1) so
    the observer's blacklist logic trips on the first guard block regardless
    of ``action.max_retries``.  A guard block is a policy decision, not a
    transient flake, so retries would be wrong.
    """
    blocking = [r for r in guard_results if not r.passed]
    reason = "; ".join(r.message for r in blocking) if blocking else "guard_blocked"
    result = _build_failure(
        action,
        RuntimeError(f"guard_blocked: {reason}"),
        state_before,
        world_state,
    )
    result["action_failure_counts"] = {action.name: action.max_retries + 1}
    return result


def _is_approved(resume_value: Any) -> bool:
    """Interpret a ``Command(resume=...)`` value as an approval decision.

    The resume value sent by the human client is interpreted generously:

    - ``None`` \u2192 approved (implicit: resume with no payload = continue)
    - ``True`` \u2192 approved
    - ``False`` \u2192 denied
    - ``{"approved": True}`` \u2192 approved
    - ``{"approved": False}`` \u2192 denied
    - Any other truthy value \u2192 approved
    """
    if resume_value is None or resume_value is True:
        return True
    if resume_value is False:
        return False
    if isinstance(resume_value, dict):
        return bool(resume_value.get("approved", True))
    return bool(resume_value)


def _is_form_model(value: Any) -> bool:
    """Return True iff ``value`` is a Pydantic ``BaseModel`` subclass."""
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover - Pydantic ships with LangChain
        return False
    return isinstance(value, type) and issubclass(value, BaseModel)


def _check_human_approval(
    action: ActionSpec,
    world_state: dict[str, Any],
    state_before: dict[str, Any],
) -> dict[str, Any] | None:
    """Gate execution behind a human-approval interrupt when required.

    Three modes (``ActionSpec.require_human_approval``):

    * ``False`` \u2014 no gate; returns ``None`` immediately.
    * ``True`` \u2014 boolean approve/deny gate.
    * Pydantic ``BaseModel`` subclass \u2014 typed-form gate; the interrupt
      payload includes the model's JSON schema; the resume payload is
      validated via ``Model.model_validate`` and merged into world
      state under :attr:`ActionSpec.human_input_key` (or the action's
      name when unset).
    """
    requirement = action.require_human_approval
    if not requirement:
        return None

    if _is_form_model(requirement):
        return _check_form_approval(action, world_state, state_before)

    resume_value = interrupt(
        {
            "type": "goap_action_approval",
            "action": action.name,
            "preconditions": dict(action.preconditions),
            "effects": dict(action.get_effects(world_state)),
            "world_state": dict(world_state),
        }
    )

    if _is_approved(resume_value):
        return None

    # Denied \u2014 build a failure result.  Pre-set the failure count to
    # max_retries + 1 so the observer's blacklist threshold trips on the
    # first denial regardless of max_retries (a human denial is a
    # deliberate decision, not a transient flake).
    reason = "denied by operator"
    if isinstance(resume_value, dict):
        reason = resume_value.get("reason", reason)
    result = _build_failure(
        action,
        RuntimeError(f"human_approval_denied: {reason}"),
        state_before,
        world_state,
    )
    result["action_failure_counts"] = {action.name: action.max_retries + 1}
    return result


def _check_form_approval(
    action: ActionSpec,
    world_state: dict[str, Any],
    state_before: dict[str, Any],
) -> dict[str, Any] | None:
    """Typed-form HITL gate.  See :func:`_check_human_approval`."""
    from pydantic import BaseModel, ValidationError

    model_cls: type[BaseModel] = action.require_human_approval  # type: ignore[assignment]
    schema = model_cls.model_json_schema()
    resume_value = interrupt(
        {
            "type": "goap_action_form",
            "action": action.name,
            "form_schema": schema,
            "model_name": model_cls.__name__,
            "preconditions": dict(action.preconditions),
            "world_state": dict(world_state),
        }
    )

    if resume_value is None or not isinstance(resume_value, (dict, BaseModel)):
        return _build_form_invalid_failure(
            action,
            "form payload missing or wrong shape",
            state_before,
            world_state,
        )
    try:
        if isinstance(resume_value, BaseModel):
            parsed = resume_value
        else:
            parsed = model_cls.model_validate(resume_value)
    except ValidationError as exc:
        return _build_form_invalid_failure(
            action,
            f"form payload failed validation: {exc.errors()!r}",
            state_before,
            world_state,
        )

    # Approved \u2014 merge the validated form (as JSON-friendly dict) into
    # world_state under human_input_key (default: action name).  Storing
    # the dict rather than the Pydantic instance keeps the value
    # round-trippable through any checkpointer serializer without
    # extra type registration.
    key = action.human_input_key or action.name
    world_state[key] = parsed.model_dump(mode="json")
    return None


def _build_form_invalid_failure(
    action: ActionSpec,
    reason: str,
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure result for a rejected typed-form HITL submission."""
    result = _build_failure(
        action,
        RuntimeError(f"human_form_invalid: {reason}"),
        state_before,
        world_state,
    )
    result["action_failure_counts"] = {action.name: action.max_retries + 1}
    return result


def _prepare_execution(
    state: GoapState,
) -> tuple[dict[str, Any], Plan, int, ActionSpec] | dict[str, Any]:
    """Validate executor pre-conditions and extract execution context.

    Returns either a ``(world_state, plan, step, action)`` tuple for the
    normal execution path, or a short-circuit result dict when the executor
    should return early without running any action.
    """
    status: str = state.get("status", "")
    plan_obj: Plan | None = state.get("plan")
    current_step: int = state.get("current_step", 0)
    world_state: dict[str, Any] = dict(state.get("world_state", {}))

    if status == "no_plan":
        return {}

    # Empty plan with no remaining steps \u2014 the planner returned a
    # zero-action Plan because the (sub-)goal was already satisfied
    # in the current world state.  Return a no-op update so the
    # observer can detect satisfaction and route accordingly;
    # emitting a ``"<none>"`` failure record here would clutter
    # execution history for legitimate pre-satisfied goals (notably
    # ``MultiGoal`` sub-goals that begin already met).
    if plan_obj is not None and len(plan_obj) == 0:
        return {}

    if plan_obj is None or current_step >= len(plan_obj):
        return {
            "status": "error",
            "execution_history": [
                ActionResult(
                    action_name="<none>",
                    success=False,
                    error="No action to execute",
                )
            ],
        }

    action = plan_obj.actions[current_step]
    return world_state, plan_obj, current_step, action


__all__ = [
    "_apply_result",
    "_build_success",
    "_build_failure",
    "_build_guard_block",
    "_is_approved",
    "_check_human_approval",
    "_prepare_execution",
]
