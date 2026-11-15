"""Shared execution helpers for the executor / parallel-executor nodes.

Pure functions that build executor result dicts (success, failure,
guard-block, human-approval-denied) and handle pre-execution
validation.  Kept module-private so the ``executor.py`` and
``parallel_executor.py`` modules do not have to import each other.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, Literal

from langgraph.types import interrupt

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, MultiGoal
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
) -> bool:
    """Merge an action's return value into world_state in place.

    If the callable returned a dict, its contents overwrite matching keys \u2014
    real runtime returns (LLM/tool outputs) stay authoritative.  Otherwise
    the executor falls back to either ``transition_model.sample`` (when
    provided) or ``action.get_effects`` (the pre-``TransitionModel``
    default).  Dynamic effects are resolved against ``world_state`` before
    merging in both paths.

    Returns:
        ``True`` when a user-returned dict was applied.  Only that case is
        checked against the declared effects by :func:`_build_success`
        (issue #31); effects the runtime applied itself are its own view.
    """
    if isinstance(raw_result, dict):
        world_state.update(raw_result)
        return True
    if raw_result is not None:
        # ``None`` is the documented "apply the declared effects" return;
        # anything else is almost certainly a mistake (a tool returning a
        # string, a forgotten dict wrapper) and used to vanish silently (#64).
        logger.warning(
            "Action %r execute() returned %s, not a dict; ignoring the value "
            "and applying the declared effects. Return a dict to update "
            "world_state, or None to apply the declared effects.",
            action.name,
            type(raw_result).__name__,
        )
    if transition_model is not None:
        sample_rng = rng if rng is not None else random.Random()
        world_state.update(transition_model.sample(world_state, action, sample_rng))
        return False
    world_state.update(action.get_effects(world_state))
    return False


def _build_success(
    action: ActionSpec,
    current_step: int,
    state_before: dict[str, Any],
    world_state: dict[str, Any],
    *,
    check_effects: bool = False,
) -> dict[str, Any]:
    """Build the success return dict for the executor.

    Effects are validated when the action declares an ``effect_validator``
    (its verdict is final) or, with ``check_effects`` set because ``execute``
    returned a dict, by the default rule: the returned dict must set every
    declared effect key.  A returned dict is authoritative for the world
    state, so a declared effect it never sets is a no-op the planner would
    re-plan into until ``max_replans``; it is an ``action_failed`` instead
    (issue #31).  Opt out per action with
    ``effect_validator=lambda pre, post: True``.

    When validation rejects the post-state, the executor
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
    rejected = False
    error = "Effect validation failed"
    if action.effect_validator is not None:
        rejected = not action.validate_effects(state_before, state_after)
    elif check_effects:
        # Default rule for a returned dict: every declared effect *key* must
        # be addressed.  Values may differ from the declaration (a declared
        # ``{"computed": 0}`` is a planning placeholder for a runtime
        # number); a value that drifts from the plan's expectation is the
        # observer's deviation check to handle, a key that is never set is
        # a no-op the planner would re-plan into forever (issue #31).
        expected = action.get_effects(state_before)
        missing = [k for k in expected if k not in state_after]
        if missing:
            rejected = True
            error = (
                "Effect validation failed: execute() returned a dict that does not "
                f"set the declared effect key(s) {missing!r}"
            )
    if rejected:
        logger.warning("Action %r %s", action.name, error)
        return {
            "status": "action_failed",
            "world_state": state_before,
            "execution_history": [
                ActionResult(
                    action_name=action.name,
                    success=False,
                    state_before=state_before,
                    state_after=state_after,
                    error=error,
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
                cost=_realised_cost(action, state_before),
            )
        ],
    }


def _realised_cost(action: ActionSpec, state_before: Mapping[str, Any]) -> float | None:
    """The action's cost on the full pre-state, or ``None`` if it cannot be evaluated."""
    try:
        return float(action.get_cost(dict(state_before)))
    except Exception:  # a dynamic cost that needs keys this state lacks
        return None


HistoryState = Literal["full", "delta", "planning"]
HISTORY_STATES: tuple[str, ...] = ("full", "delta", "planning")


def _validate_history_state(value: str) -> HistoryState:
    if value not in HISTORY_STATES:
        raise ValueError(
            f"history_state must be one of {HISTORY_STATES}, got {value!r}"
        )
    return value  # type: ignore[return-value]


def _planning_keys_for(
    state: Mapping[str, Any], plan: Plan | None, known: Iterable[ActionSpec]
) -> set[str]:
    """Keys that preconditions, effects or the goal read: the ``"planning"`` subset."""
    actions = list(known) or (list(plan.actions) if plan is not None else [])
    keys: set[str] = set()
    for action in actions:
        keys.update(str(k) for k in action.preconditions)
        keys.update(str(k) for k in action.effect_key_set())
    goal = state.get("goal")
    goals = goal.goals if isinstance(goal, MultiGoal) else [goal]
    for g in goals:
        if isinstance(g, GoalSpec):
            keys.update(str(k) for k in g.conditions)
    return keys


def _compact_entry(
    entry: ActionResult, mode: HistoryState, planning_keys: set[str] | None
) -> ActionResult:
    before, after = entry.state_before, entry.state_after
    if mode == "delta":
        changed = {k: v for k, v in after.items() if k not in before or before[k] != v}
        prior = {k: before[k] for k in changed if k in before}
    else:  # "planning"
        keys = planning_keys or set()
        changed = {k: v for k, v in after.items() if k in keys}
        prior = {k: v for k, v in before.items() if k in keys}
    return replace(entry, state_before=prior, state_after=changed)


def _compact_history(
    result: dict[str, Any], mode: HistoryState, planning_keys: set[str] | None
) -> dict[str, Any]:
    """Shrink the ``ActionResult`` snapshots in an executor update (issue #74).

    ``"full"`` returns ``result`` unchanged.  ``"delta"`` keeps the keys an
    action changed, with their prior values in ``state_before``;
    ``"planning"`` keeps the planning-relevant keys of both snapshots.  Only
    the history entries shrink; the ``world_state`` channel is untouched.
    """
    if mode == "full" or "execution_history" not in result:
        return result
    compacted = [
        _compact_entry(e, mode, planning_keys) for e in result["execution_history"]
    ]
    return {**result, "execution_history": compacted}


def _build_failure(
    action: ActionSpec,
    exc: Exception,
    state_before: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the failure return dict for the executor.

    ``state_before`` is the state the action ran in, so it includes a
    typed-form answer merged by the HITL gate; returning it as the
    ``world_state`` keeps that answer in the checkpoint after a failure
    instead of dropping it with the action's partial effects (#75).
    """
    logger.warning("Action %r failed: %s", action.name, exc)
    return {
        "status": "action_failed",
        "world_state": dict(state_before),
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


def _blacklisting_counts(
    action: ActionSpec, failure_counts: Mapping[str, int] | None
) -> dict[str, int]:
    """Merge ``action`` at ``max_retries + 1`` into the existing counts.

    A policy decision (guard block, human denial, invalid form) must
    blacklist the action immediately, but it must not erase the retry
    budgets other actions have already spent (issue #33).
    """
    merged = dict(failure_counts or {})
    merged[action.name] = action.max_retries + 1
    return merged


def _build_guard_block(
    action: ActionSpec,
    guard_results: list[GuardResult],
    state_before: dict[str, Any],
    world_state: dict[str, Any],
    *,
    failure_counts: Mapping[str, int] | None = None,
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
    result["action_failure_counts"] = _blacklisting_counts(action, failure_counts)
    return result


def _approval_decision(resume_value: Any) -> str:
    """Interpret a ``Command(resume=...)`` value for a boolean gate.

    A boolean gate asks a yes/no question, so only an explicit boolean is a
    decision:

    * ``True`` or ``{"approved": True, ...}`` -> ``"approved"``
    * ``False`` or ``{"approved": False, "reason": ...}`` -> ``"denied"``
    * anything else -> ``"invalid"``: ``None``, a string (``"no"`` used to
      count as approval), a number, a list, a dict without ``approved`` or
      with a non-boolean ``approved``.  The caller asks again.
    """
    if resume_value is True:
        return "approved"
    if resume_value is False:
        return "denied"
    if isinstance(resume_value, dict):
        approved = resume_value.get("approved")
        if approved is True:
            return "approved"
        if approved is False:
            return "denied"
    return "invalid"


_APPROVAL_SHAPE_ERROR = {
    "loc": [],
    "msg": "resume with True or False, or {'approved': bool, 'reason': str}",
    "type": "wrong_shape",
}


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
    *,
    failure_counts: Mapping[str, int] | None = None,
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
        return _check_form_approval(
            action, world_state, state_before, failure_counts=failure_counts
        )

    payload: dict[str, Any] = {
        "type": "goap_action_approval",
        "action": action.name,
        "preconditions": dict(action.preconditions),
        "effects": dict(action.get_effects(world_state)),
        "world_state": dict(world_state),
        "attempt": 1,
    }
    while True:
        resume_value = interrupt(payload)
        decision = _approval_decision(resume_value)
        if decision != "invalid":
            break
        # Not an answer to a yes/no question: ask again and say why, rather
        # than guess.  Each round needs a human resume, so nothing loops on
        # its own.
        logger.info(
            "Action %r: approval answer %d not understood (%r); asking again",
            action.name,
            payload["attempt"],
            resume_value,
        )
        payload = {
            **payload,
            "attempt": payload["attempt"] + 1,
            "validation_errors": [dict(_APPROVAL_SHAPE_ERROR)],
            "rejected_payload": resume_value,
        }

    if decision == "approved":
        return None

    # Denied: build a failure result.  Pre-set the failure count to
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
    result["action_failure_counts"] = _blacklisting_counts(action, failure_counts)
    return result


def _check_form_approval(
    action: ActionSpec,
    world_state: dict[str, Any],
    state_before: dict[str, Any],
    *,
    failure_counts: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """Typed-form HITL gate.  See :func:`_check_human_approval`.

    The resume value is interpreted as follows:

    * ``False`` or ``{"approved": False, "reason": ...}`` (when the model has
      no ``approved`` field) declines the form.  That is a decision, so it
      is recorded and blacklisted exactly like a denied boolean gate.
    * A dict (or model instance) that validates is the answer; it is stored
      under ``human_input_key`` and the action runs.
    * Anything else is an input error, not a decision: the operator is
      asked again and the new interrupt payload carries ``attempt``,
      ``validation_errors`` (``loc`` / ``msg`` / ``type`` per error) and
      ``rejected_payload``.  Each round needs a human resume, so nothing
      loops on its own.
    """
    from pydantic import BaseModel, ValidationError

    model_cls: type[BaseModel] = action.require_human_approval  # type: ignore[assignment]
    key = action.human_input_key or action.name
    existing = world_state.get(key)
    if existing is not None:
        # The operator already answered (earlier attempt of this action, or
        # a pre-filled world state).  Re-asking on every retry threw that
        # answer away (#75); an answer that still validates is reused and
        # only a missing or stale one prompts again.
        try:
            model_cls.model_validate(existing)
        except ValidationError:
            logger.info(
                "Action %r: stored %r answer no longer validates; asking again",
                action.name,
                key,
            )
        else:
            logger.info("Action %r: reusing the operator's %r answer", action.name, key)
            return None
    payload: dict[str, Any] = {
        "type": "goap_action_form",
        "action": action.name,
        "form_schema": model_cls.model_json_schema(),
        "model_name": model_cls.__name__,
        "preconditions": dict(action.preconditions),
        "effects": dict(action.get_effects(world_state)),
        "world_state": dict(world_state),
        "attempt": 1,
    }
    parsed: BaseModel
    while True:
        resume_value = interrupt(payload)
        if _is_form_denial(resume_value, model_cls):
            reason = "denied by operator"
            if isinstance(resume_value, dict):
                reason = resume_value.get("reason", reason)
            result = _build_failure(
                action,
                RuntimeError(f"human_approval_denied: {reason}"),
                state_before,
                world_state,
            )
            result["action_failure_counts"] = _blacklisting_counts(
                action, failure_counts
            )
            return result
        if isinstance(resume_value, BaseModel):
            parsed = resume_value
            break
        if isinstance(resume_value, dict):
            try:
                parsed = model_cls.model_validate(resume_value)
                break
            except ValidationError as exc:
                errors = [
                    {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                    for e in exc.errors()
                ]
        else:
            errors = [
                {
                    "loc": [],
                    "msg": "form payload must be an object matching the schema",
                    "type": "wrong_shape",
                }
            ]
        logger.info(
            "Action %r: form submission %d rejected (%d error(s)); asking again",
            action.name,
            payload["attempt"],
            len(errors),
        )
        payload = {
            **payload,
            "attempt": payload["attempt"] + 1,
            "validation_errors": errors,
            "rejected_payload": resume_value,
        }

    # Approved: merge the validated form (as a JSON-friendly dict) into
    # world_state under human_input_key (default: action name).  Storing
    # the dict rather than the Pydantic instance keeps the value
    # round-trippable through any checkpointer serializer without
    # extra type registration.
    world_state[key] = parsed.model_dump(mode="json")
    return None


def _is_form_denial(resume_value: Any, model_cls: Any) -> bool:
    """``False``, or ``{"approved": False}`` when the form has no such field."""
    if resume_value is False:
        return True
    return (
        isinstance(resume_value, dict)
        and resume_value.get("approved") is False
        and "approved" not in getattr(model_cls, "model_fields", {})
    )


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
    "_approval_decision",
    "_check_human_approval",
    "_prepare_execution",
]
