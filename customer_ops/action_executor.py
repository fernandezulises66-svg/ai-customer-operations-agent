"""Simulated action execution for the Mercora Customer Operations Agent.

`execute_action` is the one place a validated `ProposedAction` and
`ActionInputResult` become exactly one call against a `CustomerActionStore`
mutation method, returning a structured `ActionResult`. It never silently
returns a fake success: mismatched contracts, missing input, unsupported
action types, and store failures all raise `ActionExecutionError`.

Defense in depth: `issue_refund`, `investigate_billing`, and
`investigate_product_issue` require `human_approved=True` to execute here,
even though the graph in this iteration never calls this function for those
action types at all (they stop at `awaiting_approval`). This iteration's
`cancel_order`/`change_address` safe actions do not require approval.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from customer_ops.action_inputs import (
    ActionInputResult,
    CancelOrderInput,
    ChangeAddressInput,
    InvestigationInput,
    RefundInput,
)
from customer_ops.action_proposal import ProposedAction
from customer_ops.state import ActionType
from tools.action_store import ActionStoreError, CustomerActionStore
from tools.customer_data import OrderNotFoundError

# Action types that must not execute without explicit human approval.
_APPROVAL_REQUIRED_ACTION_TYPES = frozenset(
    {"issue_refund", "investigate_billing", "investigate_product_issue"}
)


class ActionResult(BaseModel):
    """The structured outcome of executing a simulated operational mutation.

    Contains only observable execution facts - no hidden reasoning, no raw
    store objects, and no customer PII.
    """

    action_type: ActionType
    success: bool
    order_id: str
    message: str
    reference_id: str | None = None


class ActionExecutionError(Exception):
    """Raised when an action cannot be executed as requested.

    Covers technical/contract inconsistencies (mismatched action type,
    input not ready, an unsupported action type, invalid parameters despite
    `ready=True`, a store precondition failure) and the approval safety
    gate - execution never silently reports a fake success.
    """


def execute_action(
    proposed_action: ProposedAction,
    action_input: ActionInputResult,
    action_store: CustomerActionStore,
    *,
    human_approved: bool = False,
) -> ActionResult:
    """Execute exactly one simulated mutation for a validated proposed action."""
    if proposed_action.action_type != action_input.action_type:
        raise ActionExecutionError(
            f"Proposed action type {proposed_action.action_type!r} does not match "
            f"action input type {action_input.action_type!r}."
        )

    if not action_input.ready or action_input.parameters is None:
        raise ActionExecutionError(
            f"Cannot execute action {proposed_action.action_type!r}: input is not ready "
            f"(missing_fields={action_input.missing_fields!r})."
        )

    action_type = proposed_action.action_type

    if action_type in _APPROVAL_REQUIRED_ACTION_TYPES and not human_approved:
        raise ActionExecutionError(
            f"Action {action_type!r} requires human approval before execution."
        )

    if action_type == "cancel_order":
        params = _validate_parameters(CancelOrderInput, action_input.parameters, action_type)
        order = _run_store_call(action_store.cancel_order, params.order_id, action_type=action_type)
        return ActionResult(
            action_type=action_type,
            success=True,
            order_id=order.order_id,
            message=f"Simulated cancel_order executed successfully for order {order.order_id}.",
        )

    if action_type == "change_address":
        params = _validate_parameters(ChangeAddressInput, action_input.parameters, action_type)
        order = _run_store_call(
            action_store.change_shipping_address,
            params.order_id,
            params.new_shipping_address,
            action_type=action_type,
        )
        return ActionResult(
            action_type=action_type,
            success=True,
            order_id=order.order_id,
            message=f"Simulated change_address executed successfully for order {order.order_id}.",
        )

    if action_type == "issue_refund":
        params = _validate_parameters(RefundInput, action_input.parameters, action_type)
        order = _run_store_call(action_store.issue_full_refund, params.order_id, action_type=action_type)
        return ActionResult(
            action_type=action_type,
            success=True,
            order_id=order.order_id,
            message=(
                f"Simulated issue_refund executed successfully for order {order.order_id} "
                f"(amount: {order.total} {order.currency})."
            ),
        )

    if action_type == "investigate_billing":
        params = _validate_parameters(InvestigationInput, action_input.parameters, action_type)
        reference_id = _run_store_call(
            action_store.create_billing_investigation, params.order_id, action_type=action_type
        )
        return ActionResult(
            action_type=action_type,
            success=True,
            order_id=params.order_id,
            message=f"Simulated investigate_billing executed successfully for order {params.order_id}.",
            reference_id=reference_id,
        )

    if action_type == "investigate_product_issue":
        params = _validate_parameters(InvestigationInput, action_input.parameters, action_type)
        reference_id = _run_store_call(
            action_store.create_product_investigation, params.order_id, action_type=action_type
        )
        return ActionResult(
            action_type=action_type,
            success=True,
            order_id=params.order_id,
            message=(
                f"Simulated investigate_product_issue executed successfully for order "
                f"{params.order_id}."
            ),
            reference_id=reference_id,
        )

    raise ActionExecutionError(f"No execution rule is defined for action type {action_type!r}.")


def _validate_parameters(model_cls, parameters: dict[str, object], action_type: str):
    try:
        return model_cls.model_validate(parameters)
    except ValidationError as exc:
        raise ActionExecutionError(
            f"Action input parameters for {action_type!r} are invalid: {exc}"
        ) from exc


def _run_store_call(store_method, *args, action_type: str):
    try:
        return store_method(*args)
    except (OrderNotFoundError, ActionStoreError) as exc:
        raise ActionExecutionError(f"Failed to execute {action_type!r}: {exc}") from exc
