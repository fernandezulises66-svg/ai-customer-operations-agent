"""Deterministic Mercora business-policy evaluation.

Applies explicit, hand-written Mercora policy rules to a classified intent
and a resolved order. This is a small, testable business-rules component,
not a general rule engine or a RAG system over policy documents - every rule
below is deliberate. It never calls OpenAI: eligibility and human-approval
requirements are business rules, never LLM improvisation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from customer_ops.order_resolution import OrderResolution
from customer_ops.state import Intent, OrderContext, PolicyCode, PolicyOutcome

_CANCELLABLE_STATUSES = frozenset({"pending", "processing"})
_ADDRESS_CHANGEABLE_STATUSES = frozenset({"pending", "processing"})


class PolicyAssessment(BaseModel):
    """The structured outcome of deterministic Mercora policy evaluation.

    `reason` is a short observable business explanation (e.g. "order status
    is shipped"), never chain-of-thought or hidden model reasoning.
    """

    outcome: PolicyOutcome
    policy_code: PolicyCode
    requires_human_approval: bool
    reason: str


class PolicyEvaluationError(Exception):
    """Raised when policy evaluation cannot proceed due to invalid input state.

    `needs_clarification` is a legitimate business outcome, not this error -
    this is reserved for state that should never occur in normal operation
    (e.g. a `selected_order_id` absent from the customer's own order
    context).
    """


def evaluate_policy(
    intent: Intent,
    order_resolution: OrderResolution,
    order_context: OrderContext | None,
) -> PolicyAssessment:
    """Evaluate deterministic Mercora policy for a classified request.

    A `selected_order_id` is looked up only inside `order_context` - the
    same customer-scoped context order resolution ran against - so another
    customer's order can never be evaluated. If `order_resolution` claims a
    selected order that is absent from `order_context`, that is treated as
    corrupted state, not a policy decision, and raises
    `PolicyEvaluationError`.
    """
    if intent == "other":
        return PolicyAssessment(
            outcome="not_applicable",
            policy_code="NOT_APPLICABLE",
            requires_human_approval=False,
            reason="Request intent does not require policy evaluation.",
        )

    if order_resolution.status == "needs_clarification":
        return PolicyAssessment(
            outcome="needs_clarification",
            policy_code="ORDER_REQUIRED",
            requires_human_approval=False,
            reason="An order reference is required but could not be resolved.",
        )

    if order_resolution.status != "selected":
        raise PolicyEvaluationError(
            f"Cannot evaluate policy for intent {intent!r} without a selected order "
            f"(order_resolution.status={order_resolution.status!r})."
        )

    selected_order = _find_selected_order(order_resolution.selected_order_id, order_context)

    if intent == "order_status":
        return PolicyAssessment(
            outcome="information_only",
            policy_code="ORDER_STATUS_INFO",
            requires_human_approval=False,
            reason="Order status information can be shared directly.",
        )

    if intent == "cancel_order":
        return _evaluate_cancel_order(selected_order)

    if intent == "address_change":
        return _evaluate_address_change(selected_order)

    if intent == "refund_request":
        return _evaluate_refund_request(selected_order)

    if intent == "billing_issue":
        return PolicyAssessment(
            outcome="review_required",
            policy_code="BILLING_REVIEW_REQUIRED",
            requires_human_approval=True,
            reason="Billing issues require manual review.",
        )

    if intent == "product_issue":
        return PolicyAssessment(
            outcome="review_required",
            policy_code="PRODUCT_REVIEW_REQUIRED",
            requires_human_approval=True,
            reason="Product issues require manual review.",
        )

    raise PolicyEvaluationError(f"No policy defined for intent {intent!r}.")


def _find_selected_order(
    selected_order_id: str | None, order_context: OrderContext | None
) -> dict[str, Any]:
    orders = (order_context or {}).get("orders", [])
    for order in orders:
        if order.get("order_id") == selected_order_id:
            return order
    raise PolicyEvaluationError(
        f"selected_order_id={selected_order_id!r} was not found in the customer's order context."
    )


def _evaluate_cancel_order(order: dict[str, Any]) -> PolicyAssessment:
    if order["status"] in _CANCELLABLE_STATUSES:
        return PolicyAssessment(
            outcome="eligible",
            policy_code="CANCEL_ALLOWED",
            requires_human_approval=False,
            reason=f"Order status '{order['status']}' allows cancellation.",
        )
    return PolicyAssessment(
        outcome="blocked",
        policy_code="CANCEL_BLOCKED_STATUS",
        requires_human_approval=False,
        reason=f"Order status '{order['status']}' does not allow cancellation.",
    )


def _evaluate_address_change(order: dict[str, Any]) -> PolicyAssessment:
    if order["status"] in _ADDRESS_CHANGEABLE_STATUSES:
        return PolicyAssessment(
            outcome="eligible",
            policy_code="ADDRESS_CHANGE_ALLOWED",
            requires_human_approval=False,
            reason=f"Order status '{order['status']}' allows an address change.",
        )
    return PolicyAssessment(
        outcome="blocked",
        policy_code="ADDRESS_CHANGE_BLOCKED_STATUS",
        requires_human_approval=False,
        reason=f"Order status '{order['status']}' does not allow an address change.",
    )


def _evaluate_refund_request(order: dict[str, Any]) -> PolicyAssessment:
    if order["payment_status"] == "refunded":
        return PolicyAssessment(
            outcome="blocked",
            policy_code="ALREADY_REFUNDED",
            requires_human_approval=False,
            reason="Order has already been refunded.",
        )
    # The "delivered and paid" case called out in the spec produces the same
    # outcome/code as every other non-refunded combination - the dataset has
    # no reliable delivery date to build a refund-window policy on, so all
    # such cases are deferred to manual review rather than auto-approved.
    return PolicyAssessment(
        outcome="review_required",
        policy_code="REFUND_REVIEW_REQUIRED",
        requires_human_approval=True,
        reason="Refund requests require manual review before approval.",
    )
