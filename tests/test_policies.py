"""Tests for deterministic Mercora business-policy evaluation.

`evaluate_policy` never queries OpenAI: eligibility and the human-approval
flag come only from the explicit rules in `customer_ops/policies.py`,
applied to a resolved order already scoped to the customer's own
`order_context`.
"""

import pytest

from customer_ops.order_resolution import OrderResolution
from customer_ops.policies import PolicyEvaluationError, evaluate_policy


def order_context_with(order_id="order-1001", status="pending", payment_status="paid"):
    return {
        "orders": [{"order_id": order_id, "status": status, "payment_status": payment_status}],
        "count": 1,
    }


def selected(order_id="order-1001"):
    return OrderResolution(status="selected", selected_order_id=order_id)


def needs_clarification():
    return OrderResolution(status="needs_clarification", selected_order_id=None)


# --- order_status ------------------------------------------------------------------


def test_order_status_with_selected_order_is_information_only():
    assessment = evaluate_policy(
        intent="order_status",
        order_resolution=selected(),
        order_context=order_context_with(status="shipped"),
    )
    assert assessment.outcome == "information_only"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "ORDER_STATUS_INFO"


# --- cancel_order --------------------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_cancel_order_eligible_statuses(status):
    assessment = evaluate_policy(
        intent="cancel_order",
        order_resolution=selected(),
        order_context=order_context_with(status=status),
    )
    assert assessment.outcome == "eligible"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "CANCEL_ALLOWED"


@pytest.mark.parametrize("status", ["shipped", "delivered", "cancelled"])
def test_cancel_order_blocked_statuses(status):
    assessment = evaluate_policy(
        intent="cancel_order",
        order_resolution=selected(),
        order_context=order_context_with(status=status),
    )
    assert assessment.outcome == "blocked"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "CANCEL_BLOCKED_STATUS"


# --- address_change ------------------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_address_change_eligible_statuses(status):
    assessment = evaluate_policy(
        intent="address_change",
        order_resolution=selected(),
        order_context=order_context_with(status=status),
    )
    assert assessment.outcome == "eligible"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "ADDRESS_CHANGE_ALLOWED"


@pytest.mark.parametrize("status", ["shipped", "delivered", "cancelled"])
def test_address_change_blocked_statuses(status):
    assessment = evaluate_policy(
        intent="address_change",
        order_resolution=selected(),
        order_context=order_context_with(status=status),
    )
    assert assessment.outcome == "blocked"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "ADDRESS_CHANGE_BLOCKED_STATUS"


# --- refund_request ------------------------------------------------------------------


def test_refund_already_refunded_is_blocked():
    assessment = evaluate_policy(
        intent="refund_request",
        order_resolution=selected(),
        order_context=order_context_with(status="cancelled", payment_status="refunded"),
    )
    assert assessment.outcome == "blocked"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "ALREADY_REFUNDED"


def test_refund_delivered_and_paid_requires_review_and_approval():
    assessment = evaluate_policy(
        intent="refund_request",
        order_resolution=selected(),
        order_context=order_context_with(status="delivered", payment_status="paid"),
    )
    assert assessment.outcome == "review_required"
    assert assessment.requires_human_approval is True
    assert assessment.policy_code == "REFUND_REVIEW_REQUIRED"


def test_refund_other_combination_requires_review_and_approval():
    assessment = evaluate_policy(
        intent="refund_request",
        order_resolution=selected(),
        order_context=order_context_with(status="processing", payment_status="paid"),
    )
    assert assessment.outcome == "review_required"
    assert assessment.requires_human_approval is True
    assert assessment.policy_code == "REFUND_REVIEW_REQUIRED"


# --- billing_issue / product_issue --------------------------------------------------


def test_billing_issue_requires_review_and_approval():
    assessment = evaluate_policy(
        intent="billing_issue", order_resolution=selected(), order_context=order_context_with()
    )
    assert assessment.outcome == "review_required"
    assert assessment.requires_human_approval is True
    assert assessment.policy_code == "BILLING_REVIEW_REQUIRED"


def test_product_issue_requires_review_and_approval():
    assessment = evaluate_policy(
        intent="product_issue", order_resolution=selected(), order_context=order_context_with()
    )
    assert assessment.outcome == "review_required"
    assert assessment.requires_human_approval is True
    assert assessment.policy_code == "PRODUCT_REVIEW_REQUIRED"


# --- other -----------------------------------------------------------------------------


def test_other_intent_is_not_applicable():
    assessment = evaluate_policy(
        intent="other",
        order_resolution=OrderResolution(status="not_required", selected_order_id=None),
        order_context=None,
    )
    assert assessment.outcome == "not_applicable"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "NOT_APPLICABLE"


# --- missing order -------------------------------------------------------------------


def test_needs_clarification_order_resolution_produces_needs_clarification_outcome():
    assessment = evaluate_policy(
        intent="cancel_order", order_resolution=needs_clarification(), order_context=order_context_with()
    )
    assert assessment.outcome == "needs_clarification"
    assert assessment.requires_human_approval is False
    assert assessment.policy_code == "ORDER_REQUIRED"


# --- contract / error handling ----------------------------------------------------------


def test_policy_assessment_serialization_is_json_friendly():
    assessment = evaluate_policy(
        intent="order_status", order_resolution=selected(), order_context=order_context_with()
    )
    dumped = assessment.model_dump(mode="json")
    assert dumped["outcome"] == "information_only"
    assert dumped["policy_code"] == "ORDER_STATUS_INFO"
    assert dumped["requires_human_approval"] is False
    assert isinstance(dumped["reason"], str) and dumped["reason"]


def test_selected_order_not_in_context_raises():
    with pytest.raises(PolicyEvaluationError):
        evaluate_policy(
            intent="cancel_order",
            order_resolution=selected(order_id="order-9999"),
            order_context=order_context_with(order_id="order-1001"),
        )


def test_corrupted_not_required_state_raises_for_non_other_intent():
    with pytest.raises(PolicyEvaluationError):
        evaluate_policy(
            intent="cancel_order",
            order_resolution=OrderResolution(status="not_required", selected_order_id=None),
            order_context=order_context_with(),
        )


def test_deterministic_behavior():
    kwargs = dict(
        intent="cancel_order",
        order_resolution=selected(),
        order_context=order_context_with(status="pending"),
    )
    assert evaluate_policy(**kwargs) == evaluate_policy(**kwargs)


def test_evaluate_policy_performs_no_network_access(no_network):
    evaluate_policy(
        intent="order_status", order_resolution=selected(), order_context=order_context_with()
    )
