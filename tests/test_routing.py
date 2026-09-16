"""Tests for deterministic case routing.

`determine_case_route` never queries OpenAI: the branch is derived purely
from `intent`, `order_resolution`, and `policy_assessment` - and internal
inconsistencies between them raise `RoutingError` rather than silently
landing on a plausible-looking branch.
"""

import pytest

from customer_ops.order_resolution import OrderResolution
from customer_ops.policies import PolicyAssessment
from customer_ops.routing import RoutingError, determine_case_route
from customer_ops.state import CASE_ROUTE_VALUES


def resolution(status="selected", selected_order_id="order-1001"):
    return OrderResolution(status=status, selected_order_id=selected_order_id)


def assessment(outcome, policy_code="ORDER_STATUS_INFO", requires_human_approval=False, reason="test"):
    return PolicyAssessment(
        outcome=outcome,
        policy_code=policy_code,
        requires_human_approval=requires_human_approval,
        reason=reason,
    )


def test_needs_clarification_routes_to_clarification():
    route = determine_case_route(
        intent="order_status",
        order_resolution=resolution(status="needs_clarification", selected_order_id=None),
        policy_assessment=assessment("needs_clarification", policy_code="ORDER_REQUIRED"),
    )
    assert route == "clarification"


def test_information_only_routes_to_information():
    route = determine_case_route(
        intent="order_status",
        order_resolution=resolution(),
        policy_assessment=assessment("information_only", policy_code="ORDER_STATUS_INFO"),
    )
    assert route == "information"


def test_eligible_without_approval_routes_to_action():
    route = determine_case_route(
        intent="cancel_order",
        order_resolution=resolution(),
        policy_assessment=assessment("eligible", policy_code="CANCEL_ALLOWED"),
    )
    assert route == "action"


def test_review_required_with_approval_routes_to_approval():
    route = determine_case_route(
        intent="refund_request",
        order_resolution=resolution(),
        policy_assessment=assessment(
            "review_required", policy_code="REFUND_REVIEW_REQUIRED", requires_human_approval=True
        ),
    )
    assert route == "approval"


def test_blocked_routes_to_blocked():
    route = determine_case_route(
        intent="cancel_order",
        order_resolution=resolution(),
        policy_assessment=assessment("blocked", policy_code="CANCEL_BLOCKED_STATUS"),
    )
    assert route == "blocked"


def test_not_applicable_routes_to_information():
    route = determine_case_route(
        intent="other",
        order_resolution=OrderResolution(status="not_required", selected_order_id=None),
        policy_assessment=assessment("not_applicable", policy_code="NOT_APPLICABLE"),
    )
    assert route == "information"


def test_eligible_with_approval_true_raises():
    with pytest.raises(RoutingError):
        determine_case_route(
            intent="cancel_order",
            order_resolution=resolution(),
            policy_assessment=assessment(
                "eligible", policy_code="CANCEL_ALLOWED", requires_human_approval=True
            ),
        )


def test_review_required_without_approval_raises():
    with pytest.raises(RoutingError):
        determine_case_route(
            intent="refund_request",
            order_resolution=resolution(),
            policy_assessment=assessment(
                "review_required", policy_code="REFUND_REVIEW_REQUIRED", requires_human_approval=False
            ),
        )


def test_route_result_is_controlled():
    route = determine_case_route(
        intent="cancel_order",
        order_resolution=resolution(),
        policy_assessment=assessment("eligible", policy_code="CANCEL_ALLOWED"),
    )
    assert route in CASE_ROUTE_VALUES


def test_deterministic_behavior():
    kwargs = dict(
        intent="cancel_order",
        order_resolution=resolution(),
        policy_assessment=assessment("eligible", policy_code="CANCEL_ALLOWED"),
    )
    assert determine_case_route(**kwargs) == determine_case_route(**kwargs)


def test_routing_performs_no_network_access(no_network):
    determine_case_route(
        intent="order_status",
        order_resolution=resolution(),
        policy_assessment=assessment("information_only"),
    )


# --- Additional state-consistency validation -----------------------------------


def test_needs_clarification_outcome_with_selected_resolution_raises():
    with pytest.raises(RoutingError):
        determine_case_route(
            intent="order_status",
            order_resolution=resolution(status="selected", selected_order_id="order-1001"),
            policy_assessment=assessment("needs_clarification", policy_code="ORDER_REQUIRED"),
        )


def test_eligible_outcome_without_selected_resolution_raises():
    with pytest.raises(RoutingError):
        determine_case_route(
            intent="cancel_order",
            order_resolution=OrderResolution(status="needs_clarification", selected_order_id=None),
            policy_assessment=assessment("eligible", policy_code="CANCEL_ALLOWED"),
        )


def test_not_applicable_outcome_with_non_other_intent_raises():
    with pytest.raises(RoutingError):
        determine_case_route(
            intent="order_status",
            order_resolution=OrderResolution(status="not_required", selected_order_id=None),
            policy_assessment=assessment("not_applicable", policy_code="NOT_APPLICABLE"),
        )
