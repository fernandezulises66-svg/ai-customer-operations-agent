"""Tests for human-in-the-loop approval contracts.

`build_approval_request`/`parse_human_approval_response` are pure functions
with no OpenAI/network dependency - the actual approve/reject decision is
external human input, never derived from policy, intent, urgency, or any
model output.
"""

import pytest
from pydantic import ValidationError

from customer_ops.action_proposal import ProposedAction
from customer_ops.approval import (
    ApprovalError,
    ApprovalRequest,
    HumanApprovalResponse,
    build_approval_request,
    parse_human_approval_response,
)


def proposed(action_type="issue_refund", order_id="order-1001"):
    return ProposedAction(action_type=action_type, order_id=order_id, requires_human_approval=True)


def order(order_id="order-1001", total=79.0, currency="USD", **overrides):
    fields = {"order_id": order_id, "total": total, "currency": currency}
    fields.update(overrides)
    return fields


# --- HumanApprovalResponse validation --------------------------------------------


def test_valid_approved_response():
    response = parse_human_approval_response({"decision": "approved"})
    assert response.decision == "approved"


def test_valid_rejected_response():
    response = parse_human_approval_response({"decision": "rejected"})
    assert response.decision == "rejected"


def test_invalid_decision_rejected():
    with pytest.raises(ApprovalError):
        parse_human_approval_response({"decision": "maybe"})


def test_missing_decision_rejected():
    with pytest.raises(ApprovalError):
        parse_human_approval_response({})


def test_non_mapping_resume_value_rejected():
    with pytest.raises(ApprovalError):
        parse_human_approval_response("approved")


def test_none_resume_value_rejected():
    with pytest.raises(ApprovalError):
        parse_human_approval_response(None)


def test_human_approval_response_model_rejects_invalid_decision_directly():
    with pytest.raises(ValidationError):
        HumanApprovalResponse(decision="maybe")


# --- ApprovalRequest construction -------------------------------------------------


def test_approval_request_is_json_serializable():
    request = build_approval_request("req-001", proposed(), order())
    dumped = request.model_dump(mode="json")
    assert dumped == {
        "request_id": "req-001",
        "action_type": "issue_refund",
        "order_id": "order-1001",
        "message": "Approve simulated full refund for order order-1001?",
        "amount": 79.0,
        "currency": "USD",
    }


def test_approval_request_excludes_pii():
    request = build_approval_request("req-001", proposed(), order())
    dumped = request.model_dump(mode="json")
    assert set(dumped) == {"request_id", "action_type", "order_id", "message", "amount", "currency"}
    assert "email" not in dumped
    assert "shipping_address" not in dumped
    assert "customer_context" not in dumped
    assert "order_context" not in dumped


def test_refund_request_uses_correct_order():
    request = build_approval_request(
        "req-001", proposed(order_id="order-2002"), order(order_id="order-2002", total=42.5)
    )
    assert request.order_id == "order-2002"
    assert request.amount == 42.5


def test_amount_and_currency_come_from_the_validated_order():
    request = build_approval_request("req-001", proposed(), order(total=123.45, currency="USD"))
    assert request.amount == 123.45
    assert request.currency == "USD"


def test_non_refund_actions_have_no_amount_or_currency():
    for action_type in ("investigate_billing", "investigate_product_issue"):
        request = build_approval_request("req-001", proposed(action_type=action_type), order())
        assert request.amount is None
        assert request.currency is None


def test_non_refund_action_messages_describe_the_simulated_operation():
    billing = build_approval_request("req-001", proposed(action_type="investigate_billing"), None)
    assert billing.message == "Approve creation of a simulated billing investigation for order order-1001?"

    product = build_approval_request(
        "req-001", proposed(action_type="investigate_product_issue"), None
    )
    assert product.message == "Approve creation of a simulated product investigation for order order-1001?"


def test_refund_without_order_data_has_no_amount():
    request = build_approval_request("req-001", proposed(), None)
    assert request.amount is None
    assert request.currency is None


def test_deterministic_request_construction():
    kwargs = dict(request_id="req-001", proposed_action=proposed(), order=order())
    first = build_approval_request(**kwargs)
    second = build_approval_request(**kwargs)
    assert first == second


def test_approval_request_has_no_hidden_reasoning_fields():
    assert set(ApprovalRequest.model_fields) == {
        "request_id",
        "action_type",
        "order_id",
        "message",
        "amount",
        "currency",
    }


def test_approval_performs_no_network_access(no_network):
    build_approval_request("req-001", proposed(), order())
    parse_human_approval_response({"decision": "approved"})
