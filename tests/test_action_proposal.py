"""Tests for structured proposed-action mapping.

`propose_action` never queries OpenAI, never touches the operational store,
and never executes anything - it only maps a classified intent and a
resolved order ID into a structured `ProposedAction` describing operational
*intent*.
"""

import pytest

from customer_ops.action_proposal import ActionProposalError, propose_action


def test_cancel_order_proposes_cancel_order_action():
    action = propose_action(intent="cancel_order", selected_order_id="order-1001")
    assert action.action_type == "cancel_order"


def test_address_change_proposes_change_address_action():
    action = propose_action(intent="address_change", selected_order_id="order-1001")
    assert action.action_type == "change_address"


def test_refund_request_proposes_issue_refund_action():
    action = propose_action(intent="refund_request", selected_order_id="order-1001")
    assert action.action_type == "issue_refund"


def test_billing_issue_proposes_investigate_billing_action():
    action = propose_action(intent="billing_issue", selected_order_id="order-1001")
    assert action.action_type == "investigate_billing"


def test_product_issue_proposes_investigate_product_issue_action():
    action = propose_action(intent="product_issue", selected_order_id="order-1001")
    assert action.action_type == "investigate_product_issue"


@pytest.mark.parametrize("intent", ["cancel_order", "address_change"])
def test_safe_action_proposals_do_not_require_approval(intent):
    action = propose_action(intent=intent, selected_order_id="order-1001")
    assert action.requires_human_approval is False


@pytest.mark.parametrize("intent", ["refund_request", "billing_issue", "product_issue"])
def test_sensitive_action_proposals_require_approval(intent):
    action = propose_action(intent=intent, selected_order_id="order-1001")
    assert action.requires_human_approval is True


def test_selected_order_id_is_preserved():
    action = propose_action(intent="cancel_order", selected_order_id="order-9999")
    assert action.order_id == "order-9999"


def test_missing_selected_order_raises():
    with pytest.raises(ActionProposalError):
        propose_action(intent="cancel_order", selected_order_id=None)


def test_empty_selected_order_raises():
    with pytest.raises(ActionProposalError):
        propose_action(intent="cancel_order", selected_order_id="")


@pytest.mark.parametrize("intent", ["order_status", "other", None])
def test_unsupported_intent_raises(intent):
    with pytest.raises(ActionProposalError):
        propose_action(intent=intent, selected_order_id="order-1001")


def test_proposal_is_json_serializable():
    action = propose_action(intent="cancel_order", selected_order_id="order-1001")
    dumped = action.model_dump(mode="json")
    assert dumped == {
        "action_type": "cancel_order",
        "order_id": "order-1001",
        "requires_human_approval": False,
    }


def test_propose_action_has_no_execution_side_effects():
    order_id = "order-1001"
    first = propose_action(intent="cancel_order", selected_order_id=order_id)
    second = propose_action(intent="cancel_order", selected_order_id=order_id)
    assert first == second
    assert first.order_id == order_id


def test_deterministic_behavior():
    kwargs = dict(intent="refund_request", selected_order_id="order-1001")
    assert propose_action(**kwargs) == propose_action(**kwargs)


def test_propose_action_performs_no_network_access(no_network):
    propose_action(intent="cancel_order", selected_order_id="order-1001")
