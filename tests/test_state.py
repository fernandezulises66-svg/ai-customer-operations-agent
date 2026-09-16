"""Tests for the foundational workflow state contracts.

These tests check our controlled vocabularies and the shape of AuditEvent /
CustomerOpsState. They do not test typing internals - TypedDict fields are
not enforced at runtime, so we assert on the documented value sets and on
constructing state dicts that match the contract.
"""

from customer_ops.state import (
    ACTION_TYPE_VALUES,
    AuditEvent,
    CASE_ROUTE_VALUES,
    CustomerOpsState,
    HUMAN_DECISION_VALUES,
    INTENT_VALUES,
    ORDER_RESOLUTION_STATUS_VALUES,
    POLICY_CODE_VALUES,
    POLICY_OUTCOME_VALUES,
    URGENCY_VALUES,
    WORKFLOW_STATUS_VALUES,
)


def test_intent_values_are_the_documented_set():
    assert set(INTENT_VALUES) == {
        "order_status",
        "refund_request",
        "cancel_order",
        "address_change",
        "billing_issue",
        "product_issue",
        "other",
    }


def test_urgency_values_are_the_documented_set():
    assert set(URGENCY_VALUES) == {"low", "medium", "high"}


def test_workflow_status_values_are_the_documented_set():
    assert set(WORKFLOW_STATUS_VALUES) == {
        "received",
        "classified",
        "context_loaded",
        "order_resolved",
        "policy_checked",
        "clarification_required",
        "information_ready",
        "action_proposed",
        "action_input_ready",
        "awaiting_approval",
        "blocked",
        "action_executed",
        "escalated",
        "completed",
        "failed",
    }


def test_human_decision_values_are_the_documented_set():
    assert set(HUMAN_DECISION_VALUES) == {"approved", "rejected"}


def test_order_resolution_status_values_are_the_documented_set():
    assert set(ORDER_RESOLUTION_STATUS_VALUES) == {"selected", "not_required", "needs_clarification"}


def test_policy_outcome_values_are_the_documented_set():
    assert set(POLICY_OUTCOME_VALUES) == {
        "information_only",
        "eligible",
        "blocked",
        "review_required",
        "needs_clarification",
        "not_applicable",
    }


def test_policy_code_values_are_the_documented_set():
    assert set(POLICY_CODE_VALUES) == {
        "ORDER_STATUS_INFO",
        "CANCEL_ALLOWED",
        "CANCEL_BLOCKED_STATUS",
        "ADDRESS_CHANGE_ALLOWED",
        "ADDRESS_CHANGE_BLOCKED_STATUS",
        "REFUND_REVIEW_REQUIRED",
        "ALREADY_REFUNDED",
        "BILLING_REVIEW_REQUIRED",
        "PRODUCT_REVIEW_REQUIRED",
        "ORDER_REQUIRED",
        "NOT_APPLICABLE",
    }


def test_case_route_values_are_the_documented_set():
    assert set(CASE_ROUTE_VALUES) == {"clarification", "information", "action", "approval", "blocked"}


def test_action_type_values_are_the_documented_set():
    assert set(ACTION_TYPE_VALUES) == {
        "cancel_order",
        "change_address",
        "issue_refund",
        "investigate_billing",
        "investigate_product_issue",
    }


def test_audit_event_accepts_required_and_optional_fields():
    event: AuditEvent = {"step": "intake", "message": "Request received.", "status": "ok"}
    assert event["step"] == "intake"
    assert event["message"] == "Request received."
    assert event["status"] == "ok"


def test_audit_event_status_is_optional():
    event: AuditEvent = {"step": "intake", "message": "Request received."}
    assert "status" not in event


def test_customer_ops_state_accepts_minimal_initial_payload():
    state: CustomerOpsState = {
        "request_id": "req-001",
        "customer_id": "cust-001",
        "customer_message": "Where is my order?",
    }
    assert state["request_id"] == "req-001"
    assert "intent" not in state
    assert "audit_log" not in state


def test_customer_ops_state_supports_progressive_population():
    state: CustomerOpsState = {
        "request_id": "req-001",
        "customer_id": "cust-001",
        "customer_message": "Where is my order?",
        "workflow_status": "received",
        "audit_log": [{"step": "intake", "message": "Request received.", "status": "ok"}],
    }
    assert state["workflow_status"] == "received"
    assert len(state["audit_log"]) == 1
