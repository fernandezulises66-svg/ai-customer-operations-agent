"""Tests for the UI-agnostic Streamlit demo runtime helpers.

Fully offline: every test injects a fake classifier/action-input-extractor/
response-generator (mirroring `tests/test_graph.py`'s doubles) - real
OpenAI-backed dependencies are never constructed here.
"""

import uuid

import pytest

from customer_ops.classifier import ClassificationDecision
from customer_ops.action_inputs import AddressExtraction
from customer_ops.demo_runtime import (
    DemoRuntime,
    DemoRuntimeError,
    create_demo_runtime,
    resume_demo_case,
    start_demo_case,
)
from customer_ops.response_generator import CustomerResponse


class FakeClassifier:
    def __init__(self, intent: str, urgency: str = "medium"):
        self.intent = intent
        self.urgency = urgency

    def classify(self, customer_message: str) -> ClassificationDecision:
        return ClassificationDecision(intent=self.intent, urgency=self.urgency)


class FakeExtractor:
    def extract_new_shipping_address(self, customer_message: str) -> AddressExtraction:
        return AddressExtraction(new_shipping_address=None)


class FakeGenerator:
    def __init__(self, message: str = "FAKE_RESPONSE"):
        self.message = message
        self.call_count = 0

    def generate(self, response_context) -> CustomerResponse:
        self.call_count += 1
        return CustomerResponse(message=self.message)


def fresh_runtime(intent: str = "order_status", urgency: str = "low") -> DemoRuntime:
    return create_demo_runtime(
        classifier=FakeClassifier(intent, urgency),
        action_input_extractor=FakeExtractor(),
        response_generator=FakeGenerator(),
    )


# --- Thread-id ownership --------------------------------------------------------------


def test_new_runtime_receives_a_thread_id():
    runtime = fresh_runtime()
    assert runtime.thread_id
    assert runtime.thread_id.startswith("demo-")


def test_thread_id_is_stable_during_one_case():
    runtime = fresh_runtime("refund_request")
    original_thread_id = runtime.thread_id
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    assert runtime.thread_id == original_thread_id
    resume_demo_case(runtime, decision="approved")
    assert runtime.thread_id == original_thread_id


def test_two_new_runtimes_receive_distinct_thread_ids():
    first = fresh_runtime()
    second = fresh_runtime()
    assert first.thread_id != second.thread_id


# --- Non-HITL workflow -----------------------------------------------------------------


def test_initial_non_hitl_workflow_reaches_completed():
    runtime = fresh_runtime("order_status", "low")
    start_demo_case(
        runtime, request_id="req-1", customer_id="cust-001", customer_message="Hola, donde esta mi pedido order-1001?"
    )
    assert runtime.phase == "completed"
    assert runtime.latest_state["final_response"] == "FAKE_RESPONSE"
    assert runtime.interrupt_payload is None


# --- Approval workflow -------------------------------------------------------------------


def test_approval_workflow_reaches_awaiting_approval():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    assert runtime.phase == "awaiting_approval"


def test_approval_interrupt_payload_is_captured():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    assert runtime.interrupt_payload is not None
    assert runtime.interrupt_payload["action_type"] == "issue_refund"
    assert runtime.interrupt_payload["order_id"] == "order-1002"


def test_no_final_response_before_approval():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    assert runtime.phase == "awaiting_approval"
    assert "final_response" not in runtime.latest_state
    assert "action_result" not in runtime.latest_state


def test_approved_resume_uses_same_thread_id():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    before = runtime.thread_id
    resume_demo_case(runtime, decision="approved")
    assert runtime.thread_id == before


def test_rejected_resume_uses_same_thread_id():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-003", customer_message="refund order-1006")
    before = runtime.thread_id
    resume_demo_case(runtime, decision="rejected")
    assert runtime.thread_id == before


def test_approved_resume_reaches_completed():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    resume_demo_case(runtime, decision="approved")
    assert runtime.phase == "completed"
    assert runtime.latest_state["final_response"] == "FAKE_RESPONSE"
    assert runtime.interrupt_payload is None


def test_rejected_resume_reaches_completed():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-003", customer_message="refund order-1006")
    resume_demo_case(runtime, decision="rejected")
    assert runtime.phase == "completed"
    assert runtime.latest_state["final_response"] == "FAKE_RESPONSE"
    assert runtime.interrupt_payload is None


def test_resume_decision_is_passed_through_command_resume():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    resume_demo_case(runtime, decision="approved")
    assert runtime.latest_state["human_decision"] == "approved"
    assert runtime.latest_state["action_result"]["success"] is True


def test_resume_is_rejected_when_no_approval_is_pending():
    runtime = fresh_runtime("order_status", "low")
    start_demo_case(
        runtime, request_id="req-1", customer_id="cust-001", customer_message="Hola, donde esta mi pedido order-1001?"
    )
    assert runtime.phase == "completed"
    with pytest.raises(DemoRuntimeError):
        resume_demo_case(runtime, decision="approved")


def test_resume_is_rejected_on_a_brand_new_idle_runtime():
    runtime = fresh_runtime()
    with pytest.raises(DemoRuntimeError):
        resume_demo_case(runtime, decision="approved")


# --- Per-runtime isolation ---------------------------------------------------------------


def test_new_runtime_gets_fresh_store_and_checkpointer():
    first = fresh_runtime()
    second = fresh_runtime()
    assert first.store is not second.store
    assert first.graph is not second.graph


def test_mutation_in_one_runtime_does_not_leak_into_a_fresh_runtime():
    mutating = fresh_runtime("cancel_order", "medium")
    start_demo_case(mutating, request_id="req-1", customer_id="cust-002", customer_message="cancel order-1005")
    assert mutating.phase == "completed"
    mutated_order = mutating.store.get_order("order-1005")
    assert mutated_order.status == "cancelled"

    fresh = fresh_runtime()
    fresh_order = fresh.store.get_order("order-1005")
    assert fresh_order.status == "pending"


def test_reset_new_case_discards_pending_runtime_cleanly():
    pending = fresh_runtime("refund_request")
    start_demo_case(pending, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    assert pending.phase == "awaiting_approval"

    # Simulates the Streamlit "New case" action: build an independent
    # runtime rather than reusing/mutating the pending one.
    fresh = fresh_runtime()
    assert fresh.phase == "idle"
    assert fresh.interrupt_payload is None
    assert fresh.thread_id != pending.thread_id
    assert fresh.store is not pending.store
    # The abandoned pending runtime is untouched by creating a new one.
    assert pending.phase == "awaiting_approval"
    assert pending.interrupt_payload is not None


# --- Public interrupt payload PII-safety --------------------------------------------------


def test_public_interrupt_payload_contains_no_email_or_address_or_full_context():
    runtime = fresh_runtime("refund_request")
    start_demo_case(runtime, request_id="req-1", customer_id="cust-001", customer_message="refund order-1002")
    payload = runtime.interrupt_payload
    assert set(payload) == {"request_id", "action_type", "order_id", "message", "amount", "currency"}
    serialized = str(payload)
    assert "@" not in serialized  # no email address
    assert "Maple Street" not in serialized  # cust-001's real shipping address
    assert "customer_context" not in serialized
    assert "audit_log" not in serialized


# --- Import safety -----------------------------------------------------------------------


def test_importing_demo_runtime_makes_no_network_call(no_network):
    import importlib

    import customer_ops.demo_runtime as demo_runtime_module

    importlib.reload(demo_runtime_module)
