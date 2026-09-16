"""Tests for the Customer Operations LangGraph workflow.

These tests cover our own behavior (validation, normalization, classification
wiring, context loading, order resolution, policy evaluation, conditional
routing, action proposal, action-input preparation, simulated execution,
audit logging, workflow status) - not LangGraph internals. No test makes a
network or OpenAI API call: `classify_request` is always exercised through
`FakeRequestClassifier`, `load_context` through `FakeCustomerOperationsStore`
(or `InMemoryCustomerActionStore` for tests that also need mutation), and
`change_address` input extraction through `FakeActionInputExtractor` - all
deterministic test doubles defined below. `resolve_order`, `evaluate_policy`,
routing, action proposal, and action execution are already fully
deterministic and offline, so the real implementations are used directly.
"""

import json

import pytest

from customer_ops.action_inputs import AddressExtraction
from customer_ops.classifier import ClassificationDecision, ClassificationError
from customer_ops.graph import InvalidCustomerRequest, build_customer_ops_graph
from customer_ops.models import CustomerRecord, OrderRecord
from tools.action_store import InMemoryCustomerActionStore
from tools.customer_data import (
    DEFAULT_CUSTOMERS_PATH,
    DEFAULT_ORDERS_PATH,
    CustomerNotFoundError,
    OrderNotFoundError,
)


class FakeRequestClassifier:
    """Deterministic `RequestClassifier` test double.

    Records every message it receives so tests can assert the node called it
    exactly once with the expected (normalized) customer message. Set `exc`
    to make `classify` raise instead of returning a decision.
    """

    def __init__(self, intent="order_status", urgency="low", exc: Exception | None = None):
        self.intent = intent
        self.urgency = urgency
        self.exc = exc
        self.calls: list[str] = []

    def classify(self, customer_message: str) -> ClassificationDecision:
        self.calls.append(customer_message)
        if self.exc is not None:
            raise self.exc
        return ClassificationDecision(intent=self.intent, urgency=self.urgency)


class FakeCustomerOperationsStore:
    """In-memory `CustomerOperationsStore` test double.

    Configured directly with `CustomerRecord`/`OrderRecord` objects so graph
    tests do not depend on the real JSON fixtures. Records every
    `customer_id` it is asked to look up.
    """

    def __init__(
        self,
        customers: dict[str, CustomerRecord],
        orders_by_customer: dict[str, list[OrderRecord]] | None = None,
    ):
        self._customers = customers
        self._orders_by_customer = orders_by_customer or {}
        self.get_customer_calls: list[str] = []
        self.list_orders_calls: list[str] = []

    def get_customer(self, customer_id: str) -> CustomerRecord:
        self.get_customer_calls.append(customer_id)
        try:
            return self._customers[customer_id]
        except KeyError:
            raise CustomerNotFoundError(f"No customer found with customer_id={customer_id!r}") from None

    def list_orders_for_customer(self, customer_id: str) -> list[OrderRecord]:
        self.list_orders_calls.append(customer_id)
        if customer_id not in self._customers:
            raise CustomerNotFoundError(f"No customer found with customer_id={customer_id!r}")
        return list(self._orders_by_customer.get(customer_id, []))

    def get_order(self, order_id: str) -> OrderRecord:
        for orders in self._orders_by_customer.values():
            for order in orders:
                if order.order_id == order_id:
                    return order
        raise OrderNotFoundError(f"No order found with order_id={order_id!r}") from None


class FakeActionInputExtractor:
    """Deterministic `ActionInputExtractor` test double.

    Returns `address` (possibly `None`) for every `change_address` proposal
    and records every message it was asked to extract from.
    """

    def __init__(self, address: str | None = None):
        self.address = address
        self.calls: list[str] = []

    def extract_new_shipping_address(self, customer_message: str) -> AddressExtraction:
        self.calls.append(customer_message)
        return AddressExtraction(new_shipping_address=self.address)


def make_customer(customer_id="cust-001", **overrides) -> CustomerRecord:
    fields = {
        "customer_id": customer_id,
        "name": "Test Customer",
        "email": "test.customer@example.com",
        "account_status": "active",
        "customer_tier": "standard",
    }
    fields.update(overrides)
    return CustomerRecord(**fields)


def make_order(order_id="order-0001", customer_id="cust-001", **overrides) -> OrderRecord:
    fields = {
        "order_id": order_id,
        "customer_id": customer_id,
        "status": "shipped",
        "payment_status": "paid",
        "total": 42.50,
        "currency": "USD",
        "shipping_address": "1 Test Street, Testville, TS 00000, USA",
        "tracking_number": "TRACK123",
        "items": [{"sku": "SKU-1", "name": "Test Item", "quantity": 1, "unit_price": 42.50}],
    }
    fields.update(overrides)
    return OrderRecord(**fields)


def default_store() -> FakeCustomerOperationsStore:
    """A store covering every customer_id used by generic tests below."""
    return FakeCustomerOperationsStore(
        customers={
            "cust-001": make_customer("cust-001"),
            "cust-042": make_customer("cust-042"),
            "cust-007": make_customer("cust-007"),
        },
        orders_by_customer={"cust-001": [make_order("order-0001", "cust-001")]},
    )


def make_action_store(
    customers: dict[str, CustomerRecord],
    orders_by_customer: dict[str, list[OrderRecord]] | None = None,
) -> InMemoryCustomerActionStore:
    """A combined read+mutate store for tests that reach action execution.

    Takes the same customers/orders_by_customer shape as
    `FakeCustomerOperationsStore` so tests can switch between the two, but
    returns the real `InMemoryCustomerActionStore` - the one class that
    implements both the read protocol `load_context` needs and the
    mutation protocol `execute_safe_action` needs against the SAME
    in-memory records, exactly as production does by default.
    """
    orders_by_customer = orders_by_customer or {}
    all_orders = [order for orders in orders_by_customer.values() for order in orders]
    return InMemoryCustomerActionStore(customers=list(customers.values()), orders=all_orders)


def make_graph(classifier=None, store=None, action_input_extractor=None, action_store=None):
    return build_customer_ops_graph(
        classifier or FakeRequestClassifier(),
        store or default_store(),
        action_input_extractor or FakeActionInputExtractor(),
        action_store,
    )


def test_graph_builds_with_injected_classifier_and_store():
    graph = make_graph()
    assert graph is not None


def test_valid_request_reaches_information_ready_status():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero saber el estado de order-0001.",
        }
    )
    assert result["workflow_status"] == "information_ready"
    assert result["route"] == "information"


def test_customer_message_is_trimmed():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "  Quiero saber donde esta mi pedido.  ",
        }
    )
    assert result["customer_message"] == "Quiero saber donde esta mi pedido."


def test_first_audit_event_is_appended():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    assert result["audit_log"][0]["step"] == "intake"
    assert result["audit_log"][0]["status"] == "ok"


def test_request_id_and_customer_id_are_preserved():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-042",
            "customer_id": "cust-042",
            "customer_message": "Where is my order?",
        }
    )
    assert result["request_id"] == "req-042"
    assert result["customer_id"] == "cust-042"


def test_no_action_or_response_state_is_fabricated():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    assert "proposed_action" not in result
    assert "human_decision" not in result
    assert "action_result" not in result
    assert "final_response" not in result
    assert "escalation_reason" not in result


def test_empty_message_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        graph.invoke(
            {"request_id": "req-001", "customer_id": "cust-001", "customer_message": ""}
        )


def test_whitespace_only_message_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        graph.invoke(
            {"request_id": "req-001", "customer_id": "cust-001", "customer_message": "   "}
        )


def test_missing_request_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        graph.invoke({"customer_id": "cust-001", "customer_message": "Where is my order?"})


def test_empty_request_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        graph.invoke(
            {"request_id": "", "customer_id": "cust-001", "customer_message": "Where is my order?"}
        )


def test_missing_customer_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        graph.invoke({"request_id": "req-001", "customer_message": "Where is my order?"})


def test_empty_customer_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        graph.invoke(
            {"request_id": "req-001", "customer_id": "", "customer_message": "Where is my order?"}
        )


def test_graph_invocation_is_deterministic():
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, default_store())
    payload = {
        "request_id": "req-007",
        "customer_id": "cust-001",
        "customer_message": "  Where is my order?  ",
    }
    result_a = graph.invoke(dict(payload))
    result_b = graph.invoke(dict(payload))
    assert result_a == result_b


def test_graph_invocation_makes_no_network_calls(no_network):
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "What is the status of order-0001?",
        }
    )
    assert result["workflow_status"] == "information_ready"


def test_state_contains_only_json_friendly_data():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    json.dumps(result)  # raises TypeError if anything is not JSON-serializable


# --- classify_request node --------------------------------------------------


def test_classifier_is_called_exactly_once():
    classifier = FakeRequestClassifier()
    graph = make_graph(classifier)
    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    assert len(classifier.calls) == 1


def test_classifier_receives_normalized_customer_message():
    classifier = FakeRequestClassifier()
    graph = make_graph(classifier)
    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "  Where is my order?  ",
        }
    )
    assert classifier.calls == ["Where is my order?"]


def test_classified_intent_enters_state():
    classifier = FakeRequestClassifier(intent="refund_request", urgency="medium")
    graph = make_graph(classifier)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero que me devuelvan el dinero.",
        }
    )
    assert result["intent"] == "refund_request"


def test_classified_urgency_enters_state():
    classifier = FakeRequestClassifier(intent="refund_request", urgency="medium")
    graph = make_graph(classifier)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero que me devuelvan el dinero.",
        }
    )
    assert result["urgency"] == "medium"


def test_classification_audit_event_is_appended():
    classifier = FakeRequestClassifier(intent="refund_request", urgency="medium")
    graph = make_graph(classifier)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero que me devuelvan el dinero.",
        }
    )
    assert result["audit_log"][1]["step"] == "classification"
    assert result["audit_log"][1]["status"] == "ok"
    assert "refund_request" in result["audit_log"][1]["message"]
    assert "medium" in result["audit_log"][1]["message"]


def test_classifier_failure_propagates():
    classifier = FakeRequestClassifier(exc=ClassificationError("simulated classification failure"))
    graph = make_graph(classifier)
    with pytest.raises(ClassificationError):
        graph.invoke(
            {
                "request_id": "req-001",
                "customer_id": "cust-001",
                "customer_message": "Where is my order?",
            }
        )


# --- load_context node -----------------------------------------------------


def test_context_store_customer_lookup_called_with_correct_customer_id():
    store = FakeCustomerOperationsStore(customers={"cust-777": make_customer("cust-777")})
    graph = make_graph(store=store)
    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert store.get_customer_calls == ["cust-777"]


def test_customer_orders_lookup_called_with_correct_customer_id():
    store = FakeCustomerOperationsStore(customers={"cust-777": make_customer("cust-777")})
    graph = make_graph(store=store)
    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert store.list_orders_calls == ["cust-777"]


def test_customer_context_populated_from_store():
    customer = make_customer("cust-777", name="Priya Shah", customer_tier="premium")
    store = FakeCustomerOperationsStore(customers={"cust-777": customer})
    graph = make_graph(store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert result["customer_context"] == customer.model_dump(mode="json")


def test_order_context_populated_from_store():
    customer = make_customer("cust-777")
    order_a = make_order("order-aaa", "cust-777")
    order_b = make_order("order-bbb", "cust-777", status="pending", tracking_number=None)
    store = FakeCustomerOperationsStore(
        customers={"cust-777": customer},
        orders_by_customer={"cust-777": [order_a, order_b]},
    )
    graph = make_graph(store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert result["order_context"] == {
        "orders": [order_a.model_dump(mode="json"), order_b.model_dump(mode="json")],
        "count": 2,
    }


def test_context_loading_audit_event_appended():
    customer = make_customer("cust-777")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": customer},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777")]},
    )
    graph = make_graph(store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert result["audit_log"][2]["step"] == "context_loading"
    assert result["audit_log"][2]["status"] == "ok"
    assert result["audit_log"][2]["message"] == "Loaded customer context and 1 order records."


def test_audit_log_does_not_contain_customer_or_order_details():
    customer = make_customer(
        "cust-777", name="Priya Shah", email="priya.shah@example.com"
    )
    order = make_order(
        "order-aaa",
        "cust-777",
        shipping_address="900 Confidential Ave, Privacy City, PC 00001, USA",
        tracking_number="SECRET-TRACK-1",
    )
    store = FakeCustomerOperationsStore(
        customers={"cust-777": customer}, orders_by_customer={"cust-777": [order]}
    )
    graph = make_graph(store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    sensitive_values = [customer.email, order.shipping_address, order.tracking_number]
    for event in result["audit_log"]:
        for value in sensitive_values:
            assert value not in event["message"]


def test_unknown_customer_failure_propagates():
    store = FakeCustomerOperationsStore(customers={})
    graph = make_graph(store=store)
    with pytest.raises(CustomerNotFoundError):
        graph.invoke(
            {
                "request_id": "req-001",
                "customer_id": "cust-does-not-exist",
                "customer_message": "Where is my order?",
            }
        )


def test_customer_with_zero_orders_is_valid():
    store = FakeCustomerOperationsStore(customers={"cust-777": make_customer("cust-777")})
    graph = make_graph(store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert result["workflow_status"] == "clarification_required"
    assert result["route"] == "clarification"
    assert result["order_context"] == {"orders": [], "count": 0}
    assert result["order_resolution"]["status"] == "needs_clarification"


# --- resolve_order node -----------------------------------------------------


def test_explicit_known_order_id_reaches_selected_order_id():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={
            "cust-777": [make_order("order-aaa", "cust-777"), make_order("order-bbb", "cust-777")]
        },
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Can you tell me the status of order-aaa?",
        }
    )
    assert result["selected_order_id"] == "order-aaa"
    assert result["order_resolution"]["status"] == "selected"


def test_no_order_id_produces_needs_clarification():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={
            "cust-777": [make_order("order-aaa", "cust-777"), make_order("order-bbb", "cust-777")]
        },
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert result["selected_order_id"] is None
    assert result["order_resolution"]["status"] == "needs_clarification"


def test_order_resolution_audit_event_appended():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777")]},
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Status of order-aaa please.",
        }
    )
    assert result["audit_log"][3]["step"] == "order_resolution"
    assert result["audit_log"][3]["status"] == "ok"
    assert "order-aaa" in result["audit_log"][3]["message"]


# --- evaluate_policy node ----------------------------------------------------


def test_policy_assessment_populated():
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status="pending")]},
    )
    classifier = FakeRequestClassifier(intent="cancel_order", urgency="medium")
    graph = make_graph(classifier, store, action_store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please cancel order-aaa.",
        }
    )
    assert result["policy_assessment"]["outcome"] == "eligible"
    assert result["policy_assessment"]["requires_human_approval"] is False
    assert result["policy_assessment"]["policy_code"] == "CANCEL_ALLOWED"


def test_policy_evaluation_audit_event_appended():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status="shipped")]},
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Status of order-aaa please.",
        }
    )
    assert result["audit_log"][4]["step"] == "policy_evaluation"
    assert result["audit_log"][4]["status"] == "ok"
    assert "information_only" in result["audit_log"][4]["message"]


def test_exactly_six_workflow_audit_stages():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status="shipped")]},
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Status of order-aaa please.",
        }
    )
    assert [event["step"] for event in result["audit_log"]] == [
        "intake",
        "classification",
        "context_loading",
        "order_resolution",
        "policy_evaluation",
        "information",
    ]


def test_customer_and_order_facts_unchanged_after_policy_evaluation():
    customer = make_customer("cust-777")
    order = make_order("order-aaa", "cust-777", status="shipped")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": customer}, orders_by_customer={"cust-777": [order]}
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Status of order-aaa please.",
        }
    )
    assert result["customer_context"] == customer.model_dump(mode="json")
    assert result["order_context"] == {"orders": [order.model_dump(mode="json")], "count": 1}


# --- conditional routing branches --------------------------------------------


def test_clarification_branch_for_ambiguous_order_reference():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={
            "cust-777": [make_order("order-aaa", "cust-777"), make_order("order-bbb", "cust-777")]
        },
    )
    classifier = FakeRequestClassifier(intent="cancel_order", urgency="medium")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I want to cancel my order.",
        }
    )
    assert result["route"] == "clarification"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "clarification_required"
    assert result["audit_log"][-1]["step"] == "clarification"


def test_information_branch_for_order_status_with_explicit_order():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status="shipped")]},
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is order-aaa?",
        }
    )
    assert result["route"] == "information"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "information_ready"
    assert result["audit_log"][-1]["step"] == "information"


def test_safe_cancellation_executes_and_updates_order_state():
    order = make_order("order-aaa", "cust-777", status="pending")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="cancel_order", urgency="medium")
    graph = make_graph(classifier, store, action_store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please cancel order-aaa.",
        }
    )
    assert result["route"] == "action"
    assert result["proposed_action"] == {
        "action_type": "cancel_order",
        "order_id": "order-aaa",
        "requires_human_approval": False,
    }
    assert result["action_input"] == {
        "ready": True,
        "action_type": "cancel_order",
        "parameters": {"order_id": "order-aaa"},
        "missing_fields": [],
    }
    assert result["action_result"]["success"] is True
    assert result["action_result"]["action_type"] == "cancel_order"
    assert result["action_result"]["order_id"] == "order-aaa"
    assert result["workflow_status"] == "action_executed"
    # State was synchronized: the order in context now shows the new status,
    # and it is the only order that changed.
    assert result["order_context"]["count"] == 1
    assert result["order_context"]["orders"][0]["status"] == "cancelled"
    assert [event["step"] for event in result["audit_log"]] == [
        "intake",
        "classification",
        "context_loading",
        "order_resolution",
        "policy_evaluation",
        "action_proposal",
        "action_input",
        "action_execution",
    ]


def test_address_change_with_explicit_address_executes():
    order = make_order("order-aaa", "cust-777", status="processing", shipping_address="Old address")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    new_address = "Calle Falsa 123, Cordoba"
    extractor = FakeActionInputExtractor(address=new_address)
    classifier = FakeRequestClassifier(intent="address_change", urgency="medium")
    graph = make_graph(classifier, store, action_input_extractor=extractor, action_store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": f"Please change the address on order-aaa to {new_address}.",
        }
    )
    assert result["route"] == "action"
    assert result["action_result"]["success"] is True
    assert result["action_result"]["action_type"] == "change_address"
    assert result["workflow_status"] == "action_executed"
    assert result["order_context"]["orders"][0]["shipping_address"] == new_address
    assert extractor.calls == [f"Please change the address on order-aaa to {new_address}."]


def test_address_change_without_new_address_requires_clarification():
    order = make_order("order-aaa", "cust-777", status="processing", shipping_address="Old address")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    extractor = FakeActionInputExtractor(address=None)
    classifier = FakeRequestClassifier(intent="address_change", urgency="medium")
    graph = make_graph(classifier, store, action_input_extractor=extractor, action_store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please change the address on order-aaa.",
        }
    )
    assert result["route"] == "clarification"
    assert result["workflow_status"] == "clarification_required"
    assert "action_result" not in result
    assert result["action_input"]["ready"] is False
    assert result["action_input"]["missing_fields"] == ["new_shipping_address"]
    # Nothing was invented and the order was never mutated.
    assert result["order_context"]["orders"][0]["shipping_address"] == "Old address"
    # The proposal is preserved for traceability even though it did not execute.
    assert result["proposed_action"]["action_type"] == "change_address"


def test_approval_branch_for_refund_on_delivered_paid_order():
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        }
    )
    assert result["route"] == "approval"
    assert result["proposed_action"] == {
        "action_type": "issue_refund",
        "order_id": "order-aaa",
        "requires_human_approval": True,
    }
    assert result["action_input"] == {
        "ready": True,
        "action_type": "issue_refund",
        "parameters": {"order_id": "order-aaa"},
        "missing_fields": [],
    }
    assert result["workflow_status"] == "awaiting_approval"
    assert result["audit_log"][-1]["step"] == "action_input"
    assert "human_decision" not in result
    assert "action_result" not in result
    # No execution occurred - payment_status remains exactly as it was.
    assert result["order_context"]["orders"][0]["payment_status"] == "paid"


def test_billing_issue_approval_branch_does_not_execute():
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="billing_issue", urgency="high")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I was charged twice for order-aaa.",
        }
    )
    assert result["route"] == "approval"
    assert result["workflow_status"] == "awaiting_approval"
    assert result["action_input"]["action_type"] == "investigate_billing"
    assert "action_result" not in result
    assert "human_decision" not in result


def test_product_issue_approval_branch_does_not_execute():
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="product_issue", urgency="medium")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "The item in order-aaa arrived broken.",
        }
    )
    assert result["route"] == "approval"
    assert result["workflow_status"] == "awaiting_approval"
    assert result["action_input"]["action_type"] == "investigate_product_issue"
    assert "action_result" not in result
    assert "human_decision" not in result


def test_audit_log_does_not_contain_new_shipping_address():
    order = make_order("order-aaa", "cust-777", status="processing", shipping_address="Old address")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    new_address = "900 Confidential Ave, Privacy City, PC 00001, USA"
    extractor = FakeActionInputExtractor(address=new_address)
    classifier = FakeRequestClassifier(intent="address_change", urgency="medium")
    graph = make_graph(classifier, store, action_input_extractor=extractor, action_store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": f"Please change the address on order-aaa to {new_address}.",
        }
    )
    for event in result["audit_log"]:
        assert new_address not in event["message"]


def test_action_execution_makes_no_network_calls(no_network):
    order = make_order("order-aaa", "cust-777", status="pending")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="cancel_order", urgency="medium")
    graph = make_graph(classifier, store, action_store=store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please cancel order-aaa.",
        }
    )
    assert result["workflow_status"] == "action_executed"


def test_graph_execution_does_not_modify_real_fixture_files():
    customers_before = DEFAULT_CUSTOMERS_PATH.read_text(encoding="utf-8")
    orders_before = DEFAULT_ORDERS_PATH.read_text(encoding="utf-8")

    action_store = InMemoryCustomerActionStore.from_json()
    classifier = FakeRequestClassifier(intent="cancel_order", urgency="medium")
    graph = build_customer_ops_graph(
        classifier=classifier,
        store=action_store,
        action_input_extractor=FakeActionInputExtractor(),
        action_store=action_store,
    )
    # order-1004 is 'processing' for cust-002 in the real fixture - eligible.
    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-002",
            "customer_message": "Please cancel order-1004.",
        }
    )

    assert DEFAULT_CUSTOMERS_PATH.read_text(encoding="utf-8") == customers_before
    assert DEFAULT_ORDERS_PATH.read_text(encoding="utf-8") == orders_before


def test_blocked_branch_for_address_change_on_shipped_order():
    order = make_order("order-aaa", "cust-777", status="shipped")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="address_change", urgency="medium")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please change the address on order-aaa.",
        }
    )
    assert result["route"] == "blocked"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "blocked"
    assert result["audit_log"][-1]["step"] == "blocked"


def test_information_branch_for_other_intent():
    store = FakeCustomerOperationsStore(customers={"cust-777": make_customer("cust-777")})
    classifier = FakeRequestClassifier(intent="other", urgency="low")
    graph = make_graph(classifier, store)
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Do you sell gift cards?",
        }
    )
    assert result["route"] == "information"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "information_ready"
    assert result["audit_log"][-1]["step"] == "information"


@pytest.mark.parametrize(
    "intent,order_status,payment_status,message",
    [
        ("cancel_order", "processing", "paid", "Please cancel order-aaa."),
        ("address_change", "shipped", "paid", "Please change the address on order-aaa."),
        ("refund_request", "delivered", "paid", "I would like a refund for order-aaa."),
        ("order_status", "shipped", "paid", "Where is order-aaa?"),
    ],
)
def test_routed_results_are_json_serializable_and_deterministic(
    intent, order_status, payment_status, message
):
    def build_graph():
        order = make_order("order-aaa", "cust-777", status=order_status, payment_status=payment_status)
        store = make_action_store(
            customers={"cust-777": make_customer("cust-777")},
            orders_by_customer={"cust-777": [order]},
        )
        classifier = FakeRequestClassifier(intent=intent, urgency="medium")
        return make_graph(classifier, store, action_store=store)

    payload = {"request_id": "req-001", "customer_id": "cust-777", "customer_message": message}
    result_a = build_graph().invoke(dict(payload))
    result_b = build_graph().invoke(dict(payload))
    json.dumps(result_a)  # raises TypeError if anything is not JSON-serializable
    assert result_a == result_b
