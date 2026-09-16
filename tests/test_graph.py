"""Tests for the Customer Operations LangGraph workflow.

These tests cover our own behavior (validation, normalization, classification
wiring, context loading, audit logging, workflow status) - not LangGraph
internals. No test makes a network or OpenAI API call: `classify_request` is
always exercised through `FakeRequestClassifier` and `load_context` through
`FakeCustomerOperationsStore`, both deterministic test doubles defined below.
"""

import json

import pytest

from customer_ops.classifier import ClassificationDecision, ClassificationError
from customer_ops.graph import InvalidCustomerRequest, build_customer_ops_graph
from customer_ops.models import CustomerRecord, OrderRecord
from tools.customer_data import CustomerNotFoundError, OrderNotFoundError


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


def make_graph(classifier=None, store=None):
    return build_customer_ops_graph(
        classifier or FakeRequestClassifier(),
        store or default_store(),
    )


def test_graph_builds_with_injected_classifier_and_store():
    graph = make_graph()
    assert graph is not None


def test_valid_request_reaches_context_loaded_status():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero saber donde esta mi pedido.",
        }
    )
    assert result["workflow_status"] == "context_loaded"


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


def test_no_policy_or_action_state_is_fabricated():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    assert "policy_assessment" not in result
    assert "proposed_action" not in result
    assert "human_decision" not in result
    assert "action_result" not in result


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
            "customer_message": "Where is my order?",
        }
    )
    assert result["workflow_status"] == "context_loaded"


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
    assert len(result["audit_log"]) == 3
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
    assert result["workflow_status"] == "context_loaded"
    assert result["order_context"] == {"orders": [], "count": 0}
