"""Tests for the Customer Operations LangGraph workflow.

These tests cover our own behavior (validation, normalization, classification
wiring, audit logging, workflow status) - not LangGraph internals. No test
makes a network or OpenAI API call: `classify_request` is always exercised
through `FakeRequestClassifier`, a deterministic `RequestClassifier` test
double defined below.
"""

import pytest

from customer_ops.classifier import ClassificationDecision, ClassificationError
from customer_ops.graph import InvalidCustomerRequest, build_customer_ops_graph


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


def make_graph(classifier=None):
    return build_customer_ops_graph(classifier or FakeRequestClassifier())


def test_graph_builds_with_injected_classifier():
    graph = make_graph()
    assert graph is not None


def test_valid_request_reaches_classified_status():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero saber donde esta mi pedido.",
        }
    )
    assert result["workflow_status"] == "classified"


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


def test_no_context_or_future_state_is_fabricated():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    assert "customer_context" not in result
    assert "order_context" not in result
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
    graph = make_graph(classifier)
    payload = {
        "request_id": "req-007",
        "customer_id": "cust-007",
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
    assert result["workflow_status"] == "classified"


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
    assert len(result["audit_log"]) == 2
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
