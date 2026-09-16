"""Tests for the minimal Customer Operations LangGraph workflow.

These tests cover our own behavior (validation, normalization, audit
logging, workflow status) - not LangGraph internals. No test makes a
network or OpenAI API call; the graph in this iteration has no node capable
of doing so.
"""

import pytest

from customer_ops.graph import InvalidCustomerRequest, build_customer_ops_graph


def make_graph():
    return build_customer_ops_graph()


def test_graph_builds_successfully():
    graph = make_graph()
    assert graph is not None


def test_valid_request_reaches_received_status():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero saber donde esta mi pedido.",
        }
    )
    assert result["workflow_status"] == "received"


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
    assert len(result["audit_log"]) == 1
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


def test_no_intent_is_fabricated():
    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    assert "intent" not in result
    assert "urgency" not in result


def test_no_customer_or_order_context_is_fabricated():
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
    graph = make_graph()
    payload = {
        "request_id": "req-007",
        "customer_id": "cust-007",
        "customer_message": "  Where is my order?  ",
    }
    result_a = graph.invoke(dict(payload))
    result_b = graph.invoke(dict(payload))
    assert result_a == result_b


def test_graph_invocation_makes_no_network_calls(monkeypatch):
    def fail_on_network(*args, **kwargs):
        raise AssertionError("Unexpected network access during graph invocation.")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)

    graph = make_graph()
    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    assert result["workflow_status"] == "received"
