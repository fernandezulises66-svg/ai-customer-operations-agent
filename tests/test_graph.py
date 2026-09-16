"""Tests for the Customer Operations LangGraph workflow.

These tests cover our own behavior (validation, normalization, classification
wiring, context loading, order resolution, policy evaluation, conditional
routing, action proposal, action-input preparation, simulated execution,
human-in-the-loop approval/checkpointing, final response generation, audit
logging, workflow status) - not LangGraph internals. No test makes a network
or OpenAI API call: `classify_request` is always exercised through
`FakeRequestClassifier`, `load_context` through `FakeCustomerOperationsStore`
(or `InMemoryCustomerActionStore` for tests that also need mutation),
`change_address` input extraction through `FakeActionInputExtractor`, and
final response generation through `FakeResponseGenerator` - all deterministic
test doubles defined below. `resolve_order`, `evaluate_policy`, routing,
action proposal, and action execution are already fully deterministic and
offline, so the real implementations are used directly.

Every completed workflow now flows through a `final_response` node before
END, so `workflow_status` is `"completed"` and `final_response` is present
for every terminal result - EXCEPT an interrupted approval-required case,
which still pauses at `interrupt()` before ever reaching `final_response`.

Every compiled graph now carries a checkpointer, so every `.invoke(...)`
call requires `config={"configurable": {"thread_id": ...}}`. Most tests
don't care about a specific thread identity, so `invoke(graph, payload)`
below supplies a shared default - safe because each test builds its own
fresh graph (and thus fresh `InMemorySaver`), so reusing the same literal
thread_id across different tests never collides. Tests that resume an
interrupt, or that specifically exercise thread identity, manage their own
`thread_config(...)` explicitly.
"""

import json

import pytest
from langgraph.types import Command

from customer_ops.action_inputs import AddressExtraction
from customer_ops.approval import ApprovalError
from customer_ops.classifier import ClassificationDecision, ClassificationError
from customer_ops.graph import InvalidCustomerRequest, build_customer_ops_graph
from customer_ops.models import CustomerRecord, OrderRecord
from customer_ops.response_generator import CustomerResponse, ResponseContext
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


class FakeResponseGenerator:
    """Deterministic `CustomerResponseGenerator` test double.

    Records every `ResponseContext` it receives and returns a configurable
    `CustomerResponse` (or raises `exc` if set), so tests can assert the
    node called it exactly once, never before an interrupt, and never twice
    on resume.
    """

    def __init__(self, message: str = "FAKE_RESPONSE", exc: Exception | None = None):
        self.message = message
        self.exc = exc
        self.calls: list[ResponseContext] = []

    def generate(self, response_context: ResponseContext) -> CustomerResponse:
        self.calls.append(response_context)
        if self.exc is not None:
            raise self.exc
        return CustomerResponse(message=self.message)


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


class MutationCountingActionStore(InMemoryCustomerActionStore):
    """`InMemoryCustomerActionStore` that records every mutating call.

    Used by the human-in-the-loop tests to prove that no mutation happens
    before `interrupt()`/on replay, and that an approved resume causes
    exactly one store mutation - never zero, never more than one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_calls: list[tuple[str, str]] = []

    def cancel_order(self, order_id):
        self.mutation_calls.append(("cancel_order", order_id))
        return super().cancel_order(order_id)

    def change_shipping_address(self, order_id, new_address):
        self.mutation_calls.append(("change_shipping_address", order_id))
        return super().change_shipping_address(order_id, new_address)

    def issue_full_refund(self, order_id):
        self.mutation_calls.append(("issue_full_refund", order_id))
        return super().issue_full_refund(order_id)

    def create_billing_investigation(self, order_id):
        self.mutation_calls.append(("create_billing_investigation", order_id))
        return super().create_billing_investigation(order_id)

    def create_product_investigation(self, order_id):
        self.mutation_calls.append(("create_product_investigation", order_id))
        return super().create_product_investigation(order_id)


def make_graph(
    classifier=None,
    store=None,
    action_input_extractor=None,
    action_store=None,
    checkpointer=None,
    response_generator=None,
):
    return build_customer_ops_graph(
        classifier or FakeRequestClassifier(),
        store or default_store(),
        action_input_extractor or FakeActionInputExtractor(),
        action_store,
        checkpointer,
        response_generator or FakeResponseGenerator(),
    )


def thread_config(thread_id: str = "test-thread-001") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def invoke(graph, payload, thread_id: str = "test-thread-001"):
    """Invoke a checkpointed graph using a default test thread_id.

    See the module docstring for why reusing this default across tests is
    safe. Tests that need multiple invokes on the SAME thread (resume) or a
    specific/different thread_id build their own `thread_config(...)`.
    """
    return graph.invoke(payload, config=thread_config(thread_id))


def test_graph_builds_with_injected_classifier_and_store():
    graph = make_graph()
    assert graph is not None


def test_valid_request_reaches_completed_status():
    graph = make_graph()
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Quiero saber el estado de order-0001.",
        }
    )
    assert result["workflow_status"] == "completed"
    assert result["route"] == "information"
    assert result["final_response"] == "FAKE_RESPONSE"


def test_customer_message_is_trimmed():
    graph = make_graph()
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "  Quiero saber donde esta mi pedido.  ",
        }
    )
    assert result["customer_message"] == "Quiero saber donde esta mi pedido."


def test_first_audit_event_is_appended():
    graph = make_graph()
    result = invoke(graph,
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
    result = invoke(graph,
        {
            "request_id": "req-042",
            "customer_id": "cust-042",
            "customer_message": "Where is my order?",
        }
    )
    assert result["request_id"] == "req-042"
    assert result["customer_id"] == "cust-042"


def test_no_action_execution_state_is_fabricated():
    graph = make_graph()
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "Where is my order?",
        }
    )
    # A clarification case: no proposed action, no human decision, no
    # execution, no escalation - but final_response IS now expected, since
    # every terminal branch (including clarification) reaches it.
    assert "proposed_action" not in result
    assert "human_decision" not in result
    assert "action_result" not in result
    assert "escalation_reason" not in result
    assert result["final_response"] == "FAKE_RESPONSE"
    assert result["workflow_status"] == "completed"


def test_empty_message_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        invoke(graph,
            {"request_id": "req-001", "customer_id": "cust-001", "customer_message": ""}
        )


def test_whitespace_only_message_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        invoke(graph,
            {"request_id": "req-001", "customer_id": "cust-001", "customer_message": "   "}
        )


def test_missing_request_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        invoke(graph, {"customer_id": "cust-001", "customer_message": "Where is my order?"})


def test_empty_request_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        invoke(graph,
            {"request_id": "", "customer_id": "cust-001", "customer_message": "Where is my order?"}
        )


def test_missing_customer_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        invoke(graph, {"request_id": "req-001", "customer_message": "Where is my order?"})


def test_empty_customer_id_is_rejected():
    graph = make_graph()
    with pytest.raises(InvalidCustomerRequest):
        invoke(graph,
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
    # Two distinct thread_ids: two independent "cases" with the same input
    # should produce the same output - this is not a resume.
    result_a = graph.invoke(dict(payload), config=thread_config("det-thread-a"))
    result_b = graph.invoke(dict(payload), config=thread_config("det-thread-b"))
    assert result_a == result_b


def test_graph_invocation_makes_no_network_calls(no_network):
    graph = make_graph()
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-001",
            "customer_message": "What is the status of order-0001?",
        }
    )
    assert result["workflow_status"] == "completed"


def test_state_contains_only_json_friendly_data():
    graph = make_graph()
    result = invoke(graph,
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
    invoke(graph,
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
    invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
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
        invoke(graph,
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
    invoke(graph,
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
    invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
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
        invoke(graph,
            {
                "request_id": "req-001",
                "customer_id": "cust-does-not-exist",
                "customer_message": "Where is my order?",
            }
        )


def test_customer_with_zero_orders_is_valid():
    store = FakeCustomerOperationsStore(customers={"cust-777": make_customer("cust-777")})
    graph = make_graph(store=store)
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is my order?",
        }
    )
    assert result["workflow_status"] == "completed"
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
    result = invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
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
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Status of order-aaa please.",
        }
    )
    assert result["audit_log"][4]["step"] == "policy_evaluation"
    assert result["audit_log"][4]["status"] == "ok"
    assert "information_only" in result["audit_log"][4]["message"]


def test_exactly_seven_workflow_audit_stages():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status="shipped")]},
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = invoke(graph,
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
        "final_response",
    ]


def test_customer_and_order_facts_unchanged_after_policy_evaluation():
    customer = make_customer("cust-777")
    order = make_order("order-aaa", "cust-777", status="shipped")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": customer}, orders_by_customer={"cust-777": [order]}
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = invoke(graph,
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
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I want to cancel my order.",
        }
    )
    assert result["route"] == "clarification"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "completed"
    assert result["audit_log"][-2]["step"] == "clarification"
    assert result["audit_log"][-1]["step"] == "final_response"
    assert result["final_response"] == "FAKE_RESPONSE"


def test_information_branch_for_order_status_with_explicit_order():
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status="shipped")]},
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    graph = make_graph(classifier, store)
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is order-aaa?",
        }
    )
    assert result["route"] == "information"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "completed"
    assert result["audit_log"][-2]["step"] == "information"
    assert result["audit_log"][-1]["step"] == "final_response"
    assert result["final_response"] == "FAKE_RESPONSE"


def test_safe_cancellation_executes_and_updates_order_state():
    order = make_order("order-aaa", "cust-777", status="pending")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="cancel_order", urgency="medium")
    graph = make_graph(classifier, store, action_store=store)
    result = invoke(graph,
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
    assert result["workflow_status"] == "completed"
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
        "final_response",
    ]
    # Safe actions complete in a single invoke - no pause, no human decision.
    assert "__interrupt__" not in result
    assert "human_decision" not in result
    # action_result is retained after final_response, and the response is
    # generated after (and grounded in) execution.
    assert result["final_response"] == "FAKE_RESPONSE"


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
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": f"Please change the address on order-aaa to {new_address}.",
        }
    )
    assert result["route"] == "action"
    assert result["action_result"]["success"] is True
    assert result["action_result"]["action_type"] == "change_address"
    assert result["workflow_status"] == "completed"
    assert result["order_context"]["orders"][0]["shipping_address"] == new_address
    assert extractor.calls == [f"Please change the address on order-aaa to {new_address}."]
    assert result["final_response"] == "FAKE_RESPONSE"
    # Safe actions complete in a single invoke - no pause, no human decision.
    assert "__interrupt__" not in result
    assert "human_decision" not in result


def test_address_change_without_new_address_requires_clarification():
    order = make_order("order-aaa", "cust-777", status="processing", shipping_address="Old address")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    extractor = FakeActionInputExtractor(address=None)
    classifier = FakeRequestClassifier(intent="address_change", urgency="medium")
    graph = make_graph(classifier, store, action_input_extractor=extractor, action_store=store)
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please change the address on order-aaa.",
        }
    )
    assert result["route"] == "clarification"
    assert result["workflow_status"] == "completed"
    assert "action_result" not in result
    assert result["action_input"]["ready"] is False
    assert result["action_input"]["missing_fields"] == ["new_shipping_address"]
    # Nothing was invented and the order was never mutated.
    assert result["order_context"]["orders"][0]["shipping_address"] == "Old address"
    # The proposal is preserved for traceability even though it did not execute.
    assert result["proposed_action"]["action_type"] == "change_address"
    assert result["final_response"] == "FAKE_RESPONSE"


def test_approval_branch_for_refund_on_delivered_paid_order():
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    generator = FakeResponseGenerator()
    graph = make_graph(classifier, store, response_generator=generator)
    result = invoke(graph,
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
    # The approval-required case pauses BEFORE reaching final_response.
    assert "__interrupt__" in result
    assert "final_response" not in result
    assert generator.calls == []


def test_billing_issue_approval_branch_does_not_execute():
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="billing_issue", urgency="high")
    generator = FakeResponseGenerator()
    graph = make_graph(classifier, store, response_generator=generator)
    result = invoke(graph,
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
    assert "final_response" not in result
    assert generator.calls == []


def test_product_issue_approval_branch_does_not_execute():
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = FakeCustomerOperationsStore(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="product_issue", urgency="medium")
    generator = FakeResponseGenerator()
    graph = make_graph(classifier, store, response_generator=generator)
    result = invoke(graph,
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
    assert "final_response" not in result
    assert generator.calls == []


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
    result = invoke(graph,
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
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please cancel order-aaa.",
        }
    )
    assert result["workflow_status"] == "completed"


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
    invoke(graph,
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
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please change the address on order-aaa.",
        }
    )
    assert result["route"] == "blocked"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "completed"
    assert result["audit_log"][-2]["step"] == "blocked"
    assert result["audit_log"][-1]["step"] == "final_response"
    assert result["final_response"] == "FAKE_RESPONSE"


def test_information_branch_for_other_intent():
    store = FakeCustomerOperationsStore(customers={"cust-777": make_customer("cust-777")})
    classifier = FakeRequestClassifier(intent="other", urgency="low")
    graph = make_graph(classifier, store)
    result = invoke(graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Do you sell gift cards?",
        }
    )
    assert result["route"] == "information"
    assert "proposed_action" not in result
    assert result["workflow_status"] == "completed"
    assert result["audit_log"][-2]["step"] == "information"
    assert result["audit_log"][-1]["step"] == "final_response"
    assert result["final_response"] == "FAKE_RESPONSE"


@pytest.mark.parametrize(
    "intent,order_status,payment_status,message",
    [
        ("cancel_order", "processing", "paid", "Please cancel order-aaa."),
        ("address_change", "shipped", "paid", "Please change the address on order-aaa."),
        # Deliberately NOT an approval-required intent here: an interrupted
        # result's `__interrupt__` entry carries a randomly-generated
        # `Interrupt.id`, so it is neither JSON-serializable as-is nor
        # equal across two separate runs - see the dedicated HITL tests
        # below for interrupt/resume-specific coverage instead.
        ("cancel_order", "shipped", "paid", "I want to cancel my order."),
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
    result_a = invoke(build_graph(), dict(payload))
    result_b = invoke(build_graph(), dict(payload))
    json.dumps(result_a)  # raises TypeError if anything is not JSON-serializable
    assert result_a == result_b


# --- human-in-the-loop approval (interrupt / checkpoint / resume) ------------


def test_refund_approval_pauses_at_interrupt():
    """A. refund pauses."""
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid", total=79.0)
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("refund-pauses")

    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=config,
    )

    assert "__interrupt__" in result
    assert result["workflow_status"] == "awaiting_approval"
    assert "action_result" not in result
    assert "human_decision" not in result
    assert result["order_context"]["orders"][0]["payment_status"] == "paid"

    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload == {
        "request_id": "req-001",
        "action_type": "issue_refund",
        "order_id": "order-aaa",
        "message": "Approve simulated full refund for order order-aaa?",
        "amount": 79.0,
        "currency": "USD",
    }


def test_refund_approved_executes_and_mutates_exactly_once():
    """B. refund approved."""
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid", total=79.0)
    store = MutationCountingActionStore(customers=[make_customer("cust-777")], orders=[order])
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    generator = FakeResponseGenerator()
    graph = make_graph(classifier, store, action_store=store, response_generator=generator)
    config = thread_config("refund-approved")

    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=config,
    )
    assert generator.calls == []  # not called before approval
    result = graph.invoke(Command(resume={"decision": "approved"}), config=config)

    assert result["human_decision"] == "approved"
    assert result["order_context"]["orders"][0]["payment_status"] == "refunded"
    assert result["action_result"]["success"] is True
    assert result["action_result"]["action_type"] == "issue_refund"
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"
    assert result["audit_log"][-2]["step"] == "action_execution"
    assert result["audit_log"][-1]["step"] == "final_response"
    assert store.mutation_calls == [("issue_full_refund", "order-aaa")]
    # Exactly one response generated - never on the interrupted pass, never twice.
    assert len(generator.calls) == 1


def test_refund_rejected_performs_no_mutation():
    """C. refund rejected."""
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid", total=79.0)
    store = MutationCountingActionStore(customers=[make_customer("cust-777")], orders=[order])
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    generator = FakeResponseGenerator()
    graph = make_graph(classifier, store, action_store=store, response_generator=generator)
    config = thread_config("refund-rejected")

    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=config,
    )
    result = graph.invoke(Command(resume={"decision": "rejected"}), config=config)

    assert result["human_decision"] == "rejected"
    assert "action_result" not in result
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"
    assert result["audit_log"][-2]["step"] == "approval_rejected"
    assert result["audit_log"][-1]["step"] == "final_response"
    assert result["order_context"]["orders"][0]["payment_status"] == "paid"
    assert len(generator.calls) == 1
    assert store.mutation_calls == []


def test_billing_investigation_approved_creates_investigation_once():
    """D. billing investigation approved."""
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = MutationCountingActionStore(customers=[make_customer("cust-777")], orders=[order])
    classifier = FakeRequestClassifier(intent="billing_issue", urgency="high")
    generator = FakeResponseGenerator()
    graph = make_graph(classifier, store, action_store=store, response_generator=generator)
    config = thread_config("billing-approved")

    initial = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I was charged twice for order-aaa.",
        },
        config=config,
    )
    assert "__interrupt__" in initial
    assert store.mutation_calls == []  # nothing happens before resume
    assert generator.calls == []

    result = graph.invoke(Command(resume={"decision": "approved"}), config=config)

    assert result["action_result"]["success"] is True
    assert result["action_result"]["action_type"] == "investigate_billing"
    assert result["action_result"]["reference_id"] == "billing-investigation-order-aaa"
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"
    assert store.mutation_calls == [("create_billing_investigation", "order-aaa")]
    assert len(generator.calls) == 1


def test_billing_investigation_rejected_creates_no_investigation():
    """E. billing investigation rejected."""
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = MutationCountingActionStore(customers=[make_customer("cust-777")], orders=[order])
    classifier = FakeRequestClassifier(intent="billing_issue", urgency="high")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("billing-rejected")

    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I was charged twice for order-aaa.",
        },
        config=config,
    )
    result = graph.invoke(Command(resume={"decision": "rejected"}), config=config)

    assert "action_result" not in result
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"
    assert store.mutation_calls == []


def test_product_investigation_approved_executes_only_after_resume():
    """F. product investigation approved."""
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = MutationCountingActionStore(customers=[make_customer("cust-777")], orders=[order])
    classifier = FakeRequestClassifier(intent="product_issue", urgency="medium")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("product-approved")

    initial = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "The item in order-aaa arrived broken.",
        },
        config=config,
    )
    assert "__interrupt__" in initial
    assert store.mutation_calls == []

    result = graph.invoke(Command(resume={"decision": "approved"}), config=config)

    assert result["action_result"]["success"] is True
    assert result["action_result"]["action_type"] == "investigate_product_issue"
    assert result["action_result"]["reference_id"] == "product-investigation-order-aaa"
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"
    assert store.mutation_calls == [("create_product_investigation", "order-aaa")]


def test_product_investigation_rejected_performs_no_mutation():
    """G. product investigation rejected."""
    order = make_order("order-aaa", "cust-777", status="delivered")
    store = MutationCountingActionStore(customers=[make_customer("cust-777")], orders=[order])
    classifier = FakeRequestClassifier(intent="product_issue", urgency="medium")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("product-rejected")

    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "The item in order-aaa arrived broken.",
        },
        config=config,
    )
    result = graph.invoke(Command(resume={"decision": "rejected"}), config=config)

    assert "action_result" not in result
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"
    assert store.mutation_calls == []


def test_blocked_information_clarification_remain_non_interrupting():
    """J. blocked/information/clarification remain non-interrupting."""
    blocked_order = make_order("order-aaa", "cust-777", status="shipped")
    blocked_store = make_action_store(
        customers={"cust-777": make_customer("cust-777")}, orders_by_customer={"cust-777": [blocked_order]}
    )
    blocked_result = invoke(
        make_graph(FakeRequestClassifier(intent="address_change", urgency="medium"), blocked_store, action_store=blocked_store),
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Please change the address on order-aaa.",
        },
        thread_id="blocked-non-interrupting",
    )
    assert "__interrupt__" not in blocked_result

    info_result = invoke(
        make_graph(FakeRequestClassifier(intent="other", urgency="low"), default_store()),
        {"request_id": "req-001", "customer_id": "cust-001", "customer_message": "Do you sell gift cards?"},
        thread_id="information-non-interrupting",
    )
    assert "__interrupt__" not in info_result

    clarification_result = invoke(
        make_graph(FakeRequestClassifier(intent="cancel_order", urgency="medium"), default_store()),
        {"request_id": "req-001", "customer_id": "cust-001", "customer_message": "I want to cancel my order."},
        thread_id="clarification-non-interrupting",
    )
    assert "__interrupt__" not in clarification_result


def test_resuming_with_wrong_thread_id_does_not_resume_pending_approval():
    """K. wrong thread ID does not resume the pending approval."""
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    graph = make_graph(classifier, store, action_store=store)
    real_config = thread_config("refund-real-thread")

    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=real_config,
    )

    wrong_config = thread_config("refund-wrong-thread")
    # No checkpoint exists for this thread_id, so the graph runs fresh from
    # START with no payload at all - intake rejects it explicitly rather
    # than silently resuming the OTHER thread's pending approval.
    with pytest.raises(InvalidCustomerRequest):
        graph.invoke(Command(resume={"decision": "approved"}), config=wrong_config)

    # The real thread's pending approval, and the order, are untouched.
    assert store.get_order("order-aaa").payment_status == "paid"


def test_malformed_resume_payload_fails_explicitly_without_mutation():
    """L. malformed resume payload fails explicitly and does not mutate."""
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("refund-malformed")

    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=config,
    )

    with pytest.raises(ApprovalError):
        graph.invoke(Command(resume={"decision": "maybe"}), config=config)

    assert store.get_order("order-aaa").payment_status == "paid"


def test_no_mutation_before_interrupt_and_exactly_one_on_approved_resume():
    """M. repeated node execution safety."""
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid", total=50.0)
    store = MutationCountingActionStore(customers=[make_customer("cust-777")], orders=[order])
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("refund-replay-safety")

    initial = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=config,
    )
    assert "__interrupt__" in initial
    assert store.mutation_calls == []

    result = graph.invoke(Command(resume={"decision": "approved"}), config=config)
    assert store.mutation_calls == [("issue_full_refund", "order-aaa")]
    assert result["action_result"]["success"] is True


def test_interrupt_payload_excludes_pii():
    """Approval payload privacy: no email, address, or full context."""
    customer = make_customer("cust-777", email="priya.shah@example.com")
    order = make_order(
        "order-aaa",
        "cust-777",
        status="delivered",
        payment_status="paid",
        total=79.0,
        shipping_address="900 Confidential Ave, Privacy City, PC 00001, USA",
    )
    store = make_action_store(customers={"cust-777": customer}, orders_by_customer={"cust-777": [order]})
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("refund-pii-check")

    result = graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=config,
    )

    interrupt_payload = result["__interrupt__"][0].value
    assert set(interrupt_payload) == {"request_id", "action_type", "order_id", "message", "amount", "currency"}
    assert customer.email not in str(interrupt_payload)
    assert order.shipping_address not in str(interrupt_payload)


def test_hitl_flow_makes_no_network_calls(no_network):
    order = make_order("order-aaa", "cust-777", status="delivered", payment_status="paid")
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [order]},
    )
    classifier = FakeRequestClassifier(intent="refund_request", urgency="high")
    graph = make_graph(classifier, store, action_store=store)
    config = thread_config("refund-no-network")

    graph.invoke(
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "I would like a refund for order-aaa.",
        },
        config=config,
    )
    result = graph.invoke(Command(resume={"decision": "approved"}), config=config)
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"


def test_audit_log_does_not_contain_the_final_response_text():
    distinctive_message = "This exact sentence must never appear in audit_log."
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status="shipped")]},
    )
    classifier = FakeRequestClassifier(intent="order_status", urgency="low")
    generator = FakeResponseGenerator(message=distinctive_message)
    graph = make_graph(classifier, store, action_store=store, response_generator=generator)
    result = invoke(
        graph,
        {
            "request_id": "req-001",
            "customer_id": "cust-777",
            "customer_message": "Where is order-aaa?",
        },
    )
    assert result["final_response"] == distinctive_message
    for event in result["audit_log"]:
        assert distinctive_message not in event["message"]
    # The final_response audit event itself is a fixed, generic marker.
    assert result["audit_log"][-1] == {
        "step": "final_response",
        "message": "Customer-facing response generated.",
        "status": "ok",
    }


@pytest.mark.parametrize(
    "intent,order_status,message",
    [
        ("order_status", "shipped", "Where is order-aaa?"),
        ("cancel_order", "shipped", "I want to cancel my order."),  # -> clarification
        ("cancel_order", "pending", "Please cancel order-aaa."),  # -> safe action
        ("address_change", "shipped", "Please change the address on order-aaa."),  # -> blocked
    ],
)
def test_response_generator_called_exactly_once_per_completed_workflow(intent, order_status, message):
    store = make_action_store(
        customers={"cust-777": make_customer("cust-777")},
        orders_by_customer={"cust-777": [make_order("order-aaa", "cust-777", status=order_status)]},
    )
    classifier = FakeRequestClassifier(intent=intent, urgency="medium")
    generator = FakeResponseGenerator()
    graph = make_graph(classifier, store, action_store=store, response_generator=generator)
    result = invoke(
        graph,
        {"request_id": "req-001", "customer_id": "cust-777", "customer_message": message},
    )
    assert result["workflow_status"] == "completed"
    assert result["final_response"] == "FAKE_RESPONSE"
    assert len(generator.calls) == 1
