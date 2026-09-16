"""LangGraph workflow for the Mercora Customer Operations Agent.

`intake` deterministically validates and normalizes the initial request.
`classify_request` then makes one model-powered call, through the injectable
`RequestClassifier` interface, to classify intent and urgency. No data
retrieval, policy evaluation, or business actions happen yet - those arrive
in later iterations.

Graph shape: START -> intake -> classify_request -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from customer_ops.classifier import OpenAIRequestClassifier, RequestClassifier
from customer_ops.state import AuditEvent, CustomerOpsState


class InvalidCustomerRequest(ValueError):
    """Raised when the initial request payload fails basic validation."""


def intake_node(state: CustomerOpsState) -> dict:
    """Validate and normalize the initial request, deterministically.

    Rejects a missing/empty request_id, a missing/empty customer_id, or an
    empty/whitespace-only customer_message. On success, trims the customer
    message, sets workflow_status to "received", and appends one audit event.
    """
    request_id = state.get("request_id")
    customer_id = state.get("customer_id")
    customer_message = state.get("customer_message")

    if not request_id or not str(request_id).strip():
        raise InvalidCustomerRequest("request_id is required and cannot be empty.")
    if not customer_id or not str(customer_id).strip():
        raise InvalidCustomerRequest("customer_id is required and cannot be empty.")
    if not customer_message or not customer_message.strip():
        raise InvalidCustomerRequest(
            "customer_message is required and cannot be empty or whitespace-only."
        )

    normalized_message = customer_message.strip()

    audit_event: AuditEvent = {
        "step": "intake",
        "message": "Request received and validated.",
        "status": "ok",
    }

    return {
        "request_id": request_id,
        "customer_id": customer_id,
        "customer_message": normalized_message,
        "workflow_status": "received",
        "audit_log": [*state.get("audit_log", []), audit_event],
    }


def make_classification_node(classifier: RequestClassifier):
    """Build the `classify_request` node bound to the given classifier.

    Dependency injection keeps the node decoupled from the OpenAI SDK: it
    only ever calls `classifier.classify(...)` through the
    `RequestClassifier` interface.
    """

    def classification_node(state: CustomerOpsState) -> dict:
        customer_message = state["customer_message"]
        decision = classifier.classify(customer_message)

        audit_event: AuditEvent = {
            "step": "classification",
            "message": (
                f"Request classified as {decision.intent} with "
                f"{decision.urgency} urgency."
            ),
            "status": "ok",
        }

        return {
            "intent": decision.intent,
            "urgency": decision.urgency,
            "workflow_status": "classified",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return classification_node


def build_customer_ops_graph(classifier: RequestClassifier | None = None) -> CompiledStateGraph:
    """Build and compile the Customer Operations graph.

    Current shape: START -> intake -> classify_request -> END.

    Pass a fake `RequestClassifier` (e.g. in tests) to avoid any OpenAI
    dependency. Omitting `classifier` uses `OpenAIRequestClassifier`, the
    production default; building the graph does not itself make a network
    call - only invoking it as far as `classify_request` does.
    """
    if classifier is None:
        classifier = OpenAIRequestClassifier()

    graph = StateGraph(CustomerOpsState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify_request", make_classification_node(classifier))
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify_request")
    graph.add_edge("classify_request", END)
    return graph.compile()
