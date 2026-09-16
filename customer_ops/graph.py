"""Minimal LangGraph workflow for the Mercora Customer Operations Agent.

Iteration 1 wires a single deterministic node (`intake`) that validates and
normalizes the initial request. It performs no LLM calls, no classification,
no data retrieval, and no business decisions - those arrive in later
iterations.

Graph shape: START -> intake -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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


def build_customer_ops_graph() -> CompiledStateGraph:
    """Build and compile the minimal Customer Operations graph.

    Current shape: START -> intake -> END.
    """
    graph = StateGraph(CustomerOpsState)
    graph.add_node("intake", intake_node)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", END)
    return graph.compile()
