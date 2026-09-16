"""LangGraph workflow for the Mercora Customer Operations Agent.

`intake` deterministically validates and normalizes the initial request.
`classify_request` then makes one model-powered call, through the injectable
`RequestClassifier` interface, to classify intent and urgency - this is the
only step the LLM participates in. `load_context` deterministically
retrieves the customer and their orders through the injectable
`CustomerOperationsStore` interface. `resolve_order` deterministically
decides which (if any) of the customer's own orders the request refers to,
never guessing. `evaluate_policy` deterministically evaluates explicit
Mercora business rules against the resolved order.

After policy evaluation, a genuine LangGraph conditional edge - driven only
by structured state (`intent`, `order_resolution`, `policy_assessment`),
never an LLM call - routes to one of five terminal branches: clarification,
information, safe-action proposal, approval preparation, or blocked. A
proposed action is a structured statement of operational *intent*, never
execution - no business mutation happens yet.

Graph shape:
START -> intake -> classify_request -> load_context -> resolve_order
      -> evaluate_policy -> (conditional) -> {clarification, information,
         propose_action, prepare_approval, blocked} -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from customer_ops.action_proposal import ActionProposalError, propose_action
from customer_ops.classifier import OpenAIRequestClassifier, RequestClassifier
from customer_ops.order_resolution import OrderResolution, resolve_order
from customer_ops.policies import PolicyAssessment, evaluate_policy
from customer_ops.routing import determine_case_route
from customer_ops.state import AuditEvent, CustomerOpsState
from tools.customer_data import CustomerOperationsStore, JsonCustomerOperationsStore


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


def make_context_loader_node(store: CustomerOperationsStore):
    """Build the `load_context` node bound to the given store.

    Dependency injection keeps the node decoupled from the JSON fixture
    files: it only ever calls `store.get_customer(...)` and
    `store.list_orders_for_customer(...)` through the
    `CustomerOperationsStore` interface. Records are serialized to plain
    dict/list values before entering state - no Pydantic objects in
    `CustomerOpsState`.
    """

    def context_loader_node(state: CustomerOpsState) -> dict:
        customer_id = state["customer_id"]
        customer = store.get_customer(customer_id)
        orders = store.list_orders_for_customer(customer_id)

        customer_context = customer.model_dump(mode="json")
        order_context = {
            "orders": [order.model_dump(mode="json") for order in orders],
            "count": len(orders),
        }

        audit_event: AuditEvent = {
            "step": "context_loading",
            "message": f"Loaded customer context and {len(orders)} order records.",
            "status": "ok",
        }

        return {
            "customer_context": customer_context,
            "order_context": order_context,
            "workflow_status": "context_loaded",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return context_loader_node


def make_order_resolution_node(resolver=resolve_order):
    """Build the `resolve_order` node.

    Purely deterministic and offline: `resolver` (the real `resolve_order`
    by default) only ever looks at `intent`, `customer_message`, and the
    already customer-scoped `order_context` already in state. It never
    calls OpenAI and never guesses an order reference.
    """

    def order_resolution_node(state: CustomerOpsState) -> dict:
        resolution = resolver(
            intent=state.get("intent"),
            customer_message=state["customer_message"],
            order_context=state.get("order_context"),
        )

        if resolution.status == "selected":
            message = f"Order {resolution.selected_order_id} selected for policy evaluation."
        elif resolution.status == "needs_clarification":
            message = "Order clarification required before policy evaluation."
        else:
            message = "No order reference required for this request."

        audit_event: AuditEvent = {
            "step": "order_resolution",
            "message": message,
            "status": "ok",
        }

        return {
            "selected_order_id": resolution.selected_order_id,
            "order_resolution": resolution.model_dump(mode="json"),
            "workflow_status": "order_resolved",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return order_resolution_node


def make_policy_evaluation_node(evaluator=evaluate_policy):
    """Build the `evaluate_policy` node.

    Purely deterministic and offline: `evaluator` (the real `evaluate_policy`
    by default) applies explicit Mercora business rules to the resolved
    order. It never calls OpenAI and never executes a business action - it
    only records eligibility and whether future human approval is required.
    """

    def policy_evaluation_node(state: CustomerOpsState) -> dict:
        resolution = OrderResolution.model_validate(state["order_resolution"])
        assessment = evaluator(
            intent=state.get("intent"),
            order_resolution=resolution,
            order_context=state.get("order_context"),
        )

        audit_event: AuditEvent = {
            "step": "policy_evaluation",
            "message": f"Policy evaluation completed with outcome {assessment.outcome}.",
            "status": "ok",
        }

        return {
            "policy_assessment": assessment.model_dump(mode="json"),
            "workflow_status": "policy_checked",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return policy_evaluation_node


def make_policy_router(router=determine_case_route):
    """Build the conditional-edge routing function used after `evaluate_policy`.

    Purely deterministic and offline: `router` (the real
    `determine_case_route` by default) decides the next branch from
    `intent`, `order_resolution`, and `policy_assessment` already in state.
    Returns a route key only, for LangGraph's conditional-edge dispatch -
    the destination branch node is the one that records `route` in state.
    """

    def policy_router(state: CustomerOpsState) -> str:
        resolution = OrderResolution.model_validate(state["order_resolution"])
        assessment = PolicyAssessment.model_validate(state["policy_assessment"])
        return router(
            intent=state.get("intent"),
            order_resolution=resolution,
            policy_assessment=assessment,
        )

    return policy_router


def make_clarification_node():
    """Build the `clarification` terminal branch node.

    The workflow cannot proceed until the customer identifies the relevant
    order. No order is guessed, no proposed action, no customer-facing
    response - response generation arrives in a later iteration.
    """

    def clarification_node(state: CustomerOpsState) -> dict:
        audit_event: AuditEvent = {
            "step": "clarification",
            "message": "Additional order identification is required.",
            "status": "ok",
        }
        return {
            "route": "clarification",
            "workflow_status": "clarification_required",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return clarification_node


def make_information_node():
    """Build the `information` terminal branch node.

    Covers both `information_only` (e.g. order_status with a selected
    order) and `not_applicable` (intent "other") policy outcomes - both are
    non-operational and terminal, so they share this branch rather than a
    separate `no_action` route (see `customer_ops/routing.py`). No proposed
    action; the selected order (if any) remains available in state for
    future response-generation iterations.
    """

    def information_node(state: CustomerOpsState) -> dict:
        audit_event: AuditEvent = {
            "step": "information",
            "message": "Request can be answered directly; no operational action required.",
            "status": "ok",
        }
        return {
            "route": "information",
            "workflow_status": "information_ready",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return information_node


def make_action_proposal_node(proposer=propose_action):
    """Build the safe-action-proposal terminal branch node.

    Only reached when policy outcome is `eligible` (e.g. cancel_order,
    address_change): proposes a structured `ProposedAction` describing what
    *could* be done - it never executes it, and never invents action input
    such as a new address.
    """

    def action_proposal_node(state: CustomerOpsState) -> dict:
        intent = state.get("intent")
        action = proposer(intent=intent, selected_order_id=state.get("selected_order_id"))
        if action.requires_human_approval:
            raise ActionProposalError(
                f"Action {action.action_type!r} for intent {intent!r} requires human "
                "approval and cannot be proposed on the safe-action branch."
            )

        audit_event: AuditEvent = {
            "step": "action_proposal",
            "message": f"Proposed action {action.action_type} for order {action.order_id}.",
            "status": "ok",
        }

        return {
            "route": "action",
            "proposed_action": action.model_dump(mode="json"),
            "workflow_status": "action_proposed",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return action_proposal_node


def make_approval_preparation_node(proposer=propose_action):
    """Build the approval-preparation terminal branch node.

    Only reached when policy outcome is `review_required` (e.g.
    refund_request, billing_issue, product_issue): proposes a structured
    `ProposedAction` with `requires_human_approval=True`. No interrupt, no
    human decision, and no execution yet - those arrive in a later
    iteration.
    """

    def approval_preparation_node(state: CustomerOpsState) -> dict:
        intent = state.get("intent")
        action = proposer(intent=intent, selected_order_id=state.get("selected_order_id"))
        if not action.requires_human_approval:
            raise ActionProposalError(
                f"Action {action.action_type!r} for intent {intent!r} does not require "
                "human approval and cannot be proposed on the approval branch."
            )

        audit_event: AuditEvent = {
            "step": "approval_required",
            "message": f"Action {action.action_type} for order {action.order_id} awaits human approval.",
            "status": "ok",
        }

        return {
            "route": "approval",
            "proposed_action": action.model_dump(mode="json"),
            "workflow_status": "awaiting_approval",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return approval_preparation_node


def make_blocked_node():
    """Build the `blocked` terminal branch node.

    Policy determined the requested operation is not allowed given the
    order's current status (e.g. cancelling a shipped order). This is a
    normal business outcome, not a technical error, and does not trigger
    automatic escalation.
    """

    def blocked_node(state: CustomerOpsState) -> dict:
        audit_event: AuditEvent = {
            "step": "blocked",
            "message": "Requested operation is not allowed for this order.",
            "status": "ok",
        }
        return {
            "route": "blocked",
            "workflow_status": "blocked",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return blocked_node


def build_customer_ops_graph(
    classifier: RequestClassifier | None = None,
    store: CustomerOperationsStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the Customer Operations graph.

    Current shape:
    START -> intake -> classify_request -> load_context -> resolve_order
          -> evaluate_policy -> (conditional routing) -> one of
             {clarification, propose_action, prepare_approval, information,
             blocked} -> END.

    Pass a fake `RequestClassifier` and/or `CustomerOperationsStore` (e.g. in
    tests) to avoid any OpenAI dependency or real fixture files. Omitting
    either uses the production defaults (`OpenAIRequestClassifier`,
    `JsonCustomerOperationsStore`); building the graph loads and validates
    the local JSON fixtures but makes no network call - only invoking the
    graph as far as `classify_request` reaches OpenAI. Every node from
    `resolve_order` onward, including the conditional edge itself, is
    deterministic and never reaches the network.
    """
    if classifier is None:
        classifier = OpenAIRequestClassifier()
    if store is None:
        store = JsonCustomerOperationsStore()

    graph = StateGraph(CustomerOpsState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify_request", make_classification_node(classifier))
    graph.add_node("load_context", make_context_loader_node(store))
    graph.add_node("resolve_order", make_order_resolution_node())
    graph.add_node("evaluate_policy", make_policy_evaluation_node())
    graph.add_node("clarification", make_clarification_node())
    graph.add_node("information", make_information_node())
    graph.add_node("propose_action", make_action_proposal_node())
    graph.add_node("prepare_approval", make_approval_preparation_node())
    graph.add_node("blocked", make_blocked_node())

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify_request")
    graph.add_edge("classify_request", "load_context")
    graph.add_edge("load_context", "resolve_order")
    graph.add_edge("resolve_order", "evaluate_policy")
    graph.add_conditional_edges(
        "evaluate_policy",
        make_policy_router(),
        {
            "clarification": "clarification",
            "information": "information",
            "action": "propose_action",
            "approval": "prepare_approval",
            "blocked": "blocked",
        },
    )
    graph.add_edge("clarification", END)
    graph.add_edge("information", END)
    graph.add_edge("propose_action", END)
    graph.add_edge("prepare_approval", END)
    graph.add_edge("blocked", END)
    return graph.compile()
