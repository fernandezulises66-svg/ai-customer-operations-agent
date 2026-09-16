"""LangGraph workflow for the Mercora Customer Operations Agent.

`intake` deterministically validates and normalizes the initial request.
`classify_request` then makes one model-powered call, through the injectable
`RequestClassifier` interface, to classify intent and urgency. `load_context`
deterministically retrieves the customer and their orders through the
injectable `CustomerOperationsStore` interface. `resolve_order` deterministically
decides which (if any) of the customer's own orders the request refers to,
never guessing. `evaluate_policy` deterministically evaluates explicit
Mercora business rules against the resolved order.

After policy evaluation, a genuine LangGraph conditional edge - driven only
by structured state, never an LLM call - routes to one of five branches:

- `clarification` / `information` / `blocked` are terminal, exactly as in
  Iteration 5.
- `action` (safe: cancel_order, address_change) additionally prepares
  validated execution input (`prepare_action_input`, model-backed only for
  `change_address`) and then, via a second conditional edge, either
  executes the simulated mutation (`execute_safe_action`) or - if required
  input is missing and was NOT invented - falls back to `clarification`.
- `approval` (sensitive: issue_refund, investigate_billing,
  investigate_product_issue) also prepares validated execution input, then
  reaches `human_approval` - a real LangGraph `interrupt()` that pauses the
  graph and exposes a minimal, PII-free `ApprovalRequest` to an external
  reviewer. Resuming with `Command(resume={"decision": ...})` on the SAME
  thread_id routes to `execute_approved_action` (approved) or
  `approval_rejected` (rejected) - execution only ever happens after an
  explicit "approved" decision, never automatically.

Mutations run only against the in-memory simulated `CustomerActionStore` -
`data/*.json` fixtures are never written to, and a successful mutation is
synchronized back into `order_context` so graph state never shows stale
data.

Because LangGraph re-executes an interrupted node from its beginning on
resume, everything in `human_approval_node` before `interrupt()` is a pure
read/validation with no mutation, network call, or audit-log append.

Graph shape:
START -> intake -> classify_request -> load_context -> resolve_order
      -> evaluate_policy -> (conditional) -> {clarification, information,
         propose_action -> prepare_action_input -> (conditional) ->
            {execute_safe_action, clarification},
         prepare_approval -> prepare_approval_input -> human_approval
            -- interrupt() --> (resume) --> {execute_approved_action,
            approval_rejected},
         blocked} -> END
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from customer_ops.action_executor import ActionExecutionError, execute_action
from customer_ops.action_inputs import (
    ActionInputExtractor,
    ActionInputResult,
    OpenAIActionInputExtractor,
    prepare_action_input,
)
from customer_ops.action_proposal import ActionProposalError, ProposedAction, propose_action
from customer_ops.approval import ApprovalError, build_approval_request, parse_human_approval_response
from customer_ops.classifier import OpenAIRequestClassifier, RequestClassifier
from customer_ops.order_resolution import OrderResolution, resolve_order
from customer_ops.policies import PolicyAssessment, evaluate_policy
from customer_ops.routing import determine_case_route
from customer_ops.state import AuditEvent, CustomerOpsState
from tools.action_store import CustomerActionStore, InMemoryCustomerActionStore
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

    Reached either when order resolution could not identify the relevant
    order, or when a safe action's required execution input turned out to
    be missing (see `prepare_action_input`'s conditional edge) - in both
    cases the specific reason was already recorded by the preceding audit
    event, so this node's own message stays generic. No order is guessed,
    no input is invented, no customer-facing response - response generation
    arrives in a later iteration.
    """

    def clarification_node(state: CustomerOpsState) -> dict:
        audit_event: AuditEvent = {
            "step": "clarification",
            "message": "Additional information is required before this request can proceed.",
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
    """Build the `propose_action` node for the safe-action branch.

    Only reached when policy outcome is `eligible` (e.g. cancel_order,
    address_change): proposes a structured `ProposedAction` describing what
    *could* be done. Feeds into `prepare_action_input` next - this node
    itself never executes anything and never invents action input.
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
    """Build the `prepare_approval` node for the approval branch.

    Only reached when policy outcome is `review_required` (e.g.
    refund_request, billing_issue, product_issue): proposes a structured
    `ProposedAction` with `requires_human_approval=True`. Feeds into
    `prepare_approval_input`, then `human_approval` - this node itself
    performs no interrupt and no execution.
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


def make_action_input_node(extractor: ActionInputExtractor, preparer=prepare_action_input):
    """Build the `prepare_action_input` node for the safe-action branch.

    Deterministic and offline for `cancel_order`; calls `extractor` exactly
    once for `change_address`. Does not itself finalize `route` or
    `workflow_status` - the conditional edge that follows sends execution
    to `execute_safe_action` (ready) or back to `clarification` (missing
    input), and that destination node records the outcome.
    """

    def action_input_node(state: CustomerOpsState) -> dict:
        proposed_action = ProposedAction.model_validate(state["proposed_action"])
        result = preparer(proposed_action, state["customer_message"], extractor)

        if result.ready:
            message = f"Action input prepared for {result.action_type}."
        else:
            message = f"Additional action input is required: {', '.join(result.missing_fields)}."

        audit_event: AuditEvent = {
            "step": "action_input",
            "message": message,
            "status": "ok",
        }

        return {
            "action_input": result.model_dump(mode="json"),
            "workflow_status": "action_input_ready",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return action_input_node


def make_approval_action_input_node(extractor: ActionInputExtractor, preparer=prepare_action_input):
    """Build the `prepare_approval_input` node for the approval branch.

    Prepares the same validated execution input as `make_action_input_node`
    so a future human reviewer knows exactly what operation is waiting -
    then always proceeds to `human_approval`. None of the three
    approval-required action types ever need the model extractor or can be
    "not ready", so no conditional branching is needed here.
    """

    def approval_action_input_node(state: CustomerOpsState) -> dict:
        proposed_action = ProposedAction.model_validate(state["proposed_action"])
        result = preparer(proposed_action, state["customer_message"], extractor)

        audit_event: AuditEvent = {
            "step": "action_input",
            "message": "Action input prepared and awaiting human approval.",
            "status": "ok",
        }

        return {
            "action_input": result.model_dump(mode="json"),
            "workflow_status": "awaiting_approval",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return approval_action_input_node


def _find_order(order_context, order_id: str) -> dict | None:
    """Look up one order dict inside `order_context` by ID, or `None`."""
    for order in (order_context or {}).get("orders", []):
        if order.get("order_id") == order_id:
            return order
    return None


def make_human_approval_node():
    """Build the `human_approval` node: a real LangGraph `interrupt()`.

    LangGraph re-executes an interrupted node from its beginning on every
    resume, so everything before `interrupt()` here is a pure read and
    validation - it builds the minimal `ApprovalRequest` but performs no
    mutation, no network call, and no audit-log append. Only after
    `interrupt()` returns the human's decision (on resume) does this node
    validate it, append the one `human_approval` audit event, and return a
    `Command` routing to `execute_approved_action` or `approval_rejected` -
    LangGraph-native resume routing, not emulated outside the graph.
    """

    def human_approval_node(state: CustomerOpsState) -> Command:
        proposed_action = ProposedAction.model_validate(state["proposed_action"])
        action_input = ActionInputResult.model_validate(state["action_input"])

        if not proposed_action.requires_human_approval:
            raise ApprovalError(
                f"Action {proposed_action.action_type!r} does not require human approval."
            )
        if not action_input.ready:
            raise ApprovalError(
                f"Cannot request approval for action {proposed_action.action_type!r}: "
                f"input is not ready (missing_fields={action_input.missing_fields!r})."
            )
        if state.get("action_result") is not None:
            raise ApprovalError("An action_result already exists; refusing to request approval again.")

        order = _find_order(state.get("order_context"), proposed_action.order_id)
        approval_request = build_approval_request(state["request_id"], proposed_action, order)

        decision_payload = interrupt(approval_request.model_dump(mode="json"))

        response = parse_human_approval_response(decision_payload)

        audit_event: AuditEvent = {
            "step": "human_approval",
            "message": f"Human reviewer {response.decision} the proposed simulated action.",
            "status": "ok",
        }
        update = {
            "human_decision": response.decision,
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

        if response.decision == "approved":
            return Command(update=update, goto="execute_approved_action")
        return Command(update=update, goto="approval_rejected")

    return human_approval_node


def make_approved_action_execution_node(action_store: CustomerActionStore, executor=execute_action):
    """Build the `execute_approved_action` node.

    Only reached after `human_approval` records an "approved" decision.
    Re-validates the preconditions the executor itself cannot infer from
    context alone (that this specific action genuinely required approval),
    then calls the same `execute_action` used for safe actions, explicitly
    passing `human_approved=True` - the executor's own defense-in-depth
    check still applies and is never bypassed.
    """

    def execute_approved_action_node(state: CustomerOpsState) -> dict:
        if state.get("human_decision") != "approved":
            raise ActionExecutionError(
                f"Cannot execute: human_decision={state.get('human_decision')!r}, expected 'approved'."
            )

        proposed_action = ProposedAction.model_validate(state["proposed_action"])
        if not proposed_action.requires_human_approval:
            raise ActionExecutionError(
                f"Action {proposed_action.action_type!r} does not require human approval."
            )

        action_input = ActionInputResult.model_validate(state["action_input"])
        if not action_input.ready:
            raise ActionExecutionError("Cannot execute: action input is not ready.")

        result = executor(proposed_action, action_input, action_store, human_approved=True)
        updated_order = action_store.get_order(result.order_id)

        audit_event: AuditEvent = {
            "step": "action_execution",
            "message": result.message,
            "status": "ok",
        }

        return {
            "action_result": result.model_dump(mode="json"),
            "order_context": _with_updated_order(state.get("order_context"), updated_order),
            "workflow_status": "action_executed",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return execute_approved_action_node


def make_approval_rejected_node():
    """Build the `approval_rejected` terminal node.

    Only reached after `human_approval` records a "rejected" decision.
    Performs zero mutation - `action_result` stays absent, and no
    escalation is created automatically.
    """

    def approval_rejected_node(state: CustomerOpsState) -> dict:
        if state.get("human_decision") != "rejected":
            raise ApprovalError(
                f"Cannot finalize rejection: human_decision={state.get('human_decision')!r}, "
                "expected 'rejected'."
            )

        audit_event: AuditEvent = {
            "step": "approval_rejected",
            "message": "Human reviewer rejected the proposed simulated action.",
            "status": "ok",
        }

        return {
            "workflow_status": "approval_rejected",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return approval_rejected_node


def _route_after_action_input(state: CustomerOpsState) -> str:
    """Conditional-edge routing key for `prepare_action_input`.

    A one-field read of `state["action_input"]["ready"]` - simple enough
    that a dependency-injection wrapper would add ceremony without value.
    """
    return "ready" if state["action_input"]["ready"] else "missing"


def make_safe_action_execution_node(action_store: CustomerActionStore, executor=execute_action):
    """Build the `execute_safe_action` node.

    Only reached when `prepare_action_input` reports `ready=True` on the
    safe-action branch (`cancel_order`/`change_address` - never an
    approval-required action type). Calls `executor` exactly once against
    the simulated `action_store`, then re-reads the affected order from
    that same store to synchronize `order_context` so graph state never
    shows stale data. No other order is touched.
    """

    def execute_safe_action_node(state: CustomerOpsState) -> dict:
        proposed_action = ProposedAction.model_validate(state["proposed_action"])
        action_input = ActionInputResult.model_validate(state["action_input"])

        result = executor(proposed_action, action_input, action_store, human_approved=False)
        updated_order = action_store.get_order(result.order_id)

        audit_event: AuditEvent = {
            "step": "action_execution",
            "message": result.message,
            "status": "ok",
        }

        return {
            "action_result": result.model_dump(mode="json"),
            "order_context": _with_updated_order(state.get("order_context"), updated_order),
            "workflow_status": "action_executed",
            "audit_log": [*state.get("audit_log", []), audit_event],
        }

    return execute_safe_action_node


def _with_updated_order(order_context, updated_order) -> dict:
    """Replace one order's serialized entry inside `order_context`.

    Leaves every other order untouched - never a cross-order mutation.
    """
    orders = (order_context or {}).get("orders", [])
    updated_dict = updated_order.model_dump(mode="json")
    new_orders = [
        updated_dict if order.get("order_id") == updated_order.order_id else order
        for order in orders
    ]
    return {"orders": new_orders, "count": len(new_orders)}


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
    action_input_extractor: ActionInputExtractor | None = None,
    action_store: CustomerActionStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the Customer Operations graph.

    Current shape:
    START -> intake -> classify_request -> load_context -> resolve_order
          -> evaluate_policy -> (conditional) -> one of:
             - clarification -> END
             - information -> END
             - propose_action -> prepare_action_input -> (conditional) ->
               execute_safe_action -> END, or clarification -> END
             - prepare_approval -> prepare_approval_input -> human_approval
               -- interrupt() -- (resume) --> Command(goto=...) -> one of:
               - execute_approved_action -> END
               - approval_rejected -> END
             - blocked -> END

    Pass fakes for any of the dependencies (e.g. in tests) to avoid any
    OpenAI dependency or real fixture files. Omitting `classifier` or
    `action_input_extractor` uses the real OpenAI-backed implementation;
    building the graph never itself makes a network call - only invoking it
    as far as `classify_request` or `prepare_action_input` (for
    `change_address`) reaches OpenAI.

    Store consistency: `store` (used for `load_context`) and `action_store`
    (used for `execute_safe_action`/`execute_approved_action`) should be the
    SAME instance so a single graph invocation reads and mutates one
    coherent snapshot rather than two independently-loaded copies that
    could drift apart. When both are omitted, this function constructs
    exactly one `InMemoryCustomerActionStore.from_json()` and uses it for
    both - `JsonCustomerOperationsStore` is used as the `store` default
    only when `store` is customized without a matching `action_store` (or
    vice versa), which is an intentionally narrow escape hatch, not the
    recommended path.

    Checkpointing: `interrupt()`/`Command(resume=...)` require a
    checkpointer, so once compiled, EVERY invocation of the returned graph
    (interrupted or not) must pass `config={"configurable": {"thread_id":
    ...}}` - the caller owns thread identity; the graph never generates or
    derives one. Omitting `checkpointer` defaults to a fresh, process-local
    `InMemorySaver()` (never a module-level global): state persists across
    separate invoke/resume calls only while this Python process and this
    saver instance stay alive - it provides no durable persistence across a
    process restart, and is not a substitute for real business-data storage
    (`tools/action_store.py`'s in-memory mutations are equally process-local
    and separate from this checkpointer).
    """
    if classifier is None:
        classifier = OpenAIRequestClassifier()
    if action_input_extractor is None:
        action_input_extractor = OpenAIActionInputExtractor()
    if checkpointer is None:
        checkpointer = InMemorySaver()

    if store is None and action_store is None:
        combined_store = InMemoryCustomerActionStore.from_json()
        store = combined_store
        action_store = combined_store
    else:
        if store is None:
            store = JsonCustomerOperationsStore()
        if action_store is None:
            action_store = InMemoryCustomerActionStore.from_json()

    graph = StateGraph(CustomerOpsState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify_request", make_classification_node(classifier))
    graph.add_node("load_context", make_context_loader_node(store))
    graph.add_node("resolve_order", make_order_resolution_node())
    graph.add_node("evaluate_policy", make_policy_evaluation_node())
    graph.add_node("clarification", make_clarification_node())
    graph.add_node("information", make_information_node())
    graph.add_node("propose_action", make_action_proposal_node())
    graph.add_node("prepare_action_input", make_action_input_node(action_input_extractor))
    graph.add_node("execute_safe_action", make_safe_action_execution_node(action_store))
    graph.add_node("prepare_approval", make_approval_preparation_node())
    graph.add_node("prepare_approval_input", make_approval_action_input_node(action_input_extractor))
    graph.add_node("human_approval", make_human_approval_node())
    graph.add_node("execute_approved_action", make_approved_action_execution_node(action_store))
    graph.add_node("approval_rejected", make_approval_rejected_node())
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
    graph.add_edge("propose_action", "prepare_action_input")
    graph.add_conditional_edges(
        "prepare_action_input",
        _route_after_action_input,
        {"ready": "execute_safe_action", "missing": "clarification"},
    )
    graph.add_edge("execute_safe_action", END)
    graph.add_edge("prepare_approval", "prepare_approval_input")
    graph.add_edge("prepare_approval_input", "human_approval")
    # human_approval routes dynamically via Command(goto=...) after resume -
    # no static edge to declare here.
    graph.add_edge("execute_approved_action", END)
    graph.add_edge("approval_rejected", END)
    graph.add_edge("clarification", END)
    graph.add_edge("information", END)
    graph.add_edge("blocked", END)
    return graph.compile(checkpointer=checkpointer)
