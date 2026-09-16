"""Typed state contracts for the Mercora Customer Operations Agent workflow.

This module defines the LangGraph state schema and the controlled vocabularies
(Literal aliases) shared across workflow nodes.

All state values must remain JSON/checkpoint friendly: strings, booleans,
numbers, lists, dictionaries, or None. Never store live OpenAI clients, tool
objects, graph instances, data-store instances, or other runtime services in
state.
"""

from typing import Any, List, Literal, NotRequired, TypedDict, get_args

# --- Controlled vocabularies -------------------------------------------------

Intent = Literal[
    "order_status",
    "refund_request",
    "cancel_order",
    "address_change",
    "billing_issue",
    "product_issue",
    "other",
]

Urgency = Literal["low", "medium", "high"]

WorkflowStatus = Literal[
    "received",
    "classified",
    "context_loaded",
    "order_resolved",
    "policy_checked",
    "clarification_required",
    "information_ready",
    "action_proposed",
    "action_input_ready",
    "awaiting_approval",
    "blocked",
    "action_executed",
    "escalated",
    "completed",
    "failed",
]

HumanDecision = Literal["approved", "rejected"]

OrderResolutionStatus = Literal["selected", "not_required", "needs_clarification"]

# The workflow branch chosen after policy evaluation. `information_only`
# and `not_applicable` policy outcomes both route to "information" - see
# customer_ops/routing.py.
CaseRoute = Literal["clarification", "information", "action", "approval", "blocked"]

# The operational action a `ProposedAction` describes. A proposal, never an
# execution - see customer_ops/action_proposal.py.
ActionType = Literal[
    "cancel_order",
    "change_address",
    "issue_refund",
    "investigate_billing",
    "investigate_product_issue",
]

PolicyOutcome = Literal[
    "information_only",
    "eligible",
    "blocked",
    "review_required",
    "needs_clarification",
    "not_applicable",
]

# Stable, machine-readable policy codes. See customer_ops/policies.py for the
# rules that produce each one.
PolicyCode = Literal[
    "ORDER_STATUS_INFO",
    "CANCEL_ALLOWED",
    "CANCEL_BLOCKED_STATUS",
    "ADDRESS_CHANGE_ALLOWED",
    "ADDRESS_CHANGE_BLOCKED_STATUS",
    "REFUND_REVIEW_REQUIRED",
    "ALREADY_REFUNDED",
    "BILLING_REVIEW_REQUIRED",
    "PRODUCT_REVIEW_REQUIRED",
    "ORDER_REQUIRED",
    "NOT_APPLICABLE",
]

# Documented value sets, derived from the Literals above so the two never
# drift apart. Useful for validation and for tests that assert on the
# controlled vocabulary without duplicating it.
INTENT_VALUES: tuple[str, ...] = get_args(Intent)
URGENCY_VALUES: tuple[str, ...] = get_args(Urgency)
WORKFLOW_STATUS_VALUES: tuple[str, ...] = get_args(WorkflowStatus)
HUMAN_DECISION_VALUES: tuple[str, ...] = get_args(HumanDecision)
ORDER_RESOLUTION_STATUS_VALUES: tuple[str, ...] = get_args(OrderResolutionStatus)
POLICY_OUTCOME_VALUES: tuple[str, ...] = get_args(PolicyOutcome)
POLICY_CODE_VALUES: tuple[str, ...] = get_args(PolicyCode)
CASE_ROUTE_VALUES: tuple[str, ...] = get_args(CaseRoute)
ACTION_TYPE_VALUES: tuple[str, ...] = get_args(ActionType)


# --- Audit trail --------------------------------------------------------------


class AuditEvent(TypedDict):
    """A single observable workflow event.

    Audit events describe *what happened* in the workflow (e.g. "intake
    validated the request"), never model reasoning or chain-of-thought.
    """

    step: str
    message: str
    status: NotRequired[Literal["ok", "error"]]


# --- Context state --------------------------------------------------------------


class OrderContext(TypedDict):
    """The envelope produced by context loading for a customer's orders.

    `orders` holds JSON-friendly order records (e.g. `OrderRecord.model_dump
    (mode="json")` from `customer_ops/models.py`), never Pydantic objects.
    """

    orders: List[dict[str, Any]]
    count: int


# --- Workflow state -----------------------------------------------------------


class CustomerOpsState(TypedDict):
    """Shared state threaded through the Mercora customer operations graph.

    `request_id`, `customer_id`, and `customer_message` are the required
    initial payload. Every other field is populated progressively by nodes
    added in later iterations and starts unset.
    """

    # Initial request payload
    request_id: str
    customer_id: str
    customer_message: str

    # Classification
    intent: NotRequired[Intent | None]
    urgency: NotRequired[Urgency | None]

    # Retrieved context
    customer_context: NotRequired[dict[str, Any] | None]
    order_context: NotRequired[OrderContext | None]

    # Order resolution
    selected_order_id: NotRequired[str | None]
    order_resolution: NotRequired[dict[str, Any] | None]

    # Policy evaluation
    policy_assessment: NotRequired[dict[str, Any] | None]

    # Routing, action proposal, and validated action input
    route: NotRequired[CaseRoute | None]
    proposed_action: NotRequired[dict[str, Any] | None]
    action_input: NotRequired[dict[str, Any] | None]

    # Human-in-the-loop (future iterations)
    human_decision: NotRequired[HumanDecision | None]

    # Execution outcome. Populated only for safe actions (cancel_order,
    # change_address) executed against the simulated in-memory operational
    # store - approval-required actions never execute in this iteration.
    action_result: NotRequired[dict[str, Any] | None]

    # Final output
    final_response: NotRequired[str | None]
    escalation_reason: NotRequired[str | None]

    # Workflow bookkeeping
    workflow_status: NotRequired[WorkflowStatus | None]
    audit_log: NotRequired[List[AuditEvent]]
