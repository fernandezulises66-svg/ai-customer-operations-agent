"""Typed state contracts for the Mercora Customer Operations Agent workflow.

This module defines the LangGraph state schema and the controlled vocabularies
(Literal aliases) shared across workflow nodes. Iteration 1 only establishes
these contracts - no classification, retrieval, policy, or action logic lives
here yet.

All state values must remain JSON/checkpoint friendly: strings, booleans,
numbers, lists, dictionaries, or None. Never store live OpenAI clients, tool
objects, graph instances, or other runtime services in state.
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
    "policy_checked",
    "action_proposed",
    "awaiting_approval",
    "action_executed",
    "escalated",
    "completed",
    "failed",
]

HumanDecision = Literal["approved", "rejected"]

# Documented value sets, derived from the Literals above so the two never
# drift apart. Useful for validation and for tests that assert on the
# controlled vocabulary without duplicating it.
INTENT_VALUES: tuple[str, ...] = get_args(Intent)
URGENCY_VALUES: tuple[str, ...] = get_args(Urgency)
WORKFLOW_STATUS_VALUES: tuple[str, ...] = get_args(WorkflowStatus)
HUMAN_DECISION_VALUES: tuple[str, ...] = get_args(HumanDecision)


# --- Audit trail --------------------------------------------------------------


class AuditEvent(TypedDict):
    """A single observable workflow event.

    Audit events describe *what happened* in the workflow (e.g. "intake
    validated the request"), never model reasoning or chain-of-thought.
    """

    step: str
    message: str
    status: NotRequired[Literal["ok", "error"]]


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

    # Classification (future iterations)
    intent: NotRequired[Intent | None]
    urgency: NotRequired[Urgency | None]

    # Retrieved context (future iterations)
    customer_context: NotRequired[dict[str, Any] | None]
    order_context: NotRequired[dict[str, Any] | None]

    # Policy and action planning (future iterations)
    policy_assessment: NotRequired[dict[str, Any] | None]
    proposed_action: NotRequired[dict[str, Any] | None]

    # Human-in-the-loop (future iterations)
    human_decision: NotRequired[HumanDecision | None]

    # Execution outcome (future iterations)
    action_result: NotRequired[dict[str, Any] | None]

    # Final output
    final_response: NotRequired[str | None]
    escalation_reason: NotRequired[str | None]

    # Workflow bookkeeping
    workflow_status: NotRequired[WorkflowStatus | None]
    audit_log: NotRequired[List[AuditEvent]]
