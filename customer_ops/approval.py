"""Human-in-the-loop approval contracts for the Mercora Customer Operations Agent.

Defines the minimal JSON-friendly payload a human reviewer sees through
LangGraph's `interrupt()` (`ApprovalRequest`) and the strictly-validated
shape their decision must take when the graph resumes via
`Command(resume=...)` (`HumanApprovalResponse`). The actual approve/reject
decision is human input, supplied externally through resume - this module
never asks OpenAI and never derives a decision from policy, intent,
urgency, or any other model output.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from customer_ops.action_proposal import ProposedAction
from customer_ops.state import ActionType, HumanDecision

# One human-readable summary per approval-required action type. Deliberately
# describes only the simulated operation being approved - never implies a
# real financial/external action will happen.
_APPROVAL_MESSAGE_BY_ACTION_TYPE: dict[str, str] = {
    "issue_refund": "Approve simulated full refund for order {order_id}?",
    "investigate_billing": (
        "Approve creation of a simulated billing investigation for order {order_id}?"
    ),
    "investigate_product_issue": (
        "Approve creation of a simulated product investigation for order {order_id}?"
    ),
}


class ApprovalRequest(BaseModel):
    """The minimal payload a human reviewer needs to decide - nothing more.

    Never includes customer email, shipping address, a full customer/order
    record, or any hidden reasoning. `amount`/`currency` are populated only
    for `issue_refund`, and only when derived from the selected order's own
    validated total - never invented.
    """

    request_id: str
    action_type: ActionType
    order_id: str
    message: str
    amount: float | None = None
    currency: str | None = None


class HumanApprovalResponse(BaseModel):
    """The strictly-validated human decision supplied via `Command(resume=...)`."""

    decision: HumanDecision


class ApprovalError(Exception):
    """Raised when approval state is invalid or a resume payload is malformed.

    Covers both a node precondition failure (e.g. requesting approval for
    an action that doesn't need it) and a resume value that cannot be
    interpreted as a `HumanApprovalResponse` - malformed human input is
    never silently treated as approved or rejected.
    """


def build_approval_request(
    request_id: str,
    proposed_action: ProposedAction,
    order: dict[str, object] | None = None,
) -> ApprovalRequest:
    """Build the minimal `ApprovalRequest` for a proposed sensitive action.

    `order` should be the already-scoped, validated order dict (e.g. looked
    up from `order_context`) - used only to derive `amount`/`currency` for
    `issue_refund`, and only from the order's own validated fields.
    """
    action_type = proposed_action.action_type
    template = _APPROVAL_MESSAGE_BY_ACTION_TYPE.get(action_type)
    if template is None:
        raise ApprovalError(
            f"No approval message template is defined for action type {action_type!r}."
        )

    amount = None
    currency = None
    if action_type == "issue_refund" and order is not None:
        amount = order.get("total")
        currency = order.get("currency")

    return ApprovalRequest(
        request_id=request_id,
        action_type=action_type,
        order_id=proposed_action.order_id,
        message=template.format(order_id=proposed_action.order_id),
        amount=amount,
        currency=currency,
    )


def parse_human_approval_response(decision_payload: object) -> HumanApprovalResponse:
    """Validate the raw value supplied through `Command(resume=...)`.

    Never silently interprets malformed or missing human input - raises
    `ApprovalError` instead of guessing a decision.
    """
    try:
        return HumanApprovalResponse.model_validate(decision_payload)
    except ValidationError as exc:
        raise ApprovalError(f"Malformed human approval response: {exc}") from exc
