"""Structured proposed-action mapping for the Mercora Customer Operations Agent.

Maps a classified intent and a resolved order into a `ProposedAction` - a
structured statement of operational *intent*, never execution. Used only on
the `action`/`approval` routes; every other route never proposes an action.
No action input is ever invented here (e.g. no new address, refund amount,
or replacement item) - those belong to a later iteration.
"""

from __future__ import annotations

from pydantic import BaseModel

from customer_ops.state import ActionType, Intent

# intent -> (action_type, requires_human_approval). Only intents with a
# defined operational action appear here - order_status, address-less
# clarification cases, and "other" never propose an action.
_ACTION_TYPE_BY_INTENT: dict[str, tuple[ActionType, bool]] = {
    "cancel_order": ("cancel_order", False),
    "address_change": ("change_address", False),
    "refund_request": ("issue_refund", True),
    "billing_issue": ("investigate_billing", True),
    "product_issue": ("investigate_product_issue", True),
}


class ProposedAction(BaseModel):
    """A structured statement of operational intent - never execution.

    Contains only what a future execution step will need to know *what*
    was proposed - no chain-of-thought, confidence score, or free-form
    execution payload.
    """

    action_type: ActionType
    order_id: str
    requires_human_approval: bool


class ActionProposalError(Exception):
    """Raised when an action cannot be proposed for the given intent/state.

    Reserved for state that should never occur in normal operation (e.g. an
    intent with no defined action mapping, or a missing selected order) -
    never used to invent a plausible-looking action.
    """


def propose_action(intent: Intent | None, selected_order_id: str | None) -> ProposedAction:
    """Build the `ProposedAction` for an `action`/`approval`-routed case.

    Only ever called for intents with a defined action mapping
    (`cancel_order`, `address_change`, `refund_request`, `billing_issue`,
    `product_issue`) and a resolved `selected_order_id`.
    """
    if intent is None:
        raise ActionProposalError("Cannot propose an action without a classified intent.")

    if intent not in _ACTION_TYPE_BY_INTENT:
        raise ActionProposalError(f"No action mapping is defined for intent {intent!r}.")

    if not selected_order_id:
        raise ActionProposalError(
            f"Cannot propose an action for intent {intent!r} without a selected order."
        )

    action_type, requires_human_approval = _ACTION_TYPE_BY_INTENT[intent]
    return ProposedAction(
        action_type=action_type,
        order_id=selected_order_id,
        requires_human_approval=requires_human_approval,
    )
