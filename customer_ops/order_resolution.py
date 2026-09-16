"""Deterministic order-reference resolution for the Mercora Customer Operations Agent.

Decides which order (if any) a customer request refers to, using only
`intent`, `customer_message`, and the customer's already-scoped
`order_context`. Never queries OpenAI and never guesses: an order is only
ever selected when exactly one known order ID for *this* customer appears
explicitly in their message. Order selection is a business rule, not an LLM
decision.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, model_validator

from customer_ops.state import Intent, OrderContext, OrderResolutionStatus

# Intents where the workflow needs a specific order to act on. `other` is
# deliberately excluded - it never requires an order reference.
ORDER_REQUIRED_INTENTS: frozenset[str] = frozenset(
    {
        "order_status",
        "refund_request",
        "cancel_order",
        "address_change",
        "billing_issue",
        "product_issue",
    }
)


class OrderResolution(BaseModel):
    """The structured result of deterministic order-reference resolution.

    Contains only the resolution decision - no rationale, hidden reasoning,
    or confidence score.
    """

    status: OrderResolutionStatus
    selected_order_id: str | None = None

    @model_validator(mode="after")
    def _check_selected_order_id_matches_status(self) -> "OrderResolution":
        if self.status == "selected" and not self.selected_order_id:
            raise ValueError("selected_order_id is required when status is 'selected'.")
        if self.status != "selected" and self.selected_order_id is not None:
            raise ValueError("selected_order_id must be None unless status is 'selected'.")
        return self


class OrderResolutionError(Exception):
    """Raised when order resolution cannot proceed due to invalid input state.

    `needs_clarification` is a legitimate business outcome, not this error -
    this is reserved for state that should never occur in normal operation
    (e.g. a missing classified intent).
    """


def resolve_order(
    intent: Intent | None,
    customer_message: str,
    order_context: OrderContext | None,
) -> OrderResolution:
    """Resolve which of the customer's known orders (if any) is referenced.

    Only inspects order IDs already present in `order_context`, which the
    caller must have scoped to this customer - the resolver performs no
    lookup of its own. It never infers an order from product names, dates,
    amounts, or vague references such as "my latest order": only an
    explicit, unambiguous order ID mention is ever selected.
    """
    if intent is None:
        raise OrderResolutionError("Cannot resolve an order reference without a classified intent.")

    if intent not in ORDER_REQUIRED_INTENTS:
        return OrderResolution(status="not_required", selected_order_id=None)

    if order_context is None:
        raise OrderResolutionError(
            "Cannot resolve an order reference without loaded order context."
        )

    known_order_ids = [order["order_id"] for order in order_context.get("orders", [])]
    mentioned_order_ids = [
        order_id for order_id in known_order_ids if _mentions_order_id(customer_message, order_id)
    ]

    if len(mentioned_order_ids) == 1:
        return OrderResolution(status="selected", selected_order_id=mentioned_order_ids[0])

    return OrderResolution(status="needs_clarification", selected_order_id=None)


def _mentions_order_id(customer_message: str, order_id: str) -> bool:
    """Case-insensitive, word-bounded exact match - never fuzzy matching."""
    pattern = rf"(?<![a-z0-9]){re.escape(order_id.lower())}(?![a-z0-9])"
    return re.search(pattern, customer_message.lower()) is not None
