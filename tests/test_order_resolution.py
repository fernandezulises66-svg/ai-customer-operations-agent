"""Tests for deterministic order-reference resolution.

`resolve_order` never queries OpenAI and never guesses: it only selects an
order ID that already appears, explicitly and unambiguously, in the
customer's message and in that customer's own (already-scoped)
`order_context`.
"""

import pytest
from pydantic import ValidationError

from customer_ops.order_resolution import OrderResolution, OrderResolutionError, resolve_order


def order_context(*order_ids):
    return {"orders": [{"order_id": order_id} for order_id in order_ids], "count": len(order_ids)}


def test_known_explicit_order_id_is_selected():
    result = resolve_order(
        intent="order_status",
        customer_message="Where is order-1001?",
        order_context=order_context("order-1001", "order-1002"),
    )
    assert result.status == "selected"
    assert result.selected_order_id == "order-1001"


def test_case_insensitive_explicit_id_matching():
    result = resolve_order(
        intent="order_status",
        customer_message="Where is ORDER-1001?",
        order_context=order_context("order-1001", "order-1002"),
    )
    assert result.status == "selected"
    assert result.selected_order_id == "order-1001"


def test_unknown_order_id_is_never_selected():
    result = resolve_order(
        intent="order_status",
        customer_message="Where is order-9999?",
        order_context=order_context("order-1001", "order-1002"),
    )
    assert result.status == "needs_clarification"
    assert result.selected_order_id is None


def test_non_customer_order_id_is_never_selected():
    # order-1004 is a real order in the dataset, but not one that belongs to
    # *this* customer's order_context - it must never be selected.
    result = resolve_order(
        intent="order_status",
        customer_message="Where is order-1004?",
        order_context=order_context("order-1001", "order-1002"),
    )
    assert result.status == "needs_clarification"
    assert result.selected_order_id is None


def test_no_order_id_needs_clarification():
    result = resolve_order(
        intent="cancel_order",
        customer_message="I want to cancel my order.",
        order_context=order_context("order-1001", "order-1002"),
    )
    assert result.status == "needs_clarification"
    assert result.selected_order_id is None


def test_multiple_known_order_ids_needs_clarification():
    result = resolve_order(
        intent="order_status",
        customer_message="Where are order-1001 and order-1002?",
        order_context=order_context("order-1001", "order-1002"),
    )
    assert result.status == "needs_clarification"
    assert result.selected_order_id is None


def test_other_intent_is_not_required():
    result = resolve_order(
        intent="other",
        customer_message="Do you sell gift cards?",
        order_context=order_context("order-1001"),
    )
    assert result.status == "not_required"
    assert result.selected_order_id is None


def test_zero_order_customer_needs_clarification():
    result = resolve_order(
        intent="order_status",
        customer_message="Where is my order?",
        order_context=order_context(),
    )
    assert result.status == "needs_clarification"
    assert result.selected_order_id is None


def test_product_name_does_not_trigger_guessing():
    result = resolve_order(
        intent="product_issue",
        customer_message="My wireless bluetooth headphones arrived broken.",
        order_context=order_context("order-1001"),
    )
    assert result.status == "needs_clarification"
    assert result.selected_order_id is None


def test_amount_does_not_trigger_guessing():
    result = resolve_order(
        intent="billing_issue",
        customer_message="I was charged 89.99 twice.",
        order_context=order_context("order-1001"),
    )
    assert result.status == "needs_clarification"
    assert result.selected_order_id is None


def test_deterministic_behavior():
    kwargs = dict(
        intent="order_status",
        customer_message="Where is order-1001?",
        order_context=order_context("order-1001", "order-1002"),
    )
    assert resolve_order(**kwargs) == resolve_order(**kwargs)


def test_resolver_performs_no_network_access(no_network):
    resolve_order(
        intent="order_status",
        customer_message="Where is order-1001?",
        order_context=order_context("order-1001"),
    )


# --- OrderResolution contract -------------------------------------------------


def test_selected_order_id_required_when_status_selected():
    with pytest.raises(ValidationError):
        OrderResolution(status="selected", selected_order_id=None)


def test_selected_order_id_must_be_none_when_not_selected():
    with pytest.raises(ValidationError):
        OrderResolution(status="not_required", selected_order_id="order-1001")


# --- Error handling -------------------------------------------------------------


def test_missing_intent_raises_order_resolution_error():
    with pytest.raises(OrderResolutionError):
        resolve_order(intent=None, customer_message="Where is my order?", order_context=order_context())


def test_missing_order_context_for_order_intent_raises():
    with pytest.raises(OrderResolutionError):
        resolve_order(intent="order_status", customer_message="Where is my order?", order_context=None)
