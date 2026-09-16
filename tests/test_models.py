"""Tests for the Pydantic operational data contracts.

These check the structural invariants the models enforce (non-empty
identifiers, positive quantities, non-negative amounts, supported controlled
vocabularies, well-formed items) and JSON-friendly serialization - not
Pydantic internals.
"""

import pytest
from pydantic import ValidationError

from customer_ops.models import CustomerRecord, OrderItem, OrderRecord


def make_customer(**overrides):
    fields = {
        "customer_id": "cust-001",
        "name": "Test Customer",
        "email": "test.customer@example.com",
        "account_status": "active",
        "customer_tier": "standard",
    }
    fields.update(overrides)
    return CustomerRecord(**fields)


def make_order_item(**overrides):
    fields = {"sku": "SKU-1", "name": "Test Item", "quantity": 1, "unit_price": 9.99}
    fields.update(overrides)
    return OrderItem(**fields)


def make_order(**overrides):
    fields = {
        "order_id": "order-0001",
        "customer_id": "cust-001",
        "status": "shipped",
        "payment_status": "paid",
        "total": 9.99,
        "currency": "USD",
        "shipping_address": "1 Test Street, Testville, TS 00000, USA",
        "tracking_number": "TRACK123",
        "items": [{"sku": "SKU-1", "name": "Test Item", "quantity": 1, "unit_price": 9.99}],
    }
    fields.update(overrides)
    return OrderRecord(**fields)


# --- CustomerRecord ------------------------------------------------------------


def test_valid_customer_record():
    customer = make_customer()
    assert customer.customer_id == "cust-001"
    assert customer.account_status == "active"
    assert customer.customer_tier == "standard"


def test_customer_id_cannot_be_empty():
    with pytest.raises(ValidationError):
        make_customer(customer_id="")


def test_customer_supports_active_and_suspended_account_status():
    assert make_customer(account_status="active").account_status == "active"
    assert make_customer(account_status="suspended").account_status == "suspended"


def test_customer_invalid_account_status_rejected():
    with pytest.raises(ValidationError):
        make_customer(account_status="banned")


def test_customer_supports_standard_and_premium_tier():
    assert make_customer(customer_tier="standard").customer_tier == "standard"
    assert make_customer(customer_tier="premium").customer_tier == "premium"


def test_customer_invalid_tier_rejected():
    with pytest.raises(ValidationError):
        make_customer(customer_tier="platinum")


# --- OrderItem -------------------------------------------------------------------


def test_valid_order_item():
    item = make_order_item(sku="SKU-9", quantity=3, unit_price=4.5)
    assert item.sku == "SKU-9"
    assert item.quantity == 3
    assert item.unit_price == 4.5


def test_order_item_quantity_must_be_positive():
    with pytest.raises(ValidationError):
        make_order_item(quantity=0)
    with pytest.raises(ValidationError):
        make_order_item(quantity=-1)


def test_order_item_unit_price_cannot_be_negative():
    with pytest.raises(ValidationError):
        make_order_item(unit_price=-0.01)


# --- OrderRecord -----------------------------------------------------------------


def test_valid_order_record():
    order = make_order()
    assert order.order_id == "order-0001"
    assert order.status == "shipped"
    assert order.payment_status == "paid"
    assert len(order.items) == 1


def test_order_invalid_status_rejected():
    with pytest.raises(ValidationError):
        make_order(status="in_transit")


def test_order_invalid_payment_status_rejected():
    with pytest.raises(ValidationError):
        make_order(payment_status="charged_back")


def test_order_negative_total_rejected():
    with pytest.raises(ValidationError):
        make_order(total=-10.0)


def test_order_malformed_item_rejected():
    with pytest.raises(ValidationError):
        make_order(items=[{"sku": "SKU-1", "name": "Bad Item", "quantity": 0, "unit_price": 9.99}])


def test_order_empty_items_rejected():
    with pytest.raises(ValidationError):
        make_order(items=[])


# --- Serialization ---------------------------------------------------------------


def test_customer_serialization_is_json_friendly():
    customer = make_customer()
    dumped = customer.model_dump(mode="json")
    assert dumped == {
        "customer_id": "cust-001",
        "name": "Test Customer",
        "email": "test.customer@example.com",
        "account_status": "active",
        "customer_tier": "standard",
    }


def test_order_serialization_is_json_friendly():
    order = make_order()
    dumped = order.model_dump(mode="json")
    assert isinstance(dumped, dict)
    assert isinstance(dumped["items"], list)
    assert dumped["items"][0] == {
        "sku": "SKU-1",
        "name": "Test Item",
        "quantity": 1,
        "unit_price": 9.99,
    }
