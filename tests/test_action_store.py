"""Tests for the simulated mutable operational store.

`InMemoryCustomerActionStore` is exercised both against small in-memory
records (constructed directly, no file I/O) and against the real fixtures
via `from_json()` (to confirm the fixture files are never modified). No
test makes a network call.
"""

import pytest

from customer_ops.models import CustomerRecord, OrderRecord
from tools.customer_data import (
    DEFAULT_CUSTOMERS_PATH,
    DEFAULT_ORDERS_PATH,
    CustomerNotFoundError,
    OrderNotFoundError,
)
from tools.action_store import ActionStoreError, InMemoryCustomerActionStore


def make_customer(customer_id="cust-001", **overrides) -> CustomerRecord:
    fields = {
        "customer_id": customer_id,
        "name": "Test Customer",
        "email": "test.customer@example.com",
        "account_status": "active",
        "customer_tier": "standard",
    }
    fields.update(overrides)
    return CustomerRecord(**fields)


def make_order(order_id="order-0001", customer_id="cust-001", **overrides) -> OrderRecord:
    fields = {
        "order_id": order_id,
        "customer_id": customer_id,
        "status": "pending",
        "payment_status": "paid",
        "total": 42.50,
        "currency": "USD",
        "shipping_address": "1 Test Street, Testville, TS 00000, USA",
        "tracking_number": None,
        "items": [{"sku": "SKU-1", "name": "Test Item", "quantity": 1, "unit_price": 42.50}],
    }
    fields.update(overrides)
    return OrderRecord(**fields)


def make_store(orders, customers=None):
    customers = customers or [make_customer()]
    return InMemoryCustomerActionStore(customers=customers, orders=orders)


# --- cancel_order --------------------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_cancel_order_from_eligible_status(status):
    order = make_order(status=status)
    store = make_store([order])
    updated = store.cancel_order(order.order_id)
    assert updated.status == "cancelled"
    assert store.get_order(order.order_id).status == "cancelled"


@pytest.mark.parametrize("status", ["shipped", "delivered", "cancelled"])
def test_cancel_order_from_ineligible_status_raises(status):
    order = make_order(status=status)
    store = make_store([order])
    with pytest.raises(ActionStoreError):
        store.cancel_order(order.order_id)
    assert store.get_order(order.order_id).status == status


def test_cancel_order_unknown_order_raises():
    store = make_store([make_order()])
    with pytest.raises(OrderNotFoundError):
        store.cancel_order("order-does-not-exist")


# --- change_shipping_address -----------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_change_address_from_eligible_status(status):
    order = make_order(status=status, shipping_address="Old address")
    store = make_store([order])
    updated = store.change_shipping_address(order.order_id, "Calle Falsa 123, Cordoba")
    assert updated.shipping_address == "Calle Falsa 123, Cordoba"
    assert store.get_order(order.order_id).shipping_address == "Calle Falsa 123, Cordoba"


@pytest.mark.parametrize("status", ["shipped", "delivered", "cancelled"])
def test_change_address_from_ineligible_status_raises(status):
    order = make_order(status=status, shipping_address="Old address")
    store = make_store([order])
    with pytest.raises(ActionStoreError):
        store.change_shipping_address(order.order_id, "Calle Falsa 123, Cordoba")
    assert store.get_order(order.order_id).shipping_address == "Old address"


def test_change_address_empty_address_rejected():
    order = make_order(status="pending", shipping_address="Old address")
    store = make_store([order])
    with pytest.raises(ActionStoreError):
        store.change_shipping_address(order.order_id, "   ")
    assert store.get_order(order.order_id).shipping_address == "Old address"


def test_change_address_only_updates_the_target_order():
    order_a = make_order("order-aaa", shipping_address="Address A")
    order_b = make_order("order-bbb", shipping_address="Address B")
    store = make_store([order_a, order_b])
    store.change_shipping_address("order-aaa", "New Address")
    assert store.get_order("order-aaa").shipping_address == "New Address"
    assert store.get_order("order-bbb").shipping_address == "Address B"


# --- issue_full_refund ------------------------------------------------------------------


def test_refund_paid_order_becomes_refunded():
    order = make_order(payment_status="paid")
    store = make_store([order])
    updated = store.issue_full_refund(order.order_id)
    assert updated.payment_status == "refunded"
    assert store.get_order(order.order_id).payment_status == "refunded"


def test_refund_already_refunded_order_rejected():
    order = make_order(payment_status="refunded")
    store = make_store([order])
    with pytest.raises(ActionStoreError):
        store.issue_full_refund(order.order_id)


def test_refund_only_changes_payment_status():
    order = make_order(status="delivered", payment_status="paid")
    store = make_store([order])
    updated = store.issue_full_refund(order.order_id)
    assert updated.status == "delivered"
    assert updated.total == order.total
    assert updated.shipping_address == order.shipping_address


def test_refund_unknown_order_raises():
    store = make_store([make_order()])
    with pytest.raises(OrderNotFoundError):
        store.issue_full_refund("order-does-not-exist")


# --- investigations ----------------------------------------------------------------------


def test_billing_investigation_reference_id_is_deterministic():
    order = make_order("order-1001")
    store = make_store([order])
    reference_id = store.create_billing_investigation("order-1001")
    assert reference_id == "billing-investigation-order-1001"


def test_product_investigation_reference_id_is_deterministic():
    order = make_order("order-1001")
    store = make_store([order])
    reference_id = store.create_product_investigation("order-1001")
    assert reference_id == "product-investigation-order-1001"


def test_investigation_does_not_mutate_the_order():
    order = make_order("order-1001", status="delivered")
    store = make_store([order])
    store.create_billing_investigation("order-1001")
    assert store.get_order("order-1001").status == "delivered"


def test_billing_investigation_unknown_order_raises():
    store = make_store([make_order()])
    with pytest.raises(OrderNotFoundError):
        store.create_billing_investigation("order-does-not-exist")


def test_product_investigation_unknown_order_raises():
    store = make_store([make_order()])
    with pytest.raises(OrderNotFoundError):
        store.create_product_investigation("order-does-not-exist")


# --- Cross-cutting guarantees ------------------------------------------------------------


def test_no_cross_order_mutation_on_cancel():
    order_a = make_order("order-aaa", status="pending")
    order_b = make_order("order-bbb", status="pending")
    store = make_store([order_a, order_b])
    store.cancel_order("order-aaa")
    assert store.get_order("order-aaa").status == "cancelled"
    assert store.get_order("order-bbb").status == "pending"


def test_deterministic_initialization_and_reads(no_network):
    store_a = InMemoryCustomerActionStore.from_json()
    store_b = InMemoryCustomerActionStore.from_json()
    assert store_a.get_customer("cust-001") == store_b.get_customer("cust-001")
    assert store_a.get_order("order-1001") == store_b.get_order("order-1001")


def test_fixture_files_are_never_written_to():
    customers_before = DEFAULT_CUSTOMERS_PATH.read_text(encoding="utf-8")
    orders_before = DEFAULT_ORDERS_PATH.read_text(encoding="utf-8")

    store = InMemoryCustomerActionStore.from_json()
    store.cancel_order("order-1004")  # order-1004 is 'processing' in the real fixture
    store.change_shipping_address("order-1005", "New Address")  # order-1005 is 'pending'
    store.issue_full_refund("order-1001")
    store.create_billing_investigation("order-1006")
    store.create_product_investigation("order-1006")

    assert DEFAULT_CUSTOMERS_PATH.read_text(encoding="utf-8") == customers_before
    assert DEFAULT_ORDERS_PATH.read_text(encoding="utf-8") == orders_before


def test_from_json_initializes_from_validated_fixture_records():
    store = InMemoryCustomerActionStore.from_json()
    customer = store.get_customer("cust-001")
    assert isinstance(customer, CustomerRecord)
    order = store.get_order("order-1001")
    assert isinstance(order, OrderRecord)


def test_customer_not_found_for_unknown_customer():
    store = make_store([make_order()])
    with pytest.raises(CustomerNotFoundError):
        store.get_customer("cust-does-not-exist")
