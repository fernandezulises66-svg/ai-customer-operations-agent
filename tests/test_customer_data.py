"""Tests for the read-only synthetic customer/order operational store.

`JsonCustomerOperationsStore` is exercised against the real fixtures in
`data/` (to validate the dataset itself) and against small temporary fixture
files (to validate loading/error behavior in isolation). No test makes a
network call.
"""

import json

import pytest

from customer_ops.models import CustomerRecord, OrderRecord
from tools.customer_data import (
    DEFAULT_CUSTOMERS_PATH,
    DEFAULT_ORDERS_PATH,
    CustomerNotFoundError,
    DataStoreError,
    JsonCustomerOperationsStore,
    OrderNotFoundError,
)


@pytest.fixture
def store():
    return JsonCustomerOperationsStore()


def write_fixture(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


# --- Loading the real fixtures --------------------------------------------------


def test_fixture_files_load_successfully():
    JsonCustomerOperationsStore()


def test_expected_customer_count():
    customers = json.loads(DEFAULT_CUSTOMERS_PATH.read_text(encoding="utf-8"))
    assert len(customers) == 6


def test_expected_order_count():
    orders = json.loads(DEFAULT_ORDERS_PATH.read_text(encoding="utf-8"))
    assert len(orders) == 10


# --- Lookups ---------------------------------------------------------------------


def test_known_customer_lookup(store):
    customer = store.get_customer("cust-001")
    assert isinstance(customer, CustomerRecord)
    assert customer.customer_id == "cust-001"
    assert customer.name == "Mariana Torres"


def test_unknown_customer_raises_explicit_error(store):
    with pytest.raises(CustomerNotFoundError):
        store.get_customer("cust-does-not-exist")


def test_known_order_lookup(store):
    order = store.get_order("order-1001")
    assert isinstance(order, OrderRecord)
    assert order.order_id == "order-1001"
    assert order.customer_id == "cust-001"


def test_unknown_order_raises_explicit_error(store):
    with pytest.raises(OrderNotFoundError):
        store.get_order("order-does-not-exist")


def test_multiple_orders_returned_for_correct_customer(store):
    orders = store.list_orders_for_customer("cust-001")
    assert {order.order_id for order in orders} == {"order-1001", "order-1002", "order-1003"}


def test_customer_with_zero_orders_returns_empty_list(store):
    orders = store.list_orders_for_customer("cust-005")
    assert orders == []


def test_orders_never_leak_across_customers(store):
    cust_001_orders = store.list_orders_for_customer("cust-001")
    cust_002_orders = store.list_orders_for_customer("cust-002")

    assert all(order.customer_id == "cust-001" for order in cust_001_orders)
    assert all(order.customer_id == "cust-002" for order in cust_002_orders)

    cust_001_ids = {order.order_id for order in cust_001_orders}
    cust_002_ids = {order.order_id for order in cust_002_orders}
    assert cust_001_ids.isdisjoint(cust_002_ids)


def test_shipped_order_preserves_tracking_number(store):
    order = store.get_order("order-1001")
    assert order.status == "shipped"
    assert order.tracking_number == "1Z999AA10123456784"


# --- Fixture variation -----------------------------------------------------------


def test_fixture_variation_includes_required_statuses(store):
    orders = json.loads(DEFAULT_ORDERS_PATH.read_text(encoding="utf-8"))
    statuses = {order["status"] for order in orders}
    assert statuses == {"pending", "processing", "shipped", "delivered", "cancelled"}


def test_fixture_variation_includes_failed_payment(store):
    order = store.get_order("order-1007")
    assert order.payment_status == "failed"


def test_fixture_variation_includes_refunded_payment(store):
    order = store.get_order("order-1003")
    assert order.payment_status == "refunded"


# --- Path handling -----------------------------------------------------------------


def test_default_paths_work_outside_repository_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store_outside_cwd = JsonCustomerOperationsStore()
    assert store_outside_cwd.get_customer("cust-001").customer_id == "cust-001"


# --- Malformed / schema-invalid fixtures --------------------------------------------


def test_malformed_customer_json_becomes_data_store_error(tmp_path):
    customers_path = tmp_path / "customers.json"
    orders_path = tmp_path / "orders.json"
    customers_path.write_text("{not valid json", encoding="utf-8")
    write_fixture(orders_path, [])

    with pytest.raises(DataStoreError):
        JsonCustomerOperationsStore(customers_path=customers_path, orders_path=orders_path)


def test_malformed_order_json_becomes_data_store_error(tmp_path):
    customers_path = tmp_path / "customers.json"
    orders_path = tmp_path / "orders.json"
    write_fixture(customers_path, [])
    orders_path.write_text("[1, 2,", encoding="utf-8")

    with pytest.raises(DataStoreError):
        JsonCustomerOperationsStore(customers_path=customers_path, orders_path=orders_path)


def test_schema_invalid_fixture_becomes_data_store_error(tmp_path):
    customers_path = tmp_path / "customers.json"
    orders_path = tmp_path / "orders.json"
    write_fixture(
        customers_path,
        [
            {
                "customer_id": "cust-001",
                "name": "Test Customer",
                "email": "test@example.com",
                "account_status": "not_a_real_status",
                "customer_tier": "standard",
            }
        ],
    )
    write_fixture(orders_path, [])

    with pytest.raises(DataStoreError):
        JsonCustomerOperationsStore(customers_path=customers_path, orders_path=orders_path)


# --- Read-only guarantees ---------------------------------------------------------


def test_read_operations_do_not_modify_fixture_files(store):
    customers_before = DEFAULT_CUSTOMERS_PATH.read_text(encoding="utf-8")
    orders_before = DEFAULT_ORDERS_PATH.read_text(encoding="utf-8")

    store.get_customer("cust-001")
    store.list_orders_for_customer("cust-001")
    store.get_order("order-1001")
    try:
        store.get_customer("nonexistent")
    except CustomerNotFoundError:
        pass

    assert DEFAULT_CUSTOMERS_PATH.read_text(encoding="utf-8") == customers_before
    assert DEFAULT_ORDERS_PATH.read_text(encoding="utf-8") == orders_before


def test_results_are_deterministic(store):
    first = store.get_customer("cust-001")
    second = store.get_customer("cust-001")
    assert first == second

    first_orders = store.list_orders_for_customer("cust-001")
    second_orders = store.list_orders_for_customer("cust-001")
    assert first_orders == second_orders
