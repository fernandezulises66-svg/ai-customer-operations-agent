"""Simulated mutable operational store for the Mercora Customer Operations Agent.

`InMemoryCustomerActionStore` implements both the read-only
`CustomerOperationsStore` protocol (see `tools/customer_data.py`) and the
mutating `CustomerActionStore` protocol defined here, against the SAME
in-memory records. A single graph invocation that reads and later mutates
through one store instance therefore always observes a coherent snapshot -
there is no second, independently-loaded copy of the data that could drift
out of sync during that invocation.

Every mutation happens only in this process's memory. `data/customers.json`
and `data/orders.json` are never written to - they remain example fixtures,
not persistent business storage. A process restart resets every simulated
mutation. No real money, orders, or external systems are touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from customer_ops.models import CustomerRecord, OrderRecord
from tools.customer_data import (
    DEFAULT_CUSTOMERS_PATH,
    DEFAULT_ORDERS_PATH,
    CustomerNotFoundError,
    OrderNotFoundError,
    load_customer_records,
    load_order_records,
)

_CANCELLABLE_STATUSES = frozenset({"pending", "processing"})
_ADDRESS_CHANGEABLE_STATUSES = frozenset({"pending", "processing"})


class ActionStoreError(Exception):
    """Raised when a simulated mutation's business precondition is not met.

    Examples: cancelling an order that is already shipped, or refunding an
    order that was already refunded. The store never mutates when raising
    this - the caller's request is simply not honored.
    """


class CustomerActionStore(Protocol):
    """Narrow interface for simulated operational mutations.

    Deliberately a separate interface from the read-only
    `CustomerOperationsStore`, even though `InMemoryCustomerActionStore`
    implements both - read-only and mutating concerns stay conceptually
    distinct.
    """

    def cancel_order(self, order_id: str) -> OrderRecord: ...

    def change_shipping_address(self, order_id: str, new_address: str) -> OrderRecord: ...

    def issue_full_refund(self, order_id: str) -> OrderRecord: ...

    def create_billing_investigation(self, order_id: str) -> str: ...

    def create_product_investigation(self, order_id: str) -> str: ...


class InMemoryCustomerActionStore:
    """Combined read + simulated-mutation store over in-memory records.

    Construct directly from already-validated `CustomerRecord`/`OrderRecord`
    objects (e.g. in tests), or via `from_json()` to load and validate the
    real synthetic fixtures the same way `JsonCustomerOperationsStore` does.
    Mutation methods never write to disk; they only replace the affected
    order in this instance's in-memory dict.
    """

    def __init__(self, customers: list[CustomerRecord], orders: list[OrderRecord]) -> None:
        self._customers_by_id: dict[str, CustomerRecord] = {
            customer.customer_id: customer for customer in customers
        }
        self._orders_by_id: dict[str, OrderRecord] = {order.order_id: order for order in orders}
        self._order_ids_by_customer: dict[str, list[str]] = {}
        for order in orders:
            self._order_ids_by_customer.setdefault(order.customer_id, []).append(order.order_id)
        self._billing_investigations: dict[str, str] = {}
        self._product_investigations: dict[str, str] = {}

    @classmethod
    def from_json(
        cls,
        customers_path: Path | str | None = None,
        orders_path: Path | str | None = None,
    ) -> "InMemoryCustomerActionStore":
        """Build the store from the synthetic JSON fixtures.

        Fixtures are read and validated once at construction, exactly like
        `JsonCustomerOperationsStore`; they are never written to.
        """
        customers = load_customer_records(customers_path or DEFAULT_CUSTOMERS_PATH)
        orders = load_order_records(orders_path or DEFAULT_ORDERS_PATH)
        return cls(customers=customers, orders=orders)

    # --- read-only protocol (CustomerOperationsStore) ----------------------------

    def get_customer(self, customer_id: str) -> CustomerRecord:
        try:
            return self._customers_by_id[customer_id]
        except KeyError:
            raise CustomerNotFoundError(f"No customer found with customer_id={customer_id!r}") from None

    def list_orders_for_customer(self, customer_id: str) -> list[OrderRecord]:
        if customer_id not in self._customers_by_id:
            raise CustomerNotFoundError(f"No customer found with customer_id={customer_id!r}")
        return [self._orders_by_id[order_id] for order_id in self._order_ids_by_customer.get(customer_id, [])]

    def get_order(self, order_id: str) -> OrderRecord:
        try:
            return self._orders_by_id[order_id]
        except KeyError:
            raise OrderNotFoundError(f"No order found with order_id={order_id!r}") from None

    # --- mutating protocol (CustomerActionStore) ---------------------------------

    def cancel_order(self, order_id: str) -> OrderRecord:
        order = self.get_order(order_id)
        if order.status not in _CANCELLABLE_STATUSES:
            raise ActionStoreError(
                f"Order {order_id!r} cannot be cancelled from status {order.status!r}."
            )
        updated = order.model_copy(update={"status": "cancelled"})
        self._orders_by_id[order_id] = updated
        return updated

    def change_shipping_address(self, order_id: str, new_address: str) -> OrderRecord:
        order = self.get_order(order_id)
        if order.status not in _ADDRESS_CHANGEABLE_STATUSES:
            raise ActionStoreError(
                f"Order {order_id!r} cannot have its address changed from status {order.status!r}."
            )
        if not new_address or not new_address.strip():
            raise ActionStoreError(
                "A non-empty new_address is required to change an order's shipping address."
            )
        updated = order.model_copy(update={"shipping_address": new_address.strip()})
        self._orders_by_id[order_id] = updated
        return updated

    def issue_full_refund(self, order_id: str) -> OrderRecord:
        """Simulate a full refund of the order's current total.

        Fictional demo rule: refund execution always refunds the order's
        full total - there is no partial-refund behavior in this project.
        """
        order = self.get_order(order_id)
        if order.payment_status == "refunded":
            raise ActionStoreError(f"Order {order_id!r} has already been refunded.")
        updated = order.model_copy(update={"payment_status": "refunded"})
        self._orders_by_id[order_id] = updated
        return updated

    def create_billing_investigation(self, order_id: str) -> str:
        self.get_order(order_id)  # raises OrderNotFoundError if unknown
        reference_id = f"billing-investigation-{order_id}"
        self._billing_investigations[order_id] = reference_id
        return reference_id

    def create_product_investigation(self, order_id: str) -> str:
        self.get_order(order_id)  # raises OrderNotFoundError if unknown
        reference_id = f"product-investigation-{order_id}"
        self._product_investigations[order_id] = reference_id
        return reference_id
