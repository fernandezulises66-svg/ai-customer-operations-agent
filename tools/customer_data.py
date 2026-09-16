"""Read-only access to Mercora's synthetic customer/order operational data.

`JsonCustomerOperationsStore` loads and validates the JSON fixtures in
`data/` once at construction time, then serves lookups against the validated
in-memory records. It is read-only: it never modifies fixture files, never
mutates records, never writes JSON, and never makes an external call. All
customer/order facts served to the workflow come from this deterministic
dataset, never from the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from customer_ops.models import CustomerRecord, OrderRecord

# Resolved relative to this module, not the process cwd, so the store works
# regardless of where the app/tests are launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CUSTOMERS_PATH = _PROJECT_ROOT / "data" / "customers.json"
DEFAULT_ORDERS_PATH = _PROJECT_ROOT / "data" / "orders.json"


class DataStoreError(Exception):
    """Raised when a fixture file is missing, malformed, or schema-invalid."""


class CustomerNotFoundError(Exception):
    """Raised when a lookup targets a customer_id absent from the store."""


class OrderNotFoundError(Exception):
    """Raised when a lookup targets an order_id absent from the store."""


class CustomerOperationsStore(Protocol):
    """Read-only interface for customer/order operational lookups."""

    def get_customer(self, customer_id: str) -> CustomerRecord: ...

    def list_orders_for_customer(self, customer_id: str) -> list[OrderRecord]: ...

    def get_order(self, order_id: str) -> OrderRecord: ...


def _load_json_array(path: Path, *, label: str) -> list[object]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataStoreError(f"Could not read {label} fixture at {path}: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DataStoreError(f"Malformed JSON in {label} fixture at {path}: {exc}") from exc

    if not isinstance(data, list):
        raise DataStoreError(f"{label} fixture at {path} must contain a JSON array.")

    return data


class JsonCustomerOperationsStore:
    """`CustomerOperationsStore` backed by the synthetic JSON fixtures.

    Fixture paths default to `data/customers.json` and `data/orders.json`
    resolved relative to this module. Explicit paths may be injected (e.g.
    for tests).
    """

    def __init__(
        self,
        customers_path: Path | str | None = None,
        orders_path: Path | str | None = None,
    ) -> None:
        customers_path = Path(customers_path) if customers_path is not None else DEFAULT_CUSTOMERS_PATH
        orders_path = Path(orders_path) if orders_path is not None else DEFAULT_ORDERS_PATH

        raw_customers = _load_json_array(customers_path, label="customers")
        raw_orders = _load_json_array(orders_path, label="orders")

        try:
            customers = [CustomerRecord.model_validate(item) for item in raw_customers]
        except ValidationError as exc:
            raise DataStoreError(f"Invalid customer record in {customers_path}: {exc}") from exc

        try:
            orders = [OrderRecord.model_validate(item) for item in raw_orders]
        except ValidationError as exc:
            raise DataStoreError(f"Invalid order record in {orders_path}: {exc}") from exc

        self._customers_by_id: dict[str, CustomerRecord] = {
            customer.customer_id: customer for customer in customers
        }
        self._orders_by_id: dict[str, OrderRecord] = {order.order_id: order for order in orders}

        self._orders_by_customer: dict[str, list[OrderRecord]] = {}
        for order in orders:
            self._orders_by_customer.setdefault(order.customer_id, []).append(order)

    def get_customer(self, customer_id: str) -> CustomerRecord:
        try:
            return self._customers_by_id[customer_id]
        except KeyError:
            raise CustomerNotFoundError(f"No customer found with customer_id={customer_id!r}") from None

    def list_orders_for_customer(self, customer_id: str) -> list[OrderRecord]:
        if customer_id not in self._customers_by_id:
            raise CustomerNotFoundError(f"No customer found with customer_id={customer_id!r}")
        return list(self._orders_by_customer.get(customer_id, []))

    def get_order(self, order_id: str) -> OrderRecord:
        try:
            return self._orders_by_id[order_id]
        except KeyError:
            raise OrderNotFoundError(f"No order found with order_id={order_id!r}") from None
