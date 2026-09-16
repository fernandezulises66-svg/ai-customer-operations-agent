"""Pydantic data contracts for Mercora's synthetic operational records.

These models validate the customer/order fixture data loaded from
`data/customers.json` and `data/orders.json` (see `tools/customer_data.py`).
They are data contracts only: structural validation (non-empty identifiers,
supported statuses, non-negative amounts, well-formed items), not business
policy, and no business-behavior methods.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AccountStatus = Literal["active", "suspended"]
CustomerTier = Literal["standard", "premium"]

OrderStatus = Literal["pending", "processing", "shipped", "delivered", "cancelled"]
PaymentStatus = Literal["pending", "paid", "failed", "refunded"]
Currency = Literal["USD"]


class CustomerRecord(BaseModel):
    """A synthetic Mercora customer record."""

    customer_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    email: str = Field(min_length=1)
    account_status: AccountStatus
    customer_tier: CustomerTier


class OrderItem(BaseModel):
    """A single purchased line item within an order."""

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    unit_price: float = Field(ge=0)


class OrderRecord(BaseModel):
    """A synthetic Mercora order record."""

    order_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    status: OrderStatus
    payment_status: PaymentStatus
    total: float = Field(ge=0)
    currency: Currency
    shipping_address: str = Field(min_length=1)
    tracking_number: str | None = None
    items: list[OrderItem] = Field(min_length=1)
