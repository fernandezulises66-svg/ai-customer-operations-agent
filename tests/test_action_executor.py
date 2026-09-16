"""Tests for the simulated action-execution layer.

`execute_action` is exercised against `InMemoryCustomerActionStore`
(constructed from small in-memory records, no file I/O) and against a
minimal fake store double that lets tests force a store-level failure. No
test makes a network call.
"""

import pytest

from customer_ops.action_executor import ActionExecutionError, ActionResult, execute_action
from customer_ops.action_inputs import ActionInputResult
from customer_ops.action_proposal import ProposedAction
from customer_ops.models import CustomerRecord, OrderRecord
from tools.action_store import ActionStoreError, InMemoryCustomerActionStore


def make_customer(customer_id="cust-001") -> CustomerRecord:
    return CustomerRecord(
        customer_id=customer_id,
        name="Test Customer",
        email="test.customer@example.com",
        account_status="active",
        customer_tier="standard",
    )


def make_order(order_id="order-1001", **overrides) -> OrderRecord:
    fields = {
        "order_id": order_id,
        "customer_id": "cust-001",
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


def make_store(*orders):
    return InMemoryCustomerActionStore(customers=[make_customer()], orders=list(orders))


def proposed(action_type, order_id="order-1001", requires_human_approval=None):
    if requires_human_approval is None:
        requires_human_approval = action_type in (
            "issue_refund",
            "investigate_billing",
            "investigate_product_issue",
        )
    return ProposedAction(
        action_type=action_type, order_id=order_id, requires_human_approval=requires_human_approval
    )


def input_for(action_type, **parameters):
    return ActionInputResult(ready=True, action_type=action_type, parameters=parameters)


class FakeFailingActionStore:
    """Minimal `CustomerActionStore` double that always fails a mutation."""

    def cancel_order(self, order_id):
        raise ActionStoreError("simulated store failure")

    def change_shipping_address(self, order_id, new_address):
        raise ActionStoreError("simulated store failure")

    def issue_full_refund(self, order_id):
        raise ActionStoreError("simulated store failure")

    def create_billing_investigation(self, order_id):
        raise ActionStoreError("simulated store failure")

    def create_product_investigation(self, order_id):
        raise ActionStoreError("simulated store failure")


# --- Action-type mapping -----------------------------------------------------------


def test_cancel_maps_to_cancel_order_store_method():
    order = make_order(status="pending")
    store = make_store(order)
    result = execute_action(proposed("cancel_order"), input_for("cancel_order", order_id="order-1001"), store)
    assert result.action_type == "cancel_order"
    assert store.get_order("order-1001").status == "cancelled"


def test_address_change_maps_with_exact_new_address():
    order = make_order(status="pending", shipping_address="Old address")
    store = make_store(order)
    result = execute_action(
        proposed("change_address"),
        input_for("change_address", order_id="order-1001", new_shipping_address="Calle Falsa 123"),
        store,
    )
    assert result.action_type == "change_address"
    assert store.get_order("order-1001").shipping_address == "Calle Falsa 123"


def test_refund_maps_to_refund_method():
    order = make_order(payment_status="paid")
    store = make_store(order)
    result = execute_action(
        proposed("issue_refund"), input_for("issue_refund", order_id="order-1001"), store, human_approved=True
    )
    assert result.action_type == "issue_refund"
    assert store.get_order("order-1001").payment_status == "refunded"


def test_billing_investigation_maps_correctly():
    order = make_order()
    store = make_store(order)
    result = execute_action(
        proposed("investigate_billing"),
        input_for("investigate_billing", order_id="order-1001"),
        store,
        human_approved=True,
    )
    assert result.action_type == "investigate_billing"
    assert result.reference_id == "billing-investigation-order-1001"


def test_product_investigation_maps_correctly():
    order = make_order()
    store = make_store(order)
    result = execute_action(
        proposed("investigate_product_issue"),
        input_for("investigate_product_issue", order_id="order-1001"),
        store,
        human_approved=True,
    )
    assert result.action_type == "investigate_product_issue"
    assert result.reference_id == "product-investigation-order-1001"


# --- Approval gate (defense in depth) ------------------------------------------------


@pytest.mark.parametrize("action_type", ["cancel_order", "change_address"])
def test_safe_actions_execute_without_approval(action_type):
    order = make_order(status="pending", shipping_address="Old")
    store = make_store(order)
    params = {"order_id": "order-1001"}
    if action_type == "change_address":
        params["new_shipping_address"] = "New Address"
    result = execute_action(proposed(action_type), input_for(action_type, **params), store, human_approved=False)
    assert result.success is True


@pytest.mark.parametrize("action_type", ["issue_refund", "investigate_billing", "investigate_product_issue"])
def test_sensitive_action_without_approval_is_blocked(action_type):
    order = make_order()
    store = make_store(order)
    with pytest.raises(ActionExecutionError):
        execute_action(
            proposed(action_type), input_for(action_type, order_id="order-1001"), store, human_approved=False
        )


@pytest.mark.parametrize("action_type", ["issue_refund", "investigate_billing", "investigate_product_issue"])
def test_sensitive_action_with_approval_executes(action_type):
    order = make_order(payment_status="paid")
    store = make_store(order)
    result = execute_action(
        proposed(action_type), input_for(action_type, order_id="order-1001"), store, human_approved=True
    )
    assert result.success is True


def test_sensitive_action_without_approval_does_not_mutate_store():
    order = make_order(payment_status="paid")
    store = make_store(order)
    with pytest.raises(ActionExecutionError):
        execute_action(
            proposed("issue_refund"), input_for("issue_refund", order_id="order-1001"), store, human_approved=False
        )
    assert store.get_order("order-1001").payment_status == "paid"


# --- Exactly one mutation, structured result --------------------------------------------


def test_exactly_one_store_mutation_per_execution():
    order = make_order(status="pending")

    class CountingStore(InMemoryCustomerActionStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cancel_calls = 0

        def cancel_order(self, order_id):
            self.cancel_calls += 1
            return super().cancel_order(order_id)

    store = CountingStore(customers=[make_customer()], orders=[order])
    execute_action(proposed("cancel_order"), input_for("cancel_order", order_id="order-1001"), store)
    assert store.cancel_calls == 1


def test_result_is_structured_and_json_friendly():
    order = make_order(status="pending")
    store = make_store(order)
    result = execute_action(proposed("cancel_order"), input_for("cancel_order", order_id="order-1001"), store)
    assert isinstance(result, ActionResult)
    dumped = result.model_dump(mode="json")
    assert dumped["action_type"] == "cancel_order"
    assert dumped["success"] is True
    assert dumped["order_id"] == "order-1001"
    assert isinstance(dumped["message"], str)


# --- Error handling ------------------------------------------------------------------------


def test_store_error_becomes_action_execution_error():
    order = make_order(status="shipped")  # ineligible for cancellation
    store = make_store(order)
    with pytest.raises(ActionExecutionError):
        execute_action(proposed("cancel_order"), input_for("cancel_order", order_id="order-1001"), store)


def test_unknown_order_store_error_becomes_action_execution_error():
    store = make_store(make_order())
    with pytest.raises(ActionExecutionError):
        execute_action(
            proposed("cancel_order", order_id="order-does-not-exist"),
            input_for("cancel_order", order_id="order-does-not-exist"),
            store,
        )


def test_mismatched_action_input_type_raises():
    order = make_order(status="pending")
    store = make_store(order)
    with pytest.raises(ActionExecutionError):
        execute_action(proposed("cancel_order"), input_for("change_address", order_id="order-1001"), store)


def test_not_ready_action_input_raises():
    order = make_order(status="pending")
    store = make_store(order)
    not_ready = ActionInputResult(ready=False, action_type="cancel_order", missing_fields=["order_id"])
    with pytest.raises(ActionExecutionError):
        execute_action(proposed("cancel_order"), not_ready, store)


def test_unsupported_action_type_raises():
    order = make_order(status="pending")
    store = make_store(order)
    # Bypass ProposedAction's own Literal validation to simulate corrupted state.
    bad_action = ProposedAction.model_construct(
        action_type="unsupported_action", order_id="order-1001", requires_human_approval=False
    )
    bad_input = ActionInputResult.model_construct(
        ready=True, action_type="unsupported_action", parameters={"order_id": "order-1001"}, missing_fields=[]
    )
    with pytest.raises(ActionExecutionError):
        execute_action(bad_action, bad_input, store)


def test_store_failure_becomes_action_execution_error():
    result_input = input_for("cancel_order", order_id="order-1001")
    with pytest.raises(ActionExecutionError):
        execute_action(proposed("cancel_order"), result_input, FakeFailingActionStore())


def test_execute_action_performs_no_network_access(no_network):
    order = make_order(status="pending")
    store = make_store(order)
    execute_action(proposed("cancel_order"), input_for("cancel_order", order_id="order-1001"), store)


def test_no_silent_success_fallback_on_store_failure():
    order = make_order(status="shipped")
    store = make_store(order)
    try:
        execute_action(proposed("cancel_order"), input_for("cancel_order", order_id="order-1001"), store)
        assert False, "execute_action should have raised ActionExecutionError"
    except ActionExecutionError:
        pass
    assert store.get_order("order-1001").status == "shipped"
