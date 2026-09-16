"""Tests for final customer-facing response generation.

`OpenAICustomerResponseGenerator` is exercised through a fake OpenAI client
double (`FakeOpenAIClient`, mirroring `tests/test_classifier.py`) that
records the kwargs it receives - no test makes a network call.
`build_response_context` is tested directly as a pure function.
"""

import importlib
import json

import pytest
from openai import OpenAIError
from pydantic import ValidationError

import customer_ops.response_generator as response_generator_module
from customer_ops.response_generator import (
    CustomerResponse,
    OpenAICustomerResponseGenerator,
    ResponseContext,
    ResponseGenerationError,
    build_response_context,
)


class _FakeParsedResponse:
    """Stand-in for `openai.types.responses.ParsedResponse`."""

    def __init__(self, parsed):
        self.output_parsed = parsed


class _FakeResponsesAPI:
    def __init__(self, parsed=None, raise_exc=None):
        self.calls: list[dict] = []
        self._parsed = parsed
        self._raise_exc = raise_exc

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeParsedResponse(self._parsed)


class FakeOpenAIClient:
    """Minimal double for `openai.OpenAI` exposing only `responses.parse`.

    Does not require an API key and makes no network call.
    """

    def __init__(self, parsed=None, raise_exc=None):
        self.responses = _FakeResponsesAPI(parsed=parsed, raise_exc=raise_exc)


def order(**overrides):
    fields = {"order_id": "order-1001", "status": "shipped", "tracking_number": "TRACK123"}
    fields.update(overrides)
    return fields


# --- CustomerResponse contract ----------------------------------------------------


def test_valid_customer_response():
    response = CustomerResponse(message="Tu pedido order-1001 fue enviado.")
    assert response.message == "Tu pedido order-1001 fue enviado."


def test_empty_message_rejected():
    with pytest.raises(ValidationError):
        CustomerResponse(message="")


# --- Model resolution precedence (shared with classifier.resolve_model) ------------


def test_explicit_model_is_used_for_the_call():
    client = FakeOpenAIClient(parsed=CustomerResponse(message="Hola"))
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-explicit")
    generator.generate(build_response_context("Hola"))
    assert client.responses.calls[0]["model"] == "gpt-explicit"


def test_environment_model_overrides_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    client = FakeOpenAIClient(parsed=CustomerResponse(message="Hola"))
    generator = OpenAICustomerResponseGenerator(client=client)
    generator.generate(build_response_context("Hola"))
    assert client.responses.calls[0]["model"] == "env-model"


def test_default_model_remains_gpt_5_6_luna(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = FakeOpenAIClient(parsed=CustomerResponse(message="Hola"))
    generator = OpenAICustomerResponseGenerator(client=client)
    generator.generate(build_response_context("Hola"))
    assert client.responses.calls[0]["model"] == "gpt-5.6-luna"


# --- Injected fake client behavior --------------------------------------------------


def test_injected_client_does_not_need_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = CustomerResponse(message="Hola, tu pedido esta en camino.")
    client = FakeOpenAIClient(parsed=response)
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-test")

    result = generator.generate(build_response_context("Donde esta mi pedido?"))

    assert result == response


def test_structured_responses_api_call_used():
    client = FakeOpenAIClient(parsed=CustomerResponse(message="Hola"))
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-test")

    generator.generate(build_response_context("Hola"))

    assert client.responses.calls[0]["text_format"] is CustomerResponse


def test_store_false_is_sent():
    client = FakeOpenAIClient(parsed=CustomerResponse(message="Hola"))
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-test")

    generator.generate(build_response_context("Hola"))

    assert client.responses.calls[0]["store"] is False


def test_exactly_one_model_call():
    client = FakeOpenAIClient(parsed=CustomerResponse(message="Hola"))
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-test")

    generator.generate(build_response_context("Hola"))

    assert len(client.responses.calls) == 1


# --- ResponseContext is narrow / PII-free -------------------------------------------


def test_response_context_serialized_without_pii():
    context = build_response_context(
        "Donde esta order-1001?",
        intent="order_status",
        route="information",
        selected_order_id="order-1001",
        order=order(),
    )
    client = FakeOpenAIClient(parsed=CustomerResponse(message="Hola"))
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-test")

    generator.generate(context)

    sent_input = client.responses.calls[0]["input"]
    assert "@example.com" not in sent_input
    assert "shipping_address" not in sent_input
    assert "payment_status" not in sent_input


def test_full_customer_context_not_included():
    assert "customer_context" not in ResponseContext.model_fields


def test_full_order_context_not_included():
    assert "order_context" not in ResponseContext.model_fields
    assert "orders" not in ResponseContext.model_fields


def test_audit_log_not_included():
    assert "audit_log" not in ResponseContext.model_fields


# --- Error handling ------------------------------------------------------------------


def test_api_failure_becomes_response_generation_error():
    client = FakeOpenAIClient(raise_exc=OpenAIError("simulated API failure"))
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-test")

    with pytest.raises(ResponseGenerationError):
        generator.generate(build_response_context("Hola"))


def test_missing_parsed_output_becomes_response_generation_error():
    client = FakeOpenAIClient(parsed=None)
    generator = OpenAICustomerResponseGenerator(client=client, model="gpt-test")

    with pytest.raises(ResponseGenerationError):
        generator.generate(build_response_context("Hola"))


# --- Language handling (context construction never translates) ---------------------


def test_spanish_customer_message_preserved():
    context = build_response_context("¿Dónde está mi pedido order-1001?")
    assert context.customer_message == "¿Dónde está mi pedido order-1001?"


def test_english_customer_message_preserved():
    context = build_response_context("Where is my order order-1001?")
    assert context.customer_message == "Where is my order order-1001?"


# --- No hidden reasoning fields -----------------------------------------------------


def test_customer_response_has_no_hidden_reasoning_fields():
    assert set(CustomerResponse.model_fields) == {"message"}


def test_response_context_has_no_hidden_reasoning_fields():
    assert "reasoning" not in ResponseContext.model_fields
    assert "rationale" not in ResponseContext.model_fields
    assert "confidence" not in ResponseContext.model_fields


# --- Import safety -------------------------------------------------------------------


def test_importing_response_generator_module_makes_no_openai_call(no_network):
    importlib.reload(response_generator_module)


# --- ResponseContext construction (branch-specific) ---------------------------------


def test_information_order_status_includes_only_selected_order():
    context = build_response_context(
        "Donde esta order-1001?",
        intent="order_status",
        route="information",
        selected_order_id="order-1001",
        order=order(status="shipped", tracking_number="1Z999"),
    )
    assert context.order_summary.order_id == "order-1001"
    assert context.order_summary.status == "shipped"
    assert context.order_summary.tracking_number == "1Z999"


def test_clarification_includes_missing_requirement():
    action_input = {
        "ready": False,
        "action_type": "change_address",
        "parameters": None,
        "missing_fields": ["new_shipping_address"],
    }
    context = build_response_context(
        "Quiero cambiar la direccion de order-1001.",
        intent="address_change",
        route="clarification",
        action_input=action_input,
    )
    assert context.action_input == action_input
    assert context.action_input["missing_fields"] == ["new_shipping_address"]


def test_blocked_includes_policy_reason():
    policy_assessment = {
        "outcome": "blocked",
        "policy_code": "CANCEL_BLOCKED_STATUS",
        "requires_human_approval": False,
        "reason": "Order status 'shipped' does not allow cancellation.",
    }
    context = build_response_context(
        "Cancela order-1001.",
        intent="cancel_order",
        route="blocked",
        policy_assessment=policy_assessment,
    )
    assert context.policy_reason == "Order status 'shipped' does not allow cancellation."
    assert context.policy_outcome == "blocked"
    assert context.policy_code == "CANCEL_BLOCKED_STATUS"


def test_executed_safe_action_includes_action_result():
    action_result = {
        "action_type": "cancel_order",
        "success": True,
        "order_id": "order-1001",
        "message": "Simulated cancel_order executed successfully for order order-1001.",
        "reference_id": None,
    }
    context = build_response_context(
        "Cancela order-1001.",
        intent="cancel_order",
        route="action",
        action_result=action_result,
    )
    assert context.action_result == action_result


def test_rejected_approval_includes_human_decision():
    context = build_response_context(
        "Reembolsa order-1001.",
        intent="refund_request",
        route="approval",
        human_decision="rejected",
    )
    assert context.human_decision == "rejected"
    assert context.action_result is None


def test_approved_refund_includes_executed_result():
    action_result = {
        "action_type": "issue_refund",
        "success": True,
        "order_id": "order-1001",
        "message": "Simulated issue_refund executed successfully for order order-1001 (amount: 79.0 USD).",
        "reference_id": None,
    }
    context = build_response_context(
        "Reembolsa order-1001.",
        intent="refund_request",
        route="approval",
        human_decision="approved",
        action_result=action_result,
    )
    assert context.human_decision == "approved"
    assert context.action_result == action_result


def test_unrelated_customer_orders_excluded():
    # build_response_context only ever accepts a single already-selected
    # order - there is no parameter through which other orders could enter.
    context = build_response_context(
        "Donde esta order-1001?",
        selected_order_id="order-1001",
        order=order(order_id="order-1001"),
    )
    dumped = context.model_dump(mode="json")
    assert dumped["order_summary"]["order_id"] == "order-1001"
    assert "orders" not in dumped
    assert "order-9999" not in json.dumps(dumped)


def test_response_context_order_summary_excludes_pii():
    full_order_like_dict = order(
        shipping_address="900 Confidential Ave, Privacy City, PC 00001, USA",
        payment_status="paid",
        customer_id="cust-777",
        total=79.0,
        currency="USD",
        items=[{"sku": "S", "name": "N", "quantity": 1, "unit_price": 79.0}],
    )
    context = build_response_context("Donde esta order-1001?", order=full_order_like_dict)
    dumped = context.model_dump(mode="json")
    assert set(dumped["order_summary"]) == {"order_id", "status", "tracking_number"}
    assert "shipping_address" not in json.dumps(dumped)
    assert "cust-777" not in json.dumps(dumped)


def test_response_context_is_json_serializable():
    context = build_response_context(
        "Donde esta order-1001?",
        intent="order_status",
        route="information",
        selected_order_id="order-1001",
        order=order(),
        policy_assessment={"outcome": "information_only", "policy_code": "ORDER_STATUS_INFO", "reason": "x"},
    )
    json.dumps(context.model_dump(mode="json"))


def test_build_response_context_is_deterministic():
    kwargs = dict(
        customer_message="Donde esta order-1001?",
        intent="order_status",
        route="information",
        selected_order_id="order-1001",
        order=order(),
    )
    first = build_response_context(**kwargs)
    second = build_response_context(**kwargs)
    assert first == second
