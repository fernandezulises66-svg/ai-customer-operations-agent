"""Tests for structured action-input preparation.

`prepare_action_input` never queries OpenAI except for `change_address`,
which is exercised through `FakeOpenAIClient` (mirroring
`tests/test_classifier.py`) so no test makes a network call.
"""

import importlib

import pytest
from openai import OpenAIError
from pydantic import ValidationError

import customer_ops.action_inputs as action_inputs_module
from customer_ops.action_inputs import (
    ActionInputError,
    ActionInputResult,
    AddressExtraction,
    OpenAIActionInputExtractor,
    prepare_action_input,
)
from customer_ops.action_proposal import ProposedAction


class FakeActionInputExtractor:
    """Deterministic `ActionInputExtractor` test double.

    Returns `address` (possibly `None`) for every call and records the
    messages it was asked to extract from.
    """

    def __init__(self, address: str | None = None):
        self.address = address
        self.calls: list[str] = []

    def extract_new_shipping_address(self, customer_message: str) -> AddressExtraction:
        self.calls.append(customer_message)
        return AddressExtraction(new_shipping_address=self.address)


def proposed(action_type, order_id="order-1001"):
    requires_approval = action_type in ("issue_refund", "investigate_billing", "investigate_product_issue")
    return ProposedAction(action_type=action_type, order_id=order_id, requires_human_approval=requires_approval)


# --- Deterministic action types (no model call) -----------------------------------


def test_cancel_order_input_ready_without_model_call():
    extractor = FakeActionInputExtractor()
    result = prepare_action_input(proposed("cancel_order"), "cancel please", extractor)
    assert result.ready is True
    assert result.parameters == {"order_id": "order-1001"}
    assert extractor.calls == []


def test_refund_input_ready_without_model_call():
    extractor = FakeActionInputExtractor()
    result = prepare_action_input(proposed("issue_refund"), "refund please", extractor)
    assert result.ready is True
    assert result.parameters == {"order_id": "order-1001"}
    assert extractor.calls == []


def test_billing_investigation_input_ready_without_model_call():
    extractor = FakeActionInputExtractor()
    result = prepare_action_input(proposed("investigate_billing"), "billing issue", extractor)
    assert result.ready is True
    assert result.parameters == {"order_id": "order-1001"}
    assert extractor.calls == []


def test_product_investigation_input_ready_without_model_call():
    extractor = FakeActionInputExtractor()
    result = prepare_action_input(proposed("investigate_product_issue"), "product issue", extractor)
    assert result.ready is True
    assert result.parameters == {"order_id": "order-1001"}
    assert extractor.calls == []


# --- change_address --------------------------------------------------------------


def test_address_change_with_supplied_address_is_ready():
    extractor = FakeActionInputExtractor(address="Calle Falsa 123, Cordoba")
    result = prepare_action_input(proposed("change_address"), "cambiar direccion a Calle Falsa 123, Cordoba", extractor)
    assert result.ready is True
    assert result.parameters == {"order_id": "order-1001", "new_shipping_address": "Calle Falsa 123, Cordoba"}


def test_address_change_missing_address_returns_not_ready():
    extractor = FakeActionInputExtractor(address=None)
    result = prepare_action_input(proposed("change_address"), "cambiar direccion", extractor)
    assert result.ready is False
    assert result.parameters is None


def test_missing_fields_contains_only_the_required_missing_field():
    extractor = FakeActionInputExtractor(address=None)
    result = prepare_action_input(proposed("change_address"), "cambiar direccion", extractor)
    assert result.missing_fields == ["new_shipping_address"]


def test_no_address_is_invented_when_none_supplied():
    extractor = FakeActionInputExtractor(address="   ")
    result = prepare_action_input(proposed("change_address"), "cambiar direccion", extractor)
    assert result.ready is False
    assert result.parameters is None


def test_address_extraction_receives_the_customer_message():
    extractor = FakeActionInputExtractor(address="Avenida Siempre Viva 742")
    prepare_action_input(proposed("change_address"), "Cambiar a Avenida Siempre Viva 742", extractor)
    assert extractor.calls == ["Cambiar a Avenida Siempre Viva 742"]


# --- ActionInputResult contract ---------------------------------------------------


def test_ready_result_requires_parameters():
    with pytest.raises(ValidationError):
        ActionInputResult(ready=True, action_type="cancel_order", parameters=None)


def test_not_ready_result_requires_missing_fields():
    with pytest.raises(ValidationError):
        ActionInputResult(ready=False, action_type="change_address", missing_fields=[])


def test_deterministic_fake_extractor_behavior():
    extractor = FakeActionInputExtractor(address="Calle Falsa 123")
    kwargs = dict(proposed_action=proposed("change_address"), customer_message="msg", extractor=extractor)
    first = prepare_action_input(**kwargs)
    second = prepare_action_input(**kwargs)
    assert first == second


def test_action_input_result_is_json_serializable():
    extractor = FakeActionInputExtractor(address="Calle Falsa 123")
    result = prepare_action_input(proposed("change_address"), "msg", extractor)
    dumped = result.model_dump(mode="json")
    assert dumped == {
        "ready": True,
        "action_type": "change_address",
        "parameters": {"order_id": "order-1001", "new_shipping_address": "Calle Falsa 123"},
        "missing_fields": [],
    }


def test_action_input_result_has_no_hidden_reasoning_fields():
    extractor = FakeActionInputExtractor(address="Calle Falsa 123")
    result = prepare_action_input(proposed("change_address"), "msg", extractor)
    assert set(ActionInputResult.model_fields) == {"ready", "action_type", "parameters", "missing_fields"}


# --- OpenAIActionInputExtractor (fake OpenAI client, no network) ------------------


class _FakeParsedResponse:
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
    """Minimal double for `openai.OpenAI` exposing only `responses.parse`."""

    def __init__(self, parsed=None, raise_exc=None):
        self.responses = _FakeResponsesAPI(parsed=parsed, raise_exc=raise_exc)


def test_injected_fake_client_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    extraction = AddressExtraction(new_shipping_address="Calle Falsa 123")
    client = FakeOpenAIClient(parsed=extraction)
    extractor = OpenAIActionInputExtractor(client=client, model="gpt-test")
    result = extractor.extract_new_shipping_address("cambiar direccion a Calle Falsa 123")
    assert result == extraction


def test_real_extractor_uses_structured_output_text_format():
    extraction = AddressExtraction(new_shipping_address="Calle Falsa 123")
    client = FakeOpenAIClient(parsed=extraction)
    extractor = OpenAIActionInputExtractor(client=client, model="gpt-test")
    extractor.extract_new_shipping_address("cambiar direccion")
    assert client.responses.calls[0]["text_format"] is AddressExtraction


def test_one_model_call_for_address_extraction():
    extraction = AddressExtraction(new_shipping_address="Calle Falsa 123")
    client = FakeOpenAIClient(parsed=extraction)
    extractor = OpenAIActionInputExtractor(client=client, model="gpt-test")
    extractor.extract_new_shipping_address("cambiar direccion")
    assert len(client.responses.calls) == 1


def test_store_false_is_sent():
    extraction = AddressExtraction(new_shipping_address="Calle Falsa 123")
    client = FakeOpenAIClient(parsed=extraction)
    extractor = OpenAIActionInputExtractor(client=client, model="gpt-test")
    extractor.extract_new_shipping_address("cambiar direccion")
    assert client.responses.calls[0]["store"] is False


def test_api_failure_becomes_action_input_error():
    client = FakeOpenAIClient(raise_exc=OpenAIError("simulated API failure"))
    extractor = OpenAIActionInputExtractor(client=client, model="gpt-test")
    with pytest.raises(ActionInputError):
        extractor.extract_new_shipping_address("cambiar direccion")


def test_missing_parsed_output_becomes_action_input_error():
    client = FakeOpenAIClient(parsed=None)
    extractor = OpenAIActionInputExtractor(client=client, model="gpt-test")
    with pytest.raises(ActionInputError):
        extractor.extract_new_shipping_address("cambiar direccion")


def test_importing_action_inputs_module_makes_no_openai_call(no_network):
    importlib.reload(action_inputs_module)
