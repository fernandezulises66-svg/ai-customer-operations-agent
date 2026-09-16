"""Tests for structured request classification.

No test makes a real OpenAI call: `OpenAIRequestClassifier` is exercised
through a fake OpenAI client double (`FakeOpenAIClient`) that records the
kwargs it receives and returns a canned `ParsedResponse`-shaped result, and
`no_network` additionally guards against accidental socket use.
"""

import importlib

import pytest
from openai import OpenAIError
from pydantic import ValidationError

import customer_ops.classifier as classifier_module
from customer_ops.classifier import (
    DEFAULT_MODEL,
    ClassificationDecision,
    ClassificationError,
    OpenAIRequestClassifier,
    resolve_model,
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


# --- ClassificationDecision validation ---------------------------------------


def test_decision_accepts_valid_intent_and_urgency():
    decision = ClassificationDecision(intent="refund_request", urgency="medium")
    assert decision.intent == "refund_request"
    assert decision.urgency == "medium"


def test_decision_rejects_invalid_intent():
    with pytest.raises(ValidationError):
        ClassificationDecision(intent="not_a_real_intent", urgency="low")


def test_decision_rejects_invalid_urgency():
    with pytest.raises(ValidationError):
        ClassificationDecision(intent="order_status", urgency="not_a_real_urgency")


# --- Model resolution precedence ---------------------------------------------


def test_explicit_model_overrides_environment_model(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert resolve_model("explicit-model") == "explicit-model"


def test_environment_model_overrides_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert resolve_model(None) == "env-model"


def test_default_model_is_gpt_5_6_luna(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_model(None) == "gpt-5.6-luna"
    assert DEFAULT_MODEL == "gpt-5.6-luna"


# --- Injected fake client behavior --------------------------------------------


def test_injected_fake_client_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    decision = ClassificationDecision(intent="order_status", urgency="low")
    client = FakeOpenAIClient(parsed=decision)

    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")
    result = classifier.classify("Where is my order?")

    assert result == decision


def test_customer_message_is_passed_to_the_model():
    decision = ClassificationDecision(intent="order_status", urgency="low")
    client = FakeOpenAIClient(parsed=decision)
    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")

    classifier.classify("Where is my order?")

    assert client.responses.calls[0]["input"] == "Where is my order?"


def test_structured_output_result_is_returned_correctly():
    decision = ClassificationDecision(intent="billing_issue", urgency="high")
    client = FakeOpenAIClient(parsed=decision)
    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")

    result = classifier.classify("Me cobraron dos veces la misma compra.")

    assert result.intent == "billing_issue"
    assert result.urgency == "high"


def test_one_responses_api_call_per_classify():
    decision = ClassificationDecision(intent="order_status", urgency="low")
    client = FakeOpenAIClient(parsed=decision)
    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")

    classifier.classify("Where is my order?")

    assert len(client.responses.calls) == 1


def test_store_false_is_sent():
    decision = ClassificationDecision(intent="order_status", urgency="low")
    client = FakeOpenAIClient(parsed=decision)
    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")

    classifier.classify("Where is my order?")

    assert client.responses.calls[0]["store"] is False


def test_explicit_model_is_used_for_the_call():
    decision = ClassificationDecision(intent="order_status", urgency="low")
    client = FakeOpenAIClient(parsed=decision)
    classifier = OpenAIRequestClassifier(client=client, model="gpt-explicit")

    classifier.classify("Where is my order?")

    assert client.responses.calls[0]["model"] == "gpt-explicit"


# --- Error handling ------------------------------------------------------------


def test_api_failure_becomes_classification_error():
    client = FakeOpenAIClient(raise_exc=OpenAIError("simulated API failure"))
    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")

    with pytest.raises(ClassificationError):
        classifier.classify("Where is my order?")


def test_missing_parsed_output_becomes_classification_error():
    client = FakeOpenAIClient(parsed=None)
    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")

    with pytest.raises(ClassificationError):
        classifier.classify("Where is my order?")


def test_failure_never_produces_a_fallback_decision():
    client = FakeOpenAIClient(raise_exc=OpenAIError("simulated API failure"))
    classifier = OpenAIRequestClassifier(client=client, model="gpt-test")

    try:
        classifier.classify("Where is my order?")
        assert False, "classify() should have raised ClassificationError"
    except ClassificationError:
        pass
    except Exception:
        assert False, "classification failures must surface as ClassificationError only"


# --- Import / construction safety ---------------------------------------------


def test_importing_classifier_module_makes_no_openai_call(no_network):
    importlib.reload(classifier_module)
