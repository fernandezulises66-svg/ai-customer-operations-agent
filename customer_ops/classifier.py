"""Structured request classification for the Mercora Customer Operations Agent.

Wraps a single call to the OpenAI Responses API structured-output mechanism
(`responses.parse`) behind the `RequestClassifier` protocol, so the LangGraph
node in `customer_ops/graph.py` depends on an interface rather than being
hard-wired to the OpenAI SDK. Classification failures are technical errors
(`ClassificationError`) and must never be silently turned into a business
decision such as intent="other".
"""

from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from customer_ops.state import Intent, Urgency

DEFAULT_MODEL = "gpt-5.6-luna"

CLASSIFICATION_INSTRUCTIONS = """
You are the request classification step of the Mercora customer operations
workflow. Read the customer's message - written in Spanish or English - and
classify it into exactly one intent and one urgency level. Interpret what the
customer actually wants; do not pattern-match on individual keywords.

Intent categories:
- order_status: questions about shipment, delivery, tracking, or where an
  order is.
- refund_request: an explicit request to receive money back / a refund.
- cancel_order: a request to cancel an order.
- address_change: a request to change a delivery/shipping address.
- billing_issue: payment problems, duplicate charges, unexpected charges, or
  payment failures.
- product_issue: a damaged, incorrect, defective, missing, or otherwise
  problematic product.
- other: requests that genuinely do not fit any of the categories above.

Urgency levels:
- low: a routine informational request with no clear immediate operational
  or financial risk.
- medium: an ordinary issue or requested action that requires resolution.
- high: a clearly time-sensitive issue, a meaningful immediate
  financial/security risk, or a situation where delay could materially
  worsen the customer's outcome.

Do not classify a message as high urgency merely because it is written
angrily or with emphatic language - judge urgency from the operational and
financial substance of the request, not its tone.

Return only the structured classification. Do not include reasoning,
explanations, or a confidence score.
""".strip()


class ClassificationDecision(BaseModel):
    """The structured operational decision produced by classification.

    Contains only the controlled decision - no chain-of-thought, rationale,
    explanation, or confidence score.
    """

    intent: Intent
    urgency: Urgency


class ClassificationError(Exception):
    """Raised when classification fails for a technical reason.

    A `ClassificationError` is a technical failure, not a business decision -
    callers must not treat it as an implicit "other"/"low" classification.
    """


class RequestClassifier(Protocol):
    """Interface the classification node depends on.

    Keeps the LangGraph node decoupled from the OpenAI SDK: production code
    injects `OpenAIRequestClassifier`, tests inject a fake.
    """

    def classify(self, customer_message: str) -> ClassificationDecision: ...


def resolve_model(explicit_model: str | None) -> str:
    """Resolve the model name using the documented precedence.

    1. explicit argument, 2. `OPENAI_MODEL` environment variable, 3. default.
    """
    if explicit_model:
        return explicit_model
    env_model = os.environ.get("OPENAI_MODEL")
    if env_model:
        return env_model
    return DEFAULT_MODEL


class OpenAIRequestClassifier:
    """`RequestClassifier` backed by the OpenAI Responses API.

    Uses Structured Outputs (`responses.parse` with `text_format`) so the
    result is validated against `ClassificationDecision` rather than
    hand-parsed from free-form text.
    """

    def __init__(self, client: OpenAI | None = None, model: str | None = None) -> None:
        if client is None:
            load_dotenv()
            client = OpenAI()
        self._client = client
        self._model = resolve_model(model)

    def classify(self, customer_message: str) -> ClassificationDecision:
        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=CLASSIFICATION_INSTRUCTIONS,
                input=customer_message,
                text_format=ClassificationDecision,
                store=False,
            )
        except (OpenAIError, ValidationError) as exc:
            raise ClassificationError(f"Request classification failed: {exc}") from exc

        decision = response.output_parsed
        if decision is None:
            raise ClassificationError(
                "Classification response did not contain a parsed structured decision."
            )
        return decision
