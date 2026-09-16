"""Structured action-input preparation for the Mercora Customer Operations Agent.

Converts a `ProposedAction` into a validated `ActionInputResult` - the exact
parameters a future execution step needs, never invented. Every action type
except `change_address` builds its input deterministically from data already
in hand (just `order_id`). `change_address` is the one case that needs a
customer-supplied value (a replacement address) that isn't already
structured anywhere, so it alone uses a single OpenAI Responses API
structured-output call, through the injectable `ActionInputExtractor`
protocol, to read it out of the customer's message - never to invent or
autocomplete it.
"""

from __future__ import annotations

from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, model_validator

from customer_ops.action_proposal import ProposedAction
from customer_ops.classifier import resolve_model
from customer_ops.state import ActionType

ADDRESS_EXTRACTION_INSTRUCTIONS = """
You are the address-extraction step of the Mercora customer operations
workflow. Read the customer's message - written in Spanish or English - and
extract only an explicit replacement shipping address the customer typed
themselves.

Rules:
- Return new_shipping_address exactly as the customer wrote it (trimming
  only surrounding whitespace) - never invented, completed, normalized, or
  guessed from partial information.
- If the customer does not explicitly supply a full replacement address in
  this message, return new_shipping_address = null. Do not reuse, infer, or
  autocomplete a street, city, or postal code that was not stated.

Return only the structured extraction. Do not include reasoning,
explanations, or a confidence score.
""".strip()


class AddressExtraction(BaseModel):
    """The structured output of address extraction from a customer message.

    Contains only the extracted value - no chain-of-thought, rationale, or
    confidence score. `new_shipping_address` is `None` when the customer
    did not explicitly supply a replacement address.
    """

    new_shipping_address: str | None = None


class ActionInputExtractor(Protocol):
    """Interface for model-backed action-input extraction.

    Only `change_address` needs this; every other action type builds its
    input deterministically with no model call.
    """

    def extract_new_shipping_address(self, customer_message: str) -> AddressExtraction: ...


class ActionInputError(Exception):
    """Raised when action-input extraction fails for a technical reason.

    A missing customer-supplied value is a legitimate business outcome
    (`ActionInputResult(ready=False, ...)`), not this error - this is
    reserved for API/structured-output failures.
    """


class OpenAIActionInputExtractor:
    """`ActionInputExtractor` backed by the OpenAI Responses API.

    Uses Structured Outputs, mirroring `OpenAIRequestClassifier`.
    """

    def __init__(self, client: OpenAI | None = None, model: str | None = None) -> None:
        if client is None:
            load_dotenv()
            client = OpenAI()
        self._client = client
        self._model = resolve_model(model)

    def extract_new_shipping_address(self, customer_message: str) -> AddressExtraction:
        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=ADDRESS_EXTRACTION_INSTRUCTIONS,
                input=customer_message,
                text_format=AddressExtraction,
                store=False,
            )
        except (OpenAIError, ValidationError) as exc:
            raise ActionInputError(f"Address extraction failed: {exc}") from exc

        extraction = response.output_parsed
        if extraction is None:
            raise ActionInputError(
                "Address extraction response did not contain a parsed structured result."
            )
        return extraction


# --- Action-input schemas (one per action type) ----------------------------------


class CancelOrderInput(BaseModel):
    order_id: str = Field(min_length=1)


class ChangeAddressInput(BaseModel):
    order_id: str = Field(min_length=1)
    new_shipping_address: str = Field(min_length=1)


class RefundInput(BaseModel):
    order_id: str = Field(min_length=1)


class InvestigationInput(BaseModel):
    order_id: str = Field(min_length=1)


# --- ActionInputResult -------------------------------------------------------------


class ActionInputResult(BaseModel):
    """The structured result of preparing an action's execution input.

    `parameters` holds the JSON-friendly dump of the matching `...Input`
    model when `ready`; `missing_fields` names what operational field is
    missing when not. No hidden reasoning, no confidence score.
    """

    ready: bool
    action_type: ActionType
    parameters: dict[str, object] | None = None
    missing_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "ActionInputResult":
        if self.ready and self.parameters is None:
            raise ValueError("parameters is required when ready is True.")
        if self.ready and self.missing_fields:
            raise ValueError("missing_fields must be empty when ready is True.")
        if not self.ready and not self.missing_fields:
            raise ValueError("missing_fields is required when ready is False.")
        return self


# --- Deterministic dispatcher --------------------------------------------------


def prepare_action_input(
    proposed_action: ProposedAction,
    customer_message: str,
    extractor: ActionInputExtractor,
) -> ActionInputResult:
    """Prepare the validated execution input for a proposed action.

    Deterministic, with no model call, for every action type except
    `change_address`, which calls `extractor.extract_new_shipping_address`
    exactly once. Never invents a missing value: a missing required field
    produces `ready=False` with `missing_fields` naming it.
    """
    action_type = proposed_action.action_type
    order_id = proposed_action.order_id

    if action_type == "cancel_order":
        return ActionInputResult(
            ready=True,
            action_type=action_type,
            parameters=CancelOrderInput(order_id=order_id).model_dump(mode="json"),
        )

    if action_type == "issue_refund":
        return ActionInputResult(
            ready=True,
            action_type=action_type,
            parameters=RefundInput(order_id=order_id).model_dump(mode="json"),
        )

    if action_type in ("investigate_billing", "investigate_product_issue"):
        return ActionInputResult(
            ready=True,
            action_type=action_type,
            parameters=InvestigationInput(order_id=order_id).model_dump(mode="json"),
        )

    if action_type == "change_address":
        extraction = extractor.extract_new_shipping_address(customer_message)
        new_address = (extraction.new_shipping_address or "").strip()
        if not new_address:
            return ActionInputResult(
                ready=False,
                action_type=action_type,
                missing_fields=["new_shipping_address"],
            )
        return ActionInputResult(
            ready=True,
            action_type=action_type,
            parameters=ChangeAddressInput(
                order_id=order_id, new_shipping_address=new_address
            ).model_dump(mode="json"),
        )

    raise ActionInputError(f"No input-preparation rule is defined for action type {action_type!r}.")
