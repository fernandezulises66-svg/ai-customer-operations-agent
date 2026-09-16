"""Final customer-facing response generation for the Mercora Customer Operations Agent.

The response generator is a communication layer, not a decision layer: it
may phrase the outcome of a completed (or paused) case naturally, but it
never decides intent, policy, routing, order selection, approval, or
execution results - those facts already exist in structured workflow state
by the time this module is involved. `ResponseContext` is the narrow,
JSON-friendly, PII-free subset of that state the model is allowed to see;
`build_response_context` builds it from already-extracted primitives (the
caller, not this module, is responsible for picking e.g. which single order
is relevant) so unrelated data structurally cannot leak in.
"""

from __future__ import annotations

from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from customer_ops.classifier import resolve_model
from customer_ops.models import OrderStatus
from customer_ops.state import CaseRoute, HumanDecision, Intent, PolicyCode, PolicyOutcome, WorkflowStatus

RESPONSE_GENERATION_INSTRUCTIONS = """
You are the final response-writing step of the Mercora customer operations
workflow. You receive a JSON object with the validated facts of a completed
(or paused) case. Write exactly one short, natural customer-facing message
based ONLY on those facts.

Language:
- Respond in Spanish by default.
- If customer_message is clearly written in English, respond in English
  instead.
- Never add a language selector or ask which language to use.
- Never translate order IDs, action type names, or other operational
  identifiers - keep them exactly as given.

Grounding rules - follow strictly:
- Use ONLY the facts supplied in the JSON context. Never invent dates,
  amounts, addresses, shipping estimates, compensation, policies, or
  support promises that are not present in the context.
- Never claim an action succeeded unless action_result.success is true.
- Never claim a refund or any sensitive action happened unless
  human_decision is "approved" AND action_result.success is true.
- If route is "clarification" and action_input is present with
  ready=false, ask specifically for the field(s) named in
  action_input.missing_fields (e.g. a new shipping address) - never invent
  a value for it.
- If route is "clarification" and action_input is absent, explain that the
  relevant order could not be identified and ask the customer to provide
  the order ID.
- If route is "blocked", explain the operation cannot be performed right
  now, grounded in policy_reason - never invent alternative compensation.
- If human_decision is "rejected", state that the proposed operation was
  not performed because it was not approved - never identify or blame a
  reviewer.
- If intent is "other" or policy_outcome is "not_applicable", respond
  neutrally that the request does not map to a currently supported
  operational flow - do not pretend it was resolved.
- policy_code is for your own understanding only - never output the raw
  code string to the customer; policy_reason is already customer-safe.
- Never mention LangGraph, OpenAI, internal architecture, or any hidden
  reasoning.
- This is a simulated/demo system - never imply real money moved or a real
  external system was contacted.
- Be concise and operationally clear.

Return only the structured response. Do not include reasoning, rationale,
confidence, or citations.
""".strip()


class OrderSummary(BaseModel):
    """A minimal, non-PII summary of one order for response grounding.

    Deliberately narrow: order ID, status, and tracking number only - never
    shipping address, payment status, totals, or purchased items.
    """

    order_id: str
    status: OrderStatus
    tracking_number: str | None = None


class ResponseContext(BaseModel):
    """The minimal, JSON-friendly facts the response generator may ground on.

    Deliberately narrow - never the full graph state, never a full
    customer/order record, never the audit log or checkpointer data.
    """

    customer_message: str
    intent: Intent | None = None
    route: CaseRoute | None = None
    workflow_status: WorkflowStatus | None = None
    selected_order_id: str | None = None
    order_summary: OrderSummary | None = None
    policy_outcome: PolicyOutcome | None = None
    policy_code: PolicyCode | None = None
    policy_reason: str | None = None
    proposed_action: dict[str, object] | None = None
    action_input: dict[str, object] | None = None
    human_decision: HumanDecision | None = None
    action_result: dict[str, object] | None = None


def build_response_context(
    customer_message: str,
    *,
    intent: Intent | None = None,
    route: CaseRoute | None = None,
    workflow_status: WorkflowStatus | None = None,
    selected_order_id: str | None = None,
    order: dict[str, object] | None = None,
    policy_assessment: dict[str, object] | None = None,
    proposed_action: dict[str, object] | None = None,
    action_input: dict[str, object] | None = None,
    human_decision: HumanDecision | None = None,
    action_result: dict[str, object] | None = None,
) -> ResponseContext:
    """Build the minimal `ResponseContext` from already-extracted facts.

    `order` should be the single already-selected order dict (e.g. looked
    up from `order_context` by the caller) - passing only one order, never
    the full `order_context`, is what keeps unrelated orders out of the
    context structurally, not just by convention. `policy_assessment`
    supplies `policy_outcome`/`policy_code`/`policy_reason`.
    """
    order_summary = None
    if order is not None:
        order_summary = OrderSummary(
            order_id=order["order_id"],
            status=order["status"],
            tracking_number=order.get("tracking_number"),
        )

    policy_assessment = policy_assessment or {}

    return ResponseContext(
        customer_message=customer_message,
        intent=intent,
        route=route,
        workflow_status=workflow_status,
        selected_order_id=selected_order_id,
        order_summary=order_summary,
        policy_outcome=policy_assessment.get("outcome"),
        policy_code=policy_assessment.get("policy_code"),
        policy_reason=policy_assessment.get("reason"),
        proposed_action=proposed_action,
        action_input=action_input,
        human_decision=human_decision,
        action_result=action_result,
    )


class CustomerResponse(BaseModel):
    """The final customer-facing message - phrasing only, never a decision.

    Contains only the message text - no reasoning, rationale, confidence,
    chain-of-thought, or citations.
    """

    message: str = Field(min_length=1)


class CustomerResponseGenerator(Protocol):
    """Interface the final-response node depends on.

    Keeps the LangGraph node decoupled from the OpenAI SDK: production code
    injects `OpenAICustomerResponseGenerator`, tests inject a fake.
    """

    def generate(self, response_context: ResponseContext) -> CustomerResponse: ...


class ResponseGenerationError(Exception):
    """Raised when response generation fails for a technical reason.

    A workflow that cannot produce a validated response must fail loudly,
    not silently fall back to a default/fake "completed" message.
    """


class OpenAICustomerResponseGenerator:
    """`CustomerResponseGenerator` backed by the OpenAI Responses API.

    Uses Structured Outputs, mirroring `OpenAIRequestClassifier` and
    `OpenAIActionInputExtractor`.
    """

    def __init__(self, client: OpenAI | None = None, model: str | None = None) -> None:
        if client is None:
            load_dotenv()
            client = OpenAI()
        self._client = client
        self._model = resolve_model(model)

    def generate(self, response_context: ResponseContext) -> CustomerResponse:
        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=RESPONSE_GENERATION_INSTRUCTIONS,
                input=response_context.model_dump_json(),
                text_format=CustomerResponse,
                store=False,
            )
        except (OpenAIError, ValidationError) as exc:
            raise ResponseGenerationError(f"Response generation failed: {exc}") from exc

        result = response.output_parsed
        if result is None:
            raise ResponseGenerationError(
                "Response generation did not contain a parsed structured result."
            )
        return result
