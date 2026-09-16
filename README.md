# AI Customer Operations Agent

**Status: intake + classification + context loading + deterministic order
resolution and policy evaluation (Iteration 4).** This is a portfolio
project and is not production software.

> Mercora is a fictional e-commerce company invented for this project. All
> customers, orders, policies, payments, addresses, and tickets referenced
> anywhere in this repository are synthetic and do not represent any real
> business or individual. All data in `data/` is fabricated for this project.

## Long-term goal

Build a portfolio-grade customer operations agent that receives a customer
request and orchestrates a multi-step workflow able to:

- classify the request (intent + urgency)
- retrieve customer information
- retrieve order information
- evaluate company policies
- decide whether the case can be answered directly
- propose operational actions
- execute safe, simulated actions
- pause for human approval before sensitive actions
- escalate cases when necessary
- generate a final customer-facing response
- expose a structured audit trail of the workflow

The project is intentionally distinct from an SQL data agent or a RAG
knowledge assistant. Its purpose is to demonstrate: LangGraph, stateful
workflows, conditional routing, tool usage, structured LLM outputs,
human-in-the-loop, checkpointing, safe action execution, evaluation, and
observability.

## Planned architecture

- **LangGraph** (`StateGraph`) orchestrates workflow state and conditional
  routing directly — no LangChain agents, `create_agent`, CrewAI, AutoGen, or
  n8n.
- The **OpenAI SDK** is called directly inside specific graph nodes that need
  LLM reasoning, using the Responses API's Structured Outputs mechanism
  rather than free-form text parsing; LangGraph orchestrates, it does not
  hide business logic.
- **Tools** (`tools/`) are narrow, independently testable interfaces used by
  graph nodes - e.g. the read-only `CustomerOperationsStore`. Read-only and
  mutating tools are kept clearly separate; no mutating tools exist yet.
- Sensitive/high-impact simulated actions require a **human-in-the-loop**
  approval step before execution.
- The workflow state is checkpoint-friendly and produces a **structured audit
  trail** of observable events (not model reasoning).

## Current implementation (Iteration 4: Order Resolution and Policy Evaluation)

This iteration establishes a core architecture principle, now fully in
place: the LLM interprets intent and urgency only; everything else is
deterministic code.

- **LLM**: intent + urgency interpretation (`customer_ops/classifier.py`).
- **Deterministic data store**: customer/order facts
  (`tools/customer_data.py`).
- **Deterministic resolver**: which order a request refers to
  (`customer_ops/order_resolution.py`).
- **Deterministic policy engine**: operational eligibility
  (`customer_ops/policies.py`).

Implemented now:

- Typed workflow state (`customer_ops/state.py`): `CustomerOpsState`,
  `AuditEvent`, `OrderContext`, and controlled `Literal` vocabularies for
  intent, urgency, workflow status, human decision, order-resolution status,
  policy outcome, and policy code.
- Deterministic intake validation, structured intent/urgency classification,
  and deterministic customer/order context loading - unchanged since
  Iterations 1-3.
- **Deterministic order-reference resolution** (`customer_ops/order_resolution.py`,
  `resolve_order`): decides which (if any) of the customer's own orders a
  request refers to, using only `intent`, `customer_message`, and the
  already customer-scoped `order_context` - never OpenAI. An order is
  selected only when exactly one known order ID appears explicitly in the
  message (case-insensitive, never fuzzy); zero or multiple matches yield
  conservative `needs_clarification` rather than a guess. Never infers from
  product names, dates, amounts, or vague references like "my latest
  order".
- **Explicit deterministic Mercora policy rules** (`customer_ops/policies.py`,
  `evaluate_policy`): hand-written business rules - not a RAG system, not an
  LLM call - covering `order_status`, `cancel_order`, `address_change`,
  `refund_request`, `billing_issue`, `product_issue`, and `other`. Produces
  a structured `PolicyAssessment` (`outcome`, `policy_code`,
  `requires_human_approval`, `reason`) with stable machine-readable policy
  codes. `requires_human_approval` is a policy *output* only - no approval
  step exists yet, and no action is proposed or executed.
- Conservative ambiguity handling throughout: an order that can't be
  resolved unambiguously, or a `selected_order_id` that doesn't appear in
  the customer's own `order_context`, is treated as a `needs_clarification`
  business outcome or a raised domain error (`OrderResolutionError`,
  `PolicyEvaluationError`) - never silently guessed or defaulted.
- LangGraph workflow:
  `START -> intake -> classify_request -> load_context -> resolve_order -> evaluate_policy -> END`,
  still linear (conditional routing is a later iteration), built via
  `build_customer_ops_graph(classifier=None, store=None)`.
- Observable workflow audit events for all five stages (`intake`,
  `classification`, `context_loading`, `order_resolution`,
  `policy_evaluation`) - the audit log records what happened, never full
  customer/order details or model reasoning.
- A minimal `app.py` placeholder entry point (no CLI, no OpenAI call).
- Unit tests for state contracts, models, the data store, the classifier,
  order resolution, policy evaluation, and graph behavior - all running
  with no network access and no API key.

Planned later (not implemented yet):

- Conditional routing between workflow branches.
- Action proposal.
- Simulated mutation tools (refunds, cancellations, address changes, etc.).
- Human-in-the-loop approval gate (interrupts).
- Checkpointing / persistence.
- Action execution.
- Final customer-facing response generation.
- Evaluation harness (`evals/`).
- Streamlit UI.
- Deployment.

## Tech stack

- Python 3.13+
- [LangGraph](https://github.com/langchain-ai/langgraph) `>=1.1,<2.0` for
  orchestration
- [OpenAI SDK](https://github.com/openai/openai-python) `>=3.0,<4.0`, used via
  the Responses API structured-output path (`responses.parse`)
- [Pydantic](https://github.com/pydantic/pydantic) `>=2.0,<3.0`
- [python-dotenv](https://github.com/theskumar/python-dotenv) for local
  environment configuration
- [pytest](https://github.com/pytest-dev/pytest) for testing

## Project structure

```
ai-customer-operations-agent/
│
├── customer_ops/
│   ├── __init__.py
│   ├── state.py            # CustomerOpsState, AuditEvent, OrderContext, controlled vocabularies
│   ├── models.py           # CustomerRecord, OrderItem, OrderRecord (Pydantic data contracts)
│   ├── classifier.py       # ClassificationDecision, RequestClassifier, OpenAIRequestClassifier
│   ├── order_resolution.py # OrderResolution, resolve_order() - deterministic order selection
│   ├── policies.py         # PolicyAssessment, evaluate_policy() - deterministic Mercora rules
│   └── graph.py            # all graph nodes, build_customer_ops_graph()
│
├── tools/
│   ├── __init__.py
│   └── customer_data.py    # CustomerOperationsStore, JsonCustomerOperationsStore (read-only)
│
├── data/
│   ├── customers.json      # synthetic Mercora customers
│   └── orders.json         # synthetic Mercora orders
│
├── evals/
│   └── __init__.py      # empty in Iteration 1
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_state.py
│   ├── test_models.py
│   ├── test_customer_data.py
│   ├── test_classifier.py
│   ├── test_order_resolution.py
│   ├── test_policies.py
│   └── test_graph.py
│
├── app.py                # minimal placeholder entry point
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

Requires Python 3.13+. Run from PowerShell in the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to run the graph for
real (i.e. invoke it without injecting a fake classifier). No key is required
to run the test suite — tests always inject a fake `RequestClassifier` and a
fake/in-memory `CustomerOperationsStore`, and make no network calls:

```powershell
Copy-Item .env.example .env
```

## Running the placeholder entry point

```powershell
.\.venv\Scripts\python app.py
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```
