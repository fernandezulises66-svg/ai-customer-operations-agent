# AI Customer Operations Agent

**Status: intake + classification + read-only context loading (Iteration
3).** This is a portfolio project and is not production software.

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

## Current implementation (Iteration 3: Synthetic Data and Read-Only Tools)

Implemented now:

- Typed workflow state (`customer_ops/state.py`): `CustomerOpsState`,
  `AuditEvent`, `OrderContext`, and controlled `Literal` vocabularies for
  intent, urgency, workflow status, and human decision.
- Deterministic intake validation (`intake_node`): validates and normalizes
  the initial request (`request_id`, `customer_id`, `customer_message`),
  sets `workflow_status` to `"received"`, and appends one audit event.
- Structured intent/urgency classification (`customer_ops/classifier.py`):
  a single call to the OpenAI Responses API **Structured Outputs**
  mechanism (`client.responses.parse(..., text_format=ClassificationDecision)`),
  validated against `ClassificationDecision` - no free-form JSON parsing, no
  chain-of-thought, no confidence score. Injectable via the
  `RequestClassifier` protocol. Unchanged since Iteration 2.
- A **synthetic Mercora customer dataset** (`data/customers.json`) and
  **order dataset** (`data/orders.json`): fabricated records only, using
  `@example.com` addresses, covering active/suspended accounts,
  standard/premium tiers, every order status, and failed/refunded payments.
- **Validated Pydantic operational records** (`customer_ops/models.py`):
  `CustomerRecord`, `OrderItem`, `OrderRecord` - structural data contracts
  (non-empty identifiers, positive quantities, non-negative amounts,
  controlled statuses/currency), not business-policy logic.
- A **read-only JSON operational store** (`tools/customer_data.py`):
  `JsonCustomerOperationsStore` loads and validates the fixtures once and
  serves `get_customer`, `list_orders_for_customer`, and `get_order` lookups.
  It never writes to the fixture files. Unknown customers/orders raise
  `CustomerNotFoundError`/`OrderNotFoundError` explicitly rather than
  fabricating a record; a known customer with no orders returns an empty
  list, never an error.
- **Deterministic customer/order context loading** (`load_context` node):
  retrieves the customer and their orders through the injectable
  `CustomerOperationsStore` interface and serializes them into
  `customer_context`/`order_context` as plain JSON-friendly data (never
  Pydantic objects) before entering graph state. The LLM never generates or
  infers customer identity, order status, totals, addresses, tracking
  numbers, or purchased items - those facts come exclusively from this
  deterministic dataset.
- Injectable classifier **and** data-store dependencies: production defaults
  to `OpenAIRequestClassifier` + `JsonCustomerOperationsStore`; tests inject
  fakes for both, so the full suite makes no network calls.
- LangGraph workflow:
  `START -> intake -> classify_request -> load_context -> END`, built via
  `build_customer_ops_graph(classifier=None, store=None)`.
- Observable workflow audit events for all three stages (`intake`,
  `classification`, `context_loading`) - the audit log records what
  happened, never full customer/order details or model reasoning.
- A minimal `app.py` placeholder entry point (no CLI, no OpenAI call).
- Unit tests for state contracts, models, the data store, the classifier,
  and graph behavior - all running with no network access and no API key.

Planned later (not implemented yet):

- Policy engine / policy evaluation.
- Case-specific reasoning over the loaded context.
- Conditional routing between workflow branches.
- Action proposal.
- Simulated mutation tools (refunds, cancellations, address changes, etc.).
- Human-in-the-loop approval gate.
- Checkpointing / persistence.
- Action execution.
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
│   ├── state.py          # CustomerOpsState, AuditEvent, OrderContext, controlled vocabularies
│   ├── models.py         # CustomerRecord, OrderItem, OrderRecord (Pydantic data contracts)
│   ├── classifier.py     # ClassificationDecision, RequestClassifier, OpenAIRequestClassifier
│   └── graph.py          # intake/classify_request/load_context nodes, build_customer_ops_graph()
│
├── tools/
│   ├── __init__.py
│   └── customer_data.py  # CustomerOperationsStore, JsonCustomerOperationsStore (read-only)
│
├── data/
│   ├── customers.json    # synthetic Mercora customers
│   └── orders.json       # synthetic Mercora orders
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
