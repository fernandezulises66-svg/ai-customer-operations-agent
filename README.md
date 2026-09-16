# AI Customer Operations Agent

**Status: intake + classification + context loading + deterministic order
resolution, policy evaluation, conditional routing, and structured action
proposals (Iteration 5).** This is a portfolio project and is not production
software.

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
  n8n. Conditional edges (`add_conditional_edges`) dispatch on structured
  state, not an if/else buried inside a single node.
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

## Current implementation (Iteration 5: Conditional Routing and Action Proposal)

The architecture now demonstrates the full separation of concerns the
project exists to show:

- **LLM**: intent + urgency interpretation only (`customer_ops/classifier.py`).
- **Deterministic data store**: customer/order facts (`tools/customer_data.py`).
- **Deterministic resolver**: which order a request refers to
  (`customer_ops/order_resolution.py`).
- **Deterministic policy engine**: operational eligibility
  (`customer_ops/policies.py`).
- **Deterministic router**: which workflow branch follows from that policy
  result (`customer_ops/routing.py`).
- **Structured action proposals**: what a future action *would* do, not
  execution (`customer_ops/action_proposal.py`).

Implemented now:

- Typed workflow state (`customer_ops/state.py`): adds `route`, controlled
  `CaseRoute` and `ActionType` vocabularies, and new `WorkflowStatus` values
  (`clarification_required`, `information_ready`, `blocked`; reuses
  `action_proposed` and `awaiting_approval`).
- Deterministic intake validation, structured intent/urgency classification,
  deterministic context loading, order resolution, and policy evaluation -
  semantically unchanged since Iterations 1-4.
- **A genuine LangGraph conditional edge** after `evaluate_policy`
  (`graph.add_conditional_edges(...)`, not an if/else buried in one node):
  routes to one of five terminal branches based only on structured state
  (`intent`, `order_resolution`, `policy_assessment`) via
  `determine_case_route` - never an LLM call, and never a fabricated branch
  for inconsistent state (`RoutingError`).
  - `needs_clarification` -> **clarification**: the order remains
    unidentified; nothing is guessed.
  - `information_only` (e.g. order_status) and `not_applicable` (intent
    "other") both -> **information**: a direct, non-operational answer.
  - `eligible` (e.g. cancel_order, address_change) -> **action**: a safe
    action is proposed only.
  - `review_required` (e.g. refund_request, billing_issue, product_issue)
    -> **approval**: a sensitive action is proposed and marked as requiring
    human approval - no interrupt or decision yet.
  - `blocked` -> **blocked**: no action is proposed; not automatically
    escalated.
- **Structured `ProposedAction`** (`action_type`, `order_id`,
  `requires_human_approval`) mapped deterministically from intent - never
  execution, and never an invented payload (no new address, refund amount,
  or replacement item; that is a later iteration).
- LangGraph workflow:
  `START -> intake -> classify_request -> load_context -> resolve_order -> evaluate_policy -> (conditional) -> {clarification, information, propose_action, prepare_approval, blocked} -> END`,
  built via `build_customer_ops_graph(classifier=None, store=None)`.
- Observable workflow audit events for all stages, including one
  branch-specific event per routed case (`clarification`, `information`,
  `action_proposal`, `approval_required`, or `blocked`) - never full
  customer/order details or model reasoning.
- A minimal `app.py` placeholder entry point (no CLI, no OpenAI call).
- Unit tests for state contracts, models, the data store, the classifier,
  order resolution, policy evaluation, routing, action proposal, and full
  graph branch coverage - all running with no network access and no API
  key.

Planned later (not implemented yet):

- Simulated mutation tools (refunds, cancellations, address changes, etc.).
- Human-in-the-loop approval gate (`interrupt()` / `Command(resume=...)`).
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
│   ├── routing.py          # determine_case_route() - deterministic branch selection
│   ├── action_proposal.py  # ProposedAction, propose_action() - structured action intent
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
│   ├── test_routing.py
│   ├── test_action_proposal.py
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
